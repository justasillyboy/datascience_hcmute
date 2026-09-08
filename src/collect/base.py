"""Lớp cơ sở cho mọi scraper của đồ án.

Thiết kế theo OOP (bài giảng Buổi 1): `BaseScraper` là lớp trừu tượng định nghĩa
hợp đồng chung — phiên HTTP, giới hạn tốc độ, cache xuống đĩa, ghi log — còn các
lớp con chỉ cần cài đặt `fetch()`.

Ba nguyên tắc đạo đức được cài cứng ở đây, không phải để lớp con tự nhớ:
  1. Mỗi request cách nhau tối thiểu `min_interval` giây.
  2. Mọi response được cache xuống đĩa; chạy lại đọc cache, không đập vào server.
  3. User-Agent khai rõ đây là đồ án môn học.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Tiki chặn User-Agent lạ ở tầng CDN (curl -> 403), nên phải giả lập trình duyệt
# thật. Chuỗi khai báo mục đích học thuật được đưa vào header `X-Purpose` để vẫn
# minh bạch với phía máy chủ.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
COURSE_TAG = "HCMUTE Data Science course project (academic, non-commercial)"

DEFAULT_MIN_INTERVAL = 1.0
DEFAULT_TIMEOUT = 20
DEFAULT_MAX_RETRY = 3
RETRY_BACKOFF = 2.0


class RateLimiter:
    """Bảo đảm hai request liên tiếp cách nhau ít nhất `min_interval` giây."""

    def __init__(self, min_interval: float = DEFAULT_MIN_INTERVAL) -> None:
        self.min_interval = min_interval
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


class DiskCache:
    """Cache JSON xuống đĩa, khoá bằng hash của (url, params).

    Mục đích không phải tăng tốc mà là **đạo đức**: cào đúng một lần. Mọi lần
    chạy lại notebook/pipeline đều đọc từ đây.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.write_errors = 0
        self._counter = 0

    def _key(self, url: str, params: dict[str, Any] | None) -> str:
        payload = json.dumps([url, params or {}], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def path_for(self, url: str, params: dict[str, Any] | None) -> Path:
        key = self._key(url, params)
        # Chia thư mục con theo 2 ký tự đầu để không dồn hàng vạn file một chỗ.
        return self.root / key[:2] / f"{key}.json"

    def get(self, url: str, params: dict[str, Any] | None) -> Any | None:
        path = self.path_for(url, params)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with path.open(encoding="utf-8") as fh:
                self.hits += 1
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            # Cache hỏng thì coi như miss, nhưng phải kêu lên chứ không nuốt lỗi.
            logger.warning("Cache hỏng tại %s (%s) — sẽ tải lại", path, exc)
            self.misses += 1
            return None

    def set(self, url: str, params: dict[str, Any] | None, value: Any) -> bool:
        """Ghi cache. Trả về True nếu thành công.

        Ghi cache là **tối ưu hoá, không phải dữ liệu**. Nếu ghi hỏng thì chỉ mất
        một lần tiết kiệm request, chứ tuyệt đối không được làm mất response đã
        tải về thành công — nên mọi lỗi ở đây được nuốt lại thành cảnh báo.
        Trong mẻ cào 2026-09-07, việc để lỗi này ném ra ngoài đã khiến 179/1717
        sản phẩm bị bỏ oan dù dữ liệu đã tải về đầy đủ.
        """
        path = self.path_for(url, params)
        # Tên tạm phải duy nhất theo tiến trình + bộ đếm: nếu hai lần ghi cùng
        # trỏ vào một tên `.tmp`, lần replace thứ hai sẽ không tìm thấy file.
        self._counter += 1
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{self._counter}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False)
            tmp.replace(path)  # ghi nguyên tử: tránh file cache dở dang
            return True
        except OSError as exc:
            logger.warning("Không ghi được cache %s (%s) — bỏ qua, dữ liệu vẫn dùng được", path, exc)
            self.write_errors += 1
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False


class ScraperError(RuntimeError):
    """Lỗi không phục hồi được khi cào dữ liệu."""


class BaseScraper(ABC):
    """Lớp trừu tượng: phiên HTTP + rate limit + cache + retry.

    Lớp con cài đặt `fetch()` để mô tả *cào cái gì*; toàn bộ phần *cào như thế
    nào cho lịch sự* đã nằm sẵn ở đây.
    """

    #: Lớp con phải khai báo, dùng cho log và cho thư mục cache.
    name: str = "base"

    def __init__(
        self,
        cache_dir: Path | str,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retry: int = DEFAULT_MAX_RETRY,
    ) -> None:
        self.cache = DiskCache(Path(cache_dir) / self.name)
        self.limiter = RateLimiter(min_interval)
        self.timeout = timeout
        self.max_retry = max_retry
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": BROWSER_UA,
                "X-Purpose": COURSE_TAG,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
            }
        )
        self.request_count = 0

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Lấy JSON, ưu tiên cache. Retry có backoff khi lỗi mạng/5xx."""
        cached = self.cache.get(url, params)
        if cached is not None:
            return cached

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retry + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                self.request_count += 1
                if resp.status_code == 200:
                    data = resp.json()
                    self.cache.set(url, params, data)
                    return data
                if resp.status_code == 404:
                    # 404 là câu trả lời hợp lệ (sản phẩm đã gỡ) — cache lại để
                    # không hỏi lại lần sau.
                    self.cache.set(url, params, None)
                    return None
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = RETRY_BACKOFF**attempt
                    logger.warning(
                        "[%s] HTTP %s, thử lại sau %.1fs (lần %d/%d) %s",
                        self.name, resp.status_code, wait, attempt, self.max_retry, url,
                    )
                    time.sleep(wait)
                    last_exc = ScraperError(f"HTTP {resp.status_code}")
                    continue
                raise ScraperError(f"HTTP {resp.status_code} khi gọi {url}")
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_exc = exc
                wait = RETRY_BACKOFF**attempt
                logger.warning(
                    "[%s] %s: %s — thử lại sau %.1fs (lần %d/%d)",
                    self.name, type(exc).__name__, exc, wait, attempt, self.max_retry,
                )
                time.sleep(wait)

        raise ScraperError(f"Thất bại sau {self.max_retry} lần với {url}") from last_exc

    @abstractmethod
    def fetch(self, *args: Any, **kwargs: Any) -> Any:
        """Cào một đơn vị dữ liệu. Lớp con định nghĩa ý nghĩa cụ thể."""

    def stats(self) -> dict[str, int]:
        return {
            "requests": self.request_count,
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
            "cache_write_errors": self.cache.write_errors,
        }
