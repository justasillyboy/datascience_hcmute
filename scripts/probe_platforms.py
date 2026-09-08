"""Kiểm chứng khả năng thu thập dữ liệu trên các sàn TMĐT lớn ở Việt Nam.

Mục đích: trả lời câu hỏi *"vì sao chọn Tiki mà không phải sàn lớn hơn?"* bằng
**bằng chứng đo được**, không phải phỏng đoán.

Nguyên tắc đạo đức — đọc kỹ trước khi chạy lại:

  * **Chỉ đọc, không ghi.** Không đăng nhập, không gửi form, không chạm endpoint
    riêng tư (`/user/`, `/order/`, `/cart/`, `/checkout/`, `/me/`).
  * **Đúng một request cho mỗi phép thử.** Đây là khảo sát schema, không phải
    thu thập dữ liệu. Tổng cộng dưới 10 request.
  * **Không lách chặn.** Nếu sàn trả 403 / CAPTCHA / anti-bot, ta ghi nhận đó là
    *kết quả* và dừng. Cố vượt rào kiểm soát truy cập là vượt quá phạm vi một đồ
    án môn học, và cũng không phải thứ trình bày được trước hội đồng.
  * **User-Agent khai rõ mục đích học thuật** ở phần chú thích bên dưới.

Chạy:  python3 scripts/probe_platforms.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import requests

# UA trình duyệt là bắt buộc: các sàn chặn UA lạ ở tầng CDN nên nếu dùng UA riêng
# ta sẽ không phân biệt được "bị chặn vì là bot" với "endpoint không tồn tại".
# Mục đích học thuật được khai trong header X-Purpose bên dưới.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
COURSE_TAG = "academic-research; HCMUTE Python for Data Science; contact via course instructor"

DELAY_SECONDS = 2.0
TIMEOUT = 25

#: Các trường chứng minh đề tài khả thi. Thiếu `delivery_date` thì không tính
#: được Delivery Promise Gap, tức là không có đề tài.
#: Sản phẩm mốc để thử Tiki — chọn từ chính dữ liệu đã cào, đảm bảo có review
#: kèm đủ cả 4 trường. Nếu Tiki gỡ sản phẩm này, đổi sang mã khác có review.
TIKI_PROBE_PRODUCT_ID = 277397107
TIKI_PROBE_SELLER_ID = 1


def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": BROWSER_UA,
            "Accept-Language": "vi,en;q=0.9",
            "X-Purpose": COURSE_TAG,
        }
    )
    return s


def probe(name: str, url: str, session: requests.Session, **kwargs: Any) -> dict[str, Any]:
    """Gọi đúng một request và ghi lại kết quả thô."""
    time.sleep(DELAY_SECONDS)
    try:
        r = session.get(url, timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        return {"phép thử": name, "url": url, "trạng thái": "LỖI MẠNG",
                "chi tiết": f"{type(exc).__name__}: {exc}"}
    body = r.text
    return {
        "phép thử": name,
        "url": url,
        "http": r.status_code,
        "kích thước": len(body),
        "trích body": body[:220].replace("\n", " ").strip(),
        "raw": body,
    }


def check_tiki(s: requests.Session) -> list[dict[str, Any]]:
    """Tiki: xác nhận API mở VÀ có đủ trường bắt buộc."""
    # Sản phẩm mốc: phải là sản phẩm CÓ review thật, nếu không `data` rỗng và
    # mọi trường đều None — khi đó phép thử không chứng minh được gì.
    res = probe(
        "Tiki — API review công khai",
        "https://tiki.vn/api/v2/reviews",
        s,
        params={"product_id": TIKI_PROBE_PRODUCT_ID, "seller_id": TIKI_PROBE_SELLER_ID,
                "limit": 5, "page": 1,
                "include": "comments,contribute_info,attribute_vote_summary"},
    )
    raw = res.pop("raw", "")
    if res.get("http") != 200:
        res["kết luận"] = f"⚠️ HTTP {res.get('http')} — không kết luận được"
        return [res]

    try:
        data = json.loads(raw)
    except ValueError as exc:
        res["kết luận"] = f"⚠️ JSON không đọc được: {exc}"
        return [res]

    items = data.get("data") or []
    res["số review trả về"] = len(items)
    if not items:
        # Sản phẩm không có review -> phép thử VÔ HIỆU, không phải "thất bại".
        # Phân biệt hai thứ này là bắt buộc: gộp chúng lại chính là cách sinh ra
        # kết luận sai. Bản chạy đầu 2026-09-08 đã báo "✅ có đủ trường" trên một
        # sản phẩm 0 review — lỗi đó dẫn tới sửa lại hàm này.
        res["kết luận"] = "⚠️ PHÉP THỬ VÔ HIỆU — sản phẩm mốc không có review, đổi TIKI_PROBE_PRODUCT_ID"
        return [res]

    # Một trường được coi là CÓ nếu xuất hiện ở ít nhất một review trong mẫu thử.
    found = {
        "ngày đặt hàng (created_by.purchased_at)":
            any((it.get("created_by") or {}).get("purchased_at") is not None for it in items),
        "ngày giao thực tế (timeline.delivery_date)":
            any((it.get("timeline") or {}).get("delivery_date") is not None for it in items),
        "điểm hài lòng (rating)":
            any(it.get("rating") is not None for it in items),
        "nhãn khách tự báo (delivery_rating)":
            any(it.get("delivery_rating") for it in items),
    }
    res["trường tìm thấy"] = found

    # Kết luận phải SUY RA từ phép kiểm, không được viết cứng.
    thiếu = [k for k, v in found.items() if not v]
    res["kết luận"] = (
        "✅ DÙNG ĐƯỢC — có đủ trường bắt buộc" if not thiếu
        else f"❌ THIẾU TRƯỜNG: {', '.join(thiếu)}"
    )
    return [res]


def check_shopee(s: requests.Session) -> list[dict[str, Any]]:
    """Shopee: robots.txt cho phép /api/, nhưng WAF có chặn không?"""
    s.headers.update({"Referer": "https://shopee.vn/", "X-API-SOURCE": "pc"})
    res = probe(
        "Shopee — API review công khai",
        "https://shopee.vn/api/v4/item/get_ratings",
        s,
        params={"filter": 0, "flag": 1, "itemid": 21899666693,
                "limit": 3, "offset": 0, "shopid": 88201679, "type": 0},
    )
    raw = res.pop("raw", "")
    if res.get("http") == 403 or "redirect_to_error_page" in raw:
        res["kết luận"] = "❌ BỊ CHẶN — WAF trả 403, không lấy được payload review"
    elif res.get("http") == 200:
        res["kết luận"] = "⚠️ Truy cập được — cần kiểm tra lại schema"
    return [res]


def check_lazada(s: requests.Session) -> list[dict[str, Any]]:
    """Lazada: trang có trả dữ liệu sản phẩm phía server không?"""
    res = probe("Lazada — trang danh mục", "https://www.lazada.vn/catalog/?q=sach", s)
    raw = res.pop("raw", "")
    lower = raw.lower()
    res["số itemId nhúng sẵn"] = lower.count('"itemid"')
    res["dấu hiệu anti-bot"] = [k for k in ("baxia", "captcha", "punish", "x5secdata") if k in lower]
    if res.get("http") == 200 and res["số itemId nhúng sẵn"] == 0:
        res["kết luận"] = "❌ KHÔNG DÙNG ĐƯỢC — dữ liệu render bằng JS, có anti-bot Baxia"
    return [res]


def check_sendo(s: requests.Session) -> list[dict[str, Any]]:
    res = probe("Sendo — trang chủ", "https://www.sendo.vn/robots.txt", s)
    res.pop("raw", None)
    if res.get("trạng thái") == "LỖI MẠNG":
        res["kết luận"] = "❌ KHÔNG TRUY CẬP ĐƯỢC"
    return [res]


def main() -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"# Kiểm chứng khả năng thu thập — các sàn TMĐT Việt Nam")
    print(f"# Thời điểm chạy: {stamp}")
    print(f"# Nguyên tắc: chỉ đọc · 1 request/phép thử · không lách chặn\n")

    for label, fn in (("TIKI", check_tiki), ("SHOPEE", check_shopee),
                      ("LAZADA", check_lazada), ("SENDO", check_sendo)):
        print(f"{'=' * 72}\n{label}\n{'=' * 72}")
        # Session mới cho mỗi sàn: cookie của sàn này không được rò sang sàn kia.
        for res in fn(new_session()):
            for k, v in res.items():
                print(f"  {k:34s}: {v}")
            print()


if __name__ == "__main__":
    main()
