"""Scraper cho Tiki: danh sách sản phẩm và review có phân tầng theo sao.

Điểm mấu chốt của module này là `TikiReviewScraper`. Cào review theo cách mặc
định cho ra mẫu 96% năm sao (xem `docs/FEASIBILITY.md` §3.1) — biến mục tiêu gần
như không có phương sai và mọi phân tích sau đó đều vô nghĩa.

Cách xử lý: **lấy mẫu phân tầng với kích thước tầng đã biết**. Mỗi response của
API đều kèm histogram `stars` ở cấp quần thể (đếm thật, không thiên lệch), và API
nhận `sort=stars|N` để lọc đúng một tầng. Ta vét cạn các tầng hiếm, lấy mẫu tầng
đông, rồi gán trọng số `w_h = N_h / n_h` để mọi ước lượng quay về đúng quần thể.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterator

from .base import BaseScraper

logger = logging.getLogger(__name__)

LISTING_URL = "https://tiki.vn/api/personalish/v1/blocks/listings"
PRODUCT_URL = "https://tiki.vn/api/v2/products/{product_id}"
REVIEW_URL = "https://tiki.vn/api/v2/reviews"

REVIEW_PAGE_SIZE = 20  # API trả tối đa 20 review/trang
LISTING_PAGE_SIZE = 40
STARS = (1, 2, 3, 4, 5)

#: Ngành hàng khảo sát. Chọn trải rộng để Gap không bị chi phối bởi một loại
#: hàng duy nhất (sách giao khác điện lạnh).
CATEGORIES: dict[int, str] = {
    1789: "Điện thoại - Máy tính bảng",
    1846: "Điện gia dụng",
    1815: "Làm đẹp - Sức khỏe",
    8322: "Nhà cửa - Đời sống",
    1883: "Sách tiếng Việt",
    915: "Nhà sách tiki",
    4384: "Thời trang nữ",
    17166: "Bách hóa online",
    1520: "Mẹ và bé",
    1801: "Thể thao - Dã ngoại",
}


class TikiListingScraper(BaseScraper):
    """Cào danh sách sản phẩm theo ngành hàng."""

    name = "tiki_listing"

    def fetch(self, category_id: int, page: int = 1, sort: str = "top_seller") -> list[dict[str, Any]]:
        data = self.get_json(
            LISTING_URL,
            {"limit": LISTING_PAGE_SIZE, "category": category_id, "page": page, "sort": sort},
        )
        if not data:
            return []
        return data.get("data", []) or []

    def iter_products(
        self,
        category_id: int,
        max_pages: int = 10,
        min_reviews: int = 5,
        sort: str = "top_seller",
    ) -> Iterator[dict[str, Any]]:
        """Duyệt sản phẩm của một ngành, chỉ giữ sản phẩm đủ review để phân tầng."""
        seen: set[int] = set()
        for page in range(1, max_pages + 1):
            items = self.fetch(category_id, page=page, sort=sort)
            if not items:
                break
            for item in items:
                pid = item.get("id")
                if pid is None or pid in seen:
                    continue
                seen.add(pid)
                if (item.get("review_count") or 0) < min_reviews:
                    continue
                yield self._normalise_listing(item, category_id)

    @staticmethod
    def _normalise_listing(item: dict[str, Any], category_id: int) -> dict[str, Any]:
        # `quantity_sold` là dict {"text": "Đã bán 1008", "value": 1008} hoặc None.
        sold = item.get("quantity_sold")
        sold_value = sold.get("value") if isinstance(sold, dict) else None
        seller = item.get("seller") or {}
        return {
            "product_id": item.get("id"),
            "spid": item.get("seller_product_id"),
            "seller_id": item.get("seller_id"),
            "seller_name": seller.get("name") if isinstance(seller, dict) else None,
            "category_id": category_id,
            "category_name": CATEGORIES.get(category_id),
            "name": item.get("name"),
            "brand_name": item.get("brand_name"),
            "price": item.get("price"),
            "list_price": item.get("list_price"),
            "original_price": item.get("original_price"),
            "discount_rate": item.get("discount_rate"),
            "rating_average": item.get("rating_average"),
            "review_count": item.get("review_count"),
            "quantity_sold": sold_value,
            "url_key": item.get("url_key"),
        }


class TikiReviewScraper(BaseScraper):
    """Cào review theo **lấy mẫu phân tầng theo sao**.

    Quy trình cho mỗi sản phẩm:
      1. `star_histogram()` — 1 request, lấy `N_h` thật của từng tầng.
      2. `plan_strata()` — quyết định lấy bao nhiêu review ở mỗi tầng.
      3. `fetch()` — cào theo kế hoạch, gắn `stratum` và `weight = N_h / n_h`.
    """

    name = "tiki_review"

    def __init__(self, *args: Any, sample_cap: int = 40, exhaust_below: int = 60, **kwargs: Any) -> None:
        """
        Args:
            sample_cap: số review tối đa lấy ở một tầng đông (mặc định 5★).
            exhaust_below: tầng có `N_h` nhỏ hơn ngưỡng này thì vét cạn.
        """
        super().__init__(*args, **kwargs)
        self.sample_cap = sample_cap
        self.exhaust_below = exhaust_below

    def _request_reviews(
        self, product_id: int, seller_id: int, page: int = 1, sort: str | None = None
    ) -> dict[str, Any] | None:
        params: dict[str, Any] = {
            "product_id": product_id,
            "seller_id": seller_id,
            "limit": REVIEW_PAGE_SIZE,
            "page": page,
            "include": "comments,contribute_info,attribute_vote_summary",
        }
        if sort:
            params["sort"] = sort
        return self.get_json(REVIEW_URL, params)

    def star_histogram(self, product_id: int, seller_id: int) -> dict[str, Any]:
        """Đếm review theo sao ở **cấp quần thể** — nền tảng của trọng số phân tầng."""
        data = self._request_reviews(product_id, seller_id, page=1)
        if not data:
            return {"total": 0, "counts": {s: 0 for s in STARS}}
        stars = data.get("stars") or {}
        counts = {s: int((stars.get(str(s)) or {}).get("count", 0) or 0) for s in STARS}
        return {"total": int(data.get("reviews_count") or 0), "counts": counts}

    def plan_strata(self, counts: dict[int, int]) -> dict[int, int]:
        """Quyết định số review cần lấy ở mỗi tầng.

        Tầng hiếm (1–3★, hoặc bất kỳ tầng nào nhỏ hơn `exhaust_below`) được vét
        cạn vì chúng chính là tín hiệu ta cần và chi phí rất rẻ. Tầng đông bị
        giới hạn ở `sample_cap` để không tốn hàng nghìn request cho thông tin lặp.
        """
        plan: dict[int, int] = {}
        for star in STARS:
            available = counts.get(star, 0)
            if available == 0:
                continue
            if star <= 3 or available <= self.exhaust_below:
                plan[star] = available
            else:
                plan[star] = min(available, self.sample_cap)
        return plan

    def fetch(self, product_id: int, seller_id: int) -> list[dict[str, Any]]:
        """Cào review phân tầng cho một sản phẩm, đã gắn trọng số."""
        hist = self.star_histogram(product_id, seller_id)
        counts = hist["counts"]
        if hist["total"] == 0:
            return []

        plan = self.plan_strata(counts)
        rows: list[dict[str, Any]] = []
        collected: dict[int, int] = {}

        for star, target in plan.items():
            got = self._fetch_stratum(product_id, seller_id, star, target)
            collected[star] = len(got)
            rows.extend(got)

        # Trọng số chỉ tính được sau khi biết n_h thực nhận (có thể nhỏ hơn kế
        # hoạch nếu API trả thiếu).
        for row in rows:
            star = row["rating"]
            n_h = collected.get(star, 0)
            N_h = counts.get(star, 0)
            row["stratum"] = star
            row["stratum_size"] = N_h
            row["stratum_sampled"] = n_h
            row["weight"] = (N_h / n_h) if n_h else 0.0

        return rows

    def _fetch_stratum(
        self, product_id: int, seller_id: int, star: int, target: int
    ) -> list[dict[str, Any]]:
        """Cào đúng một tầng sao, tối đa `target` review."""
        out: list[dict[str, Any]] = []
        pages = math.ceil(target / REVIEW_PAGE_SIZE)
        for page in range(1, pages + 1):
            data = self._request_reviews(product_id, seller_id, page=page, sort=f"stars|{star}")
            if not data:
                break
            batch = data.get("data") or []
            if not batch:
                break
            for raw in batch:
                # API đã lọc theo tầng, nhưng vẫn kiểm lại — không tin dữ liệu
                # ngoài, kể cả khi nó vừa trả đúng ở lần thử trước.
                if raw.get("rating") != star:
                    logger.warning(
                        "pid=%s tầng %s trả về rating=%s — bỏ qua",
                        product_id, star, raw.get("rating"),
                    )
                    continue
                out.append(self._normalise_review(raw, product_id, seller_id))
                if len(out) >= target:
                    return out
        return out

    @staticmethod
    def _normalise_review(raw: dict[str, Any], product_id: int, seller_id: int) -> dict[str, Any]:
        """Làm phẳng review thô thành một dòng bảng.

        Cố ý **không** tính lead time ở đây: collect chỉ ghi lại sự thật quan
        sát được, mọi phép biến đổi thuộc về bước cleaning để còn kiểm toán được.
        """
        timeline = raw.get("timeline") or {}
        created_by = raw.get("created_by") or {}
        seller = raw.get("seller") or {}
        delivery = {q["question"]: q["option"] for q in (raw.get("delivery_rating") or []) if isinstance(q, dict)}

        return {
            "review_id": raw.get("id"),
            "product_id": product_id,
            "spid": raw.get("spid"),
            "seller_id": seller_id,
            "seller_name": seller.get("name"),
            "customer_id": raw.get("customer_id"),
            "rating": raw.get("rating"),
            "title": raw.get("title"),
            "content": raw.get("content"),
            "status": raw.get("status"),
            "thank_count": raw.get("thank_count"),
            "is_photo": raw.get("is_photo"),
            "n_images": len(raw.get("images") or []),
            "attributes": raw.get("attributes"),
            # --- các mốc thời gian: xương sống của đồ án ---
            "purchased_at": created_by.get("purchased_at"),
            "delivery_date": timeline.get("delivery_date"),
            "review_created_date": timeline.get("review_created_date"),
            "created_at": raw.get("created_at"),
            "customer_joined": created_by.get("created_time"),
            "customer_purchased_flag": created_by.get("purchased"),
            # --- nhãn SLA do khách tự báo ---
            "dr_thoi_gian": delivery.get("Thời gian giao hàng?"),
            "dr_shipper": delivery.get("Thái độ của shipper?"),
            "dr_gio_giao": delivery.get("Giờ giao hàng?"),
            "dr_dong_goi": delivery.get("Cách đóng gói sản phẩm?"),
            "has_delivery_rating": bool(delivery),
        }
