"""Danh sách biến cấm và kiểm toán leakage — task C3, tầng 3 của khung chứng minh.

**Thời điểm dự báo** của toàn bộ Track C được chốt là *lúc đơn hàng được giao
tới tay khách, trước khi khách viết review*. Một biến chỉ được làm feature nếu
giá trị của nó đã biết tại thời điểm đó. Mọi biến trong `FORBIDDEN` vi phạm
nguyên tắc này theo một trong bốn cách:

1. **Chính là nhãn, đổi tên** — `stratum`, `weight`, `title`, ...
2. **Hệ quả của nhãn** (nhân quả ngược) — `thank_count`: review 1★ được người
   khác bấm "cảm ơn" nhiều hơn *sau khi* đăng.
3. **Thông tin tương lai** — ảnh chụp sản phẩm lúc cào (2026) như
   `rating_average` chứa luôn review đang cần dự báo.
4. **Trùng thông tin tuyệt đối** — `gap_days = lead_days − 5` (tương quan đúng
   1,0): đưa cả hai vào hồi quy tuyến tính thì hệ số không xác định.

Ngoài ra có nhóm **hậu nghiệm** (`EX_POST`): biến do khách khai *cùng lúc* với
số sao (bộ câu hỏi `delivery_rating`, số ảnh đính kèm, độ trễ viết review). Chúng
không được dùng ở mô hình dự báo (tầng `ex_ante`) nhưng được dùng ở mô hình giải
thích (tầng `ex_post`) — đó chính là phép so M1a/M1b/M1c của task C9.
"""

from __future__ import annotations

from collections.abc import Iterable

#: Biến không bao giờ được làm feature, kèm lý do — in thẳng vào notebook.
FORBIDDEN: dict[str, str] = {
    # 1 · chính là nhãn
    "rating": "chính là biến gốc của nhãn is_low_rating",
    "is_low_rating": "biến mục tiêu",
    "stratum": "tầng lấy mẫu = số sao của review → chính là nhãn",
    "stratum_size": "số review cùng số sao của sản phẩm → hàm của nhãn",
    "stratum_sampled": "số review đã cào ở tầng sao đó → hàm của nhãn",
    "weight": "w = N_h/n_h phụ thuộc tầng sao → hàm của nhãn; chỉ dùng để ĐÁNH GIÁ",
    "title": "tiêu đề Tiki tự sinh theo số sao (vd 'Cực kì hài lòng')",
    "content": "văn bản viết cùng lúc với số sao — dùng nó là phân tích cảm xúc, không phải dự báo",
    # 2 · hệ quả của nhãn
    "thank_count": "lượt 'cảm ơn' người khác bấm SAU khi review đăng — nhân quả ngược",
    # 3 · thông tin tương lai (ảnh chụp lúc cào 2026)
    "rating_average": "điểm TB sản phẩm lúc cào — chứa chính review cần dự báo + review tương lai",
    "review_count": "số review lúc cào — đếm cả review tương lai",
    "quantity_sold": "số lượng bán lúc cào — thông tin tương lai",
    "review_ts": "mốc đăng review — xảy ra SAU thời điểm dự báo",
    "created_at": "mốc tạo review — xảy ra SAU thời điểm dự báo",
    # 4 · trùng thông tin tuyệt đối
    "gap_days": "= lead_days − 5, tương quan đúng 1,0 với lead_days",
    "sla_status": "bản chữ của sla_breach",
    # định danh — mô hình sẽ học thuộc lòng thay vì khái quát hoá
    "review_id": "định danh",
    "customer_id": "định danh (dùng gián tiếp qua feature lịch sử khách)",
    "product_id": "định danh (dùng gián tiếp qua feature lịch sử sản phẩm)",
    "seller_id": "định danh 366 mức (dùng gián tiếp qua feature lịch sử nhà bán)",
    # không mang thông tin
    "status": "hằng số ('approved')",
    "customer_purchased_flag": "99,9% True — gần hằng số",
    "year": "năm thô: train ≤2023 thì mô hình chưa từng thấy 2024+, cây không ngoại suy được",
}

#: Biến khách tự khai cùng lúc với số sao — chỉ hợp lệ ở tầng giải thích.
EX_POST: frozenset[str] = frozenset(
    {
        "customer_says_late",
        "dr_shipper_bad",
        "dr_gio_giao_bad",
        "dr_dong_goi_bad",
        "has_delivery_rating",
        "n_images",
        "review_lag_days",
    }
)

#: Cặp biến trùng thông tin: đứng chung một mô hình *tuyến tính* thì hệ số
#: chia nhau tuỳ tiện, không diễn giải được. Mô hình cây thì không sao.
COLLINEAR_PAIRS: tuple[tuple[str, str], ...] = (("lead_days", "sla_breach"),)

TIERS = ("ex_ante", "ex_post")


class LeakageError(ValueError):
    """Một danh sách feature vi phạm quy tắc chống leakage."""


def audit_features(features: Iterable[str], tier: str = "ex_ante", linear: bool = False) -> None:
    """Chặn ngay khi khai báo mô hình — không đợi tới lúc đọc kết quả mới phát hiện.

    Args:
        features: danh sách tên cột mô hình sẽ dùng.
        tier: `"ex_ante"` (dự báo lúc giao hàng) hoặc `"ex_post"` (giải thích,
            được thêm biến khách tự khai).
        linear: True nếu là mô hình tuyến tính → cấm thêm các cặp trùng thông tin.

    Raises:
        LeakageError: kèm lý do cụ thể của từng vi phạm.
    """
    if tier not in TIERS:
        raise ValueError(f"tier phải thuộc {TIERS}, nhận {tier!r}")
    feats = list(features)
    problems = [f"{f}: {FORBIDDEN[f]}" for f in feats if f in FORBIDDEN]
    if tier == "ex_ante":
        problems += [f"{f}: biến hậu nghiệm, chỉ hợp lệ ở tầng ex_post" for f in feats if f in EX_POST]
    if linear:
        present = set(feats)
        problems += [
            f"{a} + {b}: trùng thông tin, không đứng chung mô hình tuyến tính"
            for a, b in COLLINEAR_PAIRS
            if a in present and b in present
        ]
    if problems:
        raise LeakageError("Vi phạm quy tắc chống leakage:\n  - " + "\n  - ".join(problems))
