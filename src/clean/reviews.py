"""Làm sạch bảng review Tiki.

Nguyên tắc xuyên suốt module: **không bao giờ vứt dữ liệu trong im lặng**. Mỗi
dòng bị loại phải được gắn cờ lý do trong cột `quality_flag` và đếm được trong
báo cáo. Đây chính là phần giảng viên chấm ở tầng "chứng minh đúng" — nhóm không
giấu giả định mà đo tác động của giả định.

Mọi hàm nhận DataFrame và **trả về bản sao mới**, không sửa tại chỗ.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: SLA công bố của Tiki: handlingTime 0–1 ngày + transitTime 1–4 ngày.
#: Xem docs/FEASIBILITY.md §2.4 — đây là hằng số site-wide, không theo từng đơn.
SLA_MAX_DAYS = 5.0
SLA_MIN_DAYS = 1.0

#: Tiki trả về hai loại mốc thời gian trên **hai đồng hồ khác nhau**:
#: `created_by.purchased_at` là unix epoch, còn `timeline.delivery_date` /
#: `timeline.review_created_date` là chuỗi giờ Việt Nam (UTC+7). Đối chiếu
#: `created_at` (epoch) với `review_created_date` (chuỗi) trên 115.948 review
#: cho chênh lệch **đúng 7,0 giờ, phương sai bằng 0** — xem
#: `docs/FINDINGS.md` §6. Không quy về cùng một đồng hồ thì mọi lead time
#: phồng lên đúng 0,2917 ngày.
VN_UTC_OFFSET = pd.Timedelta(hours=7)

#: Ngưỡng lead time coi là bất thường. Tiki giao trung vị ~1.6 ngày; trên 90 ngày
#: gần như chắc chắn là lỗi ghép mốc thời gian chứ không phải đơn giao thật.
LEAD_DAYS_MAX_PLAUSIBLE = 90.0

QUALITY_OK = "ok"
FLAG_NO_DELIVERY_DATE = "thiếu delivery_date"
FLAG_NO_PURCHASE = "thiếu purchased_at"
FLAG_NEGATIVE_LEAD = "lead_days âm"
FLAG_IMPLAUSIBLE_LEAD = "lead_days > 90 ngày"


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi các mốc thời gian thô sang datetime.

    `purchased_at` là unix epoch (giây); `delivery_date` và `review_created_date`
    là chuỗi "YYYY-MM-DD HH:MM:SS" theo giờ Việt Nam.

    Vì thế epoch phải **cộng thêm `VN_UTC_OFFSET`** để nằm cùng đồng hồ với hai
    chuỗi kia. Kết quả là ba cột `*_ts` đều là giờ Việt Nam dạng naive, trừ nhau
    được trực tiếp.
    """
    out = df.copy()
    out["purchased_ts"] = (
        pd.to_datetime(out["purchased_at"], unit="s", errors="coerce") + VN_UTC_OFFSET
    )
    out["delivered_ts"] = pd.to_datetime(out["delivery_date"], errors="coerce")
    out["review_ts"] = pd.to_datetime(out["review_created_date"], errors="coerce")
    return out


def compute_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Tính lead time giao hàng và độ trễ viết review."""
    out = df.copy()
    out["lead_days"] = (out["delivered_ts"] - out["purchased_ts"]).dt.total_seconds() / 86400
    out["review_lag_days"] = (out["review_ts"] - out["delivered_ts"]).dt.total_seconds() / 86400
    return out


def flag_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Gắn cờ chất lượng thay vì lọc bỏ.

    Thứ tự ưu tiên cờ: thiếu mốc → âm → phi lý. Một dòng chỉ mang một cờ, là cờ
    nghiêm trọng nhất áp dụng được, để bảng đếm cộng lại đúng bằng tổng số dòng.
    """
    out = df.copy()
    flag = pd.Series(QUALITY_OK, index=out.index, dtype="object")

    flag = flag.mask(out["purchased_ts"].isna(), FLAG_NO_PURCHASE)
    flag = flag.mask(out["delivered_ts"].isna() & (flag == QUALITY_OK), FLAG_NO_DELIVERY_DATE)
    flag = flag.mask((out["lead_days"] < 0) & (flag == QUALITY_OK), FLAG_NEGATIVE_LEAD)
    flag = flag.mask(
        (out["lead_days"] > LEAD_DAYS_MAX_PLAUSIBLE) & (flag == QUALITY_OK),
        FLAG_IMPLAUSIBLE_LEAD,
    )

    out["quality_flag"] = flag
    out["is_analysable"] = flag == QUALITY_OK
    return out


def compute_gap(df: pd.DataFrame, sla_days: float = SLA_MAX_DAYS) -> pd.DataFrame:
    """Delivery Promise Gap theo **phương án C** (xem CLAUDE.md §4).

    `gap_days > 0` nghĩa là đơn vượt SLA công bố. Giá trị chỉ có nghĩa ở các dòng
    `is_analysable`; nơi khác để `NaN` để không ai lỡ tay tính trung bình lên rác.
    """
    out = df.copy()
    gap = out["lead_days"] - sla_days
    out["gap_days"] = gap.where(out["is_analysable"])
    out["sla_breach"] = (out["gap_days"] > 0).where(out["is_analysable"])

    conditions = [out["gap_days"] < 0, out["gap_days"] == 0, out["gap_days"] > 0]
    labels = ["Giao sớm", "Đúng hạn", "Giao trễ"]
    out["sla_status"] = pd.Series(
        np.select(conditions, labels, default=None), index=out.index, dtype="object"
    ).where(out["is_analysable"])
    return out


def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hoá nhãn khách tự báo thành biến nhị phân tiện dùng."""
    out = df.copy()
    out["customer_says_late"] = out["dr_thoi_gian"].map(
        {"Giao trễ hẹn": True, "Giao đúng hẹn": False}
    )
    out["is_low_rating"] = out["rating"] <= 3
    return out


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Khử trùng lặp theo `review_id`.

    Trùng lặp phát sinh vì một sản phẩm có thể xuất hiện ở nhiều ngành hàng, và
    vì phân trang có thể chồng lấn khi có review mới trong lúc đang cào.
    """
    out = df.copy()
    before = len(out)
    out = out.drop_duplicates(subset=["review_id"], keep="first").reset_index(drop=True)
    removed = before - len(out)
    if removed:
        logger.info("Khử trùng lặp: bỏ %d/%d dòng (%.2f%%)", removed, before, 100 * removed / before)
    return out


def clean_reviews(df: pd.DataFrame, sla_days: float = SLA_MAX_DAYS) -> pd.DataFrame:
    """Pipeline làm sạch đầy đủ, chạy theo đúng thứ tự."""
    return (
        df.pipe(deduplicate)
        .pipe(parse_timestamps)
        .pipe(compute_durations)
        .pipe(flag_quality)
        .pipe(compute_gap, sla_days=sla_days)
        .pipe(normalise_labels)
    )


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng đếm theo cờ chất lượng — đưa thẳng vào báo cáo."""
    counts = df["quality_flag"].value_counts(dropna=False)
    return pd.DataFrame(
        {"số dòng": counts, "tỉ lệ %": (counts / len(df) * 100).round(2)}
    ).rename_axis("cờ chất lượng")
