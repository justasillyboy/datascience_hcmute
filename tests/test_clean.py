"""Test cho bước làm sạch — trọng tâm là các quy tắc dễ làm sai lặng lẽ."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean.reviews import (
    FLAG_IMPLAUSIBLE_LEAD,
    FLAG_NEGATIVE_LEAD,
    FLAG_NO_DELIVERY_DATE,
    FLAG_NO_PURCHASE,
    QUALITY_OK,
    SLA_MAX_DAYS,
    clean_reviews,
)

#: Mốc gốc cho mọi ca thử: **2024-06-01 00:00:00 giờ Việt Nam**.
#: Phải neo theo giờ VN chứ không phải UTC — `delivery_date` và
#: `review_created_date` mà Tiki trả về là chuỗi giờ VN, nên nếu mốc epoch này
#: được hiểu là UTC thì mọi kỳ vọng lead time trong file sẽ lệch đúng 7 giờ.
BASE_TS = 1717174800


def make_row(review_id: int, purchased_at, delivery_date, rating: int = 5, **kw) -> dict:
    row = {
        "review_id": review_id,
        "product_id": 1,
        "seller_id": 1,
        "customer_id": 100 + review_id,
        "rating": rating,
        "purchased_at": purchased_at,
        "delivery_date": delivery_date,
        "review_created_date": "2024-06-20 10:00:00",
        "stratum": rating,
        "stratum_size": 10,
        "stratum_sampled": 5,
        "weight": 2.0,
        "dr_thoi_gian": None,
    }
    row.update(kw)
    return row


@pytest.fixture
def sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # giao sau 2 ngày -> hợp lệ
            make_row(1, BASE_TS, "2024-06-03 00:00:00"),
            # giao sau 10 ngày -> hợp lệ nhưng vượt SLA
            make_row(2, BASE_TS, "2024-06-11 00:00:00", rating=3),
            # giao TRƯỚC khi đặt -> lead âm
            make_row(3, BASE_TS, "2024-05-20 00:00:00", rating=1),
            # thiếu ngày giao
            make_row(4, BASE_TS, None, rating=4),
            # thiếu ngày đặt
            make_row(5, None, "2024-06-03 00:00:00", rating=2),
            # lead phi lý (hơn 90 ngày)
            make_row(6, BASE_TS, "2025-06-01 00:00:00"),
        ]
    )


def test_lead_days_dung(sample):
    out = clean_reviews(sample).set_index("review_id")
    assert out.loc[1, "lead_days"] == pytest.approx(2.0)
    assert out.loc[2, "lead_days"] == pytest.approx(10.0)


def test_gan_co_dung_loai(sample):
    out = clean_reviews(sample).set_index("review_id")
    assert out.loc[1, "quality_flag"] == QUALITY_OK
    assert out.loc[2, "quality_flag"] == QUALITY_OK
    assert out.loc[3, "quality_flag"] == FLAG_NEGATIVE_LEAD
    assert out.loc[4, "quality_flag"] == FLAG_NO_DELIVERY_DATE
    assert out.loc[5, "quality_flag"] == FLAG_NO_PURCHASE
    assert out.loc[6, "quality_flag"] == FLAG_IMPLAUSIBLE_LEAD


def test_moi_dong_chi_mang_mot_co(sample):
    """Bảng đếm theo cờ phải cộng lại đúng bằng tổng số dòng."""
    out = clean_reviews(sample)
    assert out["quality_flag"].value_counts().sum() == len(out)


def test_gap_dung_cong_thuc(sample):
    out = clean_reviews(sample).set_index("review_id")
    assert out.loc[1, "gap_days"] == pytest.approx(2.0 - SLA_MAX_DAYS)
    assert out.loc[2, "gap_days"] == pytest.approx(10.0 - SLA_MAX_DAYS)


def test_gap_rong_o_dong_bi_loai(sample):
    """Không được để lọt giá trị gap ở dòng bị gắn cờ — nếu không ai đó sẽ
    vô tình tính trung bình lên dữ liệu rác."""
    out = clean_reviews(sample)
    bad = out[~out["is_analysable"]]
    assert bad["gap_days"].isna().all()
    assert bad["sla_breach"].isna().all()


def test_sla_breach_dung(sample):
    out = clean_reviews(sample).set_index("review_id")
    assert out.loc[1, "sla_breach"] is False or out.loc[1, "sla_breach"] == False  # noqa: E712
    assert out.loc[2, "sla_breach"] == True  # noqa: E712


def test_khu_trung_lap_theo_review_id():
    df = pd.DataFrame(
        [make_row(1, BASE_TS, "2024-06-03 00:00:00"), make_row(1, BASE_TS, "2024-06-03 00:00:00")]
    )
    assert len(clean_reviews(df)) == 1


def test_nhan_khach_duoc_chuan_hoa():
    df = pd.DataFrame(
        [
            make_row(1, BASE_TS, "2024-06-03 00:00:00", dr_thoi_gian="Giao trễ hẹn"),
            make_row(2, BASE_TS, "2024-06-03 00:00:00", dr_thoi_gian="Giao đúng hẹn"),
            make_row(3, BASE_TS, "2024-06-03 00:00:00", dr_thoi_gian=None),
        ]
    )
    out = clean_reviews(df).set_index("review_id")
    assert out.loc[1, "customer_says_late"] == True   # noqa: E712
    assert out.loc[2, "customer_says_late"] == False  # noqa: E712
    assert pd.isna(out.loc[3, "customer_says_late"])


def test_khong_sua_dataframe_goc(sample):
    """Nguyên tắc bất biến: pipeline không được đụng vào input."""
    before = sample.copy(deep=True)
    clean_reviews(sample)
    pd.testing.assert_frame_equal(sample, before)


def test_is_low_rating():
    df = pd.DataFrame(
        [make_row(i, BASE_TS, "2024-06-03 00:00:00", rating=i) for i in range(1, 6)]
    )
    out = clean_reviews(df).set_index("review_id")
    assert out.loc[[1, 2, 3], "is_low_rating"].all()
    assert not out.loc[[4, 5], "is_low_rating"].any()


# --------------------------------------------------------------------------
# Múi giờ — lỗi này từng làm mọi con số giao hàng phồng lên đúng 0,2917 ngày.
# --------------------------------------------------------------------------


def test_purchased_at_duoc_doi_sang_gio_viet_nam():
    """`purchased_at` là epoch; `delivery_date` là chuỗi giờ VN. Hai mốc phải
    được đưa về **cùng một đồng hồ** trước khi trừ nhau.

    1717174800 = 2024-06-01 00:00:00 giờ VN. Giao lúc 2024-06-03 00:00:00 giờ VN
    thì lead time đúng bằng 2 ngày chẵn — không phải 2,2917 ngày.
    """
    df = pd.DataFrame([make_row(1, 1717174800, "2024-06-03 00:00:00")])
    out = clean_reviews(df).set_index("review_id")
    assert out.loc[1, "purchased_ts"] == pd.Timestamp("2024-06-01 00:00:00")
    assert out.loc[1, "lead_days"] == pytest.approx(2.0)


def test_lead_time_khong_bi_phong_len_7_gio():
    """Ca hồi quy trực tiếp cho lỗi cũ.

    Đặt và giao **cùng một thời điểm** thì lead time phải bằng 0. Bản cũ đọc
    epoch theo UTC còn chuỗi theo giờ VN nên cho ra 0,2917 ngày (7 giờ).
    """
    df = pd.DataFrame([make_row(1, 1717174800, "2024-06-01 00:00:00")])
    out = clean_reviews(df).set_index("review_id")
    assert out.loc[1, "lead_days"] == pytest.approx(0.0, abs=1e-9)


def test_moi_moc_thoi_gian_nam_tren_cung_mot_dong_ho():
    """`review_lag_days` chỉ đúng khi cả ba mốc cùng múi giờ."""
    df = pd.DataFrame(
        [make_row(1, 1717174800, "2024-06-03 00:00:00",
                  review_created_date="2024-06-05 00:00:00")]
    )
    out = clean_reviews(df).set_index("review_id")
    assert out.loc[1, "lead_days"] == pytest.approx(2.0)
    assert out.loc[1, "review_lag_days"] == pytest.approx(2.0)
