"""Test cho điều tra `lead_days` âm.

Phép kiểm quan trọng nhất ở đây là `test_tac_dong_phai_dung_trong_so`: nó dựng
một ca mà **đếm thô và ước lượng có trọng số kết luận ngược nhau**. Đó chính là
cái bẫy đã làm hỏng hai ước lượng phụ trong bản phân tích trước (FINDINGS §5),
nên nó phải có một test giữ chỗ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate.lead_anomaly import (
    CRAWL_DATE,
    age_gradient,
    anomaly_profile,
    category_lift,
    clock_alignment,
    prepare,
    weighted_impact,
)


def make(review_id, lead_days, purchased, delivered, review, rating=5, weight=1.0,
         quality_flag="ok", category="A", **kw):
    row = {
        "review_id": review_id,
        "product_id": 1,
        "seller_id": 1,
        "rating": rating,
        "weight": weight,
        "quality_flag": quality_flag,
        "category_name": category,
        "lead_days": lead_days,
        "purchased_ts": pd.Timestamp(purchased),
        "delivered_ts": pd.Timestamp(delivered),
        "review_ts": pd.Timestamp(review),
    }
    row.update(kw)
    return row


@pytest.fixture
def sample():
    return pd.DataFrame([
        # bình thường: đặt → giao → review
        make(1, 2.0, "2024-06-01", "2024-06-03", "2024-06-05"),
        # bất thường: ngày đặt rơi SAU ngày viết review
        make(2, -60.0, "2024-08-04", "2024-06-05", "2024-06-06",
             lead_days_flag=None, quality_flag="lead_days âm"),
    ])


def test_prepare_gan_co_dung(sample):
    out = prepare(sample).set_index("review_id")
    assert not out.loc[1, "is_negative_lead"]
    assert out.loc[2, "is_negative_lead"]
    assert not out.loc[1, "purchase_after_review"]
    assert out.loc[2, "purchase_after_review"]
    assert out.loc[1, "delivery_before_review"]


def test_prepare_khong_sua_bang_goc(sample):
    before = sample.copy(deep=True)
    prepare(sample)
    pd.testing.assert_frame_equal(sample, before)


def test_tuoi_review_tinh_tu_ngay_cao(sample):
    out = prepare(sample).set_index("review_id")
    expected = (CRAWL_DATE - pd.Timestamp("2024-06-05")).total_seconds() / 86400 / 365.25
    assert out.loc[1, "review_age_years"] == pytest.approx(expected)


def test_clock_alignment_do_dung_do_lech():
    """Dựng dữ liệu lệch đúng 7 giờ thì hàm phải đo ra 7 giờ."""
    ts = pd.Timestamp("2024-06-01 03:00:00")  # UTC
    df = pd.DataFrame([{
        "created_at": ts.timestamp(),
        "review_created_date": (ts + pd.Timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"),
    }])
    off = clock_alignment(df)
    assert off.iloc[0] == pytest.approx(7.0)


def test_clock_alignment_phat_hien_lech_khac():
    """Không được hardcode 7 — đổi dữ liệu thì kết quả phải đổi theo."""
    ts = pd.Timestamp("2024-06-01 03:00:00")
    df = pd.DataFrame([{
        "created_at": ts.timestamp(),
        "review_created_date": (ts + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
    }])
    assert clock_alignment(df).iloc[0] == pytest.approx(3.0)


def test_anomaly_profile_dem_dung(sample):
    p = anomaly_profile(prepare(sample))
    assert p["n"] == 1
    assert p["purchase_after_review_pct"] == pytest.approx(100.0)
    assert p["delivery_before_review_pct"] == pytest.approx(100.0)
    assert p["purchase_after_review_pct_ok"] == pytest.approx(0.0)


def test_age_gradient_chia_nhom_dung():
    rows = [make(i, 1.0, "2020-01-01", "2020-01-03", d)
            for i, d in enumerate(["2026-06-01", "2024-06-01", "2019-06-01"])]
    g = age_gradient(prepare(pd.DataFrame(rows)))
    assert g["n"].sum() == 3
    assert list(g.index) == ["<1 năm", "2–3", "6+"]


def test_category_lift_chuan_hoa_theo_mat_bang():
    rows = [make(1, -1.0, "2024-08-01", "2024-06-01", "2024-06-02", category="X"),
            make(2, 1.0, "2024-06-01", "2024-06-02", "2024-06-03", category="X"),
            make(3, 1.0, "2024-06-01", "2024-06-02", "2024-06-03", category="Y"),
            make(4, 1.0, "2024-06-01", "2024-06-02", "2024-06-03", category="Y")]
    g = category_lift(prepare(pd.DataFrame(rows)))
    # mặt bằng chung 1/4 = 25%; X có 50% -> bội số 2, Y có 0% -> bội số 0
    assert g.loc["X", "bội số"] == pytest.approx(2.0)
    assert g.loc["Y", "bội số"] == pytest.approx(0.0)


def test_tac_dong_phai_dung_trong_so():
    """Đếm thô và ước lượng có trọng số phải cho kết luận NGƯỢC nhau ở ca này.

    Nhóm bất thường: hai review 1 sao trọng số 1, một review 5 sao trọng số 100.
    Đếm thô nói nhóm này toàn khách bất mãn (2/3 là 1 sao); có trọng số thì
    review 5 sao chi phối và nhóm hoá ra hài lòng hơn phần còn lại.
    """
    rows = [
        make(1, -1.0, "2024-08-01", "2024-06-01", "2024-06-02", rating=1, weight=1.0),
        make(2, -1.0, "2024-08-01", "2024-06-01", "2024-06-02", rating=1, weight=1.0),
        make(3, -1.0, "2024-08-01", "2024-06-01", "2024-06-02", rating=5, weight=100.0),
        make(4, 1.0, "2024-06-01", "2024-06-02", "2024-06-03", rating=4, weight=1.0),
        make(5, 1.0, "2024-06-01", "2024-06-02", "2024-06-03", rating=4, weight=1.0),
    ]
    w = weighted_impact(prepare(pd.DataFrame(rows)))
    assert w["neg_rating_tb_thô"] < w["rest_rating_tb_thô"]   # thô: nhóm âm tệ hơn
    assert w["neg_rating_tb"] > w["rest_rating_tb"]           # trọng số: ngược lại
    assert w["neg_rating_tb"] == pytest.approx((1 + 1 + 5 * 100) / 102)


def test_tac_dong_bao_ti_le_quan_the_co_trong_so():
    rows = [make(1, -1.0, "2024-08-01", "2024-06-01", "2024-06-02", weight=25.0),
            make(2, 1.0, "2024-06-01", "2024-06-02", "2024-06-03", weight=75.0)]
    w = weighted_impact(prepare(pd.DataFrame(rows)))
    assert w["pct_quần_thể"] == pytest.approx(25.0)
