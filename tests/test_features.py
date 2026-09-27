"""Test cho tầng feature (Track C · C1, C3, C5).

Ba thứ phải chứng minh được bằng test chứ không bằng lời:

1. **Không leakage** — biến cấm bị chặn ngay khi khai báo mô hình.
2. **Feature lịch sử chỉ nhìn quá khứ** — giá trị của một dòng không đổi khi ta
   sửa nhãn của chính nó hay của bất kỳ review nào đăng *sau* thời điểm dự báo.
3. **Chia theo thời gian** — mọi năm của tập train nhỏ hơn mọi năm của tập test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    EX_ANTE_FEATURES,
    EX_POST_FEATURES,
    FEATURE_GROUPS,
    build_features,
    temporal_split,
)
from src.features.history import asof_group_stats
from src.features.leakage import FORBIDDEN, LeakageError, audit_features

# --------------------------------------------------------------------- leakage


@pytest.mark.parametrize("col", ["weight", "stratum", "title", "thank_count", "gap_days"])
def test_bien_cam_bi_chan(col):
    with pytest.raises(LeakageError):
        audit_features(["lead_days", col])


def test_bien_hau_nghiem_bi_chan_o_tang_ex_ante():
    with pytest.raises(LeakageError):
        audit_features(["lead_days", "customer_says_late"], tier="ex_ante")


def test_bien_hau_nghiem_duoc_phep_o_tang_ex_post():
    audit_features(["lead_days", "customer_says_late"], tier="ex_post")


def test_lead_va_sla_breach_khong_dung_chung_mo_hinh_tuyen_tinh():
    with pytest.raises(LeakageError):
        audit_features(["lead_days", "sla_breach"], linear=True)
    # Cây thì được: sla_breach chỉ là một ngưỡng của lead_days, cây tự học được.
    audit_features(["lead_days", "sla_breach"], linear=False)


def test_moi_feature_khai_bao_deu_qua_kiem_toan():
    audit_features(EX_ANTE_FEATURES, tier="ex_ante")
    audit_features(EX_ANTE_FEATURES + EX_POST_FEATURES, tier="ex_post")
    assert not set(EX_ANTE_FEATURES) & set(FORBIDDEN)


def test_nhom_feature_phu_kin_danh_sach_ex_ante():
    grouped = [f for g in FEATURE_GROUPS.values() for f in g]
    assert sorted(grouped) == sorted(EX_ANTE_FEATURES)


# --------------------------------------------------------------------- history


def _events() -> pd.DataFrame:
    day = pd.Timestamp("2024-01-01")
    return pd.DataFrame(
        {
            "g": [1, 1, 1, 1, 2],
            # thời điểm review được đăng (sự kiện có nhãn)
            "event_ts": [day, day + pd.Timedelta(days=2), day + pd.Timedelta(days=5),
                         day + pd.Timedelta(days=9), day],
            # thời điểm dự báo của chính dòng đó (lúc giao hàng)
            "cutoff_ts": [day - pd.Timedelta(days=1), day + pd.Timedelta(days=1),
                          day + pd.Timedelta(days=4), day + pd.Timedelta(days=8),
                          day - pd.Timedelta(days=1)],
            "y": [1.0, 0.0, 1.0, 0.0, 1.0],
            "w": [1.0, 3.0, 1.0, 2.0, 1.0],
        }
    )


def test_asof_chi_cong_su_kien_truoc_moc_du_bao():
    ev = _events()
    n, rate = asof_group_stats(ev["g"], ev["event_ts"], ev["cutoff_ts"], ev["y"], ev["w"],
                               prior_rate=0.5, prior_strength=0.0)
    # dòng 0: chưa có gì trước nó · dòng 1: chỉ dòng 0 · dòng 2: dòng 0,1 · dòng 3: 0,1,2
    np.testing.assert_allclose(n, [0.0, 1.0, 4.0, 5.0, 0.0])
    assert np.isnan(rate[0]) and np.isnan(rate[4])
    np.testing.assert_allclose(rate[1:4], [1.0, 1 / 4, 2 / 5])


def test_asof_khong_doi_khi_sua_nhan_tuong_lai():
    ev = _events()
    base = asof_group_stats(ev["g"], ev["event_ts"], ev["cutoff_ts"], ev["y"], ev["w"],
                            prior_rate=0.1, prior_strength=5.0)
    future = ev.copy()
    future.loc[3, "y"] = 1.0  # sửa nhãn của review cuối — không dòng nào được thấy nó
    future.loc[2, "y"] = 0.0  # sửa nhãn của chính dòng 2
    after = asof_group_stats(future["g"], future["event_ts"], future["cutoff_ts"],
                             future["y"], future["w"], prior_rate=0.1, prior_strength=5.0)
    np.testing.assert_allclose(base[1][:3], after[1][:3])


def test_asof_loai_chinh_minh_khi_review_dang_truoc_moc_du_bao():
    """Dữ liệu lỗi: review đăng *trước* lúc giao. Không được tự đếm nhãn của mình."""
    day = pd.Timestamp("2024-01-01")
    g = pd.Series([1])
    n, rate = asof_group_stats(g, pd.Series([day]), pd.Series([day + pd.Timedelta(days=1)]),
                               pd.Series([1.0]), pd.Series([1.0]),
                               prior_rate=0.3, prior_strength=0.0)
    assert n[0] == 0.0 and np.isnan(rate[0])


def test_asof_co_lam_tron_ve_prior():
    ev = _events()
    _, rate = asof_group_stats(ev["g"], ev["event_ts"], ev["cutoff_ts"], ev["y"], ev["w"],
                               prior_rate=0.2, prior_strength=4.0)
    # dòng 1: (1·1 + 4·0.2) / (1 + 4) = 0.36 ; dòng 0: không có lịch sử → đúng bằng prior
    assert rate[1] == pytest.approx(0.36)
    assert rate[0] == pytest.approx(0.2)


# --------------------------------------------------------------------- build + split


def _tiny_clean() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    purchased = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D")
    lead = rng.uniform(0.1, 8.0, n)
    delivered = purchased + pd.to_timedelta(lead, unit="D")
    review = delivered + pd.to_timedelta(rng.uniform(0.5, 20, n), unit="D")
    rating = rng.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.05, 0.05, 0.2, 0.6])
    return pd.DataFrame(
        {
            "review_id": np.arange(n),
            "product_id": rng.choice([10, 11, 12], n),
            "seller_id": rng.choice([1, 7], n),
            "customer_id": rng.choice(np.arange(20), n),
            "category_id": rng.choice([100, 200], n),
            "category_name": "x",
            "rating": rating,
            "is_low_rating": rating <= 3,
            "weight": np.where(rating >= 4, 3.0, 1.0),
            "stratum": rating,
            "stratum_size": 10,
            "stratum_sampled": np.where(rating >= 4, 5, 10),
            "purchased_ts": purchased,
            "delivered_ts": delivered,
            "review_ts": review,
            "lead_days": lead,
            "review_lag_days": (review - delivered).total_seconds() / 86400,
            "sla_breach": lead > 5,
            "is_analysable": True,
            "customer_joined": "2019-05-01 10:00:00",
            "dr_thoi_gian": None, "dr_shipper": None, "dr_gio_giao": None, "dr_dong_goi": None,
            "customer_says_late": None,
            "has_delivery_rating": False,
            "n_images": 0,
        }
    )


def test_build_features_khong_sua_dau_vao_va_du_cot():
    clean = _tiny_clean()
    snapshot = clean.copy()
    frame = build_features(clean, products=None, train_end_year=2023)
    pd.testing.assert_frame_equal(clean, snapshot)
    for col in EX_ANTE_FEATURES + EX_POST_FEATURES + ["year", "weight", "is_low_rating"]:
        assert col in frame.columns, col
    assert frame["is_low_rating"].isin([0, 1]).all()


def test_temporal_split_khong_chong_lan():
    frame = build_features(_tiny_clean(), products=None, train_end_year=2023)
    train, test = temporal_split(frame, test_start_year=2024)
    assert train["year"].max() < test["year"].min()
    assert len(train) + len(test) == len(frame)
    assert not set(train.index) & set(test.index)


def test_ban_vector_hoa_khop_ban_vong_lap():
    """Bản np.searchsorted phải cho đúng từng số như bản vòng lặp thuần."""
    from src.features.history import asof_group_stats_loop

    rng = np.random.default_rng(3)
    n = 300
    base = pd.Timestamp("2022-01-01")
    ev = pd.Series(base + pd.to_timedelta(rng.uniform(0, 400, n), unit="D"))
    ev[rng.random(n) < 0.05] = pd.NaT
    cut = pd.Series(base + pd.to_timedelta(rng.uniform(0, 400, n), unit="D"))
    g = pd.Series(rng.integers(0, 12, n))
    y = pd.Series(rng.integers(0, 2, n).astype(float))
    w = pd.Series(rng.choice([1.0, 2.5, 7.0], n))
    fast = asof_group_stats(g, ev, cut, y, w, prior_rate=0.1, prior_strength=3.0)
    slow = asof_group_stats_loop(g, ev, cut, y, w, prior_rate=0.1, prior_strength=3.0)
    np.testing.assert_allclose(fast[0], slow[0])
    np.testing.assert_allclose(fast[1], slow[1])
