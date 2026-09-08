"""Test cho tầng thống kê có trọng số và bootstrap.

Trọng tâm: chứng minh việc **bỏ quên trọng số** cho ra số khác hẳn — đây là lỗi
âm thầm nguy hiểm nhất của thiết kế lấy mẫu phân tầng.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate.stats import (
    cluster_bootstrap,
    discriminative_power,
    weighted_mean,
    weighted_proportion,
    weighted_quantile,
)


def test_weighted_mean_bang_mean_khi_trong_so_deu():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)
    assert weighted_mean(v, w) == pytest.approx(v.mean())


def test_weighted_mean_khac_mean_thuong_khi_trong_so_lech():
    """Ca thật: mẫu phân tầng lấy quá nhiều review 1 sao so với quần thể."""
    ratings = np.array([1.0, 5.0])
    weights = np.array([1.0, 99.0])  # thực tế 1 review 1★ và 99 review 5★
    assert ratings.mean() == pytest.approx(3.0)
    assert weighted_mean(ratings, weights) == pytest.approx(4.96)


def test_weighted_proportion():
    mask = np.array([True, False, False])
    w = np.array([10.0, 1.0, 1.0])
    assert weighted_proportion(mask, w) == pytest.approx(10 / 12)


def test_weighted_quantile_trung_vi():
    v = np.array([1.0, 2.0, 3.0, 100.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert weighted_quantile(v, w, 0.5) == pytest.approx(2.5)


def test_weighted_mean_tra_nan_khi_tong_trong_so_bang_0():
    assert np.isnan(weighted_mean(np.array([1.0]), np.array([0.0])))


@pytest.fixture
def clustered() -> pd.DataFrame:
    """12 sản phẩm × 20 review, nhóm True có rating cao hơn ~1 điểm.

    Hiệu ứng **khác nhau giữa các sản phẩm** (`effect` rút ngẫu nhiên theo từng
    product_id). Đây mới là cấu trúc cụm thật: review cùng một sản phẩm chia sẻ
    cùng nhà bán, kho và tuyến giao, nên chúng không độc lập.
    """
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(12):
        effect = rng.normal(1.0, 0.5)  # hiệu ứng riêng của từng sản phẩm
        for _ in range(20):
            flag = bool(rng.integers(0, 2))
            rows.append(
                {
                    "product_id": pid,
                    "flag": flag,
                    "rating": 4.0 + (effect if flag else 0.0) + rng.normal(0, 0.1),
                    "weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_cluster_bootstrap_bao_quanh_uoc_luong_diem(clustered):
    est = cluster_bootstrap(
        clustered, discriminative_power(clustered, "flag"), n_boot=200, name="Δ"
    )
    assert est.lo <= est.value <= est.hi
    assert est.value == pytest.approx(1.0, abs=0.4)
    assert est.n == len(clustered)


def test_cluster_bootstrap_tai_lap_duoc(clustered):
    """Cùng seed phải cho cùng khoảng tin cậy — điều kiện của tầng reproducibility."""
    stat = discriminative_power(clustered, "flag")
    a = cluster_bootstrap(clustered, stat, n_boot=200, seed=7)
    b = cluster_bootstrap(clustered, stat, n_boot=200, seed=7)
    assert (a.value, a.lo, a.hi) == (b.value, b.lo, b.hi)


def test_cluster_bootstrap_rong_hon_bootstrap_theo_dong(clustered):
    """Khi hiệu ứng khác nhau giữa các sản phẩm, bootstrap theo cụm PHẢI cho
    khoảng tin cậy rộng hơn bootstrap theo dòng — vì nó tính cả phương sai giữa
    các cụm. Bootstrap theo dòng bỏ qua phần này và cho KTC hẹp giả tạo."""
    stat = discriminative_power(clustered, "flag")
    by_cluster = cluster_bootstrap(clustered, stat, cluster_col="product_id", n_boot=400, seed=1)
    row_ids = clustered.assign(_row=np.arange(len(clustered)))
    by_row = cluster_bootstrap(row_ids, stat, cluster_col="_row", n_boot=400, seed=1)
    assert (by_cluster.hi - by_cluster.lo) >= (by_row.hi - by_row.lo)


def test_discriminative_power_bang_0_khi_hai_nhom_giong_nhau():
    df = pd.DataFrame(
        {"flag": [True, False, True, False], "rating": [4.0] * 4, "weight": [1.0] * 4}
    )
    assert discriminative_power(df, "flag")(df) == pytest.approx(0.0)


def test_discriminative_power_nan_khi_thieu_mot_nhom():
    df = pd.DataFrame({"flag": [True, True], "rating": [4.0, 5.0], "weight": [1.0, 1.0]})
    assert np.isnan(discriminative_power(df, "flag")(df))


# --- Cỡ mẫu hiệu dụng (Kish) -------------------------------------------------
# Thêm 2026-09-08 sau khi phát hiện ước lượng "91.9%" dựng trên n_eff=25 và
# lệch 10 điểm khi mở rộng mẫu. Số dòng KHÔNG phải lượng thông tin.


def test_n_eff_bang_n_khi_trong_so_deu():
    from src.evaluate.stats import effective_sample_size

    assert effective_sample_size(np.ones(100)) == pytest.approx(100.0)


def test_n_eff_sup_do_khi_mot_dong_ap_dao():
    """99 dòng trọng số 1 + 1 dòng trọng số 300 → thông tin gần như chỉ còn 1 quan sát."""
    from src.evaluate.stats import effective_sample_size

    w = np.array([1.0] * 99 + [300.0])
    assert effective_sample_size(w) < 5.0


def test_n_eff_bang_0_khi_rong():
    from src.evaluate.stats import effective_sample_size

    assert effective_sample_size(np.array([])) == 0.0


def test_n_eff_bo_qua_trong_so_khong_hop_le():
    from src.evaluate.stats import effective_sample_size

    assert effective_sample_size(np.array([1.0, 1.0, 0.0, -3.0, np.nan])) == pytest.approx(2.0)


def test_uoc_luong_mang_theo_n_eff_va_co_canh_bao(clustered):
    """Estimate phải tự khai n_eff — không được để người đọc phải tự đi tính."""
    est = cluster_bootstrap(
        clustered, discriminative_power(clustered, "flag"), n_boot=100, name="Δ"
    )
    assert est.n_eff == pytest.approx(len(clustered))  # fixture dùng weight=1
    assert est.n_clusters == clustered["product_id"].nunique()
    assert not est.is_fragile
    assert "n_eff" in str(est)


def test_danh_dau_mong_manh_khi_n_eff_thap():
    """Mẫu trông lớn (200 dòng) nhưng trọng số lệch → phải bị gắn cờ MONG MANH."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "product_id": np.repeat(np.arange(20), 10),
            "flag": rng.integers(0, 2, 200).astype(bool),
            "rating": rng.normal(4, 0.5, 200),
            "weight": np.where(np.arange(200) < 5, 500.0, 1.0),
        }
    )
    est = cluster_bootstrap(df, discriminative_power(df, "flag"), n_boot=100, name="Δ")
    assert est.n == 200
    assert est.n_eff < 100
    assert est.is_fragile
    assert "MONG MANH" in str(est)
