"""Thống kê có trọng số + bootstrap — tầng 5 của khung "chứng minh đúng".

Hai điều module này bắt buộc phải làm đúng:

1. **Mọi ước lượng quần thể phải mang trọng số phân tầng.** Mẫu được lấy phân
   tầng theo sao (CLAUDE.md §3.5) nên trung bình thô là ước lượng chệch. Trong
   mẻ 2026-09-08, rating trung bình thô lệch khỏi ước lượng có trọng số rất xa
   (mẫu 61.9% năm sao, quần thể 85.7%).

2. **Bootstrap phải lấy mẫu lại theo cụm sản phẩm, không theo từng dòng.** Các
   review của cùng một sản phẩm không độc lập (cùng nhà bán, cùng kho, cùng
   tuyến giao). Bootstrap theo dòng sẽ cho khoảng tin cậy hẹp giả tạo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.05

#: Dưới ngưỡng này, ước lượng bị đánh dấu là mong manh. Chọn 100 vì mẻ
#: 2026-09-08 cho thấy nhóm "khách nói trễ hẹn" có n=182 dòng nhưng n_eff=25,
#: và ước lượng của nó lệch 10 điểm khi mở rộng mẫu — trong khi nhóm có
#: n_eff≈1000 thì tái lập gần như nguyên vẹn.
FRAGILE_N_EFF = 100.0


@dataclass(frozen=True)
class Estimate:
    """Một ước lượng điểm kèm khoảng tin cậy bootstrap."""

    name: str
    value: float
    lo: float
    hi: float
    n: int
    conf: float = 1 - DEFAULT_ALPHA
    #: Cỡ mẫu **hiệu dụng** (Kish) và số cụm. Hai con số này đi kèm ước lượng
    #: chứ không tách rời: `n` lớn mà `n_eff` bé thì ước lượng không đáng tin,
    #: và người đọc phải thấy điều đó ngay tại chỗ đọc kết quả.
    n_eff: float = float("nan")
    n_clusters: int = 0

    @property
    def is_fragile(self) -> bool:
        """True nếu ước lượng dựa trên quá ít thông tin thực."""
        return not np.isnan(self.n_eff) and self.n_eff < FRAGILE_N_EFF

    def __str__(self) -> str:
        base = (
            f"{self.name}: {self.value:.4f} "
            f"[{self.lo:.4f}, {self.hi:.4f}] ({self.conf:.0%} CI, n={self.n:,}"
        )
        if not np.isnan(self.n_eff):
            base += f", n_eff={self.n_eff:.0f}, cụm={self.n_clusters:,}"
        base += ")"
        return base + ("  ⚠️ MONG MANH" if self.is_fragile else "")


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float(np.dot(values, weights) / total)


def weighted_proportion(mask: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float(weights[mask].sum() / total)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float | np.ndarray):
    """Phân vị có trọng số, nội suy tuyến tính trên CDF thực nghiệm."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cdf, v)


def effective_sample_size(weights: np.ndarray) -> float:
    """Cỡ mẫu hiệu dụng của Kish: `n_eff = (Σw)² / Σw²`.

    Với mẫu phân tầng, **số dòng không phải là lượng thông tin**. Nếu vài dòng
    mang trọng số rất lớn thì chúng chi phối ước lượng, và mẫu 200 dòng có thể
    chỉ chứa thông tin tương đương 25 quan sát độc lập.

    Trọng số đều nhau → `n_eff == n`. Càng lệch thì `n_eff` càng nhỏ hơn `n`.
    """
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    denom = float((w ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float(w.sum() ** 2 / denom)


def cluster_bootstrap(
    df: pd.DataFrame,
    statistic,
    cluster_col: str = "product_id",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    name: str = "statistic",
    weight_col: str = "weight",
) -> Estimate:
    """Bootstrap theo cụm: lấy mẫu lại **sản phẩm** (có hoàn lại), không lấy dòng.

    Args:
        statistic: hàm nhận DataFrame và trả về một số thực.
    """
    point = float(statistic(df))
    clusters = df[cluster_col].to_numpy()
    uniq = np.unique(clusters)
    # Gom chỉ số theo cụm một lần, tránh lọc lại DataFrame 10.000 lần.
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in picked])
        draws[b] = statistic(df.iloc[idx])

    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # n_eff tính trên chính tập dòng dùng để ước lượng — báo cáo ước lượng mà
    # không kèm nó là che mất mức độ tin cậy thật.
    n_eff = effective_sample_size(df[weight_col].to_numpy()) if weight_col in df else float("nan")
    return Estimate(
        name=name, value=point, lo=float(lo), hi=float(hi), n=len(df),
        conf=1 - alpha, n_eff=n_eff, n_clusters=int(uniq.size),
    )


def weighted_mean_diff(
    df: pd.DataFrame, group_col: str, value_col: str, weight_col: str = "weight"
):
    """Trả về hàm thống kê: chênh lệch trung bình có trọng số giữa nhóm True và False."""

    def stat(d: pd.DataFrame) -> float:
        g = d[group_col].astype(bool).to_numpy()
        v = d[value_col].to_numpy(dtype=float)
        w = d[weight_col].to_numpy(dtype=float)
        if g.sum() == 0 or (~g).sum() == 0:
            return float("nan")
        return weighted_mean(v[g], w[g]) - weighted_mean(v[~g], w[~g])

    return stat


def weighted_prop(mask_col: str, weight_col: str = "weight"):
    """Trả về hàm thống kê: tỉ lệ có trọng số của `mask_col`."""

    def stat(d: pd.DataFrame) -> float:
        return weighted_proportion(d[mask_col].astype(bool).to_numpy(), d[weight_col].to_numpy(dtype=float))

    return stat


def discriminative_power(
    df: pd.DataFrame,
    indicator_col: str,
    value_col: str = "rating",
    weight_col: str = "weight",
):
    """|Δ rating| giữa hai mức của một chỉ báo — "chỉ báo này phân biệt được bao nhiêu".

    Dùng để so trực tiếp: chỉ báo SLA tính toán (`sla_breach`) và nhãn khách tự
    báo (`customer_says_late`) — cái nào bám sát mức hài lòng hơn.
    """

    def stat(d: pd.DataFrame) -> float:
        g = d[indicator_col].astype(bool).to_numpy()
        v = d[value_col].to_numpy(dtype=float)
        w = d[weight_col].to_numpy(dtype=float)
        if g.sum() == 0 or (~g).sum() == 0:
            return float("nan")
        return abs(weighted_mean(v[g], w[g]) - weighted_mean(v[~g], w[~g]))

    return stat
