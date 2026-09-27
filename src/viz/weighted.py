"""Các primitive vẽ biểu đồ có trọng số cho Track B.

Nguyên tắc: các ước lượng mô tả quần thể phải dùng cột ``weight``.
Notebook chỉ gọi các hàm ở đây, không tự viết lại logic thống kê.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


def _clean(values, weights=None):
    v = np.asarray(values, dtype=float)
    if weights is None:
        w = np.ones(v.shape, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != v.shape:
            raise ValueError("values và weights phải cùng kích thước")
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return v[mask], w[mask]


def _weighted_quantile(values, weights, q):
    v, w = _clean(values, weights)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cdf, v))


def weighted_hist(values, weights=None, bins=30, ax=None, density=False, **kwargs):
    """Vẽ histogram bằng tổng trọng số thay vì số dòng."""
    if ax is None:
        _, ax = plt.subplots()
    v, w = _clean(values, weights)
    if v.size == 0:
        raise ValueError("Không có dữ liệu hợp lệ để vẽ")
    return ax.hist(v, bins=bins, weights=w, density=density, **kwargs)


def weighted_ecdf(values, weights=None, ax=None, **kwargs):
    """Vẽ ECDF có trọng số, trả về (x, y)."""
    if ax is None:
        _, ax = plt.subplots()
    v, w = _clean(values, weights)
    if v.size == 0:
        raise ValueError("Không có dữ liệu hợp lệ để vẽ")
    order = np.argsort(v)
    x = v[order]
    y = np.cumsum(w[order]) / w.sum()
    ax.step(x, y, where="post", **kwargs)
    return x, y


def weighted_bar_ci(categories: Sequence, values, weights=None, ax=None, alpha=0.05, **kwargs):
    """Bar chart của tỷ lệ/trung bình có trọng số + CI xấp xỉ.

    ``categories`` là nhãn nhóm; ``values`` là mảng giá trị theo từng dòng.
    Hàm tính weighted mean và CI bằng phương sai có trọng số và n_eff.
    """
    if ax is None:
        _, ax = plt.subplots()
    cat = np.asarray(categories)
    v = np.asarray(values, dtype=float)
    w = np.ones(v.shape, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if not (cat.shape == v.shape == w.shape):
        raise ValueError("categories, values và weights phải cùng kích thước")

    labels = list(dict.fromkeys(cat.tolist()))
    means, errors = [], []
    for label in labels:
        x, ww = _clean(v[cat == label], w[cat == label])
        if x.size == 0:
            means.append(np.nan); errors.append(0.0); continue
        mean = float(np.dot(x, ww) / ww.sum())
        n_eff = float(ww.sum() ** 2 / np.dot(ww, ww))
        var = float(np.dot(ww, (x - mean) ** 2) / ww.sum())
        se = np.sqrt(var / n_eff) if n_eff > 0 else np.nan
        z = 1.96
        means.append(mean)
        errors.append(z * se)

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, means, yerr=errors, capsize=4, **kwargs)
    ax.set_xticks(x_pos, labels)
    return {"labels": labels, "means": np.asarray(means), "errors": np.asarray(errors)}


def weighted_box(values, groups=None, weights=None, ax=None, labels=None, **kwargs):
    """Vẽ boxplot dựa trên weighted quantile (Q1, median, Q3).

    Matplotlib không hỗ trợ weight trực tiếp cho ``boxplot``. Vì vậy hàm dựng
    box bằng các percentile có trọng số và whisker ở Q1/Q3 ± 1.5 IQR.
    """
    if ax is None:
        _, ax = plt.subplots()
    v = np.asarray(values, dtype=float)
    w = np.ones(v.shape, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if groups is None:
        groups = np.zeros(v.shape, dtype=int)
    g = np.asarray(groups)
    if not (v.shape == w.shape == g.shape):
        raise ValueError("values, groups và weights phải cùng kích thước")

    group_labels = list(dict.fromkeys(g.tolist())) if labels is None else list(labels)
    stats = []
    for label in group_labels:
        x, ww = _clean(v[g == label], w[g == label])
        if x.size == 0:
            stats.append(dict(med=np.nan, q1=np.nan, q3=np.nan, whislo=np.nan, whishi=np.nan))
            continue
        q1 = _weighted_quantile(x, ww, .25)
        med = _weighted_quantile(x, ww, .50)
        q3 = _weighted_quantile(x, ww, .75)
        iqr = q3 - q1
        lo = max(float(x.min()), q1 - 1.5 * iqr)
        hi = min(float(x.max()), q3 + 1.5 * iqr)
        stats.append(dict(med=med, q1=q1, q3=q3, whislo=lo, whishi=hi))

    ax.bxp(stats, positions=np.arange(1, len(stats) + 1), showfliers=False, **kwargs)
    ax.set_xticks(np.arange(1, len(stats) + 1), [str(x) for x in group_labels])
    return stats
