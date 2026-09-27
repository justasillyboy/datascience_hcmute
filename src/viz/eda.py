"""Các figure EDA chính của Track B.

Các hàm ở đây nhận DataFrame đã clean. Notebook chỉ gọi hàm và ghi kết luận.
"""
from __future__ import annotations
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluate.stats import effective_sample_size, weighted_mean, weighted_proportion
from src.viz.theme import PALETTE, save_fig


def plot_sla_2x2(df: pd.DataFrame, output: str = "B10_SLA Status vs. Customer-Reported Delivery Status.png"):
    """Heatmap 2x2 của SLA status × customer_says_late.

    Đây là **số dòng mẫu có nhãn**, không phải ước lượng quần thể. Hình ghi rõ
    điều này để tránh hiểu nhầm rằng các ô là weighted population counts.
    """
    d = df[df["is_analysable"] & df["customer_says_late"].notna()].copy()
    sla_label = d["sla_breach"].map({False: "Within SLA", True: "SLA Breach"})
    customer_label = d["customer_says_late"].map({False: "On-time", True: "Late"})
    table = pd.crosstab(
        pd.Categorical(sla_label, categories=["Within SLA", "SLA Breach"]),
        pd.Categorical(customer_label, categories=["On-time", "Late"]),
    ).reindex(index=["Within SLA", "SLA Breach"], columns=["On-time", "Late"], fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(table.to_numpy(), aspect="auto", cmap="Blues",
    norm=LogNorm(
        vmin=table.to_numpy().min(),
        vmax=table.to_numpy().max()
    ))
    ax.set_xticks(range(2), table.columns)
    ax.set_yticks(range(2), table.index)
    ax.set_xlabel("Customer-Reported Delivery Status")
    ax.set_ylabel("SLA Status")
    ax.set_title("SLA Status vs. Customer-Reported Delivery Status")
    for i in range(2):
        for j in range(2):
            value = int(table.iloc[i, j])
            weight = "Mismatch" if (i, j) in [(0, 1), (1, 0)] else ""
            ax.text(j, i, f"{value:,}\n{weight}", ha="center", va="center", fontsize=13)
    
    total = int(table.to_numpy().sum())
    fig.text(0.5, 0.01, f"Sample size: n = {total:,} labeled reviews · Values show observed review counts (unweighted).", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = save_fig(fig, output)
    return fig, table, path


def plot_label_coverage_by_year(df: pd.DataFrame, output: str = "B14_Customer-Reported Delivery Label Coverage by Year.png"):
    """Độ phủ nhãn delivery_rating theo năm, dùng tỷ lệ có trọng số."""
    d = df[df["is_analysable"]].copy()
    d["year"] = pd.to_datetime(d["review_ts"]).dt.year
    d["has_delivery_label"] = d["customer_says_late"].notna()

    rows = []
    for year, g in d.groupby("year", sort=True):
        w = g["weight"].to_numpy(dtype=float)
        labelled = g["has_delivery_label"].to_numpy(dtype=bool)
        coverage = weighted_proportion(labelled, w) * 100
        rows.append({"year": int(year), "coverage_pct": coverage, "n": len(g), "n_eff": effective_sample_size(w)})
    result = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(result["year"], result["coverage_pct"], width=0.72)
    ax.set_xlabel("Year")
    ax.set_ylabel("Weighted Label Coverage (%)")
    ax.set_title("Customer-Reported Delivery Label Coverage by Year")
    ax.set_xticks(result["year"])
    ax.set_ylim(0, max(5, result["coverage_pct"].max() * 1.25))
    for _, row in result.iterrows():
        ax.text(row["year"], row["coverage_pct"] + 0.05, f"n={int(row['n']):,}", ha="center", va="bottom", fontsize=8, rotation=90)
    fig.text(0.5, 0.01, "Trước năm 2023 gần như không có label; từ năm 2023 bắt đầu xuất hiện customer-reported delivery labels.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = save_fig(fig, output)
    return fig, result, path


def plot_lead_days_hist(df: pd.DataFrame, output: str = "B04_lead_days.png"):
    """Histogram raw vs weighted của lead_days, có vạch SLA 5 ngày."""
    d = df[df["is_analysable"] & df["lead_days"].notna()].copy()
    x = d["lead_days"].to_numpy(dtype=float)
    w = d["weight"].to_numpy(dtype=float)
    bins = np.linspace(0, min(30, np.nanpercentile(x, 99.5)), 31)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(x, bins=bins, alpha=0.35, label="Thô", density=True)
    ax.hist(x, bins=bins, weights=w, alpha=0.55, label="Có trọng số", density=True)
    ax.axvline(5, linestyle="--", linewidth=2, label="SLA = 5 ngày")
    ax.set_xlabel("Lead time (ngày)")
    ax.set_ylabel("Mật độ")
    ax.set_title("B4 — Phân phối lead_days: thô và có trọng số")
    ax.legend()
    fig.tight_layout()
    path = save_fig(fig, output)
    return fig, path


def plot_sla_by_year(df: pd.DataFrame, output: str = "B07_sla_by_year.png"):
    """Tỷ lệ vượt SLA theo năm, có trọng số và ghi n."""
    d = df[df["is_analysable"]].copy()
    d["year"] = pd.to_datetime(d["review_ts"]).dt.year
    rows = []
    for year, g in d.groupby("year", sort=True):
        rows.append({
            "year": int(year),
            "breach_pct": weighted_proportion(g["sla_breach"].astype(bool).to_numpy(), g["weight"].to_numpy()) * 100,
            "n": len(g),
        })
    result = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(result["year"], result["breach_pct"])
    ax.set_xlabel("Năm")
    ax.set_ylabel("Vượt SLA (%) — có trọng số")
    ax.set_title("B7 — Tỷ lệ vượt SLA theo năm")
    ax.set_xticks(result["year"])
    for bar, n in zip(bars, result["n"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"n={n:,}", ha="center", va="bottom", fontsize=8, rotation=90)
    fig.tight_layout()
    path = save_fig(fig, output)
    return fig, result, path


def plot_rating_distribution(df: pd.DataFrame, output: str = "B08_rating.png"):
    """Phân phối rating thô vs có trọng số."""
    d = df[df["rating"].notna()].copy()
    ratings = np.arange(1, 6)
    raw = d["rating"].value_counts(normalize=True).reindex(ratings, fill_value=0) * 100
    weighted = [weighted_proportion((d["rating"].to_numpy() == r), d["weight"].to_numpy()) * 100 for r in ratings]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.38
    ax.bar(ratings - width/2, raw.to_numpy(), width, label="Thô")
    ax.bar(ratings + width/2, weighted, width, label="Có trọng số")
    ax.set_xticks(ratings, [f"{r}★" for r in ratings])
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_xlabel("Rating")
    ax.set_title("B8 — Phân phối rating: thô và có trọng số")
    ax.legend()
    fig.tight_layout()
    path = save_fig(fig, output)
    return fig, path
