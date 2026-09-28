"""Các figure EDA chính của Track B.

Các hàm ở đây nhận DataFrame đã clean. Notebook chỉ gọi hàm và ghi kết luận.
"""
from __future__ import annotations
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluate.stats import effective_sample_size, weighted_mean, weighted_proportion,cluster_bootstrap,weighted_quantile
from src.viz.theme import PALETTE, save_fig
from src.viz.weighted import weighted_box


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
        ax.text(row["year"], row["coverage_pct"] + 0.05, f"n={int(row['n']):,}", ha="center", va="bottom", fontsize=8, rotation=0)
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
    ax.hist(
        x,
        bins=bins,
        histtype="step",
        linewidth=2,
        label="Raw Sample",
        density=True
    )

    ax.hist(
        x,
        bins=bins,
        weights=w,
        alpha=0.55,
        label="Weighted",
        density=True
    )

    ax.axvline(
        5,
        linestyle="--",
        linewidth=2,
        label="SLA = 5 ngày"
    )
    
    ax.set_xlim(0, 15)
    
    ax.text(
    5.2,
    ax.get_ylim()[1] * 0.85,
    "Vùng SLA Breach",
    fontsize=10
    )
    
    ax.set_xlabel("Lead Time (ngày)")
    ax.set_ylabel("Mật độ")
    ax.set_title("Phân phối Lead Time giao hàng")
    ax.legend()
    fig.tight_layout()
    path = save_fig(fig, output)
    return fig, path

def plot_lead_days_box_by_year(
    df: pd.DataFrame,
    output: str = "B05_lead_days_by_year.png",
):
    """B5: Weighted boxplot Lead Time theo năm."""
    d = df[
        df["is_analysable"]
        & df["lead_days"].notna()
        & df["weight"].notna()
    ].copy()

    d["year"] = pd.to_datetime(d["review_ts"]).dt.year

    # Chỉ giữ các năm có đủ số quan sát để boxplot có ý nghĩa.
    year_counts = d.groupby("year").size()
    years = year_counts[year_counts >= 100].index.to_list()

    d = d[d["year"].isin(years)].copy()

    fig, ax = plt.subplots(figsize=(11, 5.8))

    stats = weighted_box(
        values=d["lead_days"].to_numpy(dtype=float),
        groups=d["year"].to_numpy(),
        weights=d["weight"].to_numpy(dtype=float),
        labels=years,
        ax=ax,
        widths=0.6,
        patch_artist=True,
        boxprops={"alpha": 0.65},
        medianprops={"linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    # SLA 5 ngày
    ax.axhline(
        5,
        linestyle="--",
        linewidth=1.8,
        label="SLA = 5 ngày",
    )

    ax.set_xlabel("Năm", labelpad=32)
    ax.set_ylabel("Lead Time (ngày)")
    ax.set_title("Phân phối Lead Time theo năm")

    ax.legend()

    # Ghi sample size dưới mỗi năm.
    n_by_year = d.groupby("year").size()

    for i, year in enumerate(years, start=1):
        ax.text(
            i,
            -0.07,
            f"n={int(n_by_year.loc[year]):,}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )

    fig.text(
        0.5,
        0.01,
        "Boxplot sử dụng weighted quantiles; các năm có n < 100 không được hiển thị.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.11, 1, 1])

    path = save_fig(fig, output)

    result = pd.DataFrame(
        {
            "year": years,
            "n": [int(n_by_year.loc[y]) for y in years],
            "q1": [s["q1"] for s in stats],
            "median": [s["med"] for s in stats],
            "q3": [s["q3"] for s in stats],
        }
    )

    return fig, result, path

def plot_lead_days_quantiles(
    df: pd.DataFrame,
    output: str = "B06_lead_days_quantiles.png",
):
    """B6: Các weighted quantiles của Lead Time từ p10 đến p99."""
    d = df[
        df["is_analysable"]
        & df["lead_days"].notna()
        & df["weight"].notna()
    ].copy()

    values = d["lead_days"].to_numpy(dtype=float)
    weights = d["weight"].to_numpy(dtype=float)

    # Các phân vị cần thể hiện trong B6.
    probs = np.array([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    quantiles = np.array(
        [
            weighted_quantile(values, weights, q)
            for q in probs
        ],
        dtype=float,
    )

    percentile_labels = [
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    x = np.arange(len(probs))

    ax.plot(
        x,
        quantiles,
        marker="o",
        linewidth=2,
        markersize=7,
        label="Weighted Quantile",
    )

    # Ngưỡng SLA của hệ thống.
    ax.axhline(
        5.0,
        linestyle="--",
        linewidth=1.8,
        label="SLA = 5 ngày",
    )

    # Ghi giá trị Lead Time tại từng phân vị.
    for xi, value in zip(x, quantiles):
        ax.annotate(
            f"{value:.2f}",
            xy=(xi, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(percentile_labels)

    ax.set_xlabel("Phân vị")
    ax.set_ylabel("Lead Time (ngày)")
    ax.set_title("Các phân vị có trọng số của Lead Time")

    ax.legend()

    # Làm nổi bật median để truyền tải thông điệp của task B6.
    median_value = float(quantiles[2])
    median_sla_pct = median_value / 5.0 * 100.0

    ax.annotate(
        f"Median = {median_value:.2f} ngày\n"
        f"≈ {median_sla_pct:.0f}% của SLA 5 ngày",
        xy=(x[2], median_value),
        xytext=(45, 45),
        textcoords="offset points",
        ha="left",
        va="bottom",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.2,
        },
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.4",
            "alpha": 0.85,
        },
    )

    fig.text(
        0.5,
        0.01,
        "Các phân vị được tính với trọng số để phản ánh phân phối của quần thể.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    path = save_fig(fig, output)

    result = pd.DataFrame(
        {
            "percentile": percentile_labels,
            "probability": probs,
            "lead_days": quantiles,
        }
    )

    return fig, result, path


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
    ax.set_ylabel("SLA Breach Rate (%)")
    ax.set_title("Tỷ lệ SLA Breach theo năm")
    ax.set_xticks(result["year"])
    ax.set_ylim(
    0,
    result["breach_pct"].max() * 1.15
    )
    for bar, pct, n in zip(
    bars,
    result["breach_pct"],
    result["n"]
    ):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()

        ax.text(
            x,
            y + 0.4,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )

        ax.text(
            x,
            y + 1.5,
            f"n={n:,}",
            ha="center",
            va="bottom",
            fontsize=7
        )
    fig.text(
    0.5,
    0.01,
    "Lưu ý: các năm đầu có sample size nhỏ; SLA Breach Rate được tính có trọng số.",
    ha="center",
    fontsize=9
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
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
    bars_raw = ax.bar(
    ratings - width/2,
    raw.to_numpy(),
    width,
    label="Raw Sample"
    )

    bars_weighted = ax.bar(
    ratings + width/2,
    weighted,
    width,
    label="Weighted"
    )
    ax.set_xticks(ratings, [f"{r}★" for r in ratings])
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_xlabel("Rating")
    ax.set_title("Phân phối Rating của khách hàng")
    ax.legend()
    for bar in bars_raw:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )

    for bar in bars_weighted:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )
    fig.text(
        0.5,
        0.01,
        "Raw Sample phản ánh mẫu quan sát; Weighted phản ánh phân phối sau khi điều chỉnh stratified sampling.",
        ha="center",
        fontsize=9
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = save_fig(fig, output)
    return fig, path

def plot_rating_by_lead_time(
    df: pd.DataFrame,
    output: str = "B09_rating_by_lead_time.png"
):
    """Weighted Mean Rating theo các khoảng Lead Time."""

    d = df[
        df["is_analysable"]
        & df["lead_days"].notna()
        & (df["lead_days"] >= 0)
        & df["rating"].notna()
        & df["weight"].notna()
        
    ].copy()
    
    bins = [
        -np.inf,
        1,
        2,
        3,
        5,
        10,
        np.inf
    ]

    labels = [
        "0–1",
        "1–2",
        "2–3",
        "3–5",
        "5–10",
        ">10"
    ]
    
    d["lead_time_bin"] = pd.cut(
        d["lead_days"],
        bins=bins,
        labels=labels,
        right=True
    )
    rows = []
    def rating_statistic(x):
        return weighted_mean(
            x["rating"].to_numpy(dtype=float),
            x["weight"].to_numpy(dtype=float)
        )
    for label in labels:
        g = d[d["lead_time_bin"] == label]

        if g.empty:
            continue
        estimate = cluster_bootstrap(
            g,
            statistic=rating_statistic,
            cluster_col="product_id",
            n_boot=10000,
            name="weighted_mean_rating",
            weight_col="weight"
        )
        
        rows.append({
            "lead_time_bin": label,
            "weighted_mean_rating": estimate.value,
            "ci_low": estimate.lo,
            "ci_high": estimate.hi,
            "n": estimate.n,
            "n_eff": estimate.n_eff,
            "n_clusters": estimate.n_clusters
        })
    result = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    x = np.arange(len(result))

    y = result["weighted_mean_rating"].to_numpy()

    yerr = np.vstack([
        y - result["ci_low"].to_numpy(),
        result["ci_high"].to_numpy() - y
    ])
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        linewidth=2,
        markersize=7,
        capsize=5,
        label="Weighted Mean Rating (95% CI)"
    )
    
    # Đặt label cao hơn đầu trên của error bar
    upper_error = yerr[1]

    for xi, yi, err_up in zip(x, y, upper_error):
        label_y = yi + err_up

        ax.annotate(
            f"{yi:.2f}",
            xy=(xi, label_y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            zorder=10,
    )
    
    ax.axvline(
        3.5,
        linestyle="--",
        linewidth=1.5,
        label="SLA = 5 ngày"
    )
    
    ax.text(
        1.5,
        ax.get_ylim()[0] + 0.01,
        "Within SLA",
        ha="center",
        fontsize=9
    )

    ax.text(
        4.5,
        ax.get_ylim()[0] + 0.01,
        "SLA Breach",
        ha="center",
        fontsize=9
    )
    
    
    
    ax.set_xticks(
        x,
        result["lead_time_bin"]
    )

    ax.set_xlabel("Lead Time (ngày)")
    ax.set_ylabel("Weighted Mean Rating")
    ax.set_title("Rating theo Lead Time giao hàng")
    ax.legend(
        loc="upper right",
        fontsize=9
    )
    fig.text(
        0.5,
        0.01,
        "Điểm biểu diễn Weighted Mean Rating; error bars biểu diễn 95% CI từ cluster bootstrap theo product.",
        ha="center",
        fontsize=9
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    path = save_fig(fig, output)

    return fig, result, path

def plot_batch_forest(
    output: str = "B11_forest_plot.png"
):
    """Forest plot kiểm tra độ ổn định của kết luận chính qua các lần phân tích."""

    result = pd.DataFrame({
        "stage": [
            "Mẫu 66.851",
            "Mẫu 117.819",
            "117.819 + sửa múi giờ",
            "Mẫu 203.510"
        ],
        "estimate": [
            0.209,
            0.241,
            0.234,
            0.2082
        ],
        "ci_low": [
            0.060,
            0.129,
            0.113,
            0.097
        ],
        "ci_high": [
            0.440,
            0.398,
            0.393,
            0.353
        ],
        "n_eff": [
            1019,
            1960,
            1960,
            4689
        ]
    })

    # ---------------------------------------------------------
    # Chuẩn bị dữ liệu
    # ---------------------------------------------------------
    y = np.arange(len(result))
    estimate = result["estimate"].to_numpy()

    xerr = np.vstack([
        estimate - result["ci_low"].to_numpy(),
        result["ci_high"].to_numpy() - estimate
    ])

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 6.2))

    # 95% CI + point estimate
    ax.errorbar(
        estimate,
        y,
        xerr=xerr,
        fmt="o",
        markersize=8,
        capsize=5,
        linewidth=2
    )

    # Mốc không có chênh lệch
    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.5
    )

    # ---------------------------------------------------------
    # Nhãn các lần phân tích
    # ---------------------------------------------------------
    ax.set_yticks(y)
    ax.set_yticklabels(result["stage"], fontsize=10)

    ax.invert_yaxis()

    # Tạo khoảng trống phía trên cho header
    ax.set_ylim(3.35, -0.65)

    # ---------------------------------------------------------
    # Cột số liệu bên phải
    # ---------------------------------------------------------
    x_ci = 0.46
    x_neff = 0.64

    for yi, row in result.iterrows():

        estimate_ci = (
            f"{row['estimate']:+.3f} "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )

        ax.text(
            x_ci,
            yi,
            estimate_ci,
            ha="left",
            va="center",
            fontsize=9
        )

        ax.text(
            x_neff,
            yi,
            f"{int(row['n_eff']):,}",
            ha="center",
            va="center",
            fontsize=9
        )

    # Header của hai cột
    header_y = -0.43

    ax.text(
        x_ci,
        header_y,
        "Estimate [95% CI]",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold"
    )

    ax.text(
        x_neff,
        header_y,
        "n_eff",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold"
    )

    # ---------------------------------------------------------
    # Trục và tiêu đề
    # ---------------------------------------------------------
    ax.set_xlim(-0.02, 0.70)

    ax.set_xlabel(
        "Chênh lệch sức phân biệt (nhãn khách hàng − SLA)",
        fontsize=10
    )

    ax.set_ylabel("")

    ax.set_title(
        "Độ ổn định của kết luận chính qua các lần phân tích",
        fontsize=14,
        pad=22
    )

    # ---------------------------------------------------------
    # Chú thích mốc 0
    # ---------------------------------------------------------
    ax.annotate(
        "Không có chênh lệch",
        xy=(0, 3.35),
        xytext=(0, -22),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8,
        annotation_clip=False
    )

    # ---------------------------------------------------------
    # Ghi chú học thuật
    # ---------------------------------------------------------
    fig.text(
        0.5,
        0.025,
        "Điểm biểu diễn estimate; thanh ngang biểu diễn 95% CI. "
        "Cả bốn KTC đều không chứa 0.",
        ha="center",
        fontsize=9
    )

    # Dành riêng khoảng trống cho title và footnote
    fig.subplots_adjust(
        left=0.19,
        right=0.97,
        top=0.84,
        bottom=0.17
    )

    path = save_fig(fig, output)

    return fig, result, path


def plot_timezone_correction(
    output: str = "B12_timezone_correction.png"
):
    """Minh họa ảnh hưởng của sửa múi giờ và kiểm chứng độ lệch 7 giờ."""

    # Pilot n=490: các quantile lead time trước/sau sửa múi giờ.
    quantiles = ["p25", "Median", "p75"]

    before = np.array([
        0.47,
        1.07,
        2.12
    ])

    after = np.array([
        0.18,
        0.78,
        1.83
    ])

    y = np.arange(len(quantiles))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 5.8),
        gridspec_kw={
            "width_ratios": [1.5, 1]
        }
    )

    # =========================================================
    # PANEL A — Lead Time trước/sau sửa múi giờ
    # =========================================================
    ax1.scatter(
        before,
        y,
        s=70,
        label="Trước sửa múi giờ",
        zorder=3
    )

    ax1.scatter(
        after,
        y,
        s=70,
        label="Sau sửa múi giờ",
        zorder=3
    )

    # Nối từng cặp before/after
    for yi, x_before, x_after in zip(y, before, after):
        ax1.plot(
            [x_after, x_before],
            [yi, yi],
            linewidth=2,
            alpha=0.55,
            color="gray",
            zorder=1
        )

        ax1.annotate(
            f"{x_before:.2f}",
            (x_before, yi),
            xytext=(7, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8
        )

        ax1.annotate(
            f"{x_after:.2f}",
            (x_after, yi),
            xytext=(-7, 8),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(quantiles)
    ax1.invert_yaxis()

    ax1.set_xlabel("Lead Time (ngày)")
    ax1.set_title("A. Phân phối Lead Time dịch sau khi sửa múi giờ")
    ax1.legend(fontsize=9)

    ax1.text(
        0.5,
        -0.16,
        "Mỗi quantile dịch ≈ 0,2917 ngày ≈ 7 giờ (pilot n=490).",
        transform=ax1.transAxes,
        ha="center",
        fontsize=9
    )

    # =========================================================
    # PANEL B — Kiểm chứng hai đồng hồ
    # =========================================================
    measured_offset = 7.0

    ax2.scatter(
        [measured_offset],
        [0],
        s=120,
        zorder=3
    )

    ax2.axvline(
        7,
        linestyle="--",
        linewidth=1.5
    )

    ax2.set_xlim(6.5, 7.5)
    ax2.set_ylim(-0.7, 0.7)

    ax2.set_yticks([])
    ax2.set_xlabel("Độ lệch thời gian (giờ)")
    ax2.set_title("B. Kiểm chứng \"hai đồng hồ\"")

    ax2.text(
        7,
        0.32,
        "7.0000 giờ",
        ha="center",
        fontsize=13,
        fontweight="bold"
    )

    ax2.text(
        7,
        0.13,
        "SD = 0.000000 giờ",
        ha="center",
        fontsize=10
    )

    ax2.text(
        7,
        -0.18,
        "min = 7.0000 h\n"
        "max = 7.0000 h\n"
        "n = 200,290",
        ha="center",
        va="top",
        fontsize=9
    )

    # =========================================================
    # Tiêu đề chung
    # =========================================================
    fig.suptitle(
        "Một API, hai đồng hồ: ảnh hưởng của sai lệch múi giờ",
        fontsize=14,
        y=0.97
    )

    fig.text(
        0.5,
        0.015,
        "Timestamp dạng chuỗi và epoch UTC lệch nhau đúng 7 giờ; "
        "sửa offset làm Lead Time giảm tương ứng khoảng 0,2917 ngày.",
        ha="center",
        fontsize=9
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        top=0.82,
        bottom=0.20,
        wspace=0.28
    )

    path = save_fig(fig, output)

    result = pd.DataFrame({
        "quantile": quantiles,
        "before_days": before,
        "after_days": after,
        "shift_days": before - after
    })

    return fig, result, path

def plot_n_vs_neff(
    output: str = "B13_n_vs_neff.png"
):
    """So sánh số quan sát thô n và cỡ mẫu hiệu dụng n_eff qua 3 mẻ."""

    result = pd.DataFrame({
        "batch": [
            "Mẻ ban đầu",
            "Mẫu 117.819",
            "Mẫu 203.510"
        ],
        "n": [
            182,
            258,
            317
        ],
        "n_eff": [
            25,
            48,
            85
        ]
    })

    result["information_ratio"] = (
        result["n_eff"] / result["n"] * 100
    )

    x = np.arange(len(result))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.8))

    # ---------------------------------------------------------
    # Hai cột n và n_eff
    # ---------------------------------------------------------
    bars_n = ax.bar(
        x - width / 2,
        result["n"],
        width,
        label="n (số dòng)"
    )

    bars_neff = ax.bar(
        x + width / 2,
        result["n_eff"],
        width,
        label="n_eff (cỡ mẫu hiệu dụng)"
    )

    # ---------------------------------------------------------
    # Ngưỡng mong manh
    # ---------------------------------------------------------
        # ---------------------------------------------------------
    # Ngưỡng n_eff dùng để đánh dấu ước lượng mong manh
    # ---------------------------------------------------------
    ax.axhline(
        y=100,
        linestyle="--",
        linewidth=1.5,
        zorder=0
    )

    # Đặt chú thích ở vùng trống phía trên để không đè lên cột
    ax.text(
        0.985,
        0.92,
        "Ngưỡng mong manh\n$n_{eff}$ = 100",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.75",
            alpha=0.9
        )
    )

    # ---------------------------------------------------------
    # Nhãn trên cột
    # ---------------------------------------------------------
    for bar in bars_n:
        height = bar.get_height()

        ax.annotate(
            f"{int(height)}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    for bar, ratio in zip(
        bars_neff,
        result["information_ratio"]
    ):
        height = bar.get_height()

        ax.annotate(
            f"{int(height)}\n({ratio:.1f}% của n)",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    # ---------------------------------------------------------
    # Trục và tiêu đề
    # ---------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(result["batch"])

    ax.set_ylabel("Cỡ mẫu")
    ax.set_title(
        "Số dòng không đồng nghĩa với lượng thông tin hiệu dụng",
        fontsize=14,
        pad=15
    )

    ax.legend(
        loc="upper left",
        fontsize=9
    )

    ax.set_ylim(
        0,
        result["n"].max() * 1.22
    )

    # ---------------------------------------------------------
    # Footnote
    # ---------------------------------------------------------
    fig.text(
        0.5,
        0.025,
        "Cùng nhóm ước lượng qua ba mẻ dữ liệu: trọng số không đều làm "
        "n_eff nhỏ hơn nhiều so với n; n_eff < 100 được đánh dấu là mong manh.",
        ha="center",
        fontsize=9
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.87,
        bottom=0.17
    )

    path = save_fig(fig, output)

    return fig, result, path

def plot_seller_sla_trap(
    csv_path,
    output: str = "B19_seller_sla_trap.png",
):
    """
    B19: So sánh hai biểu hiện của bẫy SLA giữa
    Tiki Trading và bên thứ ba từ kết quả C14.

    Dữ liệu được đọc trực tiếp từ reports/models/c14_sla_trap.csv,
    không hard-code các tỷ lệ trong hàm.
    """
    d = pd.read_csv(csv_path).copy()

    required_cols = {
        "nhóm",
        "n",
        "báo_động_giả",
        "n_eff_báo_động_giả",
        "bỏ_sót",
        "n_eff_bỏ_sót",
    }
    missing = required_cols.difference(d.columns)
    if missing:
        raise ValueError(
            f"Thiếu cột trong C14: {sorted(missing)}"
        )

    # Giữ thứ tự cố định để hình dễ đọc.
    order = ["Tiki Trading", "Bên thứ ba"]
    d = d.set_index("nhóm").reindex(order)

    if d[["báo_động_giả", "bỏ_sót"]].isna().any().any():
        raise ValueError(
            "Không tìm thấy đầy đủ hai nhóm Tiki Trading/Bên thứ ba trong C14."
        )

    x = np.arange(len(order))
    width = 0.34

    false_alarm = d["báo_động_giả"].to_numpy(dtype=float) * 100
    miss = d["bỏ_sót"].to_numpy(dtype=float) * 100

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    bars_false = ax.bar(
        x - width / 2,
        false_alarm,
        width,
        label="Báo động giả: vượt SLA nhưng khách nói đúng hẹn",
        alpha=0.8,
    )
    bars_miss = ax.bar(
        x + width / 2,
        miss,
        width,
        label="Bỏ sót: trong SLA nhưng khách nói trễ",
        alpha=0.8,
    )

    # Ghi tỷ lệ trên từng cột.
    for bars in (bars_false, bars_miss):
        for bar in bars:
            value = bar.get_height()
            ax.annotate(
                f"{value:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Sample size của từng nhóm.
    for i, group in enumerate(order):
        ax.text(
            i,
            -0.08,
            f"n={int(d.loc[group, 'n']):,}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Tỷ lệ (%)")
    fig.suptitle(
    "Phân rã bẫy SLA theo nhóm nhà bán",
    fontsize=14,
    y=0.97,
    )
    ax.set_ylim(0, 100)
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        frameon=True,
        ncol=2,
        fontsize=9,
    )

    fig.text(
        0.5,
        0.01,
        "C14: khác biệt giữa hai nhóm chưa đủ bằng chứng thống kê "
        "(95% CI của cả hai chênh lệch đều chứa 0).",
        ha="center",
        fontsize=9,
    )

    fig.subplots_adjust(
    top=0.78,
    bottom=0.20,
    left=0.10,
    right=0.97,
    )
    path = save_fig(fig, output)

    result = d[
        [
            "n",
            "báo_động_giả",
            "n_eff_báo_động_giả",
            "bỏ_sót",
            "n_eff_bỏ_sót",
        ]
    ].reset_index()

    return fig, result, path