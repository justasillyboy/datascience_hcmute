"""Hình cho notebook Track C — chỉ trình bày, không tính toán logic mới.

Quy chuẩn: nền sáng `#fcfcfb`; màu phân loại theo thứ tự cố định (xanh → cam →
ngọc → vàng, đã chạy bộ kiểm tra mù màu); nét 2px; lưới mảnh 1px liền; chữ luôn
dùng màu chữ (không bao giờ tô màu dữ liệu); từ 2 chuỗi trở lên luôn có chú giải.
Hai màu ngọc/vàng dưới 3:1 tương phản → mọi hình dùng chúng đều có bảng số đi kèm
trong notebook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.evaluate.stats import Estimate

from .metrics import reliability_table

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKER = 8


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "legend.frameon": False, "legend.labelcolor": TEXT, "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round", "figure.dpi": 110, "font.size": 10,
    })


def _dot(ax, x, y, color, label=None, zorder=3):
    ax.scatter(x, y, s=MARKER ** 2, color=color, edgecolor=SURFACE, linewidth=2,
               zorder=zorder, label=label)


def plot_sign_flip(table: pd.DataFrame) -> plt.Figure:
    """AUC một biến: mẫu thô vs quần thể. Điểm nằm hai phía vạch 0,5 = quan hệ đảo chiều."""
    t = table.sort_values("auc_quan_the")
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(t) + 1.4))
    y = np.arange(len(t))
    for i, (_, r) in enumerate(t.iterrows()):
        flipped = (r["auc_mau_tho"] - 0.5) * (r["auc_quan_the"] - 0.5) < 0
        ax.plot([r["auc_mau_tho"], r["auc_quan_the"]], [i, i], color=TEXT_2 if flipped else GRID,
                linewidth=2 if flipped else 1.5, zorder=2)
    _dot(ax, t["auc_mau_tho"], y, SERIES[1], "mẫu thô (không trọng số)")
    _dot(ax, t["auc_quan_the"], y, SERIES[0], "quần thể (trọng số khảo sát)")
    ax.axvline(0.5, color=TEXT_2, linewidth=1)
    ax.set_yticks(y, t.index)
    ax.set_xlabel("ROC-AUC một biến trên tập train ≤2023  (0,5 = không phân biệt)")
    ax.set_title("Cùng một feature, hai câu trả lời ngược chiều")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_grouped_bars(df: pd.DataFrame, category: str, series: str, value: str,
                      title: str, xlabel: str) -> plt.Figure:
    """Thanh ngang nhóm (≤4 chuỗi), giá trị ghi ở đầu thanh."""
    cats, sers = list(dict.fromkeys(df[category])), list(dict.fromkeys(df[series]))
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(cats) * len(sers) / 1.6 + 1.4))
    h = 0.8 / len(sers)
    for j, s in enumerate(sers):
        sub = df[df[series] == s].set_index(category).reindex(cats)
        pos = np.arange(len(cats)) + (j - (len(sers) - 1) / 2) * h
        ax.barh(pos, sub[value], height=h * 0.85, color=SERIES[j], label=s)
        for p, v in zip(pos, sub[value]):
            ax.text(v, p, f" {v:.4f}", va="center", fontsize=8, color=TEXT_2)
    ax.set_yticks(np.arange(len(cats)), cats)
    ax.invert_yaxis()
    ax.set_xlim(0, df[value].max() * 1.18)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(sers))
    fig.tight_layout()
    return fig


def plot_tuning(results: pd.DataFrame, params: Sequence[str]) -> plt.Figure:
    """Điểm CV của từng cấu hình theo từng siêu tham số; cấu hình thắng được khoanh."""
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3.2), sharey=True)
    best = results.loc[results["rank"] == 1].iloc[0]
    for ax, prm in zip(np.atleast_1d(axes), params):
        x = pd.to_numeric(results[prm], errors="coerce")
        ax.scatter(x, results["mean_score"], s=36, color=SERIES[0], alpha=0.75, edgecolor=SURFACE)
        _dot(ax, [pd.to_numeric(best[prm], errors="coerce")], [best["mean_score"]], SERIES[1], zorder=4)
        if prm in ("learning_rate", "l2_regularization", "min_samples_leaf"):
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel(prm)
    np.atleast_1d(axes)[0].set_ylabel("PR-AUC trung bình CV")
    fig.suptitle("Random search: mỗi chấm là một cấu hình (cam = thắng)", x=0.01, ha="left",
                 fontweight="bold", color=TEXT)
    fig.tight_layout()
    return fig


def plot_lift(table: pd.DataFrame, sets: Sequence[str]) -> plt.Figure:
    """Lift PR-AUC (= PR-AUC / tỉ lệ dương) của thang mô hình trên từng tập chấm."""
    models = list(dict.fromkeys(table["mô hình"]))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(models) + 1.5))
    y = np.arange(len(models))
    for j, s in enumerate(sets):
        sub = table[table["tập"] == s].set_index("mô hình").reindex(models)
        _dot(ax, sub["lift_pr_auc"], y + (j - 0.5) * 0.18, SERIES[j], s)
    ax.axvline(1.0, color=TEXT_2, linewidth=1)
    ax.text(1.0, -0.55, " = đoán ngẫu nhiên", color=TEXT_2, fontsize=8, va="center")
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlabel("PR-AUC ÷ tỉ lệ dương  (bao nhiêu lần tốt hơn ngẫu nhiên)")
    ax.set_title("Thang mô hình trên tập test 2024–2026", pad=14)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_pr_curves(y, preds: Mapping[str, np.ndarray], w) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    prevalence = float(np.average(y, weights=w))
    for j, (name, p) in enumerate(preds.items()):
        prec, rec, _ = precision_recall_curve(y, p, sample_weight=w)
        ax.plot(rec, prec, color=SERIES[j], label=name)
    ax.axhline(prevalence, color=TEXT_2, linewidth=1)
    ax.text(1.0, prevalence * 0.92, f"ngẫu nhiên = {prevalence:.3f}", color=TEXT_2, fontsize=8,
            ha="right", va="top")
    ax.set_xlabel("Recall (tỉ lệ review xấu bắt được, có trọng số)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, min(1.0, max(0.25, 4 * prevalence)))
    ax.set_title("Đường Precision–Recall (test, có trọng số)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_reliability(y, preds: Mapping[str, np.ndarray], w, n_bins: int = 10) -> plt.Figure:
    """Biểu đồ độ tin cậy trên thang log — xác suất nhỏ nên thang tuyến tính dồn hết vào góc."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lo, hi = 1.0, 0.0
    for j, (name, p) in enumerate(preds.items()):
        tab = reliability_table(y, p, w, n_bins)
        ax.plot(tab["p_du_bao"], tab["ti_le_thuc"], color=SERIES[j], zorder=2)
        _dot(ax, tab["p_du_bao"], tab["ti_le_thuc"], SERIES[j], name)
        pos = tab[["p_du_bao", "ti_le_thuc"]].to_numpy()
        pos = pos[pos > 0]
        lo, hi = min(lo, pos.min()), max(hi, pos.max())
    ax.plot([lo, hi], [lo, hi], color=TEXT_2, linewidth=1, zorder=1)
    ax.text(hi, hi, "hiệu chỉnh hoàn hảo ", color=TEXT_2, fontsize=8, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Xác suất dự báo trung bình trong bin")
    ax.set_ylabel("Tỉ lệ review xấu thật (có trọng số)")
    ax.set_title("Độ tin cậy của xác suất (test)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def _short_pair(name: str) -> str:
    """'ΔPR-AUC (M7 · abc − M2 · xyz)' → 'M7 − M2' (mã mô hình, tên đầy đủ ở bảng)."""
    inner = re.sub(r"^ΔPR-AUC \((.*)\)$", r"\1", name)
    return " − ".join(part.split(" · ")[0].strip() for part in inner.split(" − "))


def plot_forest(estimates: Sequence[Estimate], title: str) -> plt.Figure:
    """KTC 95% của chênh lệch; vạch 0 = không khác biệt. Nhãn trục là mã mô hình."""
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(estimates) + 1.5))
    for i, e in enumerate(estimates):
        color = SERIES[0] if e.lo > 0 or e.hi < 0 else TEXT_2
        ax.plot([e.lo, e.hi], [i, i], color=color, zorder=2)
        _dot(ax, [e.value], [i], color)
        ax.text(e.hi, i, f"  {e.value:+.4f} [{e.lo:+.4f}; {e.hi:+.4f}]", va="center",
                fontsize=8, color=TEXT_2)
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_yticks(range(len(estimates)), [_short_pair(e.name) for e in estimates])
    ax.invert_yaxis()
    ax.set_xlabel("Chênh lệch PR-AUC (xanh = KTC 95% không chứa 0)")
    ax.set_title(title)
    x_lo = min(e.lo for e in estimates)
    x_hi = max(e.hi for e in estimates)
    ax.set_xlim(min(0, x_lo) - 0.1 * (x_hi - x_lo), x_hi + 0.9 * (x_hi - min(0, x_lo)))
    fig.tight_layout()
    return fig


def plot_hbar(values: pd.Series, title: str, xlabel: str, errors: pd.Series | None = None) -> plt.Figure:
    """Thanh ngang một chuỗi, sắp theo giá trị."""
    v = values.sort_values()
    fig, ax = plt.subplots(figsize=(7.0, 0.34 * len(v) + 1.3))
    ax.barh(np.arange(len(v)), v, height=0.6, color=SERIES[0],
            xerr=None if errors is None else errors.reindex(v.index), ecolor=TEXT_2)
    ax.set_yticks(np.arange(len(v)), v.index)
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_dose_response(curves: pd.DataFrame) -> plt.Figure:
    """Hai bảng riêng (không phải hai trục y): xác suất tuyệt đối và rủi ro tương đối."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for j, (era, d) in enumerate(curves.groupby("kỷ nguyên", sort=False)):
        label = f"{era} (n={d['n'].iloc[0]:,}, dương={d['n_dương'].iloc[0]:,})"
        axes[0].plot(d["lead_days"], d["p"], color=SERIES[j], label=label)
        axes[1].plot(d["lead_days"], d["rủi_ro_tương_đối"], color=SERIES[j], label=label)
    for ax in axes:
        ax.axvline(5, color=TEXT_2, linewidth=1)
        ax.set_xlabel("lead_days (ngày từ lúc mua tới lúc nhận)")
    axes[0].text(5, axes[0].get_ylim()[1], " SLA 5 ngày", color=TEXT_2, fontsize=8, va="top")
    axes[0].set_ylabel("P(review ≤3★) — quần thể")
    axes[0].set_title("Rủi ro tuyệt đối")
    axes[1].set_ylabel("Rủi ro ÷ rủi ro khi giao trong 1 ngày")
    axes[1].set_title("Rủi ro tương đối")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_line(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str,
              vline: float | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(df[x], df[y], color=SERIES[0])
    if vline is not None:
        ax.axvline(vline, color=TEXT_2, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig
