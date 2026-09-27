"""Metric có trọng số cho dữ liệu lệch lớp — task C11 (metric), C12 (calibration).

Vì sao **không dùng accuracy**: lớp dương chỉ ~2–4% quần thể; đoán "không bao giờ
có review xấu" đã đạt ~97% accuracy mà không có chút giá trị nào.

Metric chính và vai trò:

* **PR-AUC (average precision)** — đo khả năng *xếp hạng* đơn rủi ro lên đầu,
  tập trung vào lớp hiếm. Mốc ngẫu nhiên = tỉ lệ dương (không phải 0,5).
* **Brier score + Brier skill** — đo chất lượng *xác suất* (vừa xếp hạng, vừa hiệu
  chỉnh). Skill = 1 − Brier/Brier_tham_chiếu; > 0 mới là hơn đoán tỉ lệ chung.
* **ROC-AUC** — chỉ ghi kèm: trên dữ liệu lệch, lượng âm khổng lồ làm FPR luôn nhỏ
  nên ROC trông đẹp giả tạo (Saito & Rehmsmeier, 2015).
* **ECE + bảng reliability** — xác suất 10% có thật sự xảy ra 10% không.
* **Precision/recall ở top k%** — câu trả lời vận hành: "nếu CSKH gọi cho 5% đơn
  rủi ro nhất, bắt được bao nhiêu phần trăm review xấu?"

Mọi hàm nhận `w` = trọng số khảo sát → kết quả là **ước lượng quần thể**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from src.evaluate.stats import Estimate, cluster_bootstrap

_EPS = 1e-6


def _arrays(y, p, w):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    if not (len(y) == len(p) == len(w)):
        raise ValueError("y, p, w phải cùng độ dài")
    return y, p, w


def precision_recall_at_top(y, p, w=None, frac: float = 0.05) -> tuple[float, float]:
    """Precision và recall khi chỉ can thiệp `frac` phần quần thể có điểm cao nhất."""
    y, p, w = _arrays(y, p, w)
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(w[order])
    top = cum <= frac * w.sum() + 1e-9
    top[0] = True
    sel = order[top]
    prec = float(np.average(y[sel], weights=w[sel]))
    positives = float((w * y).sum())
    rec = float((w[sel] * y[sel]).sum() / positives) if positives > 0 else float("nan")
    return prec, rec


def reliability_table(y, p, w=None, n_bins: int = 10) -> pd.DataFrame:
    """Bảng calibration theo **phân vị có trọng số** của xác suất dự báo.

    Chia theo phân vị (mỗi bin ~cùng lượng quần thể) thay vì chia đều [0, 1],
    vì với lớp dương 2% gần như mọi dự báo dồn vào bin đầu tiên.
    """
    y, p, w = _arrays(y, p, w)
    order = np.argsort(p)
    cdf = np.cumsum(w[order]) / w.sum()
    edges = np.interp(np.linspace(0, 1, n_bins + 1)[1:-1], cdf, p[order])
    bins = np.searchsorted(np.unique(edges), p, side="right")
    frame = pd.DataFrame({"bin": bins, "y": y, "p": p, "w": w})
    grouped = frame.groupby("bin")
    return pd.DataFrame(
        {
            "p_du_bao": grouped.apply(lambda d: np.average(d["p"], weights=d["w"]), include_groups=False),
            "ti_le_thuc": grouped.apply(lambda d: np.average(d["y"], weights=d["w"]), include_groups=False),
            "weight": grouped["w"].sum(),
            "n": grouped.size(),
        }
    )


def expected_calibration_error(y, p, w=None, n_bins: int = 10) -> float:
    """ECE = Σ_b (W_b / W) · |p̄_b − ȳ_b| trên các bin phân vị có trọng số."""
    tab = reliability_table(y, p, w, n_bins)
    share = tab["weight"] / tab["weight"].sum()
    return float((share * (tab["p_du_bao"] - tab["ti_le_thuc"]).abs()).sum())


def evaluate_predictions(
    y, p, w=None, reference_rate: float | None = None, top_frac: float = 0.05, n_bins: int = 10
) -> dict[str, float]:
    """Toàn bộ metric của một bộ dự báo, có trọng số.

    Args:
        reference_rate: xác suất của dự báo tham chiếu cho Brier skill. Mặc định
            là tỉ lệ dương của chính tập đánh giá ("climatology" — tham chiếu mạnh
            nhất có thể của một hằng số). Notebook truyền tỉ lệ học được từ tập
            train (M0) — tham chiếu *công bằng*, vì không nhìn trước tập test.
    """
    y, p, w = _arrays(y, p, w)
    prevalence = float(np.average(y, weights=w))
    ref = prevalence if reference_rate is None else float(reference_rate)
    brier = float(np.average((p - y) ** 2, weights=w))
    brier_ref = float(np.average((ref - y) ** 2, weights=w))
    prec, rec = precision_recall_at_top(y, p, w, top_frac)
    return {
        "pr_auc": float(average_precision_score(y, p, sample_weight=w)),
        "lift_pr_auc": float(average_precision_score(y, p, sample_weight=w)) / prevalence,
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "brier": brier,
        "brier_skill": 1.0 - brier / brier_ref,
        "log_loss": float(log_loss(y, np.clip(p, _EPS, 1 - _EPS), sample_weight=w)),
        "ece": expected_calibration_error(y, p, w, n_bins),
        f"precision@{top_frac:.0%}": prec,
        f"recall@{top_frac:.0%}": rec,
        "prevalence": prevalence,
        "p_mean": float(np.average(p, weights=w)),
        "n": int(len(y)),
    }


def weighted_pr_auc(frame: pd.DataFrame, pred_col: str, y_col: str, w_col: str) -> float:
    return float(average_precision_score(frame[y_col], frame[pred_col], sample_weight=frame[w_col]))


def paired_pr_auc_bootstrap(
    frame: pd.DataFrame,
    pred_a: str,
    pred_b: str,
    y_col: str = "is_low_rating",
    w_col: str = "weight",
    cluster_col: str = "product_id",
    n_boot: int = 1000,
    seed: int = 42,
) -> Estimate:
    """KTC 95% cho **chênh lệch** PR-AUC(a) − PR-AUC(b), bootstrap theo cụm sản phẩm.

    *Ghép cặp*: mỗi lần lấy lại mẫu, hai mô hình được chấm trên **cùng** tập dòng
    → phần nhiễu chung (tập test dễ/khó) triệt tiêu, KTC của chênh lệch hẹp hơn
    nhiều so với so hai KTC riêng lẻ. *Theo cụm*: review cùng sản phẩm không độc
    lập (cùng hàng, cùng kho) — lấy lại theo dòng sẽ cho KTC hẹp giả tạo.
    Tái sử dụng `cluster_bootstrap` của `src/evaluate/stats.py` (không sửa file đó).
    """

    def stat(d: pd.DataFrame) -> float:
        if d[y_col].sum() == 0:
            return float("nan")
        return weighted_pr_auc(d, pred_a, y_col, w_col) - weighted_pr_auc(d, pred_b, y_col, w_col)

    return cluster_bootstrap(
        frame, stat, cluster_col=cluster_col, n_boot=n_boot, seed=seed,
        name=f"ΔPR-AUC ({pred_a} − {pred_b})", weight_col=w_col,
    )
