"""Tầng chứng minh của Track C: ablation, giải thích, phân tích lỗi (C9, C13, C14).

* `group_ablation` — tầng 7: bỏ từng nhóm feature, train lại, đo PR-AUC mất bao
  nhiêu. Khác permutation importance ở chỗ mô hình được **train lại** nên đo đúng
  "nhóm này có mang thông tin *không thay thế được* không".
* `permutation_importance_table` — xáo trộn một cột trên tập test, đo PR-AUC tụt.
* `partial_dependence` / `dose_response_by_era` — hình dạng quan hệ lead → rủi ro.
* `segment_report` — lỗi tập trung ở phân khúc nào (năm, ngành, nhà bán).
* `sla_trap_table` / `sla_trap_difference` — Tiki Trading vs bên thứ ba (C14).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate.stats import FRAGILE_N_EFF, Estimate, cluster_bootstrap, effective_sample_size
from src.features.build import CLUSTER, FEATURE_GROUPS, TARGET, WEIGHT

from .base import BaseModel
from .experiments import evaluation_sets, training_weights
from .linear import LogisticModel
from .metrics import paired_pr_auc_bootstrap

MIN_POSITIVES = 30  # dưới ngưỡng này PR-AUC của một phân khúc chỉ là nhiễu


def _wap(y, p, w) -> float:
    return float(average_precision_score(y, p, sample_weight=w)) if np.sum(y) > 0 else float("nan")


def univariate_auc_table(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """ROC-AUC của từng feature đứng một mình: trên mẫu thô vs trên quần thể (có trọng số).

    Bằng chứng cho việc bác D9: nếu lấy mẫu chỉ phụ thuộc nhãn thì hai cột phải
    cùng phía so với 0,5. Feature nằm hai phía = quan hệ đảo chiều do thiết kế
    lấy mẫu → mô hình train không trọng số học sai chiều. Chỉ dùng tập train.
    """
    y = train[TARGET].to_numpy()
    w = train[WEIGHT].to_numpy(dtype=float)
    rows = {}
    for f in features:
        x = train[f].fillna(train[f].median()).to_numpy(dtype=float)
        rows[f] = {"auc_mau_tho": roc_auc_score(y, x), "auc_quan_the": roc_auc_score(y, x, sample_weight=w)}
    out = pd.DataFrame(rows).T
    out["đảo_chiều"] = (out["auc_mau_tho"] - 0.5) * (out["auc_quan_the"] - 0.5) < 0
    return out


def group_ablation(model: BaseModel, features: list[str], train: pd.DataFrame, test: pd.DataFrame,
                   scheme: str) -> pd.DataFrame:
    """PR-AUC trên test khi bỏ từng nhóm feature (mô hình train lại, giữ siêu tham số).

    Dùng `model.with_features(...)` — đa hình, chạy được cho cả mô hình đơn,
    ensemble lẫn mô hình đã bọc hiệu chỉnh.
    """
    sets = {k: v for k, v in evaluation_sets(test).items() if "nhãn" not in k}
    y_te, w_te = test[TARGET].to_numpy(), test[WEIGHT].to_numpy(dtype=float)
    full = list(features)
    variants = {"(đầy đủ)": full} | {
        f"bỏ {g}": [f for f in full if f not in feats] for g, feats in FEATURE_GROUPS.items()
    }
    rows = []
    for name, feats in variants.items():
        m = model.with_features(feats)
        m.fit(train, train[TARGET], sample_weight=training_weights(train, scheme))
        p = m.predict_proba(test)[:, 1]
        rows.append({"biến thể": name, "số feature": len(feats),
                     **{s: _wap(y_te[k], p[k], w_te[k]) for s, k in sets.items()}})
    out = pd.DataFrame(rows).set_index("biến thể")
    for s in sets:
        out[f"Δ {s}"] = out[s] - out.loc["(đầy đủ)", s]
    return out


def permutation_importance_table(model: BaseModel, test: pd.DataFrame, features: list[str],
                                 n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """PR-AUC có trọng số tụt bao nhiêu khi xáo trộn từng feature trên tập test."""
    X = test[list(features)]
    r = permutation_importance(model, X, test[TARGET], scoring="average_precision",
                               n_repeats=n_repeats, random_state=seed,
                               sample_weight=test[WEIGHT].to_numpy(dtype=float))
    return (pd.DataFrame({"feature": X.columns, "giảm_pr_auc": r.importances_mean,
                          "độ_lệch_chuẩn": r.importances_std})
            .sort_values("giảm_pr_auc", ascending=False).reset_index(drop=True))


def partial_dependence(model: BaseModel, X: pd.DataFrame, feature: str, grid: np.ndarray,
                       w: np.ndarray, max_rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Rủi ro dự báo trung bình (có trọng số) khi ép `feature` = từng giá trị của `grid`."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(max_rows, len(X)), replace=False)
    Xs, ws = X.iloc[idx].copy(), np.asarray(w, dtype=float)[idx]
    values = []
    for v in grid:
        Xs[feature] = v
        values.append(float(np.average(model.predict_proba(Xs)[:, 1], weights=ws)))
    return pd.DataFrame({feature: grid, "rủi_ro_dự_báo": values})


def dose_response_by_era(frame: pd.DataFrame, eras: dict[str, np.ndarray], grid: np.ndarray) -> pd.DataFrame:
    """P(review xấu | lead_days) theo từng kỷ nguyên — logistic spline, trọng số khảo sát.

    Dùng trọng số `survey` (không cân bằng lớp) để đường cong là xác suất *quần
    thể*. Trả thêm rủi ro tương đối so với mốc giao trong 1 ngày.
    """
    rows = []
    for era, mask in eras.items():
        d = frame[mask]
        m = LogisticModel(features=["lead_days"], spline_features=("lead_days",))
        m.fit(d, d[TARGET], sample_weight=training_weights(d, "survey"))
        p = m.predict_proba(pd.DataFrame({"lead_days": grid}))[:, 1]
        ref = m.predict_proba(pd.DataFrame({"lead_days": [1.0]}))[0, 1]
        rows.append(pd.DataFrame({"kỷ nguyên": era, "lead_days": grid, "p": p, "rủi_ro_tương_đối": p / ref,
                                  "n": int(mask.sum()), "n_dương": int(d[TARGET].sum())}))
    return pd.concat(rows, ignore_index=True)


def segment_report(test: pd.DataFrame, p: np.ndarray, segment_col: str) -> pd.DataFrame:
    """Chất lượng mô hình trên từng phân khúc. Cờ ⚠️ khi quá ít ca dương để tin PR-AUC."""
    rows = []
    for seg, d in test.assign(_p=p).groupby(segment_col, observed=True):
        y, w, pp = d[TARGET].to_numpy(), d[WEIGHT].to_numpy(dtype=float), d["_p"].to_numpy()
        prev = float(np.average(y, weights=w))
        ap = _wap(y, pp, w)
        rows.append({
            segment_col: seg, "n": len(d), "n_dương": int(y.sum()),
            "n_eff": effective_sample_size(w),
            "tỉ_lệ_thật": prev, "xác_suất_TB": float(np.average(pp, weights=w)),
            "pr_auc": ap, "lift": ap / prev if prev > 0 else float("nan"),
            "cờ": "⚠️ ít ca dương" if y.sum() < MIN_POSITIVES else "",
        })
    return pd.DataFrame(rows).set_index(segment_col)


def paired_comparisons(preds: pd.DataFrame, rows: pd.DataFrame, pairs: list[tuple[str, str]],
                       n_boot: int = 1000) -> list[Estimate]:
    """Bootstrap ghép cặp theo cụm cho từng cặp (a, b): KTC 95% của PR-AUC(a) − PR-AUC(b)."""
    frame = rows[[TARGET, WEIGHT, CLUSTER]].join(preds)
    return [paired_pr_auc_bootstrap(frame, a, b, n_boot=n_boot) for a, b in pairs]


# --------------------------------------------------------------------- C14 · bẫy SLA theo nhóm nhà bán

def _rate(d: pd.DataFrame, event: str, given: str, given_value: float) -> float:
    sub = d[d[given] == given_value]
    if sub.empty:
        return float("nan")
    return float(np.average(sub[event], weights=sub[WEIGHT]))


#: Hai tỉ lệ "bẫy SLA" (định nghĩa ở CLAUDE.md §4 / TASKS.md §0.2):
#:   báo động giả = P(khách nói đúng hẹn | vượt SLA)
#:   bỏ sót       = P(trong SLA | khách nói trễ)
TRAP_RATES = {
    "báo_động_giả": ("says_on_time", "sla_breach", 1.0),
    "bỏ_sót": ("within_sla", "customer_says_late", 1.0),
}


def _with_trap_columns(labelled: pd.DataFrame) -> pd.DataFrame:
    return labelled.assign(says_on_time=1.0 - labelled["customer_says_late"],
                           within_sla=1.0 - labelled["sla_breach"])


def sla_trap_table(labelled: pd.DataFrame, group_col: str = "is_tiki_trading",
                   labels: dict | None = None) -> pd.DataFrame:
    """Bảng chéo 2×2 **có trọng số** + hai tỉ lệ bẫy, tách theo nhóm."""
    labels = labels or {1.0: "Tiki Trading", 0.0: "Bên thứ ba"}
    d = _with_trap_columns(labelled)
    rows = []
    for g, sub in d.groupby(group_col):
        w = sub[WEIGHT].to_numpy(dtype=float)
        row = {"nhóm": labels.get(g, g), "n": len(sub), "n_eff": effective_sample_size(w)}
        for breach in (0.0, 1.0):
            for late in (0.0, 1.0):
                cell = (sub["sla_breach"] == breach) & (sub["customer_says_late"] == late)
                name = f"{'vượt' if breach else 'trong'} SLA · khách {'trễ' if late else 'đúng hẹn'}"
                row[name] = int(cell.sum())
        for rate_name, (event, given, value) in TRAP_RATES.items():
            cond = sub[given] == value
            row[rate_name] = _rate(sub, event, given, value)
            n_eff = effective_sample_size(sub.loc[cond, WEIGHT].to_numpy(dtype=float))
            row[f"n_eff_{rate_name}"] = n_eff
            row[f"cờ_{rate_name}"] = "⚠️ MONG MANH" if n_eff < FRAGILE_N_EFF else ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("nhóm")


def sla_trap_difference(labelled: pd.DataFrame, rate_name: str, group_col: str = "is_tiki_trading",
                        n_boot: int = 1000, seed: int = 42) -> Estimate:
    """KTC 95% cho chênh lệch tỉ lệ bẫy (Tiki Trading − bên thứ ba), bootstrap theo cụm sản phẩm."""
    event, given, value = TRAP_RATES[rate_name]
    d = _with_trap_columns(labelled)

    def stat(x: pd.DataFrame) -> float:
        return _rate(x[x[group_col] == 1.0], event, given, value) - _rate(x[x[group_col] == 0.0], event, given, value)

    return cluster_bootstrap(d, stat, cluster_col=CLUSTER, n_boot=n_boot, seed=seed,
                             name=f"Δ {rate_name} (Tiki Trading − bên thứ ba)", weight_col=WEIGHT)


# --------------------------------------------------------------------- thiên lệch thời gian của tầng 5★

def five_star_recency(clean: pd.DataFrame, min_reviews: int = 5) -> pd.DataFrame:
    """Ở sản phẩm bị giới hạn tầng 5★: review 5★ lấy được mới hơn review 1–3★ (vét cạn) bao nhiêu?

    Nếu API trả review theo thứ tự ngẫu nhiên trong tầng, chênh lệch trung vị ngày
    đăng phải quanh 0. Chênh dương ở gần như mọi sản phẩm = API ưu tiên review mới
    → trọng số `N_h/n_h` đúng cho ước lượng *toàn thời gian*, nhưng dồn review 5★
    về các năm gần → ước lượng *theo năm* bị lệch. Đây là lý do cần tập kiểm
    chứng "sản phẩm vét cạn" (không có lấy mẫu nào).
    """
    capped = clean[(clean["stratum"] == 5) & (clean["stratum_sampled"] < clean["stratum_size"])]
    sub = clean[clean[CLUSTER].isin(capped[CLUSTER].unique())]
    gaps = []
    for _, d in sub.groupby(CLUSTER):
        low, five = d.loc[d["rating"] <= 3, "review_ts"], d.loc[d["rating"] == 5, "review_ts"]
        if low.notna().sum() >= min_reviews and five.notna().sum() >= min_reviews:
            gaps.append((five.median() - low.median()).days)
    gaps = np.asarray(gaps, dtype=float)
    return pd.Series({
        "số sản phẩm bị giới hạn tầng 5★": int(capped[CLUSTER].nunique()),
        "số sản phẩm đủ dữ liệu để so": int(gaps.size),
        "trung vị (ngày 5★ − ngày 1–3★)": float(np.median(gaps)),
        "tỉ lệ sản phẩm có 5★ mới hơn": float((gaps > 0).mean()),
    }).to_frame("giá trị")
