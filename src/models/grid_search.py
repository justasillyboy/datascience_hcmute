"""GridSearchCV trên Pipeline (tiền xử lý + estimator) — task C8b.

Cách chuẩn của scikit-learn để tune *mà không rò rỉ*
---------------------------------------------------
```
Pipeline([("prep",  ColumnTransformer(...)),     # điền thiếu, chuẩn hoá, one-hot
          ("model", <estimator>)])               # LogisticRegression / RF / HistGBM
```
`GridSearchCV` clone **cả pipeline** cho từng cấu hình × từng fold, nên trung vị để
điền thiếu, trung bình/độ lệch chuẩn để chuẩn hoá… chỉ được `fit` trên phần train
của fold đó, không bao giờ nhìn phần kiểm định (tầng 3 — leakage).

Siêu tham số của một bước được gọi bằng **`<tên bước>__<tham số>`**, lồng bao nhiêu
tầng cũng được:

* `model__C` → `C` của `LogisticRegression` trong bước `model`;
* `prep__num__impute__strategy` → bước `prep` → nhánh `num` → bước `impute` → `strategy`;
* `model` → **chính estimator** cũng là một siêu tham số: lưới dạng *danh sách dict*
  cho phép một lần `GridSearchCV` so sánh ba họ mô hình trên cùng tiền xử lý,
  cùng fold, cùng thước đo;
* `model__sample_weight` (truyền vào `fit`) → trọng số đi thẳng tới `fit` của bước
  `model`; GridSearchCV tự cắt nó theo chỉ số train của từng fold.

Mất cân bằng lớp là một siêu tham số
------------------------------------
Lớp dương (review ≤3★) chỉ ~4,45% quần thể train. `model__class_weight` thử ba cách:
`None` (chỉ trọng số khảo sát), `"balanced"` (sklearn tự cân theo **số dòng**) và
`{0: 1, 1: r}` với r = tổng trọng số lớp âm / lớp dương ≈ 21,5 (cân bằng theo **quần
thể** — chính là phương án `survey_balanced` của notebook 03). CV chọn, không chọn tay.
Không dùng SMOTE/undersample: mẫu đã được thiết kế phân tầng, và nhân bản dòng sẽ phá
trọng số khảo sát `w = N_h/n_h`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build import CATEGORICAL, EX_ANTE_FEATURES, TARGET, WEIGHT

from .splits import YearForwardSplit

SEED = 42
NUMERIC_FEATURES: list[str] = [f for f in EX_ANTE_FEATURES if f not in CATEGORICAL]
#: CV theo thời gian giống notebook 03: train ≤2021→val 2022, train ≤2022→val 2023.
CV = YearForwardSplit(val_years=(2022, 2023))

_FAMILY = {
    "LogisticRegression": "Logistic",
    "RandomForestClassifier": "RandomForest",
    "HistGradientBoostingClassifier": "HistGBM",
}


# --------------------------------------------------------------------- pipeline


def make_preprocessor(numeric: Sequence[str] = NUMERIC_FEATURES,
                      categorical: Sequence[str] = CATEGORICAL) -> ColumnTransformer:
    """Bước `prep`: số → điền thiếu (+ cờ thiếu) → chuẩn hoá; phân loại → one-hot.

    Cột không khai báo (`year`, `weight`, nhãn…) bị bỏ (`remainder="drop"`), nên X
    truyền vào được phép mang cột meta — scorer cần đọc `weight` từ đó.
    """
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num, list(numeric)), ("cat", cat, list(categorical))],
                             remainder="drop")


def make_pipeline(estimator=None, numeric: Sequence[str] = NUMERIC_FEATURES,
                  categorical: Sequence[str] = CATEGORICAL) -> Pipeline:
    """`Pipeline([("prep", ...), ("model", estimator)])` — estimator mặc định là logistic."""
    model = LogisticRegression(max_iter=2000) if estimator is None else estimator
    return Pipeline([("prep", make_preprocessor(numeric, categorical)), ("model", model)])


def family_of(estimator) -> str:
    return _FAMILY.get(type(estimator).__name__, type(estimator).__name__)


def param_grid(pos_weight: float) -> list[dict[str, list]]:
    """Lưới tìm kiếm — mỗi dict là một họ estimator, khoá viết theo `bước__tham_số`.

    Lý do chọn từng khoảng (giải thích đầy đủ ở notebook 04 §4). Nguyên tắc chung: giá trị
    thắng phải nằm **bên trong** lưới — thắng ở mép nghĩa là tối ưu có thể nằm ngoài vùng đã thử.
    Lượt chạy đầu (2026-10-01) có ba tham số thắng ở mép (`C`=1e-4, RF `min_samples_leaf`=200,
    GBM `max_leaf_nodes`=8) nên lưới được mở rộng về phía đó.

    * `model__C` (logistic): C = 1/λ; quét 7 bậc độ lớn 1e-6…1.
    * `model__min_samples_leaf`: lớp dương hiếm — lá nhỏ chỉ chứa 0–1 ca dương → học nhiễu.
    * `model__max_leaf_nodes` (GBM): số lá ~ bậc tương tác giữa các feature.
    * `model__learning_rate` (GBM): nhỏ → nhiều cây hơn, mượt hơn; số cây do dừng sớm quyết định.
    * `model__class_weight`: ba cách xử lý mất cân bằng (docstring module).
    """
    imbalance = [None, "balanced", {0: 1.0, 1: float(pos_weight)}]
    return [
        {
            "model": [LogisticRegression(max_iter=2000)],
            "model__C": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "model__class_weight": imbalance,
            "prep__num__impute__strategy": ["median", "mean"],
        },
        {
            "model": [RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)],
            "model__min_samples_leaf": [50, 200, 800, 2000],
            "model__max_features": ["sqrt", 0.5],
            "model__class_weight": [None, "balanced_subsample", {0: 1.0, 1: float(pos_weight)}],
        },
        {
            "model": [HistGradientBoostingClassifier(max_iter=2000, early_stopping=True,
                                                     n_iter_no_change=30, random_state=SEED)],
            "model__learning_rate": [0.03, 0.1],
            "model__max_leaf_nodes": [4, 8, 31],
            "model__min_samples_leaf": [20, 100, 400],
            "model__class_weight": imbalance,
        },
    ]


def quick_param_grid(pos_weight: float) -> list[dict[str, list]]:
    """Lưới rút gọn (10 cấu hình) quanh vùng thắng của `param_grid` — chạy vài phút thay vì ~20.

    Dùng để trình diễn GridSearchCV từ đầu tới cuối (`python main.py --search quick`);
    mỗi tham số vẫn có giá trị hai bên cấu hình thắng để đọc được hướng của đường CV.
    """
    population = {0: 1.0, 1: float(pos_weight)}
    return [
        {
            "model": [LogisticRegression(max_iter=2000)],
            "model__C": [1e-5, 1e-4, 1e-3],
            "model__class_weight": [None, population],
        },
        {
            "model": [RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)],
            "model__min_samples_leaf": [200, 800],
            "model__class_weight": [population],
        },
        {
            "model": [HistGradientBoostingClassifier(max_iter=2000, early_stopping=True,
                                                     n_iter_no_change=30, random_state=SEED)],
            "model__learning_rate": [0.1],
            "model__max_leaf_nodes": [4, 8],
            "model__min_samples_leaf": [400],
            "model__class_weight": ["balanced"],
        },
    ]


# --------------------------------------------------------------------- trọng số & thước đo


def normalized_survey_weights(frame: pd.DataFrame) -> np.ndarray:
    """Trọng số khảo sát `N_h/n_h` chia cho trung bình → trung bình 1.

    Không đổi tỉ lệ giữa các dòng, chỉ đổi thang đo — để `C` của logistic (vốn nhân
    với *tổng* trọng số) có cùng ý nghĩa như khi train không trọng số.
    """
    w = frame[WEIGHT].to_numpy(dtype=float)
    return w / w.mean()


def population_ratio(frame: pd.DataFrame) -> float:
    """Tổng trọng số lớp âm / lớp dương — tỉ lệ mất cân bằng *quần thể*."""
    y = frame[TARGET].to_numpy()
    w = frame[WEIGHT].to_numpy(dtype=float)
    return float(w[y == 0].sum() / w[y == 1].sum())


def weighted_pr_auc(estimator, X: pd.DataFrame, y) -> float:
    """Scorer cho GridSearchCV: PR-AUC có trọng số khảo sát, đọc từ cột `weight` của X.

    Accuracy vô nghĩa ở đây: đoán toàn "không phải review xấu" đã đúng ~98% quần thể.
    """
    p = estimator.predict_proba(X)[:, 1]
    return float(average_precision_score(y, p, sample_weight=X[WEIGHT].to_numpy(dtype=float)))


def class_balance_table(sets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Mỗi tập: số dòng, số dương, tỉ lệ dương thô và có trọng số, tỉ lệ âm:dương quần thể."""
    rows = {}
    for name, d in sets.items():
        y = d[TARGET].to_numpy()
        w = d[WEIGHT].to_numpy(dtype=float)
        rows[name] = {
            "n": len(d),
            "n_dương": int(y.sum()),
            "tỉ_lệ_thô": float(y.mean()),
            "tỉ_lệ_quần_thể": float(np.average(y, weights=w)),
            "âm_trên_dương_quần_thể": population_ratio(d),
        }
    return pd.DataFrame(rows).T.astype({"n": int, "n_dương": int})


# --------------------------------------------------------------------- chạy & đọc kết quả


def run_grid_search(train: pd.DataFrame, grid: list[dict] | None = None, cv=CV,
                    pipeline: Pipeline | None = None, verbose: int = 0) -> GridSearchCV:
    """Một lần GridSearchCV cho mọi họ estimator; `refit=True` → `best_estimator_` train trên toàn bộ train."""
    grid = param_grid(round(population_ratio(train), 1)) if grid is None else grid
    search = GridSearchCV(
        make_pipeline() if pipeline is None else pipeline,
        grid,
        scoring=weighted_pr_auc,
        cv=cv,
        refit=True,
        return_train_score=True,
        error_score="raise",
        n_jobs=1,  # HistGBM và RF đã đa luồng bên trong; song song thêm chỉ tranh CPU
        verbose=verbose,
    )
    return search.fit(train, train[TARGET].to_numpy(), model__sample_weight=normalized_survey_weights(train))


def _label(value: Any) -> Any:
    """Giá trị siêu tham số → nhãn đọc được (dict class_weight → '1:21.5')."""
    if value is None:
        return "None"
    if isinstance(value, Mapping):
        return f"1:{value[1]:g}"
    return value


def jsonable_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Bộ siêu tham số (có estimator, dict class_weight, số numpy) → dict ghi được ra JSON."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k == "model":
            out[k] = type(v).__name__
        elif isinstance(v, Mapping):
            out[k] = {str(c): float(w) for c, w in v.items()}
        else:
            out[k] = v.item() if isinstance(v, np.generic) else v
    return out


def results_table(search: GridSearchCV) -> pd.DataFrame:
    """`cv_results_` → bảng gọn: họ estimator, siêu tham số (tên đầy đủ có `__`), điểm, hạng."""
    res = search.cv_results_
    params = pd.DataFrame([{k: _label(v) for k, v in p.items() if k != "model"} for p in res["params"]])
    folds = sorted(k for k in res if k.startswith("split") and k.endswith("_test_score"))
    table = pd.concat([
        pd.DataFrame({"họ": [family_of(p["model"]) for p in res["params"]]}),
        params,
        pd.DataFrame({k: res[k] for k in ["mean_test_score", "std_test_score", *folds,
                                          "mean_train_score", "mean_fit_time"]}),
    ], axis=1)
    table.insert(0, "rank", res["rank_test_score"])
    table["rank_trong_họ"] = (table.groupby("họ")["mean_test_score"]
                              .rank(ascending=False, method="min").astype(int))
    return table.sort_values("rank").reset_index(drop=True)


def best_params_by_family(search: GridSearchCV) -> dict[str, dict[str, Any]]:
    """Cấu hình có điểm CV cao nhất **trong từng họ** (dạng dict truyền thẳng vào `set_params`)."""
    res = search.cv_results_
    best: dict[str, tuple[float, dict]] = {}
    for score, params in zip(res["mean_test_score"], res["params"]):
        fam = family_of(params["model"])
        if fam not in best or score > best[fam][0]:
            best[fam] = (float(score), params)
    return {fam: params for fam, (_, params) in best.items()}


def validation_slices(table: pd.DataFrame, family: str) -> dict[str, pd.DataFrame]:
    """Validation curve từ lưới: đổi MỘT siêu tham số, giữ các tham số khác ở cấu hình thắng của họ."""
    rows = table[table["họ"] == family].dropna(axis=1, how="all")
    meta = {"rank", "họ", "rank_trong_họ", "mean_test_score", "std_test_score",
            "mean_train_score", "mean_fit_time"}
    params = [c for c in rows.columns if c not in meta and not c.startswith("split")]
    varied = [p for p in params if rows[p].nunique() > 1]
    best = rows.loc[rows["mean_test_score"].idxmax()]
    out = {}
    for prm in varied:
        others = [p for p in varied if p != prm]
        mask = np.logical_and.reduce([rows[o] == best[o] for o in others]) if others else np.ones(len(rows), bool)
        out[prm] = rows[mask].reset_index(drop=True)
    return out


def seed_stability(best_params: Mapping[str, Mapping[str, Any]], train: pd.DataFrame,
                   test: pd.DataFrame, seeds: Sequence[int] = (0, 1, 2, 3, 4),
                   pipeline: Pipeline | None = None) -> pd.DataFrame:
    """Train lại cấu hình thắng của mỗi họ với nhiều `random_state` → độ dao động của PR-AUC test.

    Logistic (lbfgs) là bài toán lồi, không có ngẫu nhiên → mọi seed cho cùng một số.
    RF (bootstrap) và HistGBM (tập kiểm định nội bộ cho dừng sớm) thì có.
    """
    base = make_pipeline() if pipeline is None else pipeline
    y_tr = train[TARGET].to_numpy()
    sw = normalized_survey_weights(train)
    rows = []
    for fam, params in best_params.items():
        for seed in seeds:
            pipe = clone(base).set_params(**params)
            if "random_state" in pipe.named_steps["model"].get_params():
                pipe.set_params(model__random_state=seed)
            pipe.fit(train, y_tr, model__sample_weight=sw)
            rows.append({"họ": fam, "seed": seed,
                         "pr_auc_test": weighted_pr_auc(pipe, test, test[TARGET].to_numpy())})
    return pd.DataFrame(rows)
