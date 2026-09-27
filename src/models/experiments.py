"""Các bước thí nghiệm của Track C — notebook và `run_modeling` gọi chung.

Nguyên tắc chọn mô hình: **mọi lựa chọn (phương án trọng số, họ mô hình, siêu
tham số) đều quyết định bằng CV theo thời gian trên tập train ≤2023.** Tập test
2024–2026 chỉ được chấm **một lần** ở cuối, cho các mô hình đã chốt. Chọn mô hình
bằng tập test là một dạng leakage — điểm test khi đó không còn là ước lượng
không chệch của hiệu năng tương lai.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from src.features.build import (
    CATEGORICAL,
    EX_ANTE_FEATURES,
    EX_POST_FEATURES,
    SLA_FEATURE,
    TARGET,
    WEIGHT,
    build_features,
)
from src.features.leakage import audit_features

from .base import BaseModel
from .baseline import PriorBaseline
from .calibration import CalibratedModel
from .ensemble import EnsembleModel
from .linear import LogisticModel
from .metrics import evaluate_predictions
from .splits import YearForwardSplit
from .trees import GBMModel, RandomForestModel

CLEAN_PATH = Path("data/processed/reviews_clean.parquet")
PRODUCTS_PATH = Path("data/raw/tiki_products.parquet")
ARTIFACT_DIR = Path("reports/models")

#: CV theo thời gian cho mọi lựa chọn: train ≤2021→val 2022, train ≤2022→val 2023.
CV = YearForwardSplit(val_years=(2022, 2023))
WEIGHT_SCHEMES = ("none", "survey", "survey_balanced")
SEED = 42

#: Không gian tìm kiếm của GBM — lý do từng chiều ở notebook §7.
GBM_SEARCH_SPACE = {
    "learning_rate": loguniform(0.02, 0.2),
    "max_leaf_nodes": randint(8, 64),
    "max_depth": [None, 4, 6, 8],
    "min_samples_leaf": [20, 50, 100, 200, 400, 800],
    "l2_regularization": loguniform(1e-3, 10.0),
    "max_features": uniform(0.4, 0.6),
}
GBM_MAX_ITER = 2000  # trần; số cây thật do dừng sớm quyết định (`n_iter_`)
#: Logistic: `C` nhân với TỔNG trọng số mẫu trong hàm mục tiêu của sklearn, mà tổng trọng số
#: khảo sát ~ quy mô quần thể → C tối ưu rất nhỏ. Khoảng rộng để không chạm biên.
LOGISTIC_SEARCH_SPACE = {"C": loguniform(1e-6, 1.0)}


def load_frame(root: Path = Path(".")) -> pd.DataFrame:
    """Đọc bảng sạch + bảng sản phẩm → bảng mô hình."""
    clean = pd.read_parquet(root / CLEAN_PATH)
    products = pd.read_parquet(root / PRODUCTS_PATH)
    return build_features(clean, products)


def training_weights(frame: pd.DataFrame, scheme: str) -> np.ndarray | None:
    """Trọng số dùng khi *train*.

    * `none` — D9 gốc: mọi dòng như nhau.
    * `survey` — trọng số khảo sát `w = N_h/n_h`: hàm mất mát trên mẫu trở thành
      ước lượng không chệch của hàm mất mát trên quần thể (importance weighting).
    * `survey_balanced` — như `survey` rồi nhân lớp dương lên để tổng trọng số hai
      lớp bằng nhau: sửa chệch lấy mẫu *và* bắt mô hình chú ý lớp hiếm.
    """
    if scheme == "none":
        return None
    w = frame[WEIGHT].to_numpy(dtype=float)
    if scheme == "survey":
        return w
    if scheme == "survey_balanced":
        y = frame[TARGET].to_numpy()
        factor = w[y == 0].sum() / w[y == 1].sum()
        return np.where(y == 1, w * factor, w)
    raise ValueError(f"scheme phải thuộc {WEIGHT_SCHEMES}, nhận {scheme!r}")


def cv_scores(model: BaseModel, train: pd.DataFrame, scheme: str, cv=CV) -> dict[str, float]:
    """PR-AUC có trọng số trên từng fold CV theo thời gian."""
    y = train[TARGET].to_numpy()
    sw = training_weights(train, scheme)
    ew = train[WEIGHT].to_numpy(dtype=float)
    out = {}
    for (tr, va), val_year in zip(cv.split(train), cv.val_years):
        fitted = clone(model).fit(train.iloc[tr], y[tr], sample_weight=None if sw is None else sw[tr])
        p = fitted.predict_proba(train.iloc[va])[:, 1]
        out[f"val_{val_year}"] = float(average_precision_score(y[va], p, sample_weight=ew[va]))
    out["mean"] = float(np.mean(list(out.values())))
    return out


def kfold_vs_temporal(train: pd.DataFrame, scheme: str, seed: int = SEED) -> pd.DataFrame:
    """Leakage thời gian đo được: cùng một mô hình, KFold xáo trộn vs chia theo năm.

    KFold cho mô hình học từ review 2023 để đoán review 2021 — điểm sẽ đẹp hơn điểm
    thật ở tương lai. Chênh lệch giữa hai cột chính là "độ lạc quan giả" mà cách
    chia sai sẽ báo cáo. Dùng logistic cho nhanh; kết luận không phụ thuộc mô hình.
    """
    from sklearn.model_selection import KFold

    model = LogisticModel(features=EX_ANTE_FEATURES, categorical=CATEGORICAL)
    y = train[TARGET].to_numpy()
    sw = training_weights(train, scheme)
    ew = train[WEIGHT].to_numpy(dtype=float)

    def score(splits) -> float:
        out = []
        for tr, va in splits:
            m = clone(model).fit(train.iloc[tr], y[tr], sample_weight=None if sw is None else sw[tr])
            out.append(average_precision_score(y[va], m.predict_proba(train.iloc[va])[:, 1],
                                               sample_weight=ew[va]))
        return float(np.mean(out))

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed).split(train)
    return pd.DataFrame({
        "cách chia": ["KFold xáo trộn 5 fold (SAI)", "theo năm: val 2022, 2023 (ĐÚNG)"],
        "PR-AUC CV": [score(kfold), score(CV.split(train))],
    }).set_index("cách chia")


def default_models() -> dict[str, Callable[[], BaseModel]]:
    """Ba họ mô hình ở cấu hình mặc định — đầu vào cho bước "chọn họ mô hình"."""
    return {
        "Logistic (L2)": lambda: LogisticModel(features=EX_ANTE_FEATURES, categorical=CATEGORICAL),
        "Random forest": lambda: RandomForestModel(features=EX_ANTE_FEATURES),
        "Gradient boosting": lambda: GBMModel(features=EX_ANTE_FEATURES, categorical=CATEGORICAL),
    }


def compare_families_and_weights(train: pd.DataFrame, schemes=WEIGHT_SCHEMES) -> pd.DataFrame:
    """Bảng CV: họ mô hình × phương án trọng số. Quyết định D9 và "vì sao chọn GBM"."""
    rows = []
    for family, factory in default_models().items():
        for scheme in schemes:
            rows.append({"họ mô hình": family, "trọng số train": scheme,
                         **cv_scores(factory(), train, scheme)})
    return pd.DataFrame(rows)


def choose_scheme(families: pd.DataFrame, family: str = "Gradient boosting") -> str:
    """Phương án trọng số có PR-AUC CV cao nhất cho họ mô hình đã chọn — quyết định bằng CV."""
    rows = families[families["họ mô hình"] == family]
    return str(rows.loc[rows["mean"].idxmax(), "trọng số train"])


def load_or_tune(path: Path, tune_fn: Callable[[], BaseModel], use_cache: bool) -> BaseModel:
    """Đọc mô hình đã tune từ đĩa nếu có (và được phép), nếu không thì tune rồi lưu.

    Mô hình lưu bằng joblib mang theo **toàn bộ** thuộc tính của đối tượng — cả
    siêu tham số đã tune lẫn `best_params_` và `tuning_results_` — nên notebook
    đọc lại vẫn in được bảng tune mà không phải chạy lại 7 phút.
    """
    if use_cache and path.exists():
        try:
            return BaseModel.load(path)
        except Exception as err:  # vd khác phiên bản scikit-learn giữa máy lưu và máy đọc
            print(f"Không đọc được {path} ({type(err).__name__}: {err}) → tune lại")
    model = tune_fn()
    model.save(path)
    model.export_hyperparams(path.with_suffix(".json"))
    return model


def tune_gbm(
    train: pd.DataFrame,
    scheme: str,
    features: list[str] = EX_ANTE_FEATURES,
    n_iter: int = 40,
    verbose: bool = False,
) -> GBMModel:
    """Random search trên CV theo thời gian. Siêu tham số tốt nhất được ghi vào chính đối tượng."""
    audit_features(features, tier="ex_post" if set(features) & set(EX_POST_FEATURES) else "ex_ante")
    model = GBMModel(features=features, categorical=CATEGORICAL, max_iter=GBM_MAX_ITER,
                     random_state=SEED)
    return model.tune(
        train, train[TARGET], GBM_SEARCH_SPACE, cv=CV, n_iter=n_iter,
        sample_weight=training_weights(train, scheme),
        eval_weight=train[WEIGHT].to_numpy(dtype=float),
        random_state=SEED, verbose=verbose,
    )


def tune_logistic(train: pd.DataFrame, scheme: str, n_iter: int = 15) -> LogisticModel:
    """Tune `C` cho logistic — để so GBM với một logistic *cũng đã tune* (so sánh công bằng)."""
    model = LogisticModel(features=EX_ANTE_FEATURES, categorical=CATEGORICAL)
    return model.tune(
        train, train[TARGET], LOGISTIC_SEARCH_SPACE, cv=CV, n_iter=n_iter,
        sample_weight=training_weights(train, scheme),
        eval_weight=train[WEIGHT].to_numpy(dtype=float), random_state=SEED,
    )


#: Mô hình dự báo cuối cùng (ex-ante): ensemble đã hiệu chỉnh — lý do ở `final_candidates`.
FINAL_MODEL = "M7 · ensemble + hiệu chỉnh (CHỐT)"


def make_ensemble(tuned: GBMModel, tuned_logistic: LogisticModel) -> EnsembleModel:
    return EnsembleModel(members=(tuned, tuned_logistic))


def final_candidates(train: pd.DataFrame, tuned: GBMModel, tuned_logistic: LogisticModel,
                     scheme: str) -> pd.DataFrame:
    """Bảng CV chọn mô hình cuối: GBM tuned · logistic tuned · ensemble hai mô hình.

    Quy tắc chọn đặt trước: điểm CV trung bình cao nhất. Cả ba dùng đúng các fold
    và siêu tham số đã tune, nên cùng mang một độ lạc quan như nhau khi so với nhau.
    """
    rows = []
    for name, model in (("GBM tuned", tuned), ("logistic tuned", tuned_logistic),
                        ("ensemble GBM + logistic", make_ensemble(tuned, tuned_logistic))):
        rows.append({"ứng viên": name, **cv_scores(model, train, scheme)})
    return pd.DataFrame(rows).set_index("ứng viên")


def build_ladder(tuned: GBMModel, tuned_logistic: LogisticModel) -> dict[str, BaseModel]:
    """Bậc thang mô hình (tầng 4): naive → tuyến tính → cây → tuned → hiệu chỉnh.

    Mọi mô hình ex_ante được kiểm toán leakage ngay khi khai báo.
    """
    for feats, linear in ((EX_ANTE_FEATURES, True), ([SLA_FEATURE], True), (["lead_days"], True)):
        audit_features(feats, tier="ex_ante", linear=linear)
    expost = tuned.get_params() | {"features": EX_ANTE_FEATURES + EX_POST_FEATURES}
    ensemble = make_ensemble(tuned, tuned_logistic)
    return {
        "M0 · tỉ lệ chung": PriorBaseline(),
        "M1a · chỉ sla_breach": LogisticModel(features=[SLA_FEATURE]),
        "M1b · chỉ lead_days (spline)": LogisticModel(features=["lead_days"], spline_features=("lead_days",)),
        "M2 · logistic tuned": tuned_logistic,
        "M3 · random forest": RandomForestModel(features=EX_ANTE_FEATURES),
        "M4 · GBM mặc định": GBMModel(features=EX_ANTE_FEATURES, categorical=CATEGORICAL),
        "M5 · GBM tuned": tuned,
        "M6 · ensemble GBM + logistic": ensemble,
        FINAL_MODEL: CalibratedModel(base_model=ensemble, method="sigmoid"),
        "M8 · GBM tuned + tín hiệu khách (ex-post)": GBMModel(**expost),
    }


def fit_ladder(
    ladder: dict[str, BaseModel], train: pd.DataFrame, test: pd.DataFrame, scheme: str
) -> pd.DataFrame:
    """Train từng mô hình trên toàn bộ train, trả bảng xác suất dự báo trên test.

    M0 luôn học tỉ lệ *có trọng số* — định nghĩa của "tỉ lệ chung quần thể".
    """
    y = train[TARGET].to_numpy()
    sw = training_weights(train, scheme)
    preds = pd.DataFrame(index=test.index)
    for name, model in ladder.items():
        weights = train[WEIGHT].to_numpy(dtype=float) if isinstance(model, PriorBaseline) else sw
        if getattr(model, "best_params_", None) is not None:
            fitted = model  # tune(refit=True) đã fit lại trên toàn bộ train, cùng trọng số
        else:
            fitted = model.fit(train, y, sample_weight=weights)
        preds[name] = fitted.predict_proba(test)[:, 1]
    return preds


def evaluation_sets(test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Ba tập chấm điểm. Tập "sản phẩm vét cạn" không có lấy mẫu → không cần trọng số."""
    return {
        "toàn bộ test (có trọng số)": np.ones(len(test), dtype=bool),
        "sản phẩm vét cạn (không lấy mẫu)": test["is_census"].to_numpy(dtype=bool),
        "có nhãn khách (2024+)": test["has_customer_label"].to_numpy(dtype=bool),
    }


def evaluation_table(preds: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Mô hình × tập chấm → toàn bộ metric có trọng số."""
    rows = []
    y = test[TARGET].to_numpy()
    w = test[WEIGHT].to_numpy(dtype=float)
    for set_name, mask in evaluation_sets(test).items():
        for model_name in preds.columns:
            m = evaluate_predictions(y[mask], preds[model_name].to_numpy()[mask], w[mask])
            rows.append({"tập": set_name, "mô hình": model_name, **m})
    return pd.DataFrame(rows)


def c9_models() -> dict[str, tuple[BaseModel, bool]]:
    """Ablation trung tâm (C9). Giá trị: (mô hình, chỉ train trên dòng có nhãn khách?)."""
    return {
        "M1a · sla_breach (doanh nghiệp đo)": (LogisticModel(features=[SLA_FEATURE]), False),
        "M1b · lead_days (thông tin đầy đủ)": (
            LogisticModel(features=["lead_days"], spline_features=("lead_days",)), False),
        "M1c · khách nói trễ (khách cảm nhận)": (LogisticModel(features=["customer_says_late"]), True),
        "M1d · lead_days + khách nói trễ": (
            LogisticModel(features=["lead_days", "customer_says_late"],
                          spline_features=("lead_days",)), True),
    }


def fit_c9(train: pd.DataFrame, test: pd.DataFrame, scheme: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train 4 mô hình C9, chấm trên **cùng** tập test có nhãn khách → so sánh ghép cặp."""
    labelled_train = train[train["has_customer_label"]]
    labelled_test = test[test["has_customer_label"]]
    preds = pd.DataFrame(index=labelled_test.index)
    for name, (model, needs_label) in c9_models().items():
        tr = labelled_train if needs_label else train
        model.fit(tr, tr[TARGET], sample_weight=training_weights(tr, scheme))
        preds[name] = model.predict_proba(labelled_test)[:, 1]
    return preds, labelled_test
