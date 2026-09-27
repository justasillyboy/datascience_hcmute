"""Test cho tầng mô hình (Track C · C4, C5, C11, C16).

Trọng tâm là **hợp đồng interface**: mọi mô hình con của `BaseModel` phải thay
thế được cho nhau trong cùng một vòng đánh giá (nguyên lý thay thế Liskov). Nếu
một lớp con phá hợp đồng — trả xác suất ngoài [0, 1], không tương thích `clone`,
tune xong mà không ghi siêu tham số vào chính nó — test ở đây phải đỏ.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from src.models.base import BaseModel
from src.models.baseline import PriorBaseline
from src.models.calibration import CalibratedModel
from src.models.linear import LogisticModel
from src.models.metrics import (
    evaluate_predictions,
    precision_recall_at_top,
    reliability_table,
)
from src.models.splits import YearForwardSplit
from src.models.trees import GBMModel, RandomForestModel

FEATS = ["x1", "x2", "cat"]


def _data(n: int = 1500, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.integers(0, 3, n),
            "year": rng.choice([2020, 2021, 2022, 2023], n),
            "weight": rng.choice([1.0, 4.0], n),
        }
    )
    X.loc[rng.random(n) < 0.05, "x2"] = np.nan  # có NaN để kiểm tra xử lý thiếu
    logit = -2.0 + 1.5 * X["x1"] + 0.8 * (X["cat"] == 2)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def _all_models() -> list[BaseModel]:
    return [
        PriorBaseline(),
        LogisticModel(features=FEATS, categorical=("cat",)),
        LogisticModel(features=["x1"], spline_features=("x1",)),
        GBMModel(features=FEATS, categorical=("cat",), max_iter=50),
        RandomForestModel(features=FEATS, n_estimators=30),
    ]


# --------------------------------------------------------------------- interface


@pytest.mark.parametrize("model", _all_models(), ids=lambda m: type(m).__name__)
def test_moi_mo_hinh_giu_dung_hop_dong(model):
    X, y = _data()
    assert model.fit(X, y) is model  # fit trả về chính nó → gọi nối được
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert list(model.classes_) == [0, 1]


@pytest.mark.parametrize("model", _all_models(), ids=lambda m: type(m).__name__)
def test_clone_giu_sieu_tham_so_khong_giu_trang_thai_da_hoc(model):
    X, y = _data()
    model.fit(X, y)
    twin = clone(model)
    assert twin.get_params() == model.get_params()
    assert not hasattr(twin, "classes_")  # thuộc tính đã học (đuôi `_`) không bị sao chép


def test_khong_the_khoi_tao_lop_truu_tuong():
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


def test_du_doan_truoc_khi_fit_bao_loi():
    with pytest.raises(Exception):
        GBMModel(features=FEATS).predict_proba(_data()[0])


def test_score_la_average_precision_co_trong_so():
    X, y = _data()
    m = LogisticModel(features=FEATS, categorical=("cat",)).fit(X, y)
    w = X["weight"].to_numpy()
    expected = average_precision_score(y, m.predict_proba(X)[:, 1], sample_weight=w)
    assert m.score(X, y, sample_weight=w) == pytest.approx(expected)


def test_prior_baseline_hoc_ti_le_co_trong_so():
    X, y = _data()
    w = X["weight"].to_numpy()
    m = PriorBaseline().fit(X, y, sample_weight=w)
    assert m.prior_ == pytest.approx(np.average(y, weights=w))


# --------------------------------------------------------------------- tuning


def test_tune_ghi_sieu_tham_so_tot_nhat_vao_chinh_doi_tuong():
    X, y = _data()
    model = GBMModel(features=FEATS, categorical=("cat",), max_iter=40)
    space = {"learning_rate": [0.05, 0.2], "max_leaf_nodes": [4, 16]}
    model.tune(X, y, space, cv=YearForwardSplit(val_years=(2022, 2023)), n_iter=4,
               eval_weight=X["weight"].to_numpy())
    # 1) bản ghi kết quả tune là thuộc tính đã học
    assert set(model.best_params_) == {"learning_rate", "max_leaf_nodes"}
    assert len(model.tuning_results_) == 4
    # 2) siêu tham số tốt nhất đã được ghi đè vào CHÍNH thuộc tính của đối tượng
    for k, v in model.best_params_.items():
        assert getattr(model, k) == v
    # 3) sau tune mô hình đã được fit lại trên toàn bộ dữ liệu
    assert hasattr(model, "estimator_")
    # 4) clone mang theo siêu tham số đã tune → tái lập được
    assert clone(model).learning_rate == model.best_params_["learning_rate"]


def test_export_hyperparams_ra_json(tmp_path):
    X, y = _data()
    model = GBMModel(features=FEATS, categorical=("cat",), max_iter=30)
    model.tune(X, y, {"learning_rate": [0.1, 0.3]}, cv=YearForwardSplit(val_years=(2023,)),
               n_iter=2)
    path = model.export_hyperparams(tmp_path / "gbm.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["best_params"] == model.best_params_
    assert payload["params"]["learning_rate"] == model.learning_rate


def test_save_load_giu_nguyen_du_doan(tmp_path):
    X, y = _data()
    m = GBMModel(features=FEATS, categorical=("cat",), max_iter=30).fit(X, y)
    m.save(tmp_path / "m.joblib")
    loaded = GBMModel.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(loaded.predict_proba(X), m.predict_proba(X))


# --------------------------------------------------------------------- splits


def test_year_forward_split_chi_train_tren_qua_khu():
    X, _ = _data()
    cv = YearForwardSplit(val_years=(2022, 2023))
    assert cv.get_n_splits() == 2
    for (tr, va), val_year in zip(cv.split(X), (2022, 2023)):
        assert X["year"].iloc[tr].max() < val_year
        assert set(X["year"].iloc[va]) == {val_year}


# --------------------------------------------------------------------- calibration


def test_calibrated_model_boc_mo_hinh_bat_ky():
    X, y = _data(3000)
    base = GBMModel(features=FEATS, categorical=("cat",), max_iter=40)
    cal = CalibratedModel(base_model=base, method="sigmoid", calib_year=2023).fit(X, y)
    p = cal.predict_proba(X)[:, 1]
    assert np.all((p > 0) & (p < 1))
    assert cal.estimator_.learning_rate == base.learning_rate


# --------------------------------------------------------------------- metrics


def test_metric_co_trong_so_bang_metric_thuong_khi_trong_so_deu():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    m = evaluate_predictions(y, p, np.ones(400))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p))
    assert m["prevalence"] == pytest.approx(y.mean())


def test_brier_skill_cua_baseline_bang_khong():
    y = np.array([0, 0, 0, 1])
    m = evaluate_predictions(y, np.full(4, 0.25), np.ones(4), reference_rate=0.25)
    assert m["brier_skill"] == pytest.approx(0.0)


def test_precision_recall_at_top():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    prec, rec = precision_recall_at_top(y, p, np.ones(10), frac=0.2)
    assert prec == pytest.approx(1.0) and rec == pytest.approx(1.0)


def test_reliability_table_bao_toan_tong_trong_so():
    rng = np.random.default_rng(2)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    w = rng.choice([1.0, 5.0], 500)
    tab = reliability_table(y, p, w, n_bins=5)
    assert tab["weight"].sum() == pytest.approx(w.sum())


# --------------------------------------------------------------------- composite + đa hình


def test_ensemble_la_trung_binh_xac_suat_thanh_vien():
    from src.models.ensemble import EnsembleModel

    X, y = _data()
    a = LogisticModel(features=FEATS, categorical=("cat",))
    b = GBMModel(features=FEATS, categorical=("cat",), max_iter=40)
    ens = EnsembleModel(members=(a, b)).fit(X, y)
    pa = clone(a).fit(X, y).predict_proba(X)[:, 1]
    pb = clone(b).fit(X, y).predict_proba(X)[:, 1]
    np.testing.assert_allclose(ens.predict_proba(X)[:, 1], (pa + pb) / 2)
    assert not hasattr(a, "classes_")  # thành viên gốc không bị train ké


def test_with_features_da_hinh_cho_moi_loai_mo_hinh():
    from src.models.ensemble import EnsembleModel

    gbm = GBMModel(features=FEATS, categorical=("cat",), learning_rate=0.07)
    ens = EnsembleModel(members=(gbm, LogisticModel(features=FEATS)))
    cal = CalibratedModel(base_model=ens)
    new = cal.with_features(["x1"])
    assert all(m.features == ["x1"] for m in new.base_model.members)
    assert new.base_model.members[0].learning_rate == 0.07  # siêu tham số được giữ
    assert gbm.features == FEATS  # bản gốc không bị sửa
