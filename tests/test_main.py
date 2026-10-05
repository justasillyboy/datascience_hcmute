"""Test cho `main.py` (pipeline chạy bằng một lệnh) và adapter `PipelineModel`.

Hợp đồng cần giữ:
* `PipelineModel` đưa trọng số train vào đúng bước cuối (`model__sample_weight`), clone được;
* lưới `quick` chỉ chứa siêu tham số hợp lệ của pipeline thật;
* tham số dòng lệnh: CV không được chạm năm test; ghi đè siêu tham số không sửa cấu hình gốc;
* chạy trọn bước [5b]–[6] trên dữ liệu giả lập đủ cột feature thật → mô hình chốt có xác suất
  đã hiệu chỉnh và hơn hẳn baseline tỉ lệ chung.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid

import main
from src.features.build import EX_ANTE_FEATURES, SLA_FEATURE, TARGET, WEIGHT, YEAR
from src.models.calibration import CalibratedModel
from src.models.grid_search import make_pipeline, quick_param_grid
from src.models.pipeline_model import PipelineModel


def _frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Bảng giả lập có đủ mọi cột mà `main` đọc; nhãn phụ thuộc lead_days và category."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in EX_ANTE_FEATURES})
    df["category"] = rng.integers(0, 4, n)
    df["lead_days"] = rng.gamma(2.0, 1.0, n)
    df[SLA_FEATURE] = (df["lead_days"] > 5).astype(float)
    df[YEAR] = rng.choice([2020, 2021, 2022, 2023, 2024], n)
    df[WEIGHT] = rng.choice([1.0, 5.0], n)
    df.loc[rng.random(n) < 0.1, "cust_tenure_days"] = np.nan
    logit = -3.0 + 0.6 * df["lead_days"] + 0.8 * (df["category"] == 2)
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["review_id"] = np.arange(n)
    return df


# --------------------------------------------------------------------- PipelineModel


def test_pipeline_model_dua_trong_so_vao_buoc_model(monkeypatch):
    df = _frame(600)
    seen = {}
    original = LogisticRegression.fit

    def spy(self, X, y, sample_weight=None):
        seen["w"] = sample_weight
        return original(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(LogisticRegression, "fit", spy)
    w = df[WEIGHT].to_numpy()
    PipelineModel(make_pipeline()).fit(df, df[TARGET], sample_weight=w)
    np.testing.assert_array_equal(seen["w"], w)


def test_pipeline_model_clone_giu_sieu_tham_so_va_du_doan_dung_hinh_dang():
    df = _frame(600)
    model = PipelineModel(make_pipeline().set_params(model__C=0.01))
    copy = clone(model)
    assert copy.pipeline.named_steps["model"].C == 0.01
    proba = copy.fit(df, df[TARGET]).predict_proba(df)
    assert proba.shape == (len(df), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_pipeline_model_thieu_pipeline_bao_loi():
    with pytest.raises(ValueError):
        PipelineModel().fit(_frame(100), _frame(100)[TARGET])


# --------------------------------------------------------------------- lưới quick & dòng lệnh


def test_luoi_quick_chi_chua_sieu_tham_so_hop_le():
    pipe = make_pipeline()
    for params in ParameterGrid(quick_param_grid(pos_weight=21.5)):
        pipe.set_params(**params)  # khoá sai tên → ValueError


def test_cv_khong_duoc_cham_nam_test():
    with pytest.raises(SystemExit):
        main.parse_args(["--test-start-year", "2023", "--cv-years", "2022", "2023"])


def test_tham_so_mac_dinh():
    args = main.parse_args([])
    assert args.search == "none"
    assert args.models == list(main.MODEL_CHOICES)
    assert args.cv_years == [2022, 2023]


def test_ghi_de_sieu_tham_so_khong_sua_cau_hinh_goc():
    base = main.chosen_params(pos_weight=20.0)
    args = main.parse_args(["--C", "0.5", "--class-weight", "none", "--impute", "median"])
    out = main.apply_overrides(base, args, pos_weight=20.0)
    assert out["Logistic"]["model__C"] == 0.5
    assert all(p["model__class_weight"] is None for p in out.values())
    assert all(p["prep__num__impute__strategy"] == "median" for p in out.values())
    assert base["Logistic"]["model__C"] == 1e-4  # bản gốc không đổi
    assert base["Logistic"]["model__class_weight"] == {0: 1.0, 1: 20.0}


def test_cau_hinh_chon_san_dat_duoc_vao_pipeline():
    for params in main.chosen_params(pos_weight=21.5).values():
        clone(make_pipeline()).set_params(**params)


def test_luon_co_baseline_va_mo_hinh_chot_duoc_hieu_chinh():
    models = main.build_models(main.chosen_params(21.5), ["logistic", "ensemble"], "sigmoid", calib_year=2023)
    assert list(models)[:2] == list(main.BASELINES)
    assert isinstance(models[main.FINAL_NAME], CalibratedModel)
    raw = main.build_models(main.chosen_params(21.5), ["ensemble"], "none", calib_year=2023)
    assert not isinstance(raw[main.FINAL_NAME], CalibratedModel)


# --------------------------------------------------------------------- chạy trọn bước train → chấm


def test_train_va_cham_tren_du_lieu_gia_lap():
    df = _frame()
    train, test = df[df[YEAR] < 2024].copy(), df[df[YEAR] >= 2024].copy()
    params = main.chosen_params(pos_weight=5.0)
    params["RandomForest"]["model"] = params["RandomForest"]["model"].set_params(n_estimators=20)
    params["HistGBM"]["model__min_samples_leaf"] = 20
    models = main.build_models(params, list(main.MODEL_CHOICES), "sigmoid", calib_year=2023)

    preds = main.fit_and_predict(models, train, test)
    metrics = main.evaluate(preds, test)

    assert list(metrics.index) == list(models)
    assert metrics.loc[main.BASELINES[0], "lift_pr_auc"] == pytest.approx(1.0)
    final = metrics.loc[main.FINAL_NAME]
    assert final["lift_pr_auc"] > 1.5
    # class_weight="balanced" phóng xác suất của GBM lên ~0,5; hiệu chỉnh phải kéo về gần tỉ lệ thật.
    # (Không so bằng tuyệt đối: hiệu chỉnh học trên năm 2023, tỉ lệ năm đó trong dữ liệu giả lập có nhiễu.)
    raw_gap = abs(metrics.loc["HistGBM (GridSearchCV)", "p_mean"] - final["prevalence"])
    assert abs(final["p_mean"] - final["prevalence"]) < raw_gap / 2
