"""Test cho GridSearchCV trên Pipeline (Track C · C8b).

Hợp đồng cần giữ:
* Pipeline có đúng hai bước `prep` → `model`, và mọi khoá trong lưới (`model__C`,
  `prep__num__impute__strategy`…) đều là siêu tham số hợp lệ của pipeline;
* scorer chấm PR-AUC **có trọng số khảo sát** đọc từ cột `weight` của X;
* trọng số train đi vào estimator qua `model__sample_weight`, và chỉ phần train
  của từng fold được dùng (GridSearchCV tự cắt theo chỉ số);
* bảng kết quả, cấu hình tốt nhất theo từng họ mô hình, và lát cắt validation
  curve đọc đúng từ `cv_results_`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

from src.models.grid_search import (
    class_balance_table,
    family_of,
    make_pipeline,
    normalized_survey_weights,
    param_grid,
    population_ratio,
    results_table,
    run_grid_search,
    seed_stability,
    validation_slices,
    weighted_pr_auc,
)
from src.models.splits import YearForwardSplit

NUM = ["x1", "x2"]
CAT = ("cat",)


def _frame(n: int = 1200, seed: int = 0) -> pd.DataFrame:
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
    X.loc[rng.random(n) < 0.05, "x2"] = np.nan
    logit = -2.0 + 1.5 * X["x1"] + 0.8 * (X["cat"] == 2)
    X["is_low_rating"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X


def _small_grid() -> list[dict]:
    return [
        {"model": [LogisticRegression(max_iter=500)], "model__C": [0.01, 1.0],
         "prep__num__impute__strategy": ["median", "mean"]},
        {"model": [HistGradientBoostingClassifier(max_iter=30, random_state=0)],
         "model__max_leaf_nodes": [4, 8]},
    ]


def _search():
    df = _frame()
    gs = run_grid_search(df, grid=_small_grid(), cv=YearForwardSplit((2022, 2023)),
                         pipeline=make_pipeline(numeric=NUM, categorical=CAT))
    return df, gs


# --------------------------------------------------------------------- pipeline & lưới


def test_pipeline_co_hai_buoc_prep_roi_model():
    pipe = make_pipeline(numeric=NUM, categorical=CAT)
    assert [name for name, _ in pipe.steps] == ["prep", "model"]


def test_dau_gach_doi_dieu_khien_ca_tien_xu_ly_lan_estimator():
    pipe = make_pipeline(numeric=NUM, categorical=CAT)
    pipe.set_params(model__C=0.123, prep__num__impute__strategy="mean")
    assert pipe.named_steps["model"].C == 0.123
    imputer = pipe.named_steps["prep"].transformers[0][1].named_steps["impute"]
    assert imputer.strategy == "mean"


def test_moi_khoa_trong_luoi_that_la_sieu_tham_so_hop_le():
    pipe = make_pipeline(numeric=NUM, categorical=CAT)
    for params in ParameterGrid(param_grid(pos_weight=20.0)):
        pipe.set_params(**params)  # khoá sai tên → ValueError


def test_luoi_co_ba_ho_estimator_va_tham_so_mat_can_bang():
    grid = param_grid(pos_weight=21.5)
    families = [family_of(g["model"][0]) for g in grid]
    assert families == ["Logistic", "RandomForest", "HistGBM"]
    for g in grid:
        assert any(isinstance(v, dict) and v[1] == 21.5 for v in g["model__class_weight"])


# --------------------------------------------------------------------- trọng số & scorer


def test_trong_so_chuan_hoa_ve_trung_binh_1():
    df = _frame()
    w = normalized_survey_weights(df)
    assert w.mean() == pytest.approx(1.0)
    np.testing.assert_allclose(w / w[0], df["weight"].to_numpy() / df["weight"].iloc[0])


def test_ti_le_quan_the_am_tren_duong():
    df = pd.DataFrame({"is_low_rating": [1, 0, 0], "weight": [1.0, 2.0, 3.0]})
    assert population_ratio(df) == pytest.approx(5.0)


def test_scorer_dung_trong_so_khao_sat_trong_cot_weight():
    df = _frame()
    pipe = make_pipeline(numeric=NUM, categorical=CAT).fit(df, df["is_low_rating"])
    p = pipe.predict_proba(df)[:, 1]
    expected = average_precision_score(df["is_low_rating"], p, sample_weight=df["weight"])
    unweighted = average_precision_score(df["is_low_rating"], p)
    assert weighted_pr_auc(pipe, df, df["is_low_rating"]) == pytest.approx(expected)
    assert expected != pytest.approx(unweighted)


def test_bang_can_bang_lop_co_ti_le_tho_va_quan_the():
    df = _frame()
    table = class_balance_table({"train": df})
    row = table.loc["train"]
    assert row["n_dương"] == df["is_low_rating"].sum()
    assert row["tỉ_lệ_thô"] == pytest.approx(df["is_low_rating"].mean())
    assert row["tỉ_lệ_quần_thể"] == pytest.approx(np.average(df["is_low_rating"], weights=df["weight"]))


# --------------------------------------------------------------------- chạy GridSearchCV


def test_grid_search_thu_du_moi_cau_hinh_tren_moi_fold():
    _, gs = _search()
    assert len(gs.cv_results_["params"]) == len(ParameterGrid(_small_grid())) == 6
    assert "split1_test_score" in gs.cv_results_ and "split2_test_score" not in gs.cv_results_
    assert "mean_train_score" in gs.cv_results_


def test_bang_ket_qua_va_cau_hinh_tot_nhat_theo_ho():
    _, gs = _search()
    table = results_table(gs)
    assert set(table["họ"]) == {"Logistic", "HistGBM"}
    assert table["rank"].min() == 1
    best_lr = table[(table["họ"] == "Logistic") & (table["rank_trong_họ"] == 1)].iloc[0]
    family_rows = table[table["họ"] == "Logistic"]
    assert best_lr["mean_test_score"] == family_rows["mean_test_score"].max()


def test_lat_cat_validation_curve_giu_cac_tham_so_khac_o_gia_tri_tot_nhat():
    _, gs = _search()
    table = results_table(gs)
    slices = validation_slices(table, "Logistic")
    assert set(slices) == {"model__C", "prep__num__impute__strategy"}
    c_slice = slices["model__C"]
    assert len(c_slice) == 2  # hai giá trị C, strategy cố định ở giá trị thắng
    assert c_slice["prep__num__impute__strategy"].nunique() == 1


def test_seed_stability_train_lai_nhieu_lan():
    df = _frame()
    train, test = df[df["year"] <= 2022], df[df["year"] == 2023]
    params = {"HistGBM": {"model": HistGradientBoostingClassifier(max_iter=30),
                          "model__max_leaf_nodes": 8}}
    out = seed_stability(params, train, test, seeds=(0, 1, 2),
                         pipeline=make_pipeline(numeric=NUM, categorical=CAT))
    assert list(out.columns[:2]) == ["họ", "seed"]
    assert len(out) == 3 and out["pr_auc_test"].between(0, 1).all()
