"""GridSearchCV trên Pipeline bằng một lệnh — mọi con số của notebook 04.

    python -m src.models.run_grid_search            # ~20 phút trên laptop 8 nhân

Đầu ra:
  * `docs/evidence/c_gridsearch_<ngày>.txt` — log đầy đủ;
  * `reports/models/class_balance.csv` — cân bằng lớp từng tập;
  * `reports/models/grid_search_cv.csv` — mọi cấu hình × fold (từ `cv_results_`);
  * `reports/models/grid_search_best.json` — cấu hình thắng của từng họ estimator;
  * `reports/models/grid_search_test.csv` — chấm test MỘT lần cho cấu hình thắng của từng họ;
  * `reports/models/seed_stability.csv` — train lại 5 seed;
  * `reports/figures/C19_class_balance.png`, `C20_grid_validation_curves.png`.

Quan hệ với notebook 03: thang M0–M8 ở đó dùng random search tự viết trong
`BaseModel.tune`. File này làm lại bước tune theo **cách chuẩn của scikit-learn**
(`Pipeline` + `GridSearchCV` + cú pháp `__`) để (1) trình bày đúng quy trình, (2) kiểm
tra chéo: hai cách tìm có cùng chọn ra một vùng siêu tham số không. Mô hình chốt M7
không đổi — nó được chọn bằng quy tắc đặt trước ở notebook 03.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from src.features.build import TARGET, WEIGHT, temporal_split
from src.viz.theme import save_fig

from . import plots
from .experiments import ARTIFACT_DIR, load_frame
from .grid_search import (
    CV,
    best_params_by_family,
    class_balance_table,
    jsonable_params,
    make_pipeline,
    normalized_survey_weights,
    results_table,
    run_grid_search,
    seed_stability,
    validation_slices,
)
from .metrics import evaluate_predictions
from .run_modeling import Log, check_sklearn_version

FAMILIES = ("Logistic", "RandomForest", "HistGBM")
TEST_METRICS = ["pr_auc", "lift_pr_auc", "roc_auc", "precision@5%", "recall@5%", "p_mean", "prevalence"]


def balance_sets(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sets = {"train ≤2023": train, "test 2024–2026": test}
    for tr, va in CV.split(train):
        year = int(train.iloc[va]["year"].iloc[0])
        sets[f"CV val {year}"] = train.iloc[va]
    return sets


def evaluate_on_test(best: dict, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Train lại cấu hình thắng của mỗi họ trên toàn bộ train → chấm test (một lần)."""
    rows = []
    sw = normalized_survey_weights(train)
    for fam in FAMILIES:
        pipe = clone(make_pipeline()).set_params(**best[fam])
        pipe.fit(train, train[TARGET].to_numpy(), model__sample_weight=sw)
        p = pipe.predict_proba(test)[:, 1]
        m = evaluate_predictions(test[TARGET].to_numpy(), p, test[WEIGHT].to_numpy(dtype=float))
        rows.append({"họ": fam, **{k: m[k] for k in TEST_METRICS}})
    return pd.DataFrame(rows).set_index("họ")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5, help="số seed khi train lại để đo độ dao động")
    ap.add_argument("--allow-other-sklearn", action="store_true")
    args = ap.parse_args(argv)
    check_sklearn_version(args.allow_other_sklearn)
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    plots.apply_theme()
    log = Log()

    frame = load_frame()
    train, test = temporal_split(frame)

    log.section("1 · Dữ liệu có cân bằng không? (lớp dương = review ≤3★)")
    balance = class_balance_table(balance_sets(train, test))
    balance.to_csv(ARTIFACT_DIR / "class_balance.csv")
    log(balance)
    save_fig(plots.plot_class_balance(balance.loc[["train ≤2023", "test 2024–2026"]]),
             "C19_class_balance.png")

    log.section("2 · Pipeline = tiền xử lý + estimator; siêu tham số gọi bằng `bước__tham_số`")
    pipe = make_pipeline()
    log(pipe)
    keys = [k for k in pipe.get_params() if "__" in k and k.count("__") <= 3]
    log("Một số khoá hợp lệ: " + ", ".join(k for k in keys if k.startswith(("model__C", "model__class",
        "prep__num__impute__strategy", "prep__num__scale__with_mean", "prep__cat__handle"))))

    log.section("3 · GridSearchCV: 3 họ estimator × lưới riêng × 2 fold theo năm")
    started = time.perf_counter()
    search = run_grid_search(train, verbose=1)
    minutes = (time.perf_counter() - started) / 60
    table = results_table(search)
    table.to_csv(ARTIFACT_DIR / "grid_search_cv.csv", index=False)
    log(f"{len(table)} cấu hình × {CV.get_n_splits()} fold = {len(table) * CV.get_n_splits()} lần fit "
        f"· {minutes:.1f} phút")
    log(f"best_params_ = {jsonable_params(search.best_params_)}")
    log(f"best_score_ (PR-AUC CV, lạc quan vì là max của {len(table)} ước lượng) = {search.best_score_:.4f}")
    for fam in FAMILIES:
        log(f"\n--- {fam}: 5 cấu hình tốt nhất")
        sub = table[table["họ"] == fam].dropna(axis=1, how="all").head(5)
        log(sub.drop(columns=["họ"]))

    best = best_params_by_family(search)
    (ARTIFACT_DIR / "grid_search_best.json").write_text(json.dumps(
        {"best_overall": jsonable_params(search.best_params_),
         "best_score_cv": search.best_score_,
         "best_by_family": {f: jsonable_params(p) for f, p in best.items()},
         "cv": repr(CV), "n_candidates": len(table)},
        ensure_ascii=False, indent=2), encoding="utf-8")

    log.section("4 · Vì sao chọn giá trị này — validation curve đọc từ lưới")
    slices = {fam: validation_slices(table, fam) for fam in FAMILIES}
    for fam, fam_slices in slices.items():
        for prm, d in fam_slices.items():
            log(f"{fam:12s} {prm:30s} " + " · ".join(
                f"{v}: {s:.4f}" for v, s in zip(d[prm], d["mean_test_score"])))
    winner = table.iloc[0]["họ"]
    save_fig(plots.plot_validation_curves(slices, best_overall=winner), "C20_grid_validation_curves.png")

    log.section("5 · Chấm test 2024–2026 MỘT lần cho cấu hình thắng của từng họ")
    test_table = evaluate_on_test(best, train, test)
    test_table.to_csv(ARTIFACT_DIR / "grid_search_test.csv")
    log(test_table)

    log.section(f"6 · Train lại {args.seeds} lần với {args.seeds} seed — kết quả có ổn định?")
    seeds = seed_stability(best, train, test, seeds=tuple(range(args.seeds)))
    seeds.to_csv(ARTIFACT_DIR / "seed_stability.csv", index=False)
    summary = seeds.groupby("họ", sort=False)["pr_auc_test"].agg(["mean", "std", "min", "max"])
    log(seeds.pivot(index="seed", columns="họ", values="pr_auc_test"))
    log(summary)

    out = Path("docs/evidence") / f"c_gridsearch_{date.today().isoformat()}.txt"
    out.write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    print(f"\nĐã lưu evidence: {out}")


if __name__ == "__main__":
    main()
