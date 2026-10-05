"""Dự đoán review xấu (≤3★) trên Tiki từ thông tin đã biết lúc giao hàng — chạy bằng MỘT lệnh.

Đồ án "Bẻ gãy bẫy SLA trong thương mại điện tử" · Python for Data Science · HCMUTE HK1 2025–2026

BÀI TOÁN
  Input : một đơn hàng Tiki tại thời điểm giao tới tay khách, mô tả bằng 17 feature đã biết
          tại lúc đó (6 nhóm: giao hàng · thời điểm mua · sản phẩm · lịch sử sản phẩm ·
          lịch sử nhà bán · lịch sử khách).
  Output: P(review ≤3★) — xác suất khách sẽ chấm từ 3 sao trở xuống, đã hiệu chỉnh về quần thể.

PIPELINE
  [1] Nạp dữ liệu   bảng review đã làm sạch + bảng sản phẩm (tự cào từ Tiki public API)
                    --from-raw: làm sạch lại từ bảng thô, chạy data contract trước và sau
  [2] Feature       17 feature ex-ante; feature lịch sử tính "as-of" — chỉ dùng review đã đăng
                    trước lúc giao, không nhìn tương lai. Tín hiệu khách khai cùng lúc với số sao
                    (ex-post) bị loại để tránh leakage.
  [3] Chia dữ liệu  theo năm đăng review: train ≤2023 · test 2024–2026 (không KFold ngẫu nhiên)
                    CV chọn siêu tham số: train ≤2021 → val 2022, train ≤2022 → val 2023
  [4] Tiền xử lý    số: SimpleImputer(median, add_indicator) → StandardScaler
                    category: OneHotEncoder(handle_unknown="ignore")
  [5] Mô hình       Logistic · RandomForest · HistGBM, siêu tham số do GridSearchCV chọn.
                    Mô hình chốt = trung bình(Logistic, HistGBM) + hiệu chỉnh Platt trên năm 2023.
                    Train có trọng số khảo sát w = N_h/n_h (mẫu cào phân tầng theo số sao).
  [6] Đánh giá      trên test, mọi metric có trọng số khảo sát (ước lượng quần thể):
                    PR-AUC (chính), lift, ROC-AUC, Brier skill, ECE, precision/recall ở top 5%;
                    so với 2 baseline: tỉ lệ chung và "chỉ dùng cờ vượt SLA".
  [7] Lưu kết quả   outputs/: metrics_test.csv · predictions_test.csv · params.json · models/*.joblib

CÁCH CHẠY
  python main.py                     # ~2 phút: train cấu hình đã chọn + chấm test
  python main.py --describe          # chỉ in mô tả pipeline và cấu hình, không train
  python main.py --search quick      # GridSearchCV lưới rút gọn (~3 phút) rồi train cấu hình thắng
  python main.py --search full       # GridSearchCV đầy đủ 120 cấu hình × 2 fold (~20 phút)
  python main.py --models logistic gbm --C 1e-3 --class-weight balanced
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    from src.features.build import (
        EX_ANTE_FEATURES, FEATURE_GROUPS, SLA_FEATURE, TARGET, TEST_START_YEAR, WEIGHT, YEAR,
        build_features, temporal_split,
    )
    from src.models.base import BaseModel
    from src.models.baseline import PriorBaseline
    from src.models.calibration import CalibratedModel
    from src.models.ensemble import EnsembleModel
    from src.models.grid_search import (
        SEED, best_params_by_family, class_balance_table, jsonable_params, make_pipeline, normalized_survey_weights,
        param_grid, population_ratio, quick_param_grid, results_table, run_grid_search,
    )
    from src.models.linear import LogisticModel
    from src.models.metrics import evaluate_predictions
    from src.models.pipeline_model import PipelineModel
    from src.models.run_modeling import PINNED_SKLEARN
    from src.models.splits import YearForwardSplit
except ImportError as exc:  # máy chưa cài đủ thư viện
    sys.exit(f"Thiếu thư viện '{exc.name}'. Cài đủ bằng:  pip install -r requirements.txt")

FAMILIES = {"logistic": "Logistic", "rf": "RandomForest", "gbm": "HistGBM"}
MODEL_CHOICES = (*FAMILIES, "ensemble")
FINAL_NAME = "Ensemble Logistic+HistGBM + hiệu chỉnh (CHỐT)"
BASELINES = ("B0 · baseline tỉ lệ chung", "B1 · baseline chỉ cờ vượt SLA")
SHOW_METRICS = ["pr_auc", "lift_pr_auc", "roc_auc", "brier_skill", "ece",
                "precision@5%", "recall@5%", "p_mean", "prevalence"]
DATA_HELP = (
    "Dữ liệu tự cào KHÔNG có trên GitHub (cam kết không public dữ liệu thô).\n"
    "  • Dùng gói nộp bài (có sẵn thư mục data/), hoặc\n"
    "  • tự cào lại: python -m src.collect.run_crawl --max-products 500 --max-pages 20 "
    "--sample-cap 200\n    rồi chạy: python main.py --from-raw"
)


# ============================================================================ tham số dòng lệnh


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="main.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = ap.add_argument_group("dữ liệu & đầu ra")
    io.add_argument("--data-dir", type=Path, default=ROOT / "data",
                    help="thư mục chứa processed/reviews_clean.parquet và raw/tiki_products.parquet")
    io.add_argument("--from-raw", action="store_true",
                    help="làm sạch lại từ raw/tiki_reviews.parquet (chạy data contract) trước khi train")
    io.add_argument("--out-dir", type=Path, default=ROOT / "outputs", help="nơi lưu kết quả")
    io.add_argument("--no-save", action="store_true", help="không ghi gì ra đĩa")
    io.add_argument("--describe", action="store_true", help="chỉ in mô tả pipeline rồi thoát")

    split = ap.add_argument_group("chia dữ liệu theo thời gian")
    split.add_argument("--test-start-year", type=int, default=TEST_START_YEAR,
                       help="năm đầu của tập test; train = mọi năm trước đó (mặc định %(default)s)")
    split.add_argument("--cv-years", type=int, nargs="+", default=[2022, 2023],
                       help="các năm làm fold kiểm định khi GridSearchCV (mặc định %(default)s)")

    mdl = ap.add_argument_group("mô hình & chọn siêu tham số")
    mdl.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES),
                     help="mô hình cần train (mặc định: tất cả). 2 baseline luôn được train để so sánh")
    mdl.add_argument("--search", choices=("none", "quick", "full"), default="none",
                     help="none: dùng cấu hình GridSearchCV đã chọn · quick: lưới 10 cấu hình · "
                          "full: lưới 120 cấu hình (~20 phút)")
    mdl.add_argument("--calibration", choices=("sigmoid", "isotonic", "none"), default="sigmoid",
                     help="hiệu chỉnh xác suất cho mô hình chốt, học trên năm cuối của train")

    hp = ap.add_argument_group("ghi đè siêu tham số (áp lên cấu hình đã chọn / tìm được)")
    hp.add_argument("--class-weight", choices=("none", "balanced", "population"),
                    help="xử lý mất cân bằng lớp cho cả 3 họ; population = {1: âm/dương quần thể}")
    hp.add_argument("--impute", choices=("median", "mean"), help="cách điền giá trị thiếu (cột số)")
    hp.add_argument("--C", type=float, help="Logistic: C = 1/λ, nghịch đảo độ mạnh phạt L2")
    hp.add_argument("--rf-trees", type=int, help="RandomForest: số cây")
    hp.add_argument("--rf-min-samples-leaf", type=int, help="RandomForest: số dòng tối thiểu mỗi lá")
    hp.add_argument("--gbm-learning-rate", type=float, help="HistGBM: tốc độ học")
    hp.add_argument("--gbm-max-leaf-nodes", type=int, help="HistGBM: số lá tối đa mỗi cây")
    hp.add_argument("--gbm-min-samples-leaf", type=int, help="HistGBM: số dòng tối thiểu mỗi lá")
    hp.add_argument("--seed", type=int, default=SEED, help="random_state (mặc định %(default)s)")

    args = ap.parse_args(argv)
    if any(y >= args.test_start_year for y in args.cv_years):
        ap.error(f"--cv-years phải nhỏ hơn --test-start-year={args.test_start_year} (không được nhìn test)")
    args.cv_years = sorted(set(args.cv_years))
    return args


# ============================================================================ cấu hình mô hình


def chosen_params(pos_weight: float, seed: int = SEED) -> dict[str, dict]:
    """Cấu hình thắng của từng họ khi chạy GridSearchCV đầy đủ (2026-10-01).

    Nguồn: `reports/models/grid_search_best.json` · notebook 04 §3–§4. `pos_weight` = tổng trọng số
    lớp âm / lớp dương trên tập train (≈ 21,5) — cân bằng lớp theo quần thể.
    """
    population = {0: 1.0, 1: pos_weight}
    return {
        "Logistic": {
            "model": LogisticRegression(max_iter=2000),
            "model__C": 1e-4, "model__class_weight": population, "prep__num__impute__strategy": "mean",
        },
        "RandomForest": {
            "model": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
            "model__min_samples_leaf": 200, "model__max_features": "sqrt", "model__class_weight": population,
        },
        "HistGBM": {
            "model": HistGradientBoostingClassifier(max_iter=2000, early_stopping=True,
                                                    n_iter_no_change=30, random_state=seed),
            "model__learning_rate": 0.03, "model__max_leaf_nodes": 4, "model__min_samples_leaf": 400,
            "model__class_weight": "balanced",
        },
    }


def apply_overrides(params: dict[str, dict], args: argparse.Namespace, pos_weight: float) -> dict[str, dict]:
    """Trả về bản sao của `params` đã áp các tham số người chạy truyền qua dòng lệnh."""
    out = {fam: dict(p) for fam, p in params.items()}
    if args.class_weight is not None:
        cw = {"none": None, "balanced": "balanced", "population": {0: 1.0, 1: pos_weight}}[args.class_weight]
        for p in out.values():
            p["model__class_weight"] = cw
    if args.impute is not None:
        for p in out.values():
            p["prep__num__impute__strategy"] = args.impute
    per_family = {
        "Logistic": {"model__C": args.C},
        "RandomForest": {"model__n_estimators": args.rf_trees, "model__min_samples_leaf": args.rf_min_samples_leaf},
        "HistGBM": {"model__learning_rate": args.gbm_learning_rate, "model__max_leaf_nodes": args.gbm_max_leaf_nodes,
                    "model__min_samples_leaf": args.gbm_min_samples_leaf},
    }
    for fam, values in per_family.items():
        out[fam].update({k: v for k, v in values.items() if v is not None})
    return out


def build_models(params: dict[str, dict], names: list[str], calibration: str, calib_year: int) -> dict[str, BaseModel]:
    """Baseline + các mô hình được chọn. Mọi mô hình cùng interface `BaseModel` → chấm công bằng."""
    pipes = {fam: PipelineModel(clone(make_pipeline()).set_params(**p)) for fam, p in params.items()}
    models: dict[str, BaseModel] = {
        BASELINES[0]: PriorBaseline(),
        BASELINES[1]: LogisticModel(features=[SLA_FEATURE]),
    }
    for key, fam in FAMILIES.items():
        if key in names:
            models[f"{fam} (GridSearchCV)"] = pipes[fam]
    if "ensemble" in names:
        ensemble = EnsembleModel(members=(pipes["HistGBM"], pipes["Logistic"]))
        models[FINAL_NAME] = (ensemble if calibration == "none" else
                              CalibratedModel(base_model=ensemble, method=calibration, calib_year=calib_year))
    return models


# ============================================================================ các bước pipeline


def check_environment() -> None:
    if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
        sys.exit("Thiếu thư viện đọc parquet. Cài:  pip install pyarrow")
    if sklearn.__version__ != PINNED_SKLEARN:
        print(f"⚠️  scikit-learn {sklearn.__version__} ≠ {PINNED_SKLEARN} (bản sinh ra số trong báo cáo). "
              "Logistic sẽ trùng khớp; RandomForest/HistGBM có thể lệch nhẹ ở chữ số thứ 3–4.\n")


def require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"Không thấy file dữ liệu: {path}\n{DATA_HELP}")
    return path


def load_frame(args: argparse.Namespace) -> pd.DataFrame:
    """[1] Nạp dữ liệu → [2] dựng bảng feature."""
    clean_path = args.data_dir / "processed" / "reviews_clean.parquet"
    if args.from_raw:
        from src.clean.run_clean import build  # chỉ cần khi làm sạch lại

        clean = build(require(args.data_dir / "raw" / "tiki_reviews.parquet"), clean_path)
    else:
        clean = pd.read_parquet(require(clean_path))
    products_path = args.data_dir / "raw" / "tiki_products.parquet"
    products = pd.read_parquet(products_path) if products_path.exists() else None
    if products is None:
        print(f"⚠️  Không thấy {products_path} → log_price, discount_rate để trống và được điền thiếu.")
    print(f"Bảng review đã làm sạch: {len(clean):,} dòng · {clean['product_id'].nunique():,} sản phẩm · "
          f"{clean['seller_id'].nunique():,} nhà bán · {clean['category_id'].nunique()} ngành hàng")
    frame = build_features(clean, products, train_end_year=args.test_start_year - 1)
    print(f"Bảng mô hình (dòng có lead time hợp lệ): {len(frame):,} dòng · {len(EX_ANTE_FEATURES)} feature ex-ante")
    for group, feats in FEATURE_GROUPS.items():
        print(f"   {group:18s} {', '.join(feats)}")
    return frame


def split_frame(frame: pd.DataFrame, test_start_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """[3] Chia theo năm + in mức mất cân bằng lớp (thô vs quần thể)."""
    if not (frame[YEAR] < test_start_year).any() or not (frame[YEAR] >= test_start_year).any():
        sys.exit(f"--test-start-year={test_start_year} làm tập train hoặc test rỗng "
                 f"(dữ liệu trải {frame[YEAR].min()}–{frame[YEAR].max()})")
    train, test = temporal_split(frame, test_start_year)
    balance = class_balance_table({f"train ≤{test_start_year - 1}": train, f"test ≥{test_start_year}": test})
    print(balance.to_string(formatters={"tỉ_lệ_thô": "{:.2%}".format, "tỉ_lệ_quần_thể": "{:.2%}".format,
                                        "âm_trên_dương_quần_thể": "{:.1f} : 1".format}))
    print("→ Lớp dương hiếm: dùng PR-AUC thay accuracy; tỉ lệ thô ≠ quần thể nên mọi metric nhân trọng số.")
    return train, test


def search_params(train: pd.DataFrame, args: argparse.Namespace, pos_weight: float) -> tuple[dict, pd.DataFrame]:
    """[5a] GridSearchCV trên Pipeline (prep + model), CV theo năm, scorer PR-AUC có trọng số."""
    grid = (param_grid if args.search == "full" else quick_param_grid)(pos_weight)
    cv = YearForwardSplit(val_years=tuple(args.cv_years))
    n_cfg = sum(int(np.prod([len(v) for v in g.values()])) for g in grid)
    print(f"Lưới '{args.search}': {n_cfg} cấu hình × {cv.get_n_splits()} fold ({cv}) = "
          f"{n_cfg * cv.get_n_splits()} lần fit")
    started = time.perf_counter()
    search = run_grid_search(train, grid=grid, cv=cv, verbose=1)
    table = results_table(search)
    print(f"Xong sau {(time.perf_counter() - started) / 60:.1f} phút. 3 cấu hình tốt nhất mỗi họ:")
    for fam in FAMILIES.values():
        sub = table[table["họ"] == fam].dropna(axis=1, how="all").head(3)
        print(f"\n--- {fam}\n" + sub.drop(columns=["họ", "mean_fit_time"]).to_string(index=False))
    print(f"\nbest_params_ = {jsonable_params(search.best_params_)} · best_score_ = {search.best_score_:.4f}")
    return best_params_by_family(search), table


def fit_and_predict(models: dict[str, BaseModel], train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """[5b] Train mọi mô hình trên toàn bộ train (trọng số khảo sát) → xác suất trên test."""
    y = train[TARGET].to_numpy()
    sw = normalized_survey_weights(train)
    preds = pd.DataFrame(index=test.index)
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(train, y, sample_weight=sw)
        preds[name] = model.predict_proba(test)[:, 1]
        print(f"   ✓ {name:48s} {time.perf_counter() - started:6.1f} s")
    return preds


def evaluate(preds: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """[6] Metric có trọng số khảo sát trên test — chấm MỘT lần, sau khi đã chọn xong."""
    y = test[TARGET].to_numpy()
    w = test[WEIGHT].to_numpy(dtype=float)
    rows = {name: evaluate_predictions(y, preds[name].to_numpy(), w) for name in preds.columns}
    return pd.DataFrame(rows).T[SHOW_METRICS]


def summarize(metrics: pd.DataFrame) -> None:
    """Đọc bảng metric thành câu — không chọn mô hình theo test (mô hình chốt đã định trước)."""
    prevalence = metrics["prevalence"].iloc[0]
    print(f"\nĐọc bảng: PR-AUC của dự đoán ngẫu nhiên = tỉ lệ dương = {prevalence:.4f}. "
          f"Chỉ dùng cờ vượt SLA: lift {metrics.loc[BASELINES[1], 'lift_pr_auc']:.2f}× (gần như ngẫu nhiên).")
    if FINAL_NAME in metrics.index:
        m = metrics.loc[FINAL_NAME]
        print(f"Mô hình chốt: lift {m['lift_pr_auc']:.2f}× · gọi 5% đơn rủi ro nhất bắt được {m['recall@5%']:.1%} "
              f"review xấu (precision {m['precision@5%']:.1%} ≈ {m['precision@5%'] / prevalence:.1f}× mức chung) · "
              f"xác suất đã hiệu chỉnh: p_mean {m['p_mean']:.4f} ≈ {prevalence:.4f}, ECE {m['ece']:.4f}.")
    print("Brier skill âm ở mô hình đơn là do class_weight phóng to xác suất (p_mean ≫ tỉ lệ thật) — "
          "thứ hạng (PR-AUC) không đổi; bước hiệu chỉnh của mô hình chốt sửa phần này.")


def show_examples(preds: pd.DataFrame, test: pd.DataFrame, model_name: str, k: int = 5) -> None:
    """Minh hoạ OUTPUT của bài toán: k đơn bị dự báo rủi ro cao nhất trên test."""
    cols = ["review_id", "category_name", "lead_days", "seller_hist_rate", "prod_hist_rate", TARGET]
    top = test.assign(p_review_xau=preds[model_name]).nlargest(k, "p_review_xau")[cols + ["p_review_xau"]]
    print(f"{k} đơn có P(review ≤3★) cao nhất theo '{model_name}' (cột {TARGET} = thực tế):")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def save_outputs(out_dir: Path, metrics: pd.DataFrame, preds: pd.DataFrame, test: pd.DataFrame,
                 models: dict[str, BaseModel], params: dict[str, dict], args: argparse.Namespace,
                 grid_table: pd.DataFrame | None) -> None:
    """[7] Ghi kết quả để đối chiếu với báo cáo."""
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics_test.csv")
    test[["review_id", YEAR, WEIGHT, TARGET]].join(preds).to_csv(out_dir / "predictions_test.csv", index=False)
    if grid_table is not None:
        grid_table.to_csv(out_dir / "grid_search_cv.csv", index=False)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {"run_config": config, "sklearn": sklearn.__version__,
               "params_by_family": {fam: jsonable_params(p) for fam, p in params.items()}}
    (out_dir / "params.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for i, (name, model) in enumerate(models.items()):
        joblib.dump(model, out_dir / "models" / f"{i:02d}_{name.split(' ')[0].lower()}.joblib")
    print(f"Đã lưu vào {out_dir}/: metrics_test.csv · predictions_test.csv · params.json · models/ ({len(models)} file)"
          + (" · grid_search_cv.csv" if grid_table is not None else ""))


# ============================================================================ chạy


def describe(args: argparse.Namespace) -> None:
    print(__doc__.split("CÁCH CHẠY")[0])
    print("Bước [4]+[5] dưới dạng sklearn Pipeline (siêu tham số gọi bằng <bước>__<tham số>):")
    print(make_pipeline())
    print("\nCấu hình mặc định (pos_weight ≈ 21,5 trên train ≤2023):")
    for fam, p in apply_overrides(chosen_params(21.5, args.seed), args, 21.5).items():
        print(f"   {fam:12s} {jsonable_params(p)}")


def section(title: str) -> None:
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option("display.width", 200)
    if args.describe:
        describe(args)
        return
    check_environment()
    started = time.perf_counter()

    section("[1–2] Nạp dữ liệu + dựng feature")
    frame = load_frame(args)

    section(f"[3] Chia theo thời gian: train < {args.test_start_year} ≤ test")
    train, test = split_frame(frame, args.test_start_year)
    pos_weight = round(population_ratio(train), 1)

    grid_table = None
    if args.search == "none":
        params = chosen_params(pos_weight, args.seed)
        section("[4–5] Pipeline với cấu hình GridSearchCV đã chọn (python main.py --search full để tìm lại)")
    else:
        section(f"[5a] GridSearchCV — lưới '{args.search}'")
        params, grid_table = search_params(train, args, pos_weight)
        section("[4–5] Pipeline với cấu hình vừa tìm được")
    params = apply_overrides(params, args, pos_weight)
    for fam, p in params.items():
        print(f"   {fam:12s} {jsonable_params(p)}")

    models = build_models(params, args.models, args.calibration, calib_year=args.test_start_year - 1)
    preds = fit_and_predict(models, train, test)

    section(f"[6] Đánh giá trên test ≥{args.test_start_year} — metric có trọng số khảo sát (ước lượng quần thể)")
    metrics = evaluate(preds, test)
    print(metrics.to_string(float_format=lambda v: f"{v:.4f}"))
    summarize(metrics)
    print()
    show_examples(preds, test, FINAL_NAME if FINAL_NAME in preds else preds.columns[-1])

    if not args.no_save:
        section("[7] Lưu kết quả")
        save_outputs(args.out_dir, metrics, preds, test, models, params, args, grid_table)
    print(f"\nHoàn tất sau {(time.perf_counter() - started) / 60:.1f} phút.")


if __name__ == "__main__":
    main()
