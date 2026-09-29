"""Toàn bộ Track C bằng một lệnh — mọi con số của FINDINGS §11 sinh ra từ đây.

    python3 -m src.models.run_modeling              # tune lại từ đầu (~15 phút)
    python3 -m src.models.run_modeling --use-cache  # đọc mô hình đã tune trong reports/models/

Đầu ra:
  * `docs/evidence/c_modeling_<ngày>.txt` — log đầy đủ;
  * `reports/models/*.csv` — bảng kết quả cho notebook/báo cáo;
  * `reports/models/*_tuned.json` — siêu tham số đã tune (đọc được bằng mắt);
  * `reports/models/*_tuned.joblib` — đối tượng mô hình đầy đủ.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build import EX_ANTE_FEATURES, TARGET, WEIGHT, temporal_split

from .diagnostics import (
    group_ablation,
    paired_comparisons,
    permutation_importance_table,
    sla_trap_difference,
    sla_trap_table,
    univariate_auc_table,
)
from .experiments import (
    ARTIFACT_DIR,
    CV,
    FINAL_MODEL,
    build_ladder,
    choose_scheme,
    compare_families_and_weights,
    evaluation_table,
    final_candidates,
    fit_c9,
    fit_ladder,
    load_frame,
    load_or_tune,
    tune_gbm,
    tune_logistic,
)

#: Các cặp so sánh ghép cặp trên toàn bộ test (có trọng số).
TEST_PAIRS = [
    (FINAL_MODEL, "M2 · logistic tuned"),
    (FINAL_MODEL, "M5 · GBM tuned"),
    ("M5 · GBM tuned", "M4 · GBM mặc định"),
    (FINAL_MODEL, "M1b · chỉ lead_days (spline)"),
    ("M1b · chỉ lead_days (spline)", "M1a · chỉ sla_breach"),
]
#: Ablation trung tâm C9 — trên cùng tập test có nhãn khách.
C9_PAIRS = [
    ("M1b · lead_days (thông tin đầy đủ)", "M1a · sla_breach (doanh nghiệp đo)"),
    ("M1c · khách nói trễ (khách cảm nhận)", "M1a · sla_breach (doanh nghiệp đo)"),
    ("M1d · lead_days + khách nói trễ", "M1b · lead_days (thông tin đầy đủ)"),
]
EXPOST_PAIR = [("M8 · GBM tuned + tín hiệu khách (ex-post)", "M5 · GBM tuned")]
#: Mô hình con số trong C9 train bằng trọng số khảo sát thuần (không cân bằng lớp) để xác
#: suất là xác suất quần thể — với một feature, trọng số không đổi thứ hạng nên PR-AUC không đổi.
C9_SCHEME = "survey"
REPORT_METRICS = ["pr_auc", "lift_pr_auc", "roc_auc", "brier_skill", "log_loss", "ece",
                  "precision@5%", "recall@5%", "p_mean", "prevalence", "n"]
#: Phiên bản scikit-learn sinh ra kết quả chuẩn (FINDINGS §11, notebook 03, `*_tuned.json`).
#: Cây (RF, HistGBM) đổi thuật toán giữa các phiên bản: ngày 2026-09-27 một lượt `make all`
#: trên sklearn 1.9.1 đã cho bộ siêu tham số GBM khác và ghi đè kết quả chuẩn. Logistic
#: thì trùng khớp tới 16 chữ số — đúng dấu hiệu của khác biệt cài đặt, không phải lỗi dữ liệu.
PINNED_SKLEARN = "1.5.1"


def check_sklearn_version(allow_other: bool) -> None:
    """Dừng lại nếu phiên bản sklearn khác bản đã ghim — trừ khi người chạy cho phép rõ ràng."""
    import sklearn

    if sklearn.__version__ == PINNED_SKLEARN or allow_other:
        return
    raise SystemExit(
        f"scikit-learn {sklearn.__version__} ≠ {PINNED_SKLEARN} (bản sinh ra kết quả chuẩn).\n"
        f"Chạy tiếp sẽ ghi đè reports/models/ và docs/evidence/ bằng số khác.\n"
        f"Cài đúng bản: pip install scikit-learn=={PINNED_SKLEARN}\n"
        f"Hoặc cố ý so sánh giữa các phiên bản: thêm --allow-other-sklearn."
    )


class Log:
    """Vừa in ra màn hình vừa gom lại để ghi file bằng chứng."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, obj: object = "") -> None:
        text = obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else str(obj)
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-iter", type=int, default=40, help="số cấu hình random search của GBM")
    ap.add_argument("--n-boot", type=int, default=1000, help="số lần bootstrap theo cụm")
    ap.add_argument("--use-cache", action="store_true", help="đọc mô hình đã tune nếu có")
    ap.add_argument("--allow-other-sklearn", action="store_true",
                    help=f"chạy dù scikit-learn khác bản ghim {PINNED_SKLEARN}")
    args = ap.parse_args(argv)
    check_sklearn_version(args.allow_other_sklearn)
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    log = Log()

    frame = load_frame()
    train, test = temporal_split(frame)
    log.section("1 · Dữ liệu mô hình")
    for name, d in (("train ≤2023", train), ("test 2024–2026", test)):
        log(f"{name:16s} n={len(d):,}  dương={int(d[TARGET].sum()):,}  "
            f"tỉ lệ thô={d[TARGET].mean():.4f}  tỉ lệ quần thể={np.average(d[TARGET], weights=d[WEIGHT]):.4f}  "
            f"sản phẩm vét cạn={int(d['is_census'].sum()):,} dòng")
    log(CV.describe(train, train[TARGET]))

    log.section("2 · Kiểm D9: AUC một biến, mẫu thô vs quần thể (chỉ tập train)")
    flip = univariate_auc_table(train, [f for f in EX_ANTE_FEATURES if f != "category"])
    log(flip)
    log(f"Số feature đảo chiều: {int(flip['đảo_chiều'].sum())}/{len(flip)}")

    log.section("3 · Chọn họ mô hình × phương án trọng số (CV theo thời gian, PR-AUC có trọng số)")
    families = compare_families_and_weights(train)
    log(families)
    scheme = choose_scheme(families)
    log(f"→ phương án trọng số train được chọn cho GBM: {scheme}")

    log.section("4 · Tune (random search trên CV theo thời gian)")
    tuned = load_or_tune(ARTIFACT_DIR / "gbm_tuned.joblib",
                         lambda: tune_gbm(train, scheme, n_iter=args.n_iter), args.use_cache)
    tuned_lr = load_or_tune(ARTIFACT_DIR / "logistic_tuned.joblib",
                            lambda: tune_logistic(train, scheme), args.use_cache)
    log(f"GBM best_params_ = {tuned.best_params_}")
    log(f"GBM best_score_ (PR-AUC CV) = {tuned.best_score_:.4f} · n_iter_ (số cây sau dừng sớm) = {tuned.n_iter_}")
    log(tuned.tuning_results_.head(10))
    log(f"Logistic best_params_ = {tuned_lr.best_params_} · best_score_ = {tuned_lr.best_score_:.4f}")
    log("\nChọn mô hình cuối bằng CV (quy tắc đặt trước: trung bình cao nhất):")
    candidates = final_candidates(train, tuned, tuned_lr, scheme)
    candidates.to_csv(ARTIFACT_DIR / "final_candidates_cv.csv")
    log(candidates)

    log.section("5 · Thang mô hình trên tập test (chấm MỘT lần)")
    ladder = build_ladder(tuned, tuned_lr)
    preds = fit_ladder(ladder, train, test, scheme)
    table = evaluation_table(preds, test)
    table.to_csv(ARTIFACT_DIR / "ladder_test.csv", index=False)
    for set_name, sub in table.groupby("tập", sort=False):
        log(f"\n--- {set_name}")
        log(sub.set_index("mô hình")[REPORT_METRICS])
    cal = ladder[FINAL_MODEL]
    log(f"\nHiệu chỉnh Platt: slope a = {cal.calibration_slope_:.3f}, intercept b = {cal.calibration_intercept_:.3f} "
        f"(học trên {cal.n_calibration_:,} dòng năm {cal.calib_year})")

    log.section("6 · So sánh ghép cặp (bootstrap theo cụm sản phẩm, KTC 95%)")
    for est in paired_comparisons(preds, test, TEST_PAIRS, n_boot=args.n_boot):
        log(est)

    log.section("7 · C9 — ablation trung tâm trên tập test có nhãn khách")
    c9_preds, labelled = fit_c9(train, test, C9_SCHEME)
    c9_table = evaluation_table(c9_preds, labelled)
    c9_table = c9_table[c9_table["tập"].str.startswith("toàn bộ")]
    c9_table.to_csv(ARTIFACT_DIR / "c9_test.csv", index=False)
    log(c9_table.set_index("mô hình")[REPORT_METRICS])
    for est in paired_comparisons(c9_preds, labelled, C9_PAIRS, n_boot=args.n_boot):
        log(est)
    lab_mask = test["has_customer_label"].to_numpy()
    for est in paired_comparisons(preds[lab_mask], test[lab_mask], EXPOST_PAIR, n_boot=args.n_boot):
        log(est)

    log.section("8 · Ablation theo nhóm feature — ensemble M6 (train lại, giữ siêu tham số)")
    ensemble = ladder["M6 · ensemble GBM + logistic"]
    ablation = group_ablation(ensemble, EX_ANTE_FEATURES, train, test, scheme)
    ablation.to_csv(ARTIFACT_DIR / "group_ablation.csv")
    log(ablation)
    log("\nPermutation importance (PR-AUC có trọng số tụt khi xáo trộn, test):")
    importance = permutation_importance_table(ensemble, test, EX_ANTE_FEATURES)
    importance.to_csv(ARTIFACT_DIR / "permutation_importance.csv", index=False)
    log(importance)

    log.section("9 · C14 — bẫy SLA: Tiki Trading vs bên thứ ba (mọi review có nhãn, 2023+)")
    labelled_all = frame[frame["has_customer_label"]]
    trap = sla_trap_table(labelled_all)
    trap.to_csv(ARTIFACT_DIR / "c14_sla_trap.csv")
    log(trap.T)
    for rate in ("báo_động_giả", "bỏ_sót"):
        log(sla_trap_difference(labelled_all, rate, n_boot=args.n_boot))

    out = Path("docs/evidence") / f"c_modeling_{date.today().isoformat()}.txt"
    out.write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    print(f"\nĐã lưu evidence: {out}")


if __name__ == "__main__":
    main()
