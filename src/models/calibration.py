"""Hiệu chỉnh xác suất — task C12, tầng 6 (uncertainty & calibration).

Vì sao cần: theo D9, mô hình train **không trọng số** trên mẫu có 8,4% review xấu,
trong khi quần thể chỉ ~3%. Nó xếp hạng tốt nhưng xác suất bị phồng — nói "10%"
khi thực tế là 3%. Cộng thêm trôi dạt theo thời gian (tỉ lệ review xấu giảm dần
qua các năm), xác suất thô không dùng được để ra quyết định.

Cách làm (theo tinh thần `CalibratedClassifierCV`, nhưng chia theo thời gian):
  1. train bản sao mô hình trên các năm **trước** `calib_year`;
  2. dự báo năm `calib_year` (dữ liệu nó chưa thấy);
  3. học ánh xạ điểm → xác suất trên năm đó, **có trọng số khảo sát** → xác suất
     quần thể;
  4. train lại mô hình trên toàn bộ tập train, giữ ánh xạ ở bước 3.

Hai phương pháp:
  * `sigmoid` (Platt, 1999): `p = σ(a·logit(s) + b)` — 2 tham số, ổn định khi ít
    dữ liệu. Nếu lệch chỉ do prior khác nhau thì lý thuyết cho `a = 1` và
    `b = log(odds_quần_thể) − log(odds_mẫu)`; `a < 1` nghĩa là mô hình còn tự tin
    quá mức.
  * `isotonic`: hàm bậc thang đơn điệu bất kỳ (thuật toán PAV) — linh hoạt hơn
    nhưng cần nhiều dữ liệu, dễ quá khớp.
Cả hai đều **đơn điệu** nên không đổi thứ hạng → PR-AUC giữ nguyên, chỉ Brier,
log-loss và ECE thay đổi.

Thiết kế OOP: `CalibratedModel` là **decorator** — nó *là* một `BaseModel` và
*chứa* một `BaseModel`. Bọc được bất kỳ mô hình nào mà không sửa mô hình đó.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.features.build import WEIGHT, YEAR

from .base import BaseModel

_EPS = 1e-6


class CalibratedModel(BaseModel):
    """Bọc một `BaseModel`, hiệu chỉnh xác suất trên năm `calib_year` có trọng số."""

    name = "calibrated"

    def __init__(
        self,
        base_model: BaseModel | None = None,
        method: str = "sigmoid",
        calib_year: int = 2023,
        year_col: str = YEAR,
        weight_col: str = WEIGHT,
        refit_full: bool = True,
    ) -> None:
        self.base_model = base_model
        self.method = method
        self.calib_year = calib_year
        self.year_col = year_col
        self.weight_col = weight_col
        self.refit_full = refit_full

    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        if self.base_model is None:
            raise ValueError("CalibratedModel cần base_model")
        if self.method not in ("sigmoid", "isotonic"):
            raise ValueError(f"method phải là 'sigmoid' hoặc 'isotonic', nhận {self.method!r}")
        cal = (X[self.year_col] == self.calib_year).to_numpy()
        if not cal.any() or cal.all():
            raise ValueError(f"cần cả dữ liệu trước và trong năm hiệu chỉnh {self.calib_year}")

        sw_train = None if sample_weight is None else sample_weight[~cal]
        inner = clone(self.base_model).fit(X[~cal], y[~cal], sample_weight=sw_train)
        scores = inner.predict_proba(X[cal])[:, 1]
        w_cal = (X.loc[cal, self.weight_col].to_numpy(dtype=float)
                 if self.weight_col in X.columns else np.ones(int(cal.sum())))

        if self.method == "sigmoid":
            z = logit(np.clip(scores, _EPS, 1 - _EPS)).reshape(-1, 1)
            self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, y[cal], sample_weight=w_cal)
            self.calibration_slope_ = float(self.calibrator_.coef_[0, 0])
            self.calibration_intercept_ = float(self.calibrator_.intercept_[0])
        else:
            self.calibrator_ = IsotonicRegression(y_min=_EPS, y_max=1 - _EPS, out_of_bounds="clip")
            self.calibrator_.fit(scores, y[cal], sample_weight=w_cal)
        self.n_calibration_ = int(cal.sum())
        self.estimator_ = clone(self.base_model).fit(X, y, sample_weight) if self.refit_full else inner

    def _apply(self, scores: np.ndarray) -> np.ndarray:
        if self.method == "sigmoid":
            z = logit(np.clip(scores, _EPS, 1 - _EPS))
            return expit(self.calibration_slope_ * z + self.calibration_intercept_)
        return self.calibrator_.predict(scores)

    def _predict_positive(self, X: pd.DataFrame) -> np.ndarray:
        return self._apply(self.estimator_.predict_proba(X)[:, 1])

    def with_features(self, features: list[str]) -> "CalibratedModel":
        """Đa hình: đổi feature của mô hình được bọc, giữ nguyên cấu hình hiệu chỉnh."""
        return clone(self).set_params(base_model=self.base_model.with_features(features))
