"""Hồi quy logistic — bậc "tuyến tính" của thang mô hình (task C7).

Mô hình: `logit P(y=1|x) = β0 + βᵀφ(x)`, ước lượng bằng hợp lý cực đại có phạt L2
(`C` = 1/λ). Chọn logistic ở bậc này vì nó **diễn giải được**: `exp(β_j)` là tỉ số
odds khi feature j tăng 1 đơn vị (ở đây: 1 độ lệch chuẩn, vì đã chuẩn hoá).

Tiền xử lý nằm **bên trong** pipeline nên được `fit` lại trong từng fold — trung
vị để điền thiếu và trung bình/độ lệch chuẩn để chuẩn hoá không bao giờ học từ
dữ liệu kiểm định (tầng 3 — leakage).

`spline_features`: biến liên tục được khai triển B-spline bậc 3 trước khi vào mô
hình, để logistic học được quan hệ **phi tuyến** mà vẫn là mô hình tuyến tính
theo tham số. M1b dùng nó cho `lead_days`: "toàn bộ thông tin trong thời gian
giao", không ép dạng đường thẳng.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .base import BaseModel


class LogisticModel(BaseModel):
    """Logistic L2 với tiền xử lý tự động theo loại cột."""

    name = "logistic"

    def __init__(
        self,
        features: list[str] | None = None,
        categorical: tuple[str, ...] = (),
        spline_features: tuple[str, ...] = (),
        C: float = 1.0,
        class_weight: str | dict | None = None,
        n_knots: int = 6,
        max_iter: int = 2000,
    ) -> None:
        self.features = features
        self.categorical = categorical
        self.spline_features = spline_features
        self.C = C
        self.class_weight = class_weight
        self.n_knots = n_knots
        self.max_iter = max_iter

    def _preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        cats = [c for c in self.categorical if c in X.columns]
        splines = [c for c in self.spline_features if c in X.columns and c not in cats]
        nums = [c for c in X.columns if c not in cats and c not in splines]
        parts = []
        if nums:
            parts.append(("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), nums))
        if splines:
            parts.append(("spline", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("bspline", SplineTransformer(n_knots=self.n_knots, degree=3, knots="quantile",
                                              extrapolation="constant")),
            ]), splines))
        if cats:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
        return ColumnTransformer(parts)

    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        self.pipeline_ = Pipeline([
            ("prep", self._preprocessor(X)),
            ("clf", LogisticRegression(C=self.C, class_weight=self.class_weight,
                                       max_iter=self.max_iter)),
        ])
        self.pipeline_.fit(X, y, clf__sample_weight=sample_weight)

    def _predict_positive(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X)[:, 1]

    def coefficients(self) -> pd.DataFrame:
        """Hệ số và tỉ số odds theo tên feature sau tiền xử lý."""
        names = self.pipeline_.named_steps["prep"].get_feature_names_out()
        beta = self.pipeline_.named_steps["clf"].coef_.ravel()
        return (
            pd.DataFrame({"feature": names, "beta": beta, "odds_ratio": np.exp(beta)})
            .assign(abs_beta=lambda d: d["beta"].abs())
            .sort_values("abs_beta", ascending=False)
            .drop(columns="abs_beta")
            .reset_index(drop=True)
        )
