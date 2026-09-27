"""Mô hình cây — bậc "cây" và "tuned" của thang mô hình (task C8).

`GBMModel` — Gradient Boosting dạng histogram (`HistGradientBoostingClassifier`)
-------------------------------------------------------------------------------
Boosting cộng dần M cây hồi quy nhỏ: `F_m(x) = F_{m-1}(x) + η · f_m(x)`, mỗi cây
`f_m` khớp **gradient âm của log-loss** (= phần dư `y − p`) của mô hình hiện tại
— tức là gradient descent trong không gian hàm (Friedman, 2001). Cài đặt histogram
là thuật toán của LightGBM (Ke et al., 2017):

* chia mỗi feature thành ≤255 bin → tìm điểm cắt O(bin) thay vì O(n) — train
  200k dòng trong vài giây;
* **mọc cây theo lá** (`max_leaf_nodes`): luôn tách lá giảm loss nhiều nhất;
* **giá trị thiếu tự nhiên**: ở mỗi điểm cắt, học luôn nhánh nào cho NaN —
  quan trọng vì `cust_tenure_days` thiếu 10%, lịch sử trống ở sản phẩm mới;
* **biến phân loại tự nhiên** (`categorical_features`), không cần one-hot;
* trọng số lá `w* = −G / (H + λ)` với λ = `l2_regularization` (G, H: tổng gradient
  và Hessian trong lá) — λ lớn co giá trị lá về 0, chống quá khớp.

`RandomForestModel` — đối chứng cùng họ cây nhưng *bagging* thay vì *boosting*:
nhiều cây sâu độc lập rồi lấy trung bình → giảm phương sai; boosting thì giảm
độ chệch tuần tự. So hai mô hình này trả lời "vì sao chọn boosting".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .base import BaseModel


class GBMModel(BaseModel):
    """Gradient boosting histogram với dừng sớm. Mọi siêu tham số là thuộc tính công khai."""

    name = "hist_gbm"

    def __init__(
        self,
        features: list[str] | None = None,
        categorical: tuple[str, ...] = (),
        learning_rate: float = 0.1,
        max_iter: int = 500,
        max_leaf_nodes: int = 31,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_features: float = 1.0,
        max_bins: int = 255,
        early_stopping: bool = True,
        n_iter_no_change: int = 30,
        validation_fraction: float = 0.1,
        class_weight: str | dict | None = None,
        random_state: int = 42,
    ) -> None:
        self.features = features
        self.categorical = categorical
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.max_bins = max_bins
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.class_weight = class_weight
        self.random_state = random_state

    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        mask = np.array([c in self.categorical for c in X.columns])
        self.estimator_ = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_features=self.max_features,
            max_bins=self.max_bins,
            categorical_features=mask if mask.any() else None,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            scoring="loss",
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y, sample_weight=sample_weight)

    def _predict_positive(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(X)[:, 1]

    @property
    def n_iter_(self) -> int:
        """Số cây thực sự dùng sau dừng sớm — một *tham số học được*, không phải siêu tham số."""
        return int(self.estimator_.n_iter_)


class RandomForestModel(BaseModel):
    """Random forest (bagging + chọn ngẫu nhiên feature ở mỗi nút). Điền thiếu bằng trung vị."""

    name = "random_forest"

    def __init__(
        self,
        features: list[str] | None = None,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        max_features: str | float = "sqrt",
        class_weight: str | dict | None = None,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.features = features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        self.pipeline_ = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("rf", RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                class_weight=self.class_weight, n_jobs=self.n_jobs, random_state=self.random_state,
            )),
        ])
        self.pipeline_.fit(X, y, rf__sample_weight=sample_weight)

    def _predict_positive(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X)[:, 1]
