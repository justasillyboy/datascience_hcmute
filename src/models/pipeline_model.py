"""`PipelineModel` — bọc một sklearn `Pipeline` (prep + model) thành `BaseModel`.

Repo có hai cách dựng mô hình:

* notebook 04 / `grid_search.py`: `Pipeline([("prep", ...), ("model", ...)])` + GridSearchCV
  — cách chuẩn của scikit-learn để chọn siêu tham số;
* notebook 03 / `base.py`: `BaseModel` — interface chung để ghép `EnsembleModel` và
  `CalibratedModel` (mô hình chốt M7).

Lớp này là **adapter** nối hai thế giới: cấu hình thắng GridSearch được bọc lại và dùng
thẳng trong ensemble + hiệu chỉnh, không phải viết lại mô hình nào. `main.py` dùng nó.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .base import BaseModel


class PipelineModel(BaseModel):
    """`BaseModel` có bên trong một `Pipeline`; trọng số train đi vào bước cuối qua `<bước>__sample_weight`."""

    name = "pipeline"

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self.pipeline = pipeline

    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        if self.pipeline is None:
            raise ValueError("PipelineModel cần một sklearn Pipeline")
        last_step = self.pipeline.steps[-1][0]
        fit_params = {} if sample_weight is None else {f"{last_step}__sample_weight": sample_weight}
        self.pipeline_ = clone(self.pipeline).fit(X, y, **fit_params)

    def _predict_positive(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X)[:, 1]

    def with_features(self, features: list[str]) -> "PipelineModel":
        raise NotImplementedError("PipelineModel chọn cột trong bước `prep` — đổi feature ở đó")
