"""Ensemble trung bình xác suất — **Composite pattern** trên interface `BaseModel`.

`EnsembleModel` *là* một `BaseModel` và *chứa* nhiều `BaseModel`. Vòng đánh giá,
`CalibratedModel`, `tune()`… đều dùng nó y như một mô hình đơn — không dòng code
nào bên ngoài phải biết bên trong có hai mô hình.

Vì sao trung bình GBM với logistic lại có lợi (phân rã bias–variance của
ensemble): sai số bình phương của trung bình hai dự báo bằng trung bình sai số
riêng **trừ đi** phần "bất đồng" giữa chúng (Krogh & Vedelsby, 1995). Hai họ mô
hình sai theo hai kiểu khác nhau — cây học tương tác nhưng không ngoại suy được
ngoài miền đã thấy; logistic ngoại suy tuyến tính mượt nhưng bỏ sót tương tác —
nên phần bất đồng lớn, và lợi ích lớn nhất đúng ở chỗ dữ liệu tương lai trôi
khỏi miền train.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from .base import BaseModel


class EnsembleModel(BaseModel):
    """Trung bình (có trọng số) xác suất của các mô hình thành viên."""

    name = "ensemble"

    def __init__(self, members: Sequence[BaseModel] = (), weights: Sequence[float] | None = None) -> None:
        self.members = members
        self.weights = weights

    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        if not self.members:
            raise ValueError("EnsembleModel cần ít nhất một mô hình thành viên")
        if self.weights is not None and len(self.weights) != len(self.members):
            raise ValueError("weights phải cùng số phần tử với members")
        # clone giữ siêu tham số (kể cả bộ đã tune) của từng thành viên, bỏ trạng thái cũ.
        self.members_ = [clone(m).fit(X, y, sample_weight=sample_weight) for m in self.members]

    def _predict_positive(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.vstack([m.predict_proba(X)[:, 1] for m in self.members_])
        return np.average(probs, axis=0, weights=self.weights)

    def with_features(self, features: list[str]) -> "EnsembleModel":
        """Đa hình: đổi feature thì đổi cho **mọi** thành viên."""
        return EnsembleModel(members=tuple(m.with_features(features) for m in self.members),
                             weights=self.weights)
