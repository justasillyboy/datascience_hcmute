"""M0 — baseline tỉ lệ chung (task C6). Bậc đầu tiên của thang mô hình (tầng 4).

Dự báo **cùng một xác suất** cho mọi đơn: tỉ lệ review xấu học từ tập train.
Nó không xếp hạng được gì (PR-AUC = đúng tỉ lệ dương), nhưng là thước đo bắt
buộc: mô hình nào không hơn được nó thì không mang thông tin gì.

Nếu dùng làm bộ phân loại cứng ở ngưỡng 0,5 thì nó luôn đoán "không phải review
xấu" và đạt accuracy ≈ 1 − tỉ lệ dương ≈ 97% — chính là lý do accuracy bị loại.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseModel


class PriorBaseline(BaseModel):
    """Dự báo hằng số `prior_` = tỉ lệ dương (có trọng số nếu truyền `sample_weight`)."""

    name = "M0_prior"

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = features

    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        self.prior_ = float(np.average(y, weights=sample_weight))

    def _predict_positive(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.prior_)
