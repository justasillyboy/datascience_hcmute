"""Chia dữ liệu theo thời gian cho cross-validation — task C5, tầng 3 (leakage).

Vì sao không KFold ngẫu nhiên: KFold trộn review 2023 vào tập train để dự báo
review 2021 — mô hình được "nhìn trước tương lai" (xu hướng giao hàng, sản phẩm
mới, feature lịch sử đã tích luỹ). Điểm CV sẽ đẹp hơn thực tế và chọn sai siêu
tham số. Ở đây mỗi fold chỉ train trên các năm **trước** năm kiểm định
(expanding window / rolling-origin), đúng như lúc triển khai thật.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd


class YearForwardSplit:
    """Splitter tương thích scikit-learn: fold k train = năm < v_k, val = năm == v_k.

    Ví dụ `val_years=(2022, 2023)`:
        fold 1 — train ≤2021 · val 2022
        fold 2 — train ≤2022 · val 2023

    Trả về **chỉ số vị trí** (như `KFold`), dùng được với `X.iloc[...]`.
    """

    def __init__(self, val_years: Sequence[int] = (2022, 2023), year_col: str = "year") -> None:
        self.val_years = tuple(val_years)
        self.year_col = year_col

    def split(
        self, X: pd.DataFrame, y: np.ndarray | None = None, groups: np.ndarray | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        years = X[self.year_col].to_numpy()
        for val_year in self.val_years:
            train_idx = np.flatnonzero(years < val_year)
            val_idx = np.flatnonzero(years == val_year)
            if train_idx.size == 0 or val_idx.size == 0:
                raise ValueError(f"fold val={val_year}: train {train_idx.size} dòng, val {val_idx.size} dòng")
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.val_years)

    def describe(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Bảng mô tả từng fold — in thẳng vào notebook."""
        rows = []
        years = X[self.year_col].to_numpy()
        for k, (tr, va) in enumerate(self.split(X), start=1):
            rows.append(
                {
                    "fold": k,
                    "train": f"{years[tr].min()}–{years[tr].max()}",
                    "val": int(years[va][0]),
                    "n_train": tr.size,
                    "n_val": va.size,
                    "dương_train": int(np.asarray(y)[tr].sum()),
                    "dương_val": int(np.asarray(y)[va].sum()),
                }
            )
        return pd.DataFrame(rows).set_index("fold")

    def __repr__(self) -> str:
        return f"YearForwardSplit(val_years={self.val_years})"
