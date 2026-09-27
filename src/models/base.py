"""`BaseModel` — interface chung cho mọi mô hình của Track C (task C4).

Vì sao cần một interface
------------------------
Bậc thang mô hình (M0 baseline → logistic → cây → tuned → hiệu chỉnh) chỉ so sánh
được **công bằng** khi mọi mô hình đi qua *cùng một* vòng đánh giá: cùng tập
dòng, cùng cách chọn cột, cùng metric có trọng số. `BaseModel` là hợp đồng bảo
đảm điều đó. Vòng đánh giá chỉ biết tới `fit` / `predict_proba`, không biết mình
đang chấm lớp nào — **đa hình** (polymorphism). Thêm mô hình mới = viết thêm một
lớp con, không sửa một dòng nào của code đánh giá (nguyên lý đóng–mở).

Bốn khái niệm OOP dùng ở đây
----------------------------
1. **Lớp trừu tượng** (`abc.ABC` + `@abstractmethod`): không tạo được `BaseModel()`
   trực tiếp; lớp con *bắt buộc* cài `_fit` và `_predict_positive`, nếu thiếu
   Python báo lỗi ngay lúc khởi tạo chứ không đợi tới lúc chạy.
2. **Template method**: `fit()` và `predict_proba()` được cài *một lần* ở lớp cha
   (kiểm tra đầu vào, chọn cột, ghi `classes_`, kẹp xác suất vào [0, 1]) rồi gọi
   "móc" `_fit` / `_predict_positive` do lớp con cài. Lớp con không thể quên kiểm
   tra đầu vào vì nó không phải tự viết phần đó.
3. **Kế thừa**: `tune()`, `save()`, `export_hyperparams()` viết một lần, mọi lớp
   con dùng được — kể cả lớp chưa ra đời.
4. **Đóng gói trạng thái** theo quy ước scikit-learn (phần dưới).

Siêu tham số được lưu ở đâu — quy ước scikit-learn
--------------------------------------------------
* **Siêu tham số** = tham số của `__init__`, được gán *nguyên văn* thành thuộc
  tính cùng tên (`self.learning_rate = learning_rate`), không xử lý gì thêm.
  Nhờ vậy `get_params()` đọc được chúng bằng introspection, `set_params()` ghi
  đè được, và `sklearn.base.clone()` tạo được bản sao *chưa huấn luyện* với đúng
  bộ siêu tham số đó.
* **Thuộc tính học được** = có **dấu gạch dưới ở cuối** (`estimator_`,
  `classes_`, `best_params_`) và chỉ xuất hiện *sau* `fit`/`tune`.
  `check_is_fitted` dựa đúng vào quy ước này để biết mô hình đã train chưa.
* Sau `tune()`: bộ tốt nhất vừa được **ghi đè vào chính thuộc tính siêu tham số
  của đối tượng** (qua `set_params`), vừa được lưu thành bản ghi `best_params_`,
  `best_score_`, `tuning_results_` (bảng mọi cấu hình đã thử). Đối tượng tự mang
  theo lịch sử tune của nó; `clone()` hay `save()` đều giữ bộ siêu tham số đã
  tune, và `export_hyperparams()` ghi ra JSON để đưa vào báo cáo.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from sklearn.utils.validation import check_is_fitted


def _to_python(value: Any) -> Any:
    """numpy scalar → kiểu Python thuần, để so sánh và ghi JSON không bất ngờ."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return type(value)(_to_python(v) for v in value)
    return value


def _jsonable(value: Any) -> Any:
    value = _to_python(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


class BaseModel(ClassifierMixin, BaseEstimator, ABC):
    """Lớp cha trừu tượng: bộ phân loại nhị phân trả xác suất `P(is_low_rating = 1)`.

    Lớp con phải:
      * khai báo **mọi** siêu tham số tường minh trong `__init__` (không `**kwargs`)
        và gán nguyên văn — nếu không `clone()`/`get_params()` sẽ sai;
      * cài `_fit(X, y, sample_weight)` và `_predict_positive(X)`.

    Thuộc tính `features` (nếu lớp con có) là danh sách cột mô hình được phép
    đọc. `X` truyền vào có thể chứa thêm cột meta (năm, trọng số…) — mô hình tự
    chọn đúng cột của mình, nên mọi mô hình nhận chung một `X`.
    """

    #: Tên ngắn dùng trong bảng kết quả. Lớp con ghi đè.
    name: ClassVar[str] = "base"

    # ------------------------------------------------------------------ móc cho lớp con
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        """Huấn luyện trên các cột đã chọn. Ghi mọi trạng thái học được vào thuộc tính `*_`."""

    @abstractmethod
    def _predict_positive(self, X: pd.DataFrame) -> np.ndarray:
        """Trả mảng 1 chiều `P(y = 1)` cho từng dòng."""

    # ------------------------------------------------------------------ template methods
    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} nhận pandas.DataFrame, nhận {type(X).__name__}")
        features = getattr(self, "features", None)
        if features is None:
            return X
        missing = [c for c in features if c not in X.columns]
        if missing:
            raise KeyError(f"{type(self).__name__}: thiếu cột {missing}")
        return X[list(features)]

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "BaseModel":
        """Kiểm tra đầu vào → chọn cột → gọi `_fit` của lớp con. Trả về chính nó."""
        Xs = self._select(X)
        y_arr = np.asarray(y).astype(int).ravel()
        if len(y_arr) != len(Xs):
            raise ValueError(f"X có {len(Xs)} dòng nhưng y có {len(y_arr)}")
        if not np.isin(y_arr, (0, 1)).all():
            raise ValueError("y phải là nhãn nhị phân 0/1")
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
        if sw is not None and (len(sw) != len(y_arr) or np.any(sw < 0)):
            raise ValueError("sample_weight phải cùng độ dài với y và không âm")

        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.asarray(Xs.columns, dtype=object)
        self.n_features_in_ = Xs.shape[1]
        self._fit(Xs, y_arr, sw)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ma trận `(n, 2)`: cột 0 = P(không phải review xấu), cột 1 = P(review xấu)."""
        check_is_fitted(self, "classes_")
        p1 = np.clip(np.asarray(self._predict_positive(self._select(X)), dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Nhãn cứng theo ngưỡng. Ngưỡng 0,5 hiếm khi đúng với lớp hiếm — xem notebook."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def score(self, X: pd.DataFrame, y, sample_weight=None) -> float:
        """**Ghi đè** `ClassifierMixin.score` (vốn là accuracy) thành PR-AUC có trọng số.

        Accuracy vô nghĩa khi lớp dương chỉ ~3%: đoán toàn "0" đã được 97%.
        """
        return float(average_precision_score(y, self.predict_proba(X)[:, 1], sample_weight=sample_weight))

    def with_features(self, features: list[str]) -> "BaseModel":
        """Bản sao chưa train, cùng siêu tham số, nhưng dùng tập feature khác (cho ablation).

        Lớp bọc (`CalibratedModel`, `EnsembleModel`) **ghi đè** phương thức này để
        truyền xuống mô hình bên trong — code ablation gọi một lệnh cho mọi loại
        mô hình mà không cần biết nó là lớp nào (đa hình).
        """
        return clone(self).set_params(features=list(features))

    # ------------------------------------------------------------------ tuning (kế thừa cho mọi lớp con)
    def tune(
        self,
        X: pd.DataFrame,
        y,
        param_distributions: Mapping[str, Any],
        cv,
        n_iter: int = 30,
        sample_weight=None,
        eval_weight=None,
        random_state: int = 42,
        refit: bool = True,
        verbose: bool = False,
    ) -> "BaseModel":
        """Random search siêu tham số trên CV theo thời gian, rồi **tự cập nhật chính mình**.

        Với mỗi cấu hình ứng viên: `clone(self)` → `set_params(ứng viên)` → train
        trên phần train của từng fold → chấm PR-AUC *có trọng số* (`eval_weight`)
        trên phần kiểm định. Cấu hình có điểm trung bình cao nhất thắng.

        Sau khi chạy xong, đối tượng có:
          * các thuộc tính siêu tham số (`self.learning_rate`, …) = bộ tốt nhất;
          * `best_params_`, `best_score_`, `tuning_results_`, `tuning_space_`;
          * đã được `fit` lại trên toàn bộ `X` nếu `refit=True`.

        Args:
            cv: splitter có `split(X)` — dùng `YearForwardSplit`, không KFold.
            sample_weight: trọng số khi *train* (D9: None).
            eval_weight: trọng số khi *chấm điểm* (trọng số khảo sát).
        """
        y_arr = np.asarray(y).astype(int).ravel()
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        ew = np.ones(len(y_arr)) if eval_weight is None else np.asarray(eval_weight, dtype=float)
        candidates = [
            {k: _to_python(v) for k, v in c.items()}
            for c in ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state)
        ]
        folds = list(cv.split(X))

        rows = []
        for i, params in enumerate(candidates):
            started = time.perf_counter()
            scores = []
            for tr, va in folds:
                model = clone(self).set_params(**params)
                model.fit(X.iloc[tr], y_arr[tr], sample_weight=None if sw is None else sw[tr])
                p = model.predict_proba(X.iloc[va])[:, 1]
                scores.append(average_precision_score(y_arr[va], p, sample_weight=ew[va]))
            row = {**params, "mean_score": float(np.mean(scores)), "std_score": float(np.std(scores))}
            row.update({f"fold{k + 1}": s for k, s in enumerate(scores)})
            row["seconds"] = time.perf_counter() - started
            rows.append(row)
            if verbose:
                print(f"[{i + 1:>3}/{len(candidates)}] PR-AUC={row['mean_score']:.4f}  {params}")

        results = pd.DataFrame(rows)
        best_idx = int(results["mean_score"].to_numpy().argmax())
        best = candidates[best_idx]
        results.insert(0, "rank", results["mean_score"].rank(ascending=False, method="min").astype(int))

        self.set_params(**best)  # ← bộ tốt nhất ghi vào CHÍNH thuộc tính của đối tượng
        self.best_params_ = best
        self.best_score_ = float(results.loc[best_idx, "mean_score"])
        self.tuning_results_ = results.sort_values("rank").reset_index(drop=True)
        self.tuning_space_ = dict(param_distributions)
        self.tuning_cv_ = repr(cv)
        if refit:
            self.fit(X, y_arr, sample_weight=sw)
        return self

    # ------------------------------------------------------------------ lưu trữ
    def get_hyperparams(self) -> dict[str, Any]:
        """Siêu tham số hiện hành (không đệ quy vào mô hình con)."""
        return {k: _to_python(v) for k, v in self.get_params(deep=False).items()}

    def export_hyperparams(self, path: str | Path) -> Path:
        """Ghi siêu tham số + kết quả tune ra JSON — bằng chứng đưa vào báo cáo."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": type(self).__name__,
            "name": self.name,
            "params": _jsonable(self.get_hyperparams()),
            "best_params": _jsonable(getattr(self, "best_params_", None)),
            "best_score_pr_auc_cv": getattr(self, "best_score_", None),
            "tuning_space": _jsonable(getattr(self, "tuning_space_", None)),
            "tuning_cv": getattr(self, "tuning_cv_", None),
            "n_candidates": len(getattr(self, "tuning_results_", [])),
            "sklearn": sklearn.__version__,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def save(self, path: str | Path) -> Path:
        """Lưu toàn bộ đối tượng (siêu tham số + trạng thái đã học) bằng joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} chứa {type(obj).__name__}, không phải {cls.__name__}")
        return obj
