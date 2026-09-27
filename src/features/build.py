"""Dựng bảng feature cho Track C — task C1 (feature), C2 (trọng số), C5 (chia thời gian).

Bài toán
--------
Mục tiêu `is_low_rating = rating ≤ 3` (quyết định D2). **Thời điểm dự báo**: lúc
đơn được giao tới tay khách, trước khi khách viết review. Mỗi feature dưới đây
đã biết tại thời điểm đó (tầng `ex_ante`), trừ nhóm `EX_POST_FEATURES` do khách
khai cùng lúc với số sao — nhóm này chỉ dùng cho mô hình *giải thích* (C9).

Quyết định D9 — trọng số khi train và khi đánh giá (task C2)
------------------------------------------------------------
Nhóm chốt ngày 2026-09-22: **train KHÔNG trọng số · đánh giá CÓ trọng số.**

* **Đánh giá CÓ trọng số — giữ nguyên, đúng.** Metric là ước lượng quần thể
  ("mô hình tốt cỡ nào trên *mọi* đơn Tiki"), nên phải nhân `w = N_h/n_h` như mọi
  ước lượng khác trong đồ án.
* **Train KHÔNG trọng số — đã kiểm bằng thực nghiệm và KHÔNG đứng vững.** Lập
  luận gốc là *case-control sampling*: nếu xác suất được lấy mẫu chỉ phụ thuộc
  nhãn thì logistic không trọng số chỉ lệch hệ số chặn, độ dốc vẫn nhất quán
  (Prentice & Pyke, 1979). Nhưng ở đây xác suất lấy mẫu một review 5★ là
  `n_h/N_h` — **phụ thuộc sản phẩm**: sản phẩm càng đông review thì tầng 5★ càng
  bị cắt mạnh. Điều kiện của định lý bị vi phạm, và hệ quả đo được là quan hệ
  feature–nhãn **đảo chiều** giữa mẫu thô và quần thể (vd `prod_hist_n`: AUC
  0,55 trên mẫu thô, 0,38 trên quần thể). Mô hình train không trọng số học đúng
  chiều sai: PR-AUC có trọng số trên test ≈ mức ngẫu nhiên. Train **có trọng số
  khảo sát** (ước lượng rủi ro quần thể — importance weighting) sửa được điều
  này. Bằng chứng đầy đủ: notebook `03_modeling.ipynb` §4 và
  `docs/evidence/c_modeling_*.txt`. Phương án train được chọn bằng CV theo thời
  gian trên tập train, **không** nhìn tập test.
* Hệ quả phụ: tỉ lệ review xấu giảm dần theo năm (trôi dạt prior), nên xác suất
  vẫn phải **hiệu chỉnh** trên năm gần nhất của tập train — `CalibratedModel`.

Mọi hàm trả về DataFrame mới, không sửa đầu vào.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .history import asof_group_stats
from .leakage import audit_features

logger = logging.getLogger(__name__)

TARGET = "is_low_rating"
WEIGHT = "weight"
YEAR = "year"
CLUSTER = "product_id"
TEST_START_YEAR = 2024  # quyết định D3: train ≤2023 · test 2024–2026
TIKI_TRADING_SELLER_ID = 1

#: Số "review ảo" của prior khi làm trơn tỉ lệ lịch sử — đơn vị là review quần
#: thể (đã nhân trọng số). 20 nghĩa là cần ~20 review thật thì dữ liệu riêng của
#: nhóm mới nặng ngang tỉ lệ chung.
PRIOR_STRENGTH = 20.0

#: Nhóm feature — đơn vị của phép ablation "bỏ từng nhóm" (tầng 7).
FEATURE_GROUPS: dict[str, list[str]] = {
    "giao_hang": ["lead_days", "delivered_hour", "delivered_weekend"],
    "thoi_diem_mua": ["purchase_month", "purchase_weekday", "purchase_hour"],
    "san_pham": ["category", "is_tiki_trading", "log_price", "discount_rate"],
    "lich_su_san_pham": ["prod_hist_n", "prod_hist_rate"],
    "lich_su_nha_ban": ["seller_hist_n", "seller_hist_rate"],
    "khach_hang": ["cust_tenure_days", "cust_hist_n", "cust_hist_rate"],
}
EX_ANTE_FEATURES: list[str] = [f for g in FEATURE_GROUPS.values() for f in g]
EX_POST_FEATURES: list[str] = [
    "customer_says_late",
    "dr_shipper_bad",
    "dr_gio_giao_bad",
    "dr_dong_goi_bad",
    "has_delivery_rating",
    "n_images",
    "review_lag_days",
]
#: Chỉ báo mà doanh nghiệp đang dùng — dùng riêng cho M1a, không vào mô hình đầy đủ
#: (mô hình đầy đủ đã có `lead_days`, mà `sla_breach` chỉ là một ngưỡng của nó).
SLA_FEATURE = "sla_breach"
CATEGORICAL: tuple[str, ...] = ("category",)

META_COLUMNS = [
    "review_id", CLUSTER, "seller_id", YEAR, WEIGHT, TARGET,
    "category_name", "is_census", "has_customer_label",
]

_BAD_ANSWER = {
    "dr_thoi_gian": ("customer_says_late", "Giao trễ hẹn", "Giao đúng hẹn"),
    "dr_shipper": ("dr_shipper_bad", "Thô lỗ", "Lịch sự"),
    "dr_gio_giao": ("dr_gio_giao_bad", "Không hẹn trước", "Có hẹn giờ trước"),
    "dr_dong_goi": ("dr_dong_goi_bad", "Cẩu thả", "Cẩn thận"),
}

audit_features(EX_ANTE_FEATURES, tier="ex_ante")
audit_features(EX_ANTE_FEATURES + EX_POST_FEATURES, tier="ex_post")


def _census_products(df: pd.DataFrame) -> pd.Series:
    """Sản phẩm được cào **vét cạn mọi tầng sao** → mẫu chính là quần thể.

    Ở các sản phẩm này không có lấy mẫu nên cũng không có thiên lệch lấy mẫu nào,
    kể cả thiên lệch thời gian của tầng 5★ (xem notebook §2). Đây là tập kiểm
    chứng "sạch" cho mọi kết luận của Track C.
    """
    full = (df["stratum_sampled"] >= df["stratum_size"]).groupby(df[CLUSTER]).transform("all")
    return full.astype(bool)


def _history_features(df: pd.DataFrame, prior_rate: float) -> pd.DataFrame:
    y = df[TARGET].astype(float)
    out = {}
    for prefix, key in (("prod", CLUSTER), ("seller", "seller_id"), ("cust", "customer_id")):
        sum_w, rate = asof_group_stats(
            df[key], df["review_ts"], df["delivered_ts"], y, df[WEIGHT],
            prior_rate=prior_rate, prior_strength=PRIOR_STRENGTH,
        )
        out[f"{prefix}_hist_n"] = np.log1p(sum_w)
        out[f"{prefix}_hist_rate"] = rate
    return pd.DataFrame(out, index=df.index)


def _customer_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for raw, (name, bad, good) in _BAD_ANSWER.items():
        out[name] = df[raw].map({bad: 1.0, good: 0.0}).astype(float)
    return pd.DataFrame(out, index=df.index)


def _product_attributes(df: pd.DataFrame, products: pd.DataFrame | None) -> pd.DataFrame:
    """Giá và giảm giá — **ảnh chụp lúc cào**, giả định là thuộc tính tĩnh của sản phẩm.

    Không lấy `rating_average`, `review_count`, `quantity_sold` (xem `FORBIDDEN`).
    """
    if products is None:
        return pd.DataFrame({"log_price": np.nan, "discount_rate": np.nan}, index=df.index)
    attrs = products.drop_duplicates(CLUSTER).set_index(CLUSTER)
    price = df[CLUSTER].map(attrs["price"]).astype(float)
    return pd.DataFrame(
        {
            "log_price": np.log1p(price.where(price > 0)),
            "discount_rate": df[CLUSTER].map(attrs["discount_rate"]).astype(float),
        },
        index=df.index,
    )


def build_features(
    clean: pd.DataFrame,
    products: pd.DataFrame | None = None,
    train_end_year: int = TEST_START_YEAR - 1,
) -> pd.DataFrame:
    """Bảng đã làm sạch → bảng mô hình: cột meta + feature ex_ante + ex_post.

    Chỉ giữ các dòng `is_analysable` (có `lead_days` hợp lệ). Lịch sử as-of thì
    tính trên **toàn bộ** review có mốc đăng, kể cả dòng không phân tích được —
    chúng vẫn là thông tin công khai tại thời điểm dự báo.

    Args:
        train_end_year: tỉ lệ prior `p0` của phép làm trơn chỉ được ước lượng từ
            các năm ≤ mốc này, để không một thông tin nào của tập test lọt vào.
    """
    df = clean.copy()
    df[TARGET] = df[TARGET].astype(bool)
    df["is_census"] = _census_products(df)
    df[YEAR] = df["review_ts"].dt.year.fillna(df["delivered_ts"].dt.year)

    train_rows = df[YEAR] <= train_end_year
    prior_rate = float(np.average(df.loc[train_rows, TARGET], weights=df.loc[train_rows, WEIGHT]))
    hist = _history_features(df, prior_rate)

    tenure = (df["purchased_ts"] - pd.to_datetime(df["customer_joined"], errors="coerce"))
    tenure_days = tenure.dt.total_seconds() / 86400
    n_bad_tenure = int((tenure_days < 0).sum())
    if n_bad_tenure:
        logger.info("cust_tenure_days: %d dòng âm (tham gia sau lúc mua) → NaN", n_bad_tenure)

    feats = pd.DataFrame(
        {
            "lead_days": df["lead_days"],
            "delivered_hour": df["delivered_ts"].dt.hour,
            "delivered_weekend": (df["delivered_ts"].dt.dayofweek >= 5).astype(float),
            "purchase_month": df["purchased_ts"].dt.month,
            "purchase_weekday": df["purchased_ts"].dt.dayofweek,
            "purchase_hour": df["purchased_ts"].dt.hour,
            "category": pd.Categorical(df["category_id"]).codes.astype(int),
            "is_tiki_trading": (df["seller_id"] == TIKI_TRADING_SELLER_ID).astype(float),
            "cust_tenure_days": tenure_days.where(tenure_days >= 0),
            SLA_FEATURE: df[SLA_FEATURE].astype(float),
            "has_delivery_rating": df["has_delivery_rating"].astype(float),
            "n_images": df["n_images"].astype(float),
            "review_lag_days": df["review_lag_days"],
        },
        index=df.index,
    )
    frame = pd.concat(
        [df[[c for c in META_COLUMNS if c in df.columns]], feats, hist,
         _product_attributes(df, products), _customer_labels(df)],
        axis=1,
    )
    frame[TARGET] = frame[TARGET].astype(int)
    frame["has_customer_label"] = frame["customer_says_late"].notna()
    # Lọc TRƯỚC khi ép kiểu năm: dòng không phân tích được có thể thiếu cả hai mốc.
    frame = frame[df["is_analysable"].astype(bool)].reset_index(drop=True)
    frame[YEAR] = frame[YEAR].astype(int)
    return frame[META_COLUMNS + [c for c in frame.columns if c not in META_COLUMNS]]


def temporal_split(
    frame: pd.DataFrame, test_start_year: int = TEST_START_YEAR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia theo năm đăng review (D3) — không bao giờ KFold ngẫu nhiên.

    Chia theo năm *đăng review* bảo đảm mọi nhãn của tập train đều đã quan sát
    được trước ngày 01/01 của năm test — đúng tình huống triển khai thật.
    """
    is_test = frame[YEAR] >= test_start_year
    train, test = frame[~is_test].copy(), frame[is_test].copy()
    assert train[YEAR].max() < test[YEAR].min(), "chồng lấn thời gian giữa train và test"
    return train, test
