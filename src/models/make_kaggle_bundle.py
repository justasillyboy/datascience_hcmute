"""Đóng gói Track C để chạy notebook trên Kaggle — `python3 -m src.models.make_kaggle_bundle`.

Tạo `dist/trackc_kaggle.zip` gồm mã nguồn cần thiết, notebook, mô hình đã tune và
**bản rút gọn** của dữ liệu:

* chỉ giữ các cột mô hình thật sự dùng — bỏ `content`, `title`, `attributes`,
  `seller_name`… (nội dung do người dùng viết không cần cho Track C);
* `customer_id` thay bằng mã số thứ tự (`pd.factorize`) — giữ nguyên việc nhóm
  theo khách nên feature lịch sử khách không đổi, nhưng không còn ID Tiki thật.

⚠️ Ràng buộc đạo đức của đồ án (README — "không public dữ liệu thô"): khi upload lên
Kaggle phải chọn **Private**. Không chia sẻ dataset công khai.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from src.features.build import build_features
from src.models.experiments import ARTIFACT_DIR, CLEAN_PATH, PRODUCTS_PATH

OUT = Path("dist/trackc_kaggle.zip")
PREFIX = "trackc"

CLEAN_COLUMNS = [
    "review_id", "product_id", "seller_id", "customer_id", "category_id", "category_name",
    "rating", "is_low_rating", "weight", "stratum", "stratum_size", "stratum_sampled",
    "purchased_ts", "delivered_ts", "review_ts", "customer_joined", "lead_days",
    "review_lag_days", "sla_breach", "is_analysable", "has_delivery_rating", "n_images",
    "dr_thoi_gian", "dr_shipper", "dr_gio_giao", "dr_dong_goi",
]
PRODUCT_COLUMNS = ["product_id", "price", "discount_rate"]
SOURCE_GLOBS = ["src/__init__.py", "src/evaluate/__init__.py", "src/evaluate/stats.py",
                "src/features/*.py", "src/models/*.py"]

README = """# Track C trên Kaggle

1. Kaggle → Datasets → New Dataset → upload `trackc_kaggle.zip` → **Private** → Create.
2. Kaggle → Code → New Notebook → File → Import Notebook → chọn `trackc/notebooks/03_modeling.ipynb`
   (lấy từ file zip hoặc từ repo).
3. Bên phải: Add Input → chọn dataset vừa tạo · Settings → Internet **On** (để ghim scikit-learn 1.5.1).
4. Run All (~20–25 phút CPU). Xong: File → Download notebook → chép vào `notebooks/03_modeling.ipynb` của repo.
"""


def _slim_clean() -> pd.DataFrame:
    clean = pd.read_parquet(CLEAN_PATH, columns=CLEAN_COLUMNS)
    return clean.assign(customer_id=pd.factorize(clean["customer_id"])[0])


def _check_same_features(slim: pd.DataFrame, products: pd.DataFrame) -> None:
    """Bản rút gọn phải cho **đúng** bảng feature như bản gốc — nếu không thì gói hỏng."""
    original = build_features(pd.read_parquet(CLEAN_PATH), pd.read_parquet(PRODUCTS_PATH))
    rebuilt = build_features(slim, products)
    pd.testing.assert_frame_equal(original.drop(columns="customer_id", errors="ignore"),
                                  rebuilt.drop(columns="customer_id", errors="ignore"))


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def main() -> None:
    slim = _slim_clean()
    products = pd.read_parquet(PRODUCTS_PATH, columns=PRODUCT_COLUMNS)
    _check_same_features(slim, products)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    root = Path(".")
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{PREFIX}/README_KAGGLE.md", README)
        zf.writestr(f"{PREFIX}/{CLEAN_PATH.as_posix()}", _parquet_bytes(slim))
        zf.writestr(f"{PREFIX}/{PRODUCTS_PATH.as_posix()}", _parquet_bytes(products))
        files = [p for g in SOURCE_GLOBS for p in sorted(root.glob(g))]
        files += sorted(ARTIFACT_DIR.glob("*_tuned.*"))
        files += [Path("notebooks/03_modeling.ipynb"), Path("requirements.txt")]
        for path in files:
            zf.write(path, f"{PREFIX}/{path.as_posix()}")
    print(f"Đã tạo {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(files) + 3} file). "
          "Nhớ chọn Private khi upload lên Kaggle.")


if __name__ == "__main__":
    main()
