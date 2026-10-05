"""Đóng gói bài nộp cho giảng viên — `python scripts/make_submission.py`.

Tạo `dist/nop_bai_SLA_tiki.zip`: `main.py` + `src/` + `tests/` + `notebooks/EDA.ipynb` + kết quả GridSearchCV +
**dữ liệu rút gọn**, kèm `README.md` hướng dẫn chạy. Giải nén rồi `python main.py` là chạy.

Dữ liệu rút gọn:
* bảng review đã làm sạch, bỏ văn bản tự do khách viết (`content`, `attributes`);
* `customer_id` thay bằng mã số thứ tự (`pd.factorize`) — vẫn nhóm đúng theo khách nên feature lịch sử khách không
  đổi, nhưng không còn ID Tiki thật;
* bảng sản phẩm chỉ giữ các cột pipeline và notebook dùng.
Trước khi ghi, script kiểm bảng feature dựng từ bản rút gọn **trùng khớp** bản gốc — lệch thì dừng.

⚠️ Gói chứa dữ liệu tự cào → chỉ nộp riêng cho giảng viên, KHÔNG đưa lên GitHub (cam kết "không public dữ liệu
thô" trong README; `dist/` đã nằm trong `.gitignore`).
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build import build_features  # noqa: E402

OUT = ROOT / "dist" / "nop_bai_SLA_tiki.zip"
PREFIX = "nop_bai_SLA_tiki"
CLEAN = Path("data/processed/reviews_clean.parquet")
PRODUCTS = Path("data/raw/tiki_products.parquet")
DROP_CLEAN = ["content", "attributes"]
PRODUCT_COLUMNS = ["product_id", "category_id", "category_name", "price", "discount_rate", "review_count"]
FILES = ["main.py", "requirements.txt", "pyproject.toml", "notebooks/EDA.ipynb", "docs/FINDINGS.md",
         "reports/models/grid_search_best.json", "reports/models/grid_search_cv.csv",
         "reports/models/grid_search_test.csv"]
GLOBS = ["src/**/*.py", "tests/*.py"]

README = """# Bẻ gãy bẫy SLA trong thương mại điện tử — bài nộp nhóm

Python for Data Science · PGS.TS Nguyễn Mạnh Hùng · HCMUTE HK1 2025–2026
Phan Ngô Quốc An (24139002) · Đỗ Minh Hiển (24139013) · Nguyễn Ngọc Ngân (24139031) · Hồ Ngọc Sỹ (24139047)

## Chạy (Python 3.10–3.12)

```bash
pip install -r requirements.txt
python main.py
```

Khoảng 1–2 phút trên laptop: nạp dữ liệu → dựng 17 feature → chia train ≤2023 / test 2024–2026 → train
Logistic, RandomForest, HistGBM và mô hình chốt (ensemble đã hiệu chỉnh) cùng 2 baseline → in bảng metric trên
test → lưu kết quả vào `outputs/`.

Python 3.13 (scikit-learn 1.5.1 chưa có bản cho 3.13): cài bản mới
`pip install pandas numpy scikit-learn scipy pyarrow joblib matplotlib seaborn jupyter` rồi `python main.py` —
vẫn chạy; Logistic ra đúng số, RandomForest/HistGBM có thể lệch nhẹ ở chữ số thứ 3–4.

## Các tuỳ chọn

| Lệnh | Việc làm |
|---|---|
| `python main.py --describe` | chỉ in mô tả pipeline (input/output, các bước, siêu tham số) |
| `python main.py --search quick` | GridSearchCV lưới rút gọn (10 cấu hình × 2 fold, ~1–3 phút), train cấu hình thắng |
| `python main.py --search full` | GridSearchCV đầy đủ 120 cấu hình × 2 fold (~20 phút) |
| `python main.py --models logistic gbm --C 1e-3 --class-weight balanced` | chọn mô hình, ghi đè siêu tham số |
| `python main.py --help` | toàn bộ tham số |
| `python -m pytest` | 111 test |

## Nội dung gói

| Đường dẫn | Là gì |
|---|---|
| `main.py` | pipeline chạy bằng một lệnh, có `argparse` |
| `notebooks/EDA.ipynb` | notebook EDA (đã chạy sẵn output); mở bằng Jupyter để chạy lại |
| `src/` | mã nguồn: thu thập (`collect`), làm sạch (`clean`), data contract (`validate`), feature, mô hình, thống kê |
| `tests/` | pytest |
| `reports/models/grid_search_*.{json,csv}` | kết quả GridSearchCV đầy đủ đã chạy (120 cấu hình) |
| `data/processed/reviews_clean.parquet` | 203.510 review đã làm sạch (tự cào từ Tiki public API) |
| `data/raw/tiki_products.parquet` | 2.438 sản phẩm (giá, giảm giá, ngành hàng) |

Dữ liệu đã bỏ nội dung review khách viết và thay mã khách hàng bằng số thứ tự. Dữ liệu tự cào nên **không công
khai** — gói này chỉ dùng cho việc chấm bài.
"""


def slim_clean() -> pd.DataFrame:
    clean = pd.read_parquet(ROOT / CLEAN).drop(columns=DROP_CLEAN)
    return clean.assign(customer_id=pd.factorize(clean["customer_id"])[0])


def check_same_features(slim: pd.DataFrame, products: pd.DataFrame) -> None:
    """Bản rút gọn phải cho đúng bảng feature như bản gốc — nếu không thì gói hỏng."""
    original = build_features(pd.read_parquet(ROOT / CLEAN), pd.read_parquet(ROOT / PRODUCTS))
    rebuilt = build_features(slim, products)
    pd.testing.assert_frame_equal(original, rebuilt)


def parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def main() -> None:
    slim = slim_clean()
    products = pd.read_parquet(ROOT / PRODUCTS, columns=PRODUCT_COLUMNS)
    check_same_features(slim, products)

    files = [ROOT / f for f in FILES]
    files += sorted(p for g in GLOBS for p in ROOT.glob(g) if "__pycache__" not in p.parts)
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"Thiếu file: {missing}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{PREFIX}/README.md", README)
        zf.writestr(f"{PREFIX}/{CLEAN.as_posix()}", parquet_bytes(slim))
        zf.writestr(f"{PREFIX}/{PRODUCTS.as_posix()}", parquet_bytes(products))
        for path in files:
            zf.write(path, f"{PREFIX}/{path.relative_to(ROOT).as_posix()}")
    print(f"Đã tạo {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(files) + 3} file). "
          "Chỉ nộp riêng cho giảng viên — không đưa lên GitHub.")


if __name__ == "__main__":
    main()
