"""Sinh `data/processed/reviews_clean.parquet` từ `data/raw/` — một lệnh.

Trước đây bảng đã làm sạch được tạo rời rạc trong phiên làm việc, không có
đường tái lập. Hệ quả: khi bước cleaning đổi (ví dụ sửa lỗi múi giờ ngày
2026-09-09) thì không ai biết bảng trên đĩa đã cũ hay chưa. Module này đóng lỗ
hổng đó — bảng processed luôn là kết quả chạy được lại từ raw.

Data contract chạy **hai lần**: trên bảng thô và trên bảng đã sạch. Vi phạm ở
bất kỳ đâu thì dừng, không ghi file.

Chạy:  python3 -m src.clean.run_clean
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.clean.reviews import SLA_MAX_DAYS, clean_reviews, quality_report
from src.validate.contract import validate_clean_reviews, validate_raw_reviews

logger = logging.getLogger(__name__)


def build(raw_path: Path, out_path: Path, sla_days: float = SLA_MAX_DAYS) -> pd.DataFrame:
    raw = pd.read_parquet(raw_path)
    print(f"Đọc {raw_path} — {len(raw):,} dòng thô\n")

    print(validate_raw_reviews(raw).raise_if_failed())

    clean = clean_reviews(raw, sla_days=sla_days)
    print(f"\nSau làm sạch: {len(clean):,} dòng\n")
    print(validate_clean_reviews(clean, sla_days=sla_days).raise_if_failed())

    print("\n" + quality_report(clean).to_string())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path, index=False)
    print(f"\nĐã ghi {out_path}")
    return clean


def main() -> None:
    ap = argparse.ArgumentParser(description="Làm sạch bảng review Tiki")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/tiki_reviews.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/reviews_clean.parquet"))
    ap.add_argument("--sla-days", type=float, default=SLA_MAX_DAYS)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build(args.raw, args.out, sla_days=args.sla_days)


if __name__ == "__main__":
    main()
