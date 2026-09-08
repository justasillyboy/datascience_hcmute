"""Điểm chạy bước Data Collection.

    python -m src.collect.run_crawl --max-products 50 --out data/raw

Chạy lại an toàn: mọi response đã nằm trong cache đĩa nên lần chạy thứ hai gần
như không phát sinh request nào tới Tiki.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .tiki import CATEGORIES, TikiListingScraper, TikiReviewScraper

logger = logging.getLogger(__name__)


def crawl(
    out_dir: Path,
    cache_dir: Path,
    max_products_per_cat: int,
    max_pages: int,
    min_reviews: int,
    sample_cap: int,
    min_interval: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    listing = TikiListingScraper(cache_dir=cache_dir, min_interval=min_interval)
    reviews = TikiReviewScraper(cache_dir=cache_dir, min_interval=min_interval, sample_cap=sample_cap)

    products: list[dict] = []
    for cat_id, cat_name in CATEGORIES.items():
        taken = 0
        for product in listing.iter_products(cat_id, max_pages=max_pages, min_reviews=min_reviews):
            products.append(product)
            taken += 1
            if taken >= max_products_per_cat:
                break
        logger.info("Ngành %-28s → %d sản phẩm", cat_name, taken)

    products_df = pd.DataFrame(products).drop_duplicates(subset=["product_id"])
    logger.info("Tổng sản phẩm: %d | %s", len(products_df), listing.stats())

    review_rows: list[dict] = []
    for i, row in enumerate(products_df.itertuples(index=False), start=1):
        seller_id = int(row.seller_id) if pd.notna(row.seller_id) else 1
        try:
            batch = reviews.fetch(int(row.product_id), seller_id)
        except Exception as exc:  # noqa: BLE001 - một sản phẩm hỏng không được giết cả mẻ
            logger.warning("Bỏ qua product_id=%s: %s", row.product_id, exc)
            continue
        for r in batch:
            r["category_id"] = row.category_id
            r["category_name"] = row.category_name
        review_rows.extend(batch)
        if i % 25 == 0:
            logger.info("  %d/%d sản phẩm · %d review · %s", i, len(products_df), len(review_rows), reviews.stats())

    reviews_df = pd.DataFrame(review_rows)
    logger.info("Tổng review: %d | %s", len(reviews_df), reviews.stats())

    out_dir.mkdir(parents=True, exist_ok=True)
    products_df.to_parquet(out_dir / "tiki_products.parquet", index=False)
    reviews_df.to_parquet(out_dir / "tiki_reviews.parquet", index=False)
    logger.info("Đã ghi %s", out_dir)

    _report(reviews_df)
    return products_df, reviews_df


def _report(df: pd.DataFrame) -> None:
    """In vài con số để phát hiện ngay nếu mẻ cào bị lệch hoặc thiếu trường."""
    if df.empty:
        logger.warning("Không thu được review nào.")
        return
    n = len(df)
    print(f"\n=== MẺ CÀO: {n} review ===")
    print("\nPhân bố sao (thô — ĐÃ phân tầng nên KHÔNG phải phân bố quần thể):")
    print((df["rating"].value_counts().sort_index() / n * 100).round(1).to_string())
    if "weight" in df:
        w = df.groupby("rating")["weight"].sum()
        print("\nPhân bố sao ước lượng quần thể (đã nhân trọng số):")
        print((w / w.sum() * 100).round(1).to_string())
    print("\nĐộ phủ các trường then chốt:")
    for col in ["purchased_at", "delivery_date", "review_created_date", "dr_thoi_gian"]:
        if col in df:
            print(f"  {col:22s} {df[col].notna().mean() * 100:5.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="Cào dữ liệu Tiki cho đồ án KHDL")
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--max-products", type=int, default=50, help="mỗi ngành hàng")
    p.add_argument("--max-pages", type=int, default=10, help="số trang listing mỗi ngành")
    p.add_argument("--min-reviews", type=int, default=5)
    p.add_argument("--sample-cap", type=int, default=40, help="trần lấy mẫu cho tầng đông")
    p.add_argument("--min-interval", type=float, default=1.0, help="giây giữa 2 request")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    crawl(
        out_dir=args.out,
        cache_dir=args.cache,
        max_products_per_cat=args.max_products,
        max_pages=args.max_pages,
        min_reviews=args.min_reviews,
        sample_cap=args.sample_cap,
        min_interval=args.min_interval,
    )


if __name__ == "__main__":
    main()
