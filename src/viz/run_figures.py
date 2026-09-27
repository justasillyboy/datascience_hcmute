"""Chạy các figure Track B đã hoàn thành trong ngày 27/09.

Chạy:
    python3 -m src.viz.run_figures
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.viz.theme import setup_theme
from src.viz.eda import plot_sla_2x2, plot_label_coverage_by_year

DATA_PATH = Path("data/processed/reviews_clean.parquet")


def main() -> None:
    setup_theme()
    df = pd.read_parquet(DATA_PATH)
    print(f"Đọc dữ liệu: {DATA_PATH}")
    print(f"n = {len(df):,}")

    _, table, path = plot_sla_2x2(df)
    print("\nB10 — SLA 2×2")
    print(table.to_string())
    print(f"Đã lưu: {path}")

    _, result, path = plot_label_coverage_by_year(df)
    print("\nB14 — Label coverage")
    print(result.to_string(index=False))
    print(f"Đã lưu: {path}")


if __name__ == "__main__":
    main()
