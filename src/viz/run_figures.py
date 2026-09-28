"""Chạy các figure Track B đã hoàn thành trong ngày 27/09.

Chạy:
    python3 -m src.viz.run_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import pandas as pd

from src.viz.theme import setup_theme
from src.viz.eda import (
    plot_sla_2x2, 
    plot_label_coverage_by_year, 
    plot_lead_days_hist, 
    plot_sla_by_year,
    plot_rating_distribution,
    plot_rating_by_lead_time,
    plot_batch_forest,
    plot_timezone_correction,
    plot_n_vs_neff,
    plot_lead_days_box_by_year,
    plot_lead_days_quantiles,
    plot_seller_sla_trap,
)

DATA_PATH = Path("data/processed/reviews_clean.parquet")


def main() -> None:
    setup_theme()
    df = pd.read_parquet(DATA_PATH)
    print(f"Đọc dữ liệu: {DATA_PATH}")
    print(f"n = {len(df):,}")

    #B10
    _, table, path = plot_sla_2x2(df)
    print("\nB10 — SLA 2×2")
    print(table.to_string())
    print(f"Đã lưu: {path}")

    #B14
    _, result, path = plot_label_coverage_by_year(df)
    print("\nB14 — Label coverage")
    print(result.to_string(index=False))
    print(f"Đã lưu: {path}")
    
    # B4 - Lead Time Distribution
    print("\nB4 — Phân phối Lead Time")
    fig, path = plot_lead_days_hist(df)
    print(f"Đã lưu: {path}")
    plt.close(fig)
    
    # B5 - Lead_days_box_by_year   
    print("Generating B05: Lead Time boxplot by year...")
    fig, result_b5, path_b5 = plot_lead_days_box_by_year(df)
    plt.close(fig)
    print(result_b5.to_string(index=False))
    print(f"Saved: {path_b5}")
    
    # B6 Weighted Lead Time quantiles
    print("\nGenerating B06: Weighted Lead Time quantiles...")
    fig, result_b6, path_b6 = plot_lead_days_quantiles(df)
    plt.close(fig)
    print(result_b6.to_string(index=False))
    print(f"Saved: {path_b6}")
    
    # B7 - SLA Breach Rate by Year
    print("\nB7 — Tỷ lệ SLA Breach theo năm")
    fig, result, path = plot_sla_by_year(df)
    print(result.to_string(index=False))
    print(f"Đã lưu: {path}")
    plt.close(fig)
    
    # B8 - Rating Distribution
    print("\nB8 — Phân phối Rating")
    fig, path = plot_rating_distribution(df)
    print(f"Đã lưu: {path}")
    plt.close(fig)
    
    # B9 - Rating by Lead Time
    print("\nB9 — Weighted Mean Rating theo Lead Time")
    fig, result_b9, path = plot_rating_by_lead_time(df)
    print(result_b9.to_string(index=False))
    print(f"Đã lưu: {path}")
    plt.close(fig)
    
    # B11 - Forest Plot
    print("\nB11 — Forest Plot độ ổn định kết luận chính")
    fig, result_b11, path = plot_batch_forest()
    print(result_b11.to_string(index=False))
    print(f"Đã lưu: {path}")
    plt.close(fig)
    
    # B12 - Timezone Correction
    print("\nB12 — Trước/sau sửa múi giờ")
    fig, result_b12, path = plot_timezone_correction()
    print(result_b12.to_string(index=False))
    print(f"Đã lưu: {path}")
    plt.close(fig)
    
    # B13 - n vs n_eff
    print("\nB13 — So sánh n và n_eff qua 3 mẻ")
    fig, result_b13, path = plot_n_vs_neff()
    print(result_b13.to_string(index=False))
    print(f"Đã lưu: {path}")
    plt.close(fig)

print("\nGenerating B19: SLA trap by seller group...")

c14_path = Path("reports/models/c14_sla_trap.csv")

fig, result_b19, path_b19 = plot_seller_sla_trap(
    c14_path
)

plt.close(fig)

print(result_b19.to_string(index=False))
print(f"Saved: {path_b19}")

if __name__ == "__main__":
    main()
