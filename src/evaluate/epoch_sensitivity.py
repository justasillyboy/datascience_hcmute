"""
A7 — Độ nhạy kỷ nguyên.

Chạy lại KẾT LUẬN CHÍNH trên 4 lát:
    1. Full
    2. Exclude 2021
    3. 2022+
    4. 2023+

Kết luận chính:
    Chênh lệch sức phân biệt rating:
        |Δrating theo nhãn khách| - |Δrating theo SLA|

Giữ nguyên:
    - trọng số
    - is_analysable
    - customer_says_late
    - product_id cluster bootstrap
    - N_BOOT = 2_000
    - SEED = 1

Output:
    docs/evidence/a7_epoch_sensitivity.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluate.run_analysis import _power_gap
from src.evaluate.stats import cluster_bootstrap


# ============================================================
# CONFIG — giữ nguyên pipeline hiện tại
# ============================================================

DATA_PATH = Path("data/processed/reviews_clean.parquet")
OUTPUT_PATH = Path("docs/evidence/a7_epoch_sensitivity.csv")

N_BOOT = 2_000
SEED = 1


# ============================================================
# TÍNH KẾT LUẬN CHÍNH
# ============================================================

def compute_main_estimate(df: pd.DataFrame):
    """
    Tính kết luận chính trên một lát dữ liệu.

    Giữ nguyên logic của run_analysis.py:
        1. Chỉ lấy is_analysable
        2. Chỉ lấy nhóm có customer_says_late
        3. Bootstrap theo product_id
        4. Có trọng số
    """

    # Giống key_estimates() trong run_analysis.py
    a = df[df["is_analysable"]]

    lab = a[a["customer_says_late"].notna()].copy()

    if lab.empty:
        return None

    lab["rating"] = lab["rating"].astype(float)

    estimate = cluster_bootstrap(
        lab,
        _power_gap,
        n_boot=N_BOOT,
        seed=SEED,
        name="Chênh lệch sức phân biệt (nhãn khách − SLA)",
    )

    return estimate


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Đọc dữ liệu
    # --------------------------------------------------------

    print("=" * 70)
    print("A7 — ĐỘ NHẠY KỶ NGUYÊN")
    print("=" * 70)

    print(f"\nĐọc dữ liệu: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    print(f"Số dòng ban đầu: {len(df):,}")

    # --------------------------------------------------------
    # 2. Tạo biến năm
    # --------------------------------------------------------

    df = df.copy()

    df["year"] = df["review_ts"].dt.year

    # --------------------------------------------------------
    # 3. Tạo 4 lát cắt
    # --------------------------------------------------------

    slices = {
        "Full": df,
        "Exclude 2021": df[df["year"] != 2021],
        "2022+": df[df["year"] >= 2022],
        "2023+": df[df["year"] >= 2023],
    }

    # --------------------------------------------------------
    # 4. Tính kết luận chính
    # --------------------------------------------------------

    results = []

    print("\n" + "-" * 70)
    print("KẾT QUẢ")
    print("-" * 70)

    for epoch, data in slices.items():

        print(f"\n[{epoch}]")
        print(f"  Raw n: {len(data):,}")

        estimate = compute_main_estimate(data)

        if estimate is None:
            print("  Không có dữ liệu có nhãn.")
            continue

        result = {
            "epoch": epoch,
            "estimate": estimate.value,
            "ci_low": estimate.lo,
            "ci_high": estimate.hi,
            "n": estimate.n,
            "n_eff": estimate.n_eff,
            "n_clusters": estimate.n_clusters,
        }

        results.append(result)

        print(
            f"  Estimate : {estimate.value:+.4f}"
        )
        print(
            f"  95% CI   : [{estimate.lo:+.4f}, "
            f"{estimate.hi:+.4f}]"
        )
        print(
            f"  n        : {estimate.n:,}"
        )
        print(
            f"  n_eff    : {estimate.n_eff:.2f}"
        )
        print(
            f"  clusters : {estimate.n_clusters:,}"
        )

    # --------------------------------------------------------
    # 5. Tạo bảng kết quả
    # --------------------------------------------------------

    result_df = pd.DataFrame(results)

    # Đảm bảo đúng thứ tự 4 epoch
    epoch_order = [
        "Full",
        "Exclude 2021",
        "2022+",
        "2023+",
    ]

    result_df["epoch"] = pd.Categorical(
        result_df["epoch"],
        categories=epoch_order,
        ordered=True,
    )

    result_df = result_df.sort_values("epoch")

    # --------------------------------------------------------
    # 6. Lưu CSV
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 7. In bảng cuối cùng
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BẢNG A7")
    print("=" * 70)

    display_df = result_df.copy()

    display_df["estimate"] = display_df["estimate"].map(
        lambda x: f"{x:+.4f}"
    )

    display_df["95% CI"] = result_df.apply(
        lambda row: (
            f"[{row['ci_low']:+.4f}, "
            f"{row['ci_high']:+.4f}]"
        ),
        axis=1,
    )

    display_df["n_eff"] = display_df["n_eff"].map(
        lambda x: f"{x:.2f}"
    )

    print(
        display_df[
            [
                "epoch",
                "estimate",
                "95% CI",
                "n",
                "n_eff",
                "n_clusters",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print(f"Đã lưu: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()