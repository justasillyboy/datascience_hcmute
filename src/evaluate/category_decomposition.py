"""
C15 — Phân rã bẫy SLA theo ngành hàng.

Nguyên tắc:
- Chỉ dùng review có thể phân tích.
- Chỉ dùng review có customer_says_late.
- Mọi tỷ lệ quần thể đều có trọng số.
- Luôn báo cáo n_eff.
- Không xếp hạng ngành có dữ liệu quá mỏng.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluate.stats import (
    effective_sample_size,
    weighted_proportion,
)


DATA_PATH = Path(
    "data/processed/reviews_clean.parquet"
)

OUTPUT_PATH = Path(
    "docs/evidence/c15_category_decomposition.txt"
)

MIN_N_EFF = 100.0

GROUP_MAP = {
    "Nhà cửa - Đời sống": "Nhà cửa – Gia dụng",
    "Điện gia dụng": "Nhà cửa – Gia dụng",

    "Làm đẹp - Sức khỏe": "Cá nhân – Gia đình",
    "Thời trang nữ": "Cá nhân – Gia đình",
    "Mẹ và bé": "Cá nhân – Gia đình",
    "Bách hóa online": "Cá nhân – Gia đình",

    "Sách tiếng Việt": "Sách – Thể thao – Công nghệ",
    "Nhà sách tiki": "Sách – Thể thao – Công nghệ",
    "Thể thao - Dã ngoại": "Sách – Thể thao – Công nghệ",
    "Điện thoại - Máy tính bảng": "Sách – Thể thao – Công nghệ",
}

def category_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính phân rã SLA theo ngành hàng.

    Output gồm:
        - n
        - n_eff
        - weighted 2x2 proportions
        - hai ô lệch SLA
        - cờ đủ/mỏng dữ liệu
    """

    required = {
        "category_name",
        "is_analysable",
        "customer_says_late",
        "sla_breach",
        "weight",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"Thiếu cột bắt buộc: {sorted(missing)}"
        )

    d = df[
    df["is_analysable"]
    & df["customer_says_late"].notna()
    ].copy()
    d["category_group"] = d["category_name"].map(GROUP_MAP)

    unknown = sorted(
        d.loc[d["category_group"].isna(), "category_name"]
        .dropna()
        .unique()
        .tolist()
    )

    if unknown:
        raise ValueError(
            "Có category chưa được gán group:\n"
            + "\n".join(f"  - {x}" for x in unknown)
        )

    d["sla_breach"] = d["sla_breach"].astype(bool)
    d["customer_says_late"] = (
        d["customer_says_late"].astype(bool)
    )

    rows = []

    for category, g in d.groupby(
        "category_group",
        dropna=False,
        observed=True,
    ):

        w = g["weight"].to_numpy(float)

        n = len(g)

        n_eff = effective_sample_size(w)

        # ----------------------------------------------------
        # 4 ô của bảng 2x2
        # ----------------------------------------------------

        in_sla_ontime = (
            (~g["sla_breach"])
            & (~g["customer_says_late"])
        )

        in_sla_late = (
            (~g["sla_breach"])
            & g["customer_says_late"]
        )

        breach_ontime = (
            g["sla_breach"]
            & (~g["customer_says_late"])
        )

        breach_late = (
            g["sla_breach"]
            & g["customer_says_late"]
        )

        # ----------------------------------------------------
        # Tỷ lệ trên toàn nhóm ngành
        # ----------------------------------------------------

        def cell_pct(mask):
            return (
                weighted_proportion(
                    mask.to_numpy(),
                    w,
                )
                * 100.0
            )

        pct_in_sla_ontime = cell_pct(
            in_sla_ontime
        )

        pct_in_sla_late = cell_pct(
            in_sla_late
        )

        pct_breach_ontime = cell_pct(
            breach_ontime
        )

        pct_breach_late = cell_pct(
            breach_late
        )

        # ----------------------------------------------------
        # n thô của hai ô lệch
        # ----------------------------------------------------

        n_in_sla_late = int(
            in_sla_late.sum()
        )

        n_breach_ontime = int(
            breach_ontime.sum()
        )

        # ----------------------------------------------------
        # n_eff riêng cho hai nhóm lệch
        # ----------------------------------------------------

        n_eff_in_sla_late = effective_sample_size(
            g.loc[in_sla_late, "weight"].to_numpy(float)
        ) if n_in_sla_late else 0.0

        n_eff_breach_ontime = effective_sample_size(
            g.loc[
                breach_ontime,
                "weight",
            ].to_numpy(float)
        ) if n_breach_ontime else 0.0

        # ----------------------------------------------------
        # Cổng dữ liệu
        #
        # Ngành chỉ được xem là "đủ dữ liệu" khi cả hai
        # ô lệch đều có n_eff >= MIN_N_EFF.
        # ----------------------------------------------------

        enough_data = (
            n_eff_in_sla_late >= MIN_N_EFF
            and n_eff_breach_ontime >= MIN_N_EFF
        )

        rows.append(
            {
                "category_name": category,
                "n": n,
                "n_eff": n_eff,

                "in_sla_ontime_pct":
                    pct_in_sla_ontime,

                "in_sla_late_pct":
                    pct_in_sla_late,

                "breach_ontime_pct":
                    pct_breach_ontime,

                "breach_late_pct":
                    pct_breach_late,

                "n_in_sla_late":
                    n_in_sla_late,

                "n_eff_in_sla_late":
                    n_eff_in_sla_late,

                "n_breach_ontime":
                    n_breach_ontime,

                "n_eff_breach_ontime":
                    n_eff_breach_ontime,

                "status":
                    "ĐỦ DỮ LIỆU"
                    if enough_data
                    else "⚠️ KHÔNG ĐỦ DỮ LIỆU",
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "n",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def main():
    print("=" * 78)
    print("C15 — PHÂN RÃ BẪY SLA THEO NGÀNH HÀNG")
    print("=" * 78)

    print(f"\nĐọc dữ liệu: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    print(
        f"Số dòng ban đầu: {len(df):,}"
    )

    result = category_stats(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        def emit(text=""):
            print(text)
            f.write(text + "\n")

        emit(
            "\nCổng đủ dữ liệu:"
            f" n_eff >= {MIN_N_EFF:.0f}"
        )

        emit(
            "Ngành chỉ được báo cáo đầy đủ khi "
            "CẢ HAI ô lệch đều đạt cổng n_eff."
        )

        emit()

        for _, row in result.iterrows():

            emit(
                f"[{row['status']}] "
                f"{row['category_name']}"
            )

            emit(
                f"  n = {int(row['n']):,}"
            )

            emit(
                f"  n_eff = {row['n_eff']:.2f}"
            )

            emit(
                f"  Trong SLA / khách đúng: "
                f"{row['in_sla_ontime_pct']:.4f}%"
            )

            emit(
                f"  Trong SLA / khách trễ:   "
                f"{row['in_sla_late_pct']:.4f}%"
            )

            emit(
                f"  Vượt SLA / khách đúng:   "
                f"{row['breach_ontime_pct']:.4f}%"
            )

            emit(
                f"  Vượt SLA / khách trễ:     "
                f"{row['breach_late_pct']:.4f}%"
            )

            emit(
                f"  Ô trong SLA / khách trễ:"
                f" n={int(row['n_in_sla_late'])}"
                f" · n_eff={row['n_eff_in_sla_late']:.2f}"
            )

            emit(
                f"  Ô vượt SLA / khách đúng:"
                f" n={int(row['n_breach_ontime'])}"
                f" · n_eff={row['n_eff_breach_ontime']:.2f}"
            )

            emit()

        emit("=" * 78)
        emit(
            f"Đã lưu evidence: {OUTPUT_PATH}"
        )
        emit("=" * 78)


if __name__ == "__main__":
    main()