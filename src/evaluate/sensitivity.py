from pathlib import Path
import sys

import pandas as pd

from src.clean.reviews import compute_gap, SLA_MAX_DAYS
from src.evaluate.stats import (
    weighted_mean,
    weighted_proportion,
    effective_sample_size,
)


DATA_PATH = Path("data/processed/reviews_clean.parquet")
LOG_PATH = Path("docs/evidence/a6_sla_sensitivity.txt")

SLA_VALUES = [3, 4, 5, 6, 7]


def pct_sla_breach(df: pd.DataFrame) -> float:
    d = df[df["is_analysable"]].copy()

    return (
        weighted_proportion(
            d["sla_breach"].astype(bool).to_numpy(),
            d["weight"].to_numpy(),
        )
        * 100
    )


def pct_customer_says_late(df: pd.DataFrame) -> float:
    d = df[
        df["is_analysable"]
        & df["customer_says_late"].notna()
    ].copy()

    return (
        weighted_proportion(
            d["customer_says_late"].astype(bool).to_numpy(),
            d["weight"].to_numpy(),
        )
        * 100
    )


def rating_gap_customer_vs_sla(df: pd.DataFrame) -> float:
    d = df[
        df["is_analysable"]
        & df["customer_says_late"].notna()
        & df["rating"].notna()
    ].copy()

    d["rating"] = d["rating"].astype(float)

    def group_gap(indicator):
        true_mask = indicator.astype(bool).to_numpy()
        false_mask = ~true_mask

        mean_true = weighted_mean(
            d.loc[true_mask, "rating"].to_numpy(),
            d.loc[true_mask, "weight"].to_numpy(),
        )

        mean_false = weighted_mean(
            d.loc[false_mask, "rating"].to_numpy(),
            d.loc[false_mask, "weight"].to_numpy(),
        )

        return abs(mean_true - mean_false)

    customer_gap = group_gap(d["customer_says_late"])
    sla_gap = group_gap(d["sla_breach"])

    return customer_gap - sla_gap


def run_sensitivity():

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    with open(LOG_PATH, "w", encoding="utf-8") as log_file:

        original_stdout = sys.stdout
        sys.stdout = Tee(
            original_stdout,
            log_file,
        )

        try:

            print("=" * 70)
            print("A6 - SLA THRESHOLD SENSITIVITY")
            print("=" * 70)

            print(
                f"Đọc dữ liệu: {DATA_PATH}"
            )

            df = pd.read_parquet(DATA_PATH)

            print(
                f"Số dòng dữ liệu: {len(df):,}"
            )

            print(
                f"SLA mặc định: {SLA_MAX_DAYS} ngày"
            )

            print()

            results = []

            for sla_days in SLA_VALUES:

                print(
                    f"--- SLA = {sla_days} ngày ---"
                )

                d = compute_gap(
                    df,
                    sla_days=sla_days,
                )

                analysable = d[
                    d["is_analysable"]
                ].copy()

                labelled = d[
                    d["is_analysable"]
                    & d["customer_says_late"].notna()
                ].copy()

                n = len(analysable)

                n_eff = effective_sample_size(
                    analysable["weight"].to_numpy()
                )

                breach_pct = pct_sla_breach(d)

                customer_late_pct = (
                    pct_customer_says_late(d)
                )

                rating_gap = (
                    rating_gap_customer_vs_sla(d)
                )

                results.append(
                    {
                        "sla_days": sla_days,
                        "sla_breach_pct": breach_pct,
                        "customer_says_late_pct":
                            customer_late_pct,
                        "rating_gap":
                            rating_gap,
                        "n": n,
                        "n_labelled":
                            len(labelled),
                        "n_eff": n_eff,
                    }
                )

                print(
                    f"  SLA breach:         "
                    f"{breach_pct:.4f}%"
                )

                print(
                    f"  Customer says late: "
                    f"{customer_late_pct:.4f}%"
                )

                print(
                    f"  Rating gap:         "
                    f"{rating_gap:.4f}"
                )

                print(
                    f"  n:                  "
                    f"{n:,}"
                )

                print(
                    f"  n_labelled:         "
                    f"{len(labelled):,}"
                )

                print(
                    f"  n_eff:              "
                    f"{n_eff:.2f}"
                )

                print()

            print("=" * 70)
            print("KẾT QUẢ A6")
            print("=" * 70)

            result_df = pd.DataFrame(results)

            print(
                result_df.to_string(
                    index=False,
                    float_format=lambda x:
                        f"{x:.4f}",
                )
            )

            print()
            print(
                f"Đã lưu evidence: {LOG_PATH}"
            )

        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    run_sensitivity()