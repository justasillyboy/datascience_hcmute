"""Sinh toàn bộ con số trong `docs/FINDINGS.md` — một lệnh, tái lập được.

Trước 2026-09-08 các kết quả được tính rời rạc rồi chép tay vào tài liệu. Cách
đó khiến không ai kiểm chứng lại được, và khi mẫu đổi thì tài liệu âm thầm sai.
Module này thay thế hoàn toàn cách làm đó.

Chạy:  python3 -m src.evaluate.run_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean.reviews import SLA_MAX_DAYS
from src.evaluate.stats import (
    Estimate,
    cluster_bootstrap,
    discriminative_power,
    effective_sample_size,
    weighted_mean,
    weighted_proportion,
    weighted_quantile,
)

N_BOOT = 2_000
SEED = 1


def _pct_in_sla(d: pd.DataFrame) -> float:
    """% (có trọng số) than phiền trễ hẹn đến từ đơn KHÔNG vượt SLA."""
    return weighted_proportion((~d["sla_breach"].astype(bool)).to_numpy(), d["weight"].to_numpy()) * 100


def _pct_says_ontime(d: pd.DataFrame) -> float:
    """% (có trọng số) đơn vượt SLA mà khách vẫn nói đúng hẹn."""
    return weighted_proportion((d["customer_says_late"] == False).to_numpy(), d["weight"].to_numpy()) * 100  # noqa: E712


def _power_gap(d: pd.DataFrame) -> float:
    """|Δrating theo nhãn khách| − |Δrating theo chỉ báo SLA|.

    Dương nghĩa là nhãn khách phân biệt mức hài lòng **tốt hơn** SLA.
    """
    return abs(discriminative_power(d, "customer_says_late")(d)) - abs(
        discriminative_power(d, "sla_breach")(d)
    )


def describe_sample(c: pd.DataFrame) -> None:
    a = c[c["is_analysable"]]
    print("## Quy mô mẫu\n")
    print(f"  review                 {len(c):,}")
    print(f"  sản phẩm               {c['product_id'].nunique():,}")
    print(f"  nhà bán                {c['seller_id'].nunique():,}")
    print(f"  ngành hàng             {c['category_id'].nunique()}")
    print(f"  quần thể ước lượng     {c['weight'].sum():,.0f}")
    print(f"  phân tích được         {len(a):,} ({len(a)/len(c)*100:.2f}%)")
    print("\n## Chất lượng\n")
    q = c["quality_flag"].value_counts()
    for k, v in q.items():
        print(f"  {k:24s} {v:8,d}  {v/len(c)*100:6.2f}%")


def describe_delivery(c: pd.DataFrame) -> None:
    a = c[c["is_analysable"]]
    v, w = a["lead_days"].to_numpy(), a["weight"].to_numpy()
    print("\n## Thời gian giao (có trọng số)\n")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  p{int(q*100):02d}  {weighted_quantile(v, w, q):7.2f} ngày")
    print(f"  trung bình {weighted_mean(v, w):.2f} · tối đa {v.max():.1f}")
    breach = weighted_proportion(a["sla_breach"].astype(bool).to_numpy(), w) * 100
    print(f"\n  Vượt SLA ({SLA_MAX_DAYS:.0f} ngày): {breach:.2f}% (có trọng số)")
    print("\n  Theo năm — mẫu trải 12 năm nên KHÔNG được gộp khi nói 'hiện tại':")
    a = a.assign(nam=a["review_ts"].dt.year)
    for y, g in a.groupby("nam"):
        if len(g) < 200:
            continue
        gb = weighted_proportion(g["sla_breach"].astype(bool).to_numpy(), g["weight"].to_numpy()) * 100
        print(f"    {int(y)}  n={len(g):6,d}  p90={weighted_quantile(g['lead_days'].to_numpy(), g['weight'].to_numpy(), .9):5.2f}  vượt SLA={gb:5.2f}%")


def key_estimates(c: pd.DataFrame) -> list[Estimate]:
    a = c[c["is_analysable"]]
    lab = a[a["customer_says_late"].notna()].copy()
    lab["rating"] = lab["rating"].astype(float)

    print(f"\n## Bảng chéo (n = {len(lab):,} review có nhãn)\n")
    ct = pd.crosstab(
        lab["sla_status"],
        lab["customer_says_late"].map({True: "Khách: trễ hẹn", False: "Khách: đúng hẹn"}),
    )
    print(ct.to_string())

    late = lab[lab["customer_says_late"] == True]  # noqa: E712
    breach = lab[lab["sla_breach"].astype(bool)]

    ests = [
        cluster_bootstrap(late, _pct_in_sla, n_boot=N_BOOT, seed=SEED,
                          name="% than phiền trễ hẹn đến từ đơn TRONG SLA"),
        cluster_bootstrap(breach, _pct_says_ontime, n_boot=N_BOOT, seed=SEED,
                          name="% đơn VƯỢT SLA mà khách nói đúng hẹn"),
        cluster_bootstrap(lab, _power_gap, n_boot=N_BOOT, seed=SEED,
                          name="Chênh lệch sức phân biệt (nhãn khách − SLA)"),
    ]
    print("\n## Ước lượng chính\n")
    for e in ests:
        print(f"  {e}")
    print(
        "\n  Ghi chú: ước lượng bị gắn ⚠️ MONG MANH có cỡ mẫu hiệu dụng thấp —\n"
        "  báo cáo dưới dạng khoảng, KHÔNG dùng làm kết luận chính."
    )
    return ests


def main() -> None:
    ap = argparse.ArgumentParser(description="Sinh kết quả cho FINDINGS.md")
    ap.add_argument("--data", type=Path, default=Path("data/processed/reviews_clean.parquet"))
    args = ap.parse_args()

    c = pd.read_parquet(args.data)
    print(f"# Kết quả phân tích — nguồn: {args.data}\n")
    describe_sample(c)
    describe_delivery(c)
    key_estimates(c)


if __name__ == "__main__":
    main()
