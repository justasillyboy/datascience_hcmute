"""Độ nhạy MNAR cho nhãn `delivery_rating` — **ĐANG LÀM DỞ**.

Trạng thái 2026-09-09: mới xong **phần mô tả cơ chế thiếu nhãn** (§A và §B dưới).
Phần chặn Manski và quét điểm gãy θ chưa viết — xem `CLAUDE.md` §9.1 để biết kế
hoạch và ba phương án đã cân nhắc.

Phát hiện đã có, và nó đủ để sửa lại cách tài liệu mô tả MNAR:

**Thiếu nhãn ở đây gồm hai cơ chế khác hẳn nhau, tài liệu cũ gộp làm một.**

1. *Thiếu do thiết kế* — trường `delivery_rating` chưa tồn tại trước 2023, độ phủ
   đúng bằng 0 ở mọi năm 2017–2022. Đây không phải chọn lọc, và **không chặn được
   bằng bất kỳ giả định nào**: 62% quần thể chưa bao giờ có cơ hội mang nhãn.
2. *Chọn lọc thật* — trong kỷ nguyên 2023+, 29,5% (có trọng số) có nhãn.

Gộp hai cơ chế lại làm nhóm có nhãn **trông** khác nhóm không nhãn xa hơn thực tế,
vì nhóm không nhãn bị trộn thêm toàn bộ giai đoạn giao hàng kém trước 2023.

Chạy:  python3 -m src.evaluate.mnar_sensitivity
"""

from __future__ import annotations

import argparse
from pathlib import Path
from src.evaluate.stats import effective_sample_size
import numpy as np
import pandas as pd

#: Năm đầu tiên `delivery_rating` tồn tại. Trước mốc này độ phủ đúng bằng 0 —
#: đo được, không phải giả định.
LABEL_ERA_START = 2023

MIN_YEAR_ROWS = 200


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["is_analysable"]].copy()
    out["labelled"] = out["customer_says_late"].notna()
    out["nam"] = out["review_ts"].dt.year
    return out


def _w(d: pd.DataFrame, col: str) -> float:
    w = d["weight"].to_numpy(float)
    return float(np.dot(d[col].to_numpy(float), w) / w.sum())


def _wpct(d: pd.DataFrame, mask: np.ndarray) -> float:
    w = d["weight"].to_numpy(float)
    return float(w[mask].sum() / w.sum() * 100)


def coverage_by_year(d: pd.DataFrame) -> pd.DataFrame:
    """Độ phủ nhãn theo năm — cho thấy mốc 2023 là ranh giới cứng."""
    t = d.groupby("nam").agg(n=("labelled", "size"), có_nhãn=("labelled", "sum"))
    t["% có nhãn"] = (t["có_nhãn"] / t["n"] * 100).round(2)
    return t[t["n"] >= MIN_YEAR_ROWS]


def profile(d: pd.DataFrame) -> dict[str, float]:
    """Chân dung một nhóm trên các biến **quan sát được** (không cần nhãn)."""
    return {
        "n": len(d),
        "rating": _w(d, "rating"),
        "lead_tb": _w(d, "lead_days"),
        "vượt_sla_%": _wpct(d, d["sla_breach"].astype(bool).to_numpy()),
    }


def selection_decomposition(d: pd.DataFrame) -> dict[str, dict]:
    """Tách chênh lệch có-nhãn/không-nhãn thành *thời kỳ* và *chọn lọc thật*.

    So gộp toàn mẫu thì nhóm không nhãn gánh thêm cả giai đoạn trước 2023 —
    giai đoạn giao hàng kém hơn hẳn. So **trong cùng kỷ nguyên** mới là chọn lọc.
    """
    era = d[d["nam"] >= LABEL_ERA_START]
    return {
        "gộp_có_nhãn": profile(d[d["labelled"]]),
        "gộp_không_nhãn": profile(d[~d["labelled"]]),
        "kỷ_nguyên_có_nhãn": profile(era[era["labelled"]]),
        "kỷ_nguyên_không_nhãn": profile(era[~era["labelled"]]),
        "trước_kỷ_nguyên": profile(d[d["nam"] < LABEL_ERA_START]),
    }

def manski_bounds(d: pd.DataFrame) -> dict[str, float]:
    """
    Manski bounds cho tỷ lệ customer_says_late trong kỷ nguyên 2023+.

    Chỉ sử dụng các dòng is_analysable thuộc kỷ nguyên nhãn tồn tại.

    Lower bound:
        toàn bộ review không nhãn được giả định là không trễ.

    Upper bound:
        toàn bộ review không nhãn được giả định là trễ.

    Mọi ước lượng đều dùng trọng số phân tầng.
    """
    era = d[
        (d["nam"] >= LABEL_ERA_START)
        & d["is_analysable"]
    ].copy()

    labelled = era[era["labelled"]].copy()
    unlabelled = era[~era["labelled"]].copy()

    w_labelled = labelled["weight"].to_numpy(float)
    w_unlabelled = unlabelled["weight"].to_numpy(float)

    y = labelled["customer_says_late"].astype(bool).to_numpy()

    total_weight = (
        w_labelled.sum()
        + w_unlabelled.sum()
    )

    late_weight = w_labelled[y].sum()

    # Cực dưới:
    # toàn bộ phần thiếu nhãn = không trễ.
    lower = (
        late_weight
        / total_weight
        * 100
    )

    # Cực trên:
    # toàn bộ phần thiếu nhãn = trễ.
    upper = (
        (late_weight + w_unlabelled.sum())
        / total_weight
        * 100
    )

    n_eff = effective_sample_size(
        era["weight"].to_numpy(float)
    )

    return {
        "n_era": len(era),
        "n_labelled": len(labelled),
        "n_unlabelled": len(unlabelled),
        "weighted_labelled_pct": (
            w_labelled.sum()
            / total_weight
            * 100
        ),
        "lower_pct": lower,
        "upper_pct": upper,
        "n_eff": n_eff,
    }
def main() -> None:
    ap = argparse.ArgumentParser(description="Độ nhạy MNAR (đang làm dở)")
    ap.add_argument("--data", type=Path, default=Path("data/processed/reviews_clean.parquet"))
    args = ap.parse_args()
    d = prepare(pd.read_parquet(args.data))
    era = d[d["nam"] >= LABEL_ERA_START]

    print(f"# Độ nhạy MNAR — nguồn: {args.data}\n")
    print("## A. Độ phủ nhãn\n")
    print(f"  toàn mẫu phân tích được  thô {d['labelled'].mean()*100:.2f}%"
          f"   trọng số {_wpct(d, d['labelled'].to_numpy()):.2f}%")
    print(f"  kỷ nguyên {LABEL_ERA_START}+ ({len(era):,} dòng)  thô "
          f"{era['labelled'].mean()*100:.2f}%   trọng số "
          f"{_wpct(era, era['labelled'].to_numpy()):.2f}%\n")
    print(coverage_by_year(d).to_string())
    print(f"\n  → Trước {LABEL_ERA_START} độ phủ đúng bằng 0: trường chưa tồn tại.")
    print("    Đây là thiếu DO THIẾT KẾ, không phải chọn lọc — và không chặn được")
    print("    bằng giả định, vì nhóm đó chưa bao giờ có cơ hội mang nhãn.\n")

    print("## B. Chọn lọc thật nhỏ hơn vẻ ngoài\n")
    dec = selection_decomposition(d)
    hdr = f"  {'nhóm':26s} {'n':>8s} {'rating':>8s} {'lead TB':>9s} {'vượt SLA':>10s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for k in ("gộp_có_nhãn", "gộp_không_nhãn", "kỷ_nguyên_có_nhãn",
              "kỷ_nguyên_không_nhãn", "trước_kỷ_nguyên"):
        p = dec[k]
        print(f"  {k.replace('_',' '):26s} {p['n']:8,d} {p['rating']:8.3f} "
              f"{p['lead_tb']:9.2f} {p['vượt_sla_%']:9.2f}%")
    g_l, g_u = dec["gộp_có_nhãn"], dec["gộp_không_nhãn"]
    e_l, e_u = dec["kỷ_nguyên_có_nhãn"], dec["kỷ_nguyên_không_nhãn"]
    print(f"\n  Bội số 'vượt SLA' không-nhãn / có-nhãn:")
    print(f"    so gộp             {g_u['vượt_sla_%']/g_l['vượt_sla_%']:.1f}×  ← phần lớn là hiệu ứng thời kỳ")
    print(f"    so trong kỷ nguyên {e_u['vượt_sla_%']/e_l['vượt_sla_%']:.1f}×  ← chọn lọc thật")
    print(f"\n  Chênh rating: gộp {g_l['rating']-g_u['rating']:+.3f} → "
          f"trong kỷ nguyên {e_l['rating']-e_u['rating']:+.3f}")

    print("\n\n## C. Chặn Manski — A1\n")
    bounds = manski_bounds(d)

    print(
        f"  Kỷ nguyên: {LABEL_ERA_START}+"
    )

    print(
        f"  n phân tích được: "
        f"{bounds['n_era']:,}"
    )

    print(
        f"  n có nhãn: "
        f"{bounds['n_labelled']:,}"
    )

    print(
        f"  n không nhãn: "
        f"{bounds['n_unlabelled']:,}"
    )

    print(
        f"  Tỷ trọng có nhãn: "
        f"{bounds['weighted_labelled_pct']:.2f}%"
    )

    print()

    print(
        "  Manski lower bound: "
        f"{bounds['lower_pct']:.2f}%"
    )

    print(
        "  Manski upper bound: "
        f"{bounds['upper_pct']:.2f}%"
    )

    print(
        f"  n_eff: "
        f"{bounds['n_eff']:.2f}"
    )

    print()

    print(
        "  Diễn giải:"
    )

    print(
        "    Lower = mọi review không nhãn "
        "đều được giả định là không trễ."
    )

    print(
        "    Upper = mọi review không nhãn "
        "đều được giả định là trễ."
    )


if __name__ == "__main__":
    main()
