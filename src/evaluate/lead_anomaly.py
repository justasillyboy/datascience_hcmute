"""Điều tra `lead_days` âm — sinh mọi con số ở `docs/FINDINGS.md` §6.

Câu hỏi: 1.181 dòng có ngày giao **trước** ngày đặt. Chúng là gì, và loại chúng
ra khỏi phân tích có làm lệch kết luận không?

Module này không đưa ra ý kiến — nó chạy năm phép kiểm, mỗi phép kiểm có một dự
đoán rơi ra từ giả thuyết, để giả thuyết có thể **sai**:

* `clock_alignment` — hai đồng hồ của Tiki lệch đúng bao nhiêu? (kiểm chứng
  hằng số `VN_UTC_OFFSET`, đồng thời là chốt chặn hồi quy)
* `anomaly_profile`  — mốc nào trong ba mốc bị lệch chỗ?
* `age_gradient`     — tỉ lệ bất thường có tăng theo tuổi review không?
* `category_lift`    — có tập trung ở ngành hàng mua lặp không?
* `weighted_impact`  — loại chúng ra có cắt mất nhóm khách bất mãn không?

Chạy:  python3 -m src.evaluate.lead_anomaly
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean.reviews import VN_UTC_OFFSET
from src.evaluate.stats import weighted_mean, weighted_proportion

#: Ngày cào mẻ dữ liệu — `timeline.current_date` trong mọi response.
CRAWL_DATE = pd.Timestamp("2026-09-08")

AGE_BINS = [0, 1, 2, 3, 4, 5, 6, 100]
AGE_LABELS = ["<1 năm", "1–2", "2–3", "3–4", "4–5", "5–6", "6+"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm các cột phụ trợ cho điều tra. Không sửa bảng gốc."""
    out = df.copy()
    out["is_negative_lead"] = out["lead_days"] < 0
    out["purchase_after_review"] = out["purchased_ts"] > out["review_ts"]
    out["delivery_before_review"] = out["delivered_ts"] < out["review_ts"]
    out["review_age_years"] = (CRAWL_DATE - out["review_ts"]).dt.total_seconds() / 86400 / 365.25
    return out


def clock_alignment(df: pd.DataFrame) -> pd.Series:
    """Độ lệch giữa hai đồng hồ của Tiki, tính bằng giờ.

    `created_at` (epoch) và `review_created_date` (chuỗi) mô tả **cùng một sự
    kiện**, nên hiệu của chúng đo trực tiếp múi giờ của các chuỗi trong
    `timeline`. Đây là phép đo, không phải giả định.
    """
    sub = df.dropna(subset=["created_at", "review_created_date"])
    as_utc = pd.to_datetime(sub["created_at"], unit="s", errors="coerce")
    as_str = pd.to_datetime(sub["review_created_date"], errors="coerce")
    return (as_str - as_utc).dt.total_seconds() / 3600.0


def anomaly_profile(df: pd.DataFrame) -> dict[str, float]:
    """Mốc thời gian nào bị lệch chỗ ở các dòng âm?

    Nếu `delivery_date` sai thì quan hệ giao→review sẽ gãy. Nếu `purchased_at`
    sai thì quan hệ giao→review còn nguyên mà ngày đặt rơi ra ngoài.
    """
    neg = df[df["is_negative_lead"]]
    ok = df[df["quality_flag"] == "ok"]
    lag_neg = (neg["review_ts"] - neg["delivered_ts"]).dt.total_seconds() / 86400
    lag_ok = (ok["review_ts"] - ok["delivered_ts"]).dt.total_seconds() / 86400
    return {
        "n": len(neg),
        "purchase_after_review_pct": float(neg["purchase_after_review"].mean() * 100),
        "delivery_before_review_pct": float(neg["delivery_before_review"].mean() * 100),
        "purchase_after_review_pct_ok": float(ok["purchase_after_review"].mean() * 100),
        "review_lag_median_neg": float(lag_neg.median()),
        "review_lag_median_ok": float(lag_ok.median()),
        "staleness_median_days": float(
            ((neg["purchased_ts"] - neg["review_ts"]).dt.total_seconds() / 86400).median()
        ),
        "n_products": int(neg["product_id"].nunique()),
        "n_sellers": int(neg["seller_id"].nunique()),
    }


def age_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """Tỉ lệ bất thường theo tuổi review.

    Giả thuyết "`purchased_at` bị đơn mua lại ghi đè" dự đoán tỉ lệ **tăng theo
    tuổi**: review càng cũ thì càng nhiều thời gian để khách mua lại.
    """
    sub = df.dropna(subset=["review_age_years"]).copy()
    sub["nhóm tuổi"] = pd.cut(sub["review_age_years"], AGE_BINS, labels=AGE_LABELS)
    g = sub.groupby("nhóm tuổi", observed=True).agg(
        n=("is_negative_lead", "size"), bất_thường=("is_negative_lead", "sum")
    )
    g["tỉ lệ %"] = (g["bất_thường"] / g["n"] * 100).round(2)
    return g


def category_lift(df: pd.DataFrame) -> pd.DataFrame:
    """Bội số tỉ lệ bất thường theo ngành hàng so với mặt bằng chung."""
    base = df["is_negative_lead"].mean()
    g = df.groupby("category_name").agg(
        n=("is_negative_lead", "size"), bất_thường=("is_negative_lead", "sum")
    )
    g["tỉ lệ %"] = (g["bất_thường"] / g["n"] * 100).round(2)
    g["bội số"] = (g["bất_thường"] / g["n"] / base).round(2)
    return g.sort_values("bội số", ascending=False)


def weighted_impact(df: pd.DataFrame) -> dict[str, float]:
    """Loại các dòng này ra có cắt mất nhóm khách bất mãn không?

    **Phải dùng trọng số.** Mẫu lấy phân tầng theo sao, nên đếm thô sẽ phóng đại
    tỉ lệ sao thấp trong bất kỳ nhóm con nào — đúng cái bẫy đã làm hỏng hai ước
    lượng ở §5.
    """
    neg, rest = df[df["is_negative_lead"]], df[~df["is_negative_lead"]]

    def pack(d: pd.DataFrame) -> dict[str, float]:
        w = d["weight"].to_numpy(float)
        r = d["rating"].to_numpy(float)
        return {
            "rating_tb": weighted_mean(r, w),
            "pct_1sao": weighted_proportion((r == 1), w) * 100,
            "pct_thấp": weighted_proportion((r <= 3), w) * 100,
            "rating_tb_thô": float(r.mean()),
            "pct_1sao_thô": float((r == 1).mean() * 100),
        }

    a, b = pack(neg), pack(rest)
    return {
        "pct_quần_thể": float(neg["weight"].sum() / df["weight"].sum() * 100),
        **{f"neg_{k}": v for k, v in a.items()},
        **{f"rest_{k}": v for k, v in b.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Điều tra lead_days âm")
    ap.add_argument("--data", type=Path, default=Path("data/processed/reviews_clean.parquet"))
    args = ap.parse_args()
    d = prepare(pd.read_parquet(args.data))

    print(f"# Điều tra `lead_days` âm — nguồn: {args.data}\n")

    off = clock_alignment(d)
    print("## 1. Hai đồng hồ của Tiki lệch bao nhiêu?\n")
    print(f"  (review_created_date dạng chuỗi) − (created_at epoch đọc theo UTC)")
    print(f"  n = {len(off):,} · trung bình {off.mean():.4f} h · độ lệch chuẩn {off.std():.6f} h")
    print(f"  min {off.min():.4f} h · max {off.max():.4f} h")
    print(f"  → hằng số dùng trong code: VN_UTC_OFFSET = {VN_UTC_OFFSET.total_seconds()/3600:.0f} h")
    print("  Phương sai bằng 0 nghĩa là đây là quy ước múi giờ, không phải nhiễu.\n")

    p = anomaly_profile(d)
    print("## 2. Mốc nào bị lệch chỗ?\n")
    print(f"  số dòng âm                         {p['n']:,}")
    print(f"  ngày đặt SAU ngày viết review      {p['purchase_after_review_pct']:.1f}%"
          f"   (nhóm bình thường: {p['purchase_after_review_pct_ok']:.2f}%)")
    print(f"  ngày giao TRƯỚC ngày viết review   {p['delivery_before_review_pct']:.1f}%")
    print(f"  trễ viết review (ngày) — âm {p['review_lag_median_neg']:.2f} · bình thường {p['review_lag_median_ok']:.2f}")
    print(f"  ngày đặt muộn hơn review (trung vị) {p['staleness_median_days']:.1f} ngày")
    print(f"  trải trên {p['n_products']:,} sản phẩm · {p['n_sellers']:,} nhà bán")
    print("\n  → Quan hệ giao→review còn nguyên; chính `purchased_at` rơi ra ngoài.\n")

    g = age_gradient(d)
    print("## 3. Tỉ lệ bất thường theo tuổi review\n")
    print(g.to_string())
    rho = float(np.corrcoef(np.arange(len(g)), g["tỉ lệ %"].to_numpy())[0, 1])
    print(f"\n  tương quan hạng với tuổi: {rho:+.3f}"
          f"  ({g['tỉ lệ %'].iloc[0]:.2f}% → {g['tỉ lệ %'].iloc[-1]:.2f}%,"
          f" gấp {g['tỉ lệ %'].iloc[-1]/g['tỉ lệ %'].iloc[0]:.1f} lần)\n")

    print("## 4. Theo ngành hàng\n")
    print(category_lift(d).to_string())

    w = weighted_impact(d)
    print("\n\n## 5. Loại chúng ra có cắt mất khách bất mãn không?\n")
    print(f"  chiếm {w['pct_quần_thể']:.2f}% quần thể (có trọng số)\n")
    print(f"  {'':22s} {'dòng âm':>10s} {'còn lại':>10s}")
    print(f"  {'rating TB (trọng số)':22s} {w['neg_rating_tb']:>10.3f} {w['rest_rating_tb']:>10.3f}")
    print(f"  {'% 1 sao (trọng số)':22s} {w['neg_pct_1sao']:>10.2f} {w['rest_pct_1sao']:>10.2f}")
    print(f"  {'% ≤3 sao (trọng số)':22s} {w['neg_pct_thấp']:>10.2f} {w['rest_pct_thấp']:>10.2f}")
    print(f"\n  Đếm THÔ (sai — mẫu phân tầng, chỉ để đối chiếu):")
    print(f"  {'rating TB thô':22s} {w['neg_rating_tb_thô']:>10.3f} {w['rest_rating_tb_thô']:>10.3f}")
    print(f"  {'% 1 sao thô':22s} {w['neg_pct_1sao_thô']:>10.2f} {w['rest_pct_1sao_thô']:>10.2f}")


if __name__ == "__main__":
    main()
