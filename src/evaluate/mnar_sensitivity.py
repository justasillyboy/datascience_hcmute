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
import re
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
# ============================================================
# A2 — THETA SENSITIVITY
# ============================================================

THETA_STEP = 0.001


def _weighted_mean(values, weights):
    """Weighted mean, dùng cho cả trọng số nguyên và phân đoạn."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    mask = np.isfinite(v) & np.isfinite(w) & (w >= 0)

    if not np.any(mask):
        raise ValueError("Không có dữ liệu hợp lệ để tính weighted mean.")

    v = v[mask]
    w = w[mask]

    total_w = w.sum()

    if total_w <= 0:
        raise ValueError("Tổng trọng số phải > 0.")

    return float(np.dot(v, w) / total_w)


def fractional_weights(weights, p):
    """
    Trọng số phân đoạn cho một dòng:

        late_weight    = w * p
        ontime_weight  = w * (1-p)

    Hai phần phải cộng lại đúng bằng w.
    """
    w = np.asarray(weights, dtype=float)
    p = np.asarray(p, dtype=float)

    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights phải hữu hạn và >= 0.")

    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p phải nằm trong [0, 1].")

    late_weight = w * p
    ontime_weight = w * (1.0 - p)

    return late_weight, ontime_weight


def _prepare_theta_components(d):
    """
    Tính các thành phần cố định cho A2:

    - p_bar: tỷ lệ khách nói trễ trong toàn nhóm có nhãn
    - p_MAR(rating, sla_breach): tỷ lệ khách nói trễ trong từng ô
    """
    required = {
        "rating",
        "sla_breach",
        "customer_says_late",
        "weight",
    }

    missing = required - set(d.columns)

    if missing:
        raise KeyError(
            f"Thiếu cột bắt buộc cho A2: {sorted(missing)}"
        )

    labelled = d[d["customer_says_late"].notna()].copy()

    if labelled.empty:
        raise ValueError(
            "A2 không có review có nhãn để ước lượng p_MAR."
        )

    labelled["rating"] = labelled["rating"].astype(float)
    labelled["sla_breach"] = labelled["sla_breach"].astype(bool)

    y = labelled["customer_says_late"].astype(bool).astype(float).to_numpy()
    w = labelled["weight"].to_numpy(float)

    p_bar = _weighted_mean(y, w)

    p_mar = {}

    for key, g in labelled.groupby(
        ["rating", "sla_breach"],
        dropna=False,
        observed=True,
    ):
        gw = g["weight"].to_numpy(float)
        gy = (
            g["customer_says_late"]
            .astype(bool)
            .astype(float)
            .to_numpy()
        )

        if gw.sum() <= 0:
            raise ValueError(
                f"Ô {key} có tổng trọng số <= 0."
            )

        p_mar[(float(key[0]), bool(key[1]))] = _weighted_mean(
            gy,
            gw,
        )

    # Theo đặc tả A2, p_MAR phải được ước lượng riêng từng ô.
    # Không tự động lấp ô thiếu nhãn bằng p_bar.
    expected_cells = {
        (float(r), bool(b))
        for r in sorted(d["rating"].dropna().unique())
        for b in [False, True]
    }

    missing_cells = sorted(
        expected_cells - set(p_mar.keys())
    )

    if missing_cells:
        raise ValueError(
            "Không thể tính p_MAR cho các ô không có nhãn: "
            f"{missing_cells}. "
            "Không tự ý thay bằng p_bar vì như vậy đã thêm giả định."
        )

    return p_bar, p_mar


def _build_theta_probabilities(
    d,
    theta,
    *,
    hard_labels=False,
    p_bar=None,
    p_mar=None,
):
    """
    Tạo xác suất 'khách nói trễ' cho từng dòng.

    Bình thường:
        - dòng có nhãn  -> nhãn thật
        - dòng không nhãn -> p_i(theta)

    A3:
        hard_labels=True
        -> ép toàn bộ p bằng nhãn thật.
    """
    theta = float(theta)

    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta phải nằm trong [0, 1].")

    labelled_mask = d["customer_says_late"].notna().to_numpy()

    if hard_labels:
        if not labelled_mask.all():
            raise ValueError(
                "hard_labels=True nhưng d vẫn chứa dòng không nhãn."
            )

        return (
            d["customer_says_late"]
            .astype(bool)
            .astype(float)
            .to_numpy()
        )

    if p_bar is None or p_mar is None:
        p_bar, p_mar = _prepare_theta_components(d)

    rating = d["rating"].astype(float).to_numpy()
    sla = d["sla_breach"].astype(bool).to_numpy()

    p = np.empty(len(d), dtype=float)

    # Nhãn thật giữ nguyên.
    p[labelled_mask] = (
        d.loc[labelled_mask, "customer_says_late"]
        .astype(bool)
        .astype(float)
        .to_numpy()
    )

    # Nhóm không nhãn dùng p_i(theta).
    unlabelled_mask = ~labelled_mask

    if np.any(unlabelled_mask):
        cell_p = np.array(
            [
                p_mar[(float(r), bool(s))]
                for r, s in zip(
                    rating[unlabelled_mask],
                    sla[unlabelled_mask],
                )
            ],
            dtype=float,
        )

        p[unlabelled_mask] = (
            theta * cell_p
            + (1.0 - theta) * p_bar
        )

    return p


def theta_estimate(
    d,
    theta,
    *,
    hard_labels=False,
    p_bar=None,
    p_mar=None,
):
    """
    Tính kết luận chính tại một giá trị theta.

    Không bootstrap.
    Trọng số được chia thành:
        w*p
        w*(1-p)
    """
    work = d.copy()

    work["rating"] = work["rating"].astype(float)
    work["sla_breach"] = work["sla_breach"].astype(bool)

    if not hard_labels and (p_bar is None or p_mar is None):
        p_bar, p_mar = _prepare_theta_components(work)

    p = _build_theta_probabilities(
        work,
        theta,
        hard_labels=hard_labels,
        p_bar=p_bar,
        p_mar=p_mar,
    )

    ratings = work["rating"].to_numpy(float)
    weights = work["weight"].to_numpy(float)
    sla = work["sla_breach"].to_numpy(bool)

    # --------------------------------------------------------
    # 1. Customer label — trọng số phân đoạn
    # --------------------------------------------------------

    customer_late_w, customer_ontime_w = fractional_weights(
        weights,
        p,
    )

    customer_late_mean = _weighted_mean(
        ratings,
        customer_late_w,
    )

    customer_ontime_mean = _weighted_mean(
        ratings,
        customer_ontime_w,
    )

    customer_power = abs(
        customer_late_mean
        - customer_ontime_mean
    )

    # --------------------------------------------------------
    # 2. SLA — hoàn toàn quan sát được
    # --------------------------------------------------------

    sla_breach_w = weights * sla.astype(float)
    sla_ontime_w = weights * (~sla).astype(float)

    sla_breach_mean = _weighted_mean(
        ratings,
        sla_breach_w,
    )

    sla_ontime_mean = _weighted_mean(
        ratings,
        sla_ontime_w,
    )

    sla_power = abs(
        sla_breach_mean
        - sla_ontime_mean
    )

    # --------------------------------------------------------
    # 3. Kết luận chính
    # --------------------------------------------------------

    return float(customer_power - sla_power)


def theta_scan(
    d,
    *,
    theta_step=THETA_STEP,
    hard_labels=False,
):
    """
    Quét theta từ 0 -> 1.

    Trả về DataFrame:
        theta
        estimate
    """
    if theta_step <= 0 or theta_step > 1:
        raise ValueError("theta_step phải thuộc (0, 1].")

    p_bar = None
    p_mar = None

    if not hard_labels:
        p_bar, p_mar = _prepare_theta_components(d)

    thetas = np.arange(
        0.0,
        1.0 + theta_step / 2.0,
        theta_step,
    )

    rows = []

    for theta in thetas:
        rows.append(
            {
                "theta": float(theta),
                "estimate": theta_estimate(
                    d,
                    theta,
                    hard_labels=hard_labels,
                    p_bar=p_bar,
                    p_mar=p_mar,
                ),
            }
        )

    return pd.DataFrame(rows)


def find_theta_star(scan_df):
    """
    Tìm theta* nơi estimate chạm 0.

    Nếu có một khoảng đổi dấu:
        nội suy tuyến tính trong chính khoảng đó.

    Không dùng simulation.
    """
    x = scan_df["theta"].to_numpy(float)
    y = scan_df["estimate"].to_numpy(float)

    roots = []

    for i in range(len(scan_df)):
        if np.isclose(y[i], 0.0, atol=1e-12):
            roots.append(float(x[i]))

    for i in range(len(scan_df) - 1):
        y1 = y[i]
        y2 = y[i + 1]

        if y1 == 0 or y2 == 0:
            continue

        if y1 * y2 < 0:
            x1 = x[i]
            x2 = x[i + 1]

            # Nội suy tuyến tính trong khoảng đổi dấu.
            root = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)

            roots.append(float(root))

    roots = sorted(set(round(r, 10) for r in roots))

    if not roots:
        return None

    if len(roots) > 1:
        raise ValueError(
            f"Có nhiều điểm gãy theta*: {roots}. "
            "Không tự chọn một nghiệm."
        )

    return roots[0]


# ============================================================
# A3 — ĐỌC GIÁ TRỊ HIỆN HÀNH TỪ FINDINGS.MD
# ============================================================

def _find_findings_path():
    candidates = [
        Path("docs/FINDINGS.md"),
        Path("FINDINGS.md"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Không tìm thấy docs/FINDINGS.md hoặc FINDINGS.md."
    )


def read_current_main_estimate(findings_path=None):
    """
    Đọc estimate hiện hành từ §4 của FINDINGS.md.

    Không hard-code 0.2082.
    """
    path = findings_path or _find_findings_path()

    text = path.read_text(
        encoding="utf-8",
    )

    match_section = re.search(
        r"(?ms)^##\s*4\.\s*Kết luận chính\b"
        r"(.*?)(?=^##\s|\Z)",
        text,
    )

    if not match_section:
        raise ValueError(
            f"Không tìm thấy §4 trong {path}."
        )

    section = match_section.group(1)

    match_value = re.search(
        r"chênh lệch\s+"
        r"([+-]?\d+(?:[.,]\d+)?)"
        r"\s*điểm\s+rating",
        section,
        flags=re.IGNORECASE,
    )

    if not match_value:
        raise ValueError(
            f"Không tìm được estimate trong §4 của {path}."
        )

    value = match_value.group(1).replace(",", ".")

    return float(value)


def a3_self_check(
    d,
    findings_path=None,
    *,
    tolerance=5e-5,
):
    """
    A3:

    - chỉ dùng nhóm có nhãn
    - ép p bằng đúng nhãn thật
    - chạy chính theta_scan()
    - so với FINDINGS.md hiện hành
    """
    labelled = d[d["customer_says_late"].notna()].copy()

    if labelled.empty:
        raise ValueError(
            "A3 không có dữ liệu có nhãn."
        )

    target = read_current_main_estimate(
        findings_path
    )

    scan = theta_scan(
        labelled,
        theta_step=THETA_STEP,
        hard_labels=True,
    )

    observed = float(scan.iloc[0]["estimate"])

    if not np.isclose(
        observed,
        target,
        atol=tolerance,
        rtol=0.0,
    ):
        raise AssertionError(
            "A3 FAIL:\n"
            f"  FINDINGS.md = {target:+.6f}\n"
            f"  theta_scan  = {observed:+.6f}\n"
            f"  sai khác    = {observed - target:+.6f}"
        )

    return {
        "target": target,
        "observed": observed,
        "difference": observed - target,
        "passed": True,
        "scan": scan,
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
        # ========================================================
    # A2 — QUÉT THETA
    # ========================================================

    print("\n" + "=" * 70)
    print("A2 — QUÉT ĐIỂM GÃY THETA")
    print("=" * 70)

    data = pd.read_parquet(args.data)

    data = prepare(data)

    # Chỉ chạy trong kỷ nguyên 2023+
    era = data[data["nam"] >= LABEL_ERA_START].copy()

    print(
        f"\nKỷ nguyên: {LABEL_ERA_START}+"
    )
    print(
        f"n phân tích được: {len(era):,}"
    )

    labelled = era[
        era["customer_says_late"].notna()
    ]

    print(
        f"n có nhãn: {len(labelled):,}"
    )

    print(
        f"n không nhãn: "
        f"{len(era) - len(labelled):,}"
    )

    p_bar, p_mar = _prepare_theta_components(era)

    print(
        f"\np̄ = {p_bar:.6f}"
    )

    print("\np_MAR theo từng ô (rating × sla_breach):")

    for key in sorted(p_mar):
        rating, breach = key

        print(
            f"  rating={rating:.0f}, "
            f"sla_breach={int(breach)}"
            f" -> {p_mar[key]:.6f}"
        )

    scan = theta_scan(
        era,
        theta_step=THETA_STEP,
    )

    theta_star = find_theta_star(scan)

    print("\nĐiểm đầu/cuối:")
    print(
        f"  θ = 0.000 -> "
        f"{scan.iloc[0]['estimate']:+.6f}"
    )
    print(
        f"  θ = 1.000 -> "
        f"{scan.iloc[-1]['estimate']:+.6f}"
    )

    if theta_star is None:
        print(
            "\nKhông tìm thấy θ* trong [0, 1]."
        )
    else:
        loss = (1.0 - theta_star) * 100.0

        print(
            f"\nθ* = {theta_star:.6f}"
        )

        print(
            "Diễn giải:"
        )

        print(
            f'  Kết luận hiện hành chỉ sụp khi '
            f'nhãn khách còn khoảng '
            f'{theta_star * 100:.2f}% '
            f'thông tin so với MAR.'
        )

        print(
            f'  Tương đương phải kém hơn khoảng '
            f'{loss:.2f}% so với giả định MAR.'
        )

    # ========================================================
    # A3 — SELF CHECK
    # ========================================================

    print("\n" + "=" * 70)
    print("A3 — SELF CHECK")
    print("=" * 70)

    check = a3_self_check(
        labelled,
        findings_path=_find_findings_path(),
    )

    print(
        f"FINDINGS.md : {check['target']:+.6f}"
    )

    print(
        f"theta_scan  : {check['observed']:+.6f}"
    )

    print(
        f"sai khác    : "
        f"{check['difference']:+.6f}"
    )

    print("\n✅ A3 PASS — theta_scan tái lập kết luận chính.")

if __name__ == "__main__":
    main()
