"""Data contract — bộ kiểm tra tự động chạy sau mỗi bước xử lý.

Đây là tầng 1 của khung "chứng minh đúng" (CLAUDE.md §7). Ý tưởng: mọi giả định
về dữ liệu phải được viết ra thành một phép kiểm chạy được, để khi Tiki đổi API
hoặc khi ai đó sửa nhầm bước cleaning thì pipeline **gãy ngay và ồn ào**, chứ
không âm thầm cho ra số sai.

    from src.validate.contract import validate_clean_reviews
    validate_clean_reviews(df).raise_if_failed()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Ngưỡng độ phủ tối thiểu, lấy từ số đo thực tế trong docs/FEASIBILITY.md §2
# (100% / 98.0% / 98.0%), hạ xuống một biên an toàn để không báo động giả khi
# mẻ cào khác chệch vài phần trăm.
MIN_COVERAGE = {
    "purchased_at": 0.95,
    "delivery_date": 0.90,
    "review_created_date": 0.90,
}
#: `delivery_rating` đo được 18–27% tuỳ mẻ; dưới 5% là dấu hiệu API đã đổi.
MIN_COVERAGE_DELIVERY_RATING = 0.05

REQUIRED_COLUMNS = [
    "review_id", "product_id", "seller_id", "customer_id", "rating",
    "purchased_at", "delivery_date", "review_created_date",
    "stratum", "stratum_size", "stratum_sampled", "weight",
]


class ContractViolation(AssertionError):
    """Dữ liệu vi phạm hợp đồng — dừng pipeline."""


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ContractReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name, passed, detail))

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    def raise_if_failed(self) -> "ContractReport":
        if self.failures:
            lines = "\n".join(f"  ✗ {r.name}: {r.detail}" for r in self.failures)
            raise ContractViolation(
                f"{len(self.failures)}/{len(self.results)} phép kiểm thất bại:\n{lines}"
            )
        return self

    def __str__(self) -> str:
        lines = [f"Data contract: {len(self.results) - len(self.failures)}/{len(self.results)} đạt"]
        for r in self.results:
            mark = "✓" if r.passed else "✗"
            lines.append(f"  {mark} {r.name}" + (f" — {r.detail}" if r.detail else ""))
        return "\n".join(lines)


def validate_raw_reviews(df: pd.DataFrame) -> ContractReport:
    """Kiểm tra bảng review **thô** ngay sau bước cào."""
    rep = ContractReport()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    rep.add("đủ cột bắt buộc", not missing, f"thiếu {missing}" if missing else "")
    if missing:
        return rep

    rep.add("bảng không rỗng", len(df) > 0, f"n={len(df)}")

    # Trùng `review_id` ở bảng THÔ là bình thường và đã được dự liệu: một sản
    # phẩm có thể nằm ở hai ngành hàng, và phân trang chồng lấn khi có review
    # mới trong lúc cào (xem `src.clean.reviews.deduplicate`). Cái *không* bình
    # thường là hai bản sao mang nội dung khác nhau — đó mới là hỏng dữ liệu.
    # Vì vậy tầng thô kiểm "bản sao phải trùng khớp", còn ràng buộc khoá duy
    # nhất được kiểm ở tầng đã làm sạch, sau bước khử trùng lặp.
    dup_ids = int(df["review_id"].duplicated().sum())
    key_cols = [c for c in ("review_id", "rating", "purchased_at", "delivery_date",
                            "review_created_date") if c in df.columns]
    conflicting = int(
        df.duplicated(subset=["review_id"], keep=False).sum()
        - df.duplicated(subset=key_cols, keep=False).sum()
    )
    rep.add(
        "bản sao review_id phải trùng khớp nội dung",
        conflicting == 0,
        f"{dup_ids} bản sao, {conflicting} bản sao mâu thuẫn",
    )

    bad_rating = int((~df["rating"].isin([1, 2, 3, 4, 5])).sum())
    rep.add("rating ∈ {1..5}", bad_rating == 0, f"{bad_rating} dòng ngoài khoảng")

    for col, threshold in MIN_COVERAGE.items():
        cov = float(df[col].notna().mean())
        rep.add(f"độ phủ {col} ≥ {threshold:.0%}", cov >= threshold, f"thực tế {cov:.1%}")

    if "dr_thoi_gian" in df.columns:
        cov = float(df["dr_thoi_gian"].notna().mean())
        rep.add(
            f"độ phủ nhãn giao hàng ≥ {MIN_COVERAGE_DELIVERY_RATING:.0%}",
            cov >= MIN_COVERAGE_DELIVERY_RATING,
            f"thực tế {cov:.1%}",
        )

    # --- tính toàn vẹn của thiết kế lấy mẫu phân tầng ---
    rep.add(
        "stratum khớp rating",
        bool((df["stratum"] == df["rating"]).all()),
        f"{int((df['stratum'] != df['rating']).sum())} dòng lệch",
    )
    over = int((df["stratum_sampled"] > df["stratum_size"]).sum())
    rep.add("số lấy mẫu ≤ kích thước tầng", over == 0, f"{over} dòng vi phạm")

    bad_w = int(((df["weight"] < 1) | df["weight"].isna()).sum())
    rep.add("trọng số ≥ 1 và không rỗng", bad_w == 0, f"{bad_w} dòng sai")

    return rep


def validate_clean_reviews(df: pd.DataFrame, sla_days: float = 5.0) -> ContractReport:
    """Kiểm tra bảng review **đã làm sạch**."""
    rep = validate_raw_reviews(df)

    for col in ["lead_days", "gap_days", "quality_flag", "is_analysable", "sla_breach"]:
        rep.add(f"có cột {col}", col in df.columns)
    if not all(c in df.columns for c in ["lead_days", "gap_days", "is_analysable"]):
        return rep

    # Sau khử trùng lặp thì khoá duy nhất là ràng buộc cứng.
    dup = int(df["review_id"].duplicated().sum())
    rep.add("review_id là khoá duy nhất (sau khử trùng lặp)", dup == 0, f"{dup} dòng trùng")

    ok = df["is_analysable"]
    rep.add("còn dòng phân tích được", bool(ok.any()), f"{int(ok.sum())}/{len(df)} dòng")

    if ok.any():
        sub = df.loc[ok]
        neg = int((sub["lead_days"] < 0).sum())
        rep.add("dòng phân tích được không có lead_days âm", neg == 0, f"{neg} dòng")

        # gap = lead − SLA, đúng tới sai số dấu phẩy động.
        residual = (sub["gap_days"] - (sub["lead_days"] - sla_days)).abs().max()
        rep.add(
            "gap_days = lead_days − SLA",
            bool(residual < 1e-9),
            f"sai lệch lớn nhất {residual:.2e}",
        )

        null_gap = int(sub["gap_days"].isna().sum())
        rep.add("dòng phân tích được đều có gap_days", null_gap == 0, f"{null_gap} dòng rỗng")

    # Dòng KHÔNG phân tích được phải để gap rỗng, tránh ai đó lỡ tay tính trung bình.
    if (~ok).any():
        leaked = int(df.loc[~ok, "gap_days"].notna().sum())
        rep.add("dòng bị loại có gap_days rỗng", leaked == 0, f"{leaked} dòng rò rỉ")

    return rep
