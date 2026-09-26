from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.clean.reviews import compute_gap, SLA_MAX_DAYS
from src.evaluate.sensitivity import rating_gap_customer_vs_sla


DATA_PATH = Path("data/processed/reviews_clean.parquet")
FINDINGS_PATH = Path("docs/FINDINGS.md")


def read_current_main_estimate():
    """
    Đọc estimate hiện hành từ FINDINGS.md §4.
    Không hard-code 0.2082.
    """
    text = FINDINGS_PATH.read_text(encoding="utf-8")

    match_section = re.search(
        r"(?ms)^##\s*4\.\s*Kết luận chính\b"
        r"(.*?)(?=^##\s|\Z)",
        text,
    )

    assert match_section is not None, (
        "Không tìm thấy §4 trong FINDINGS.md"
    )

    section = match_section.group(1)

    match_value = re.search(
        r"chênh lệch\s+"
        r"([+-]?\d+(?:[.,]\d+)?)"
        r"\s*điểm\s+rating",
        section,
        flags=re.IGNORECASE,
    )

    assert match_value is not None, (
        "Không tìm thấy estimate trong §4 FINDINGS.md"
    )

    return float(
        match_value.group(1).replace(",", ".")
    )


def test_sla_days_5_reproduces_current_main_estimate():
    """
    A10:
    sla_days = 5 phải tái lập đúng estimate hiện hành
    trong FINDINGS.md §4.
    """
    df = pd.read_parquet(DATA_PATH)

    current_estimate = read_current_main_estimate()

    # A10 yêu cầu đúng ngưỡng SLA chuẩn hiện hành.
    assert SLA_MAX_DAYS == 5

    d = compute_gap(
        df,
        sla_days=5,
    )

    observed = rating_gap_customer_vs_sla(d)

    np.testing.assert_allclose(
        observed,
        current_estimate,
        atol=5e-5,
        rtol=0.0,
    )