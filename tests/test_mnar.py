import numpy as np
import pandas as pd

from src.evaluate.mnar_sensitivity import (
    _prepare_theta_components,
    _build_theta_probabilities,
    a3_self_check,
    fractional_weights,
    theta_estimate,
    prepare,
)

from pathlib import Path

# ============================================================
# A9.1 — A3 SELF-CHECK
# ============================================================

def test_a3_reproduces_current_findings():
    """
    Cổng A3:
    theta_scan trên riêng nhóm có nhãn phải tái lập
    đúng estimate hiện hành trong FINDINGS.md.
    """
    df = pd.read_parquet(
        "data/processed/reviews_clean.parquet"
    )

    data = prepare(df)

    era = data[data["nam"] >= 2023].copy()

    result = a3_self_check(
        era,
        findings_path=Path("docs/FINDINGS.md"),
    )

    assert result["passed"] is True

    np.testing.assert_allclose(
        result["observed"],
        result["target"],
        atol=5e-5,
        rtol=0.0,
    )


# ============================================================
# A9.2 — TRỌNG SỐ PHÂN ĐOẠN
# ============================================================

def test_fractional_weights_sum_to_original():
    """
    Với mỗi dòng:

        w*p + w*(1-p) = w

    Đây là bất biến bắt buộc của trọng số phân đoạn.
    """
    weights = np.array([
        1.0,
        2.0,
        5.0,
        10.0,
    ])

    probabilities = np.array([
        0.0,
        0.25,
        0.70,
        1.0,
    ])

    late_w, ontime_w = fractional_weights(
        weights,
        probabilities,
    )

    np.testing.assert_allclose(
        late_w + ontime_w,
        weights,
        rtol=0.0,
        atol=1e-12,
    )


# ============================================================
# A9.3 — THETA = 1 PHẢI BẰNG MAR THUẦN
# ============================================================

def test_theta_one_equals_pure_mar():
    """
    θ = 1:

        p_i(1) = p_MAR(i)

    Kết quả theta_estimate phải đúng bằng
    phép tính MAR thuần thủ công.
    """

    # Có đủ cả 10 ô:
    # rating 1..5 × sla_breach 0/1
    # và có cả nhãn lẫn không nhãn.
    df = pd.DataFrame(
        {
            "rating": [
                1, 1,
                2, 2,
                3, 3,
                4, 4,
                5, 5,
                5, 5,
            ],
            "sla_breach": [
                False, True,
                False, True,
                False, True,
                False, True,
                False, True,
                False, True,
            ],
            "customer_says_late": [
                True, False,
                False, True,
                True, False,
                False, True,
                False, True,
                np.nan, np.nan,
            ],
            "weight": [
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0,
                2.0, 3.0,
            ],
        }
    )

    p_bar, p_mar = _prepare_theta_components(df)

    # --------------------------------------------------------
    # Kết quả bằng module
    # --------------------------------------------------------

    observed = theta_estimate(
        df,
        theta=1.0,
        p_bar=p_bar,
        p_mar=p_mar,
    )

    # --------------------------------------------------------
    # Tự tính MAR thuần
    # --------------------------------------------------------

    p = _build_theta_probabilities(
        df,
        theta=1.0,
        hard_labels=False,
        p_bar=p_bar,
        p_mar=p_mar,
    )

    rating = df["rating"].to_numpy(float)
    weight = df["weight"].to_numpy(float)
    sla = df["sla_breach"].to_numpy(bool)

    late_weight = weight * p
    ontime_weight = weight * (1.0 - p)

    late_mean = (
        np.sum(rating * late_weight)
        / np.sum(late_weight)
    )

    ontime_mean = (
        np.sum(rating * ontime_weight)
        / np.sum(ontime_weight)
    )

    # SLA giữ nguyên như mô hình chính.
    sla_breach_weight = weight * sla.astype(float)
    sla_ontime_weight = weight * (~sla).astype(float)

    sla_breach_mean = (
        np.sum(rating * sla_breach_weight)
        / np.sum(sla_breach_weight)
    )

    sla_ontime_mean = (
        np.sum(rating * sla_ontime_weight)
        / np.sum(sla_ontime_weight)
    )

    expected = (
        abs(late_mean - ontime_mean)
        - abs(sla_breach_mean - sla_ontime_mean)
    )

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=0.0,
        atol=1e-12,
    )