import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.viz.weighted import weighted_hist, weighted_ecdf, weighted_bar_ci, weighted_box


def test_equal_weights_match_basic_hist_counts():
    values = np.array([0.0, 1.0, 1.0, 2.0])
    _, _, patches = weighted_hist(values, weights=np.ones(4), bins=[0, 1, 2, 3])
    heights = np.array([p.get_height() for p in patches])
    assert np.allclose(heights, [1, 2, 1])


def test_skewed_weights_change_ecdf():
    values = np.array([0.0, 10.0])
    x, y = weighted_ecdf(values, weights=np.array([9.0, 1.0]))
    assert np.allclose(x, [0.0, 10.0])
    assert y[0] > 0.8


def test_weighted_bar_ci_returns_group_summary():
    result = weighted_bar_ci(
        np.array(["A", "A", "B", "B"]),
        np.array([1.0, 3.0, 10.0, 10.0]),
        np.ones(4),
    )
    assert result["labels"] == ["A", "B"]
    assert np.allclose(result["means"], [2.0, 10.0])


def test_weighted_box_accepts_equal_weights():
    stats = weighted_box(
        np.array([1.0, 2.0, 3.0, 4.0]),
        groups=np.array([0, 0, 0, 0]),
        weights=np.ones(4),
    )
    assert np.isclose(stats[0]["med"], 2.5)
