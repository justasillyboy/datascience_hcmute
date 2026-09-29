"""Theme và tiện ích lưu figure cho Track B."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Tuyệt đối theo vị trí file này, không theo thư mục đang chạy: Jupyter mặc định
# chạy notebook trong `notebooks/`, đường dẫn tương đối sẽ ghi hình vào
# `notebooks/reports/figures/` thay vì `reports/figures/`.
FIGURE_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

# Màu mặc định theo thứ tự, không gắn ý nghĩa tốt/xấu vào màu.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def setup_theme() -> None:
    plt.rcParams.update({
        "figure.figsize": (10, 5.6),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, filename: str, directory: Path = FIGURE_DIR) -> Path:
    """Lưu figure PNG 300 dpi và trả về đường dẫn."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path
