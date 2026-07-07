"""Consistent chart styling and a single save helper.

Keeping all styling in one place means every figure in the project shares the
same look, and saved charts land in reports/figures automatically.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES

# A restrained, readable theme.
PALETTE = ["#2f6f9f", "#3fb950", "#d9822b", "#b5495b", "#6c5ce7", "#5d6d7e"]
ACCENT = "#0d1b2a"


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.titleweight": "600",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": "#cdd5df",
        "axes.grid": True,
        "grid.color": "#e8edf2",
        "font.size": 10,
        "legend.frameon": False,
    })
    sns.set_palette(PALETTE)


def save_fig(fig, name: str) -> str:
    """Save a figure to reports/figures and return its path as a string."""
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    return str(path)
