"""Track how a stock moves through correlation space over time.

Each stock is described, in a given time window, by its vector of correlations
to every other stock (its "correlation profile"). Stacking those profiles across
all windows and fitting a single 2D PCA gives one consistent map, so a stock's
position can be followed from window to window. A stock that stays inside its
sector cluster sits still; one whose relationships break drifts away and (often)
returns.

This is descriptive (unsupervised) analysis. Its job here is twofold: it makes
regime change visible, and it motivates the detachment features used by the
supervised model in ``labeling.py``.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

from .config import sector_of, SECTORS
from .features import log_returns
from .plotting import save_fig

_SECTOR_COLORS = {
    "Tech": "#2f6f9f", "Financials": "#d9822b", "Health": "#b5495b",
    "Consumer": "#3fb950", "Staples": "#6c5ce7", "Energy": "#5d6d7e",
}


def rolling_embeddings(prices: pd.DataFrame, window: int = 126, step: int = 42) -> pd.DataFrame:
    """Return a tidy frame [ticker, date, x, y, sector] across rolling windows.

    A single PCA is fit on all windows' correlation profiles so the 2D axes are
    comparable over time (positions can be tracked, not just compared in shape).
    """
    rets = log_returns(prices)
    tickers = list(rets.columns)
    ends = list(range(window, len(rets) + 1, step))

    profiles, index = [], []
    for e in ends:
        corr = rets.iloc[e - window:e].corr().reindex(index=tickers, columns=tickers)
        date = rets.index[e - 1]
        for t in tickers:
            profiles.append(corr.loc[t].values)
            index.append((t, date))
    X = np.nan_to_num(np.array(profiles), nan=0.0)

    coords = PCA(n_components=2, random_state=0).fit_transform(X)
    emb = pd.DataFrame(coords, columns=["x", "y"])
    emb["ticker"] = [t for t, _ in index]
    emb["date"] = [d for _, d in index]
    emb["sector"] = emb["ticker"].map(sector_of)
    return emb


def sector_centroid_distance(emb: pd.DataFrame, ticker: str) -> pd.Series:
    """Distance from a ticker to its sector's centroid in each window.

    A clean read-out of detachment: it rises when the stock leaves its group and
    falls when it rejoins.
    """
    sec = sector_of(ticker)
    peers = [t for t in SECTORS if sector_of(t) == sec and t != ticker]
    out = {}
    for date, g in emb.groupby("date"):
        gp = g.set_index("ticker")
        if ticker not in gp.index:
            continue
        cx, cy = gp.loc[peers, ["x", "y"]].mean()
        out[date] = float(np.hypot(gp.loc[ticker, "x"] - cx, gp.loc[ticker, "y"] - cy))
    return pd.Series(out, name=f"{ticker}_dist_to_{sec}")


def plot_trajectory(emb: pd.DataFrame, ticker: str, fname: str):
    """2D map: sector centroids for context + the ticker's path over time."""
    last = emb["date"].max()
    bg = emb[emb["date"] == last]
    path = emb[emb["ticker"] == ticker].sort_values("date")

    fig, ax = plt.subplots(figsize=(9, 7))
    # faint individual stocks at the final window
    for sec, gs in bg.groupby("sector"):
        ax.scatter(gs["x"], gs["y"], s=28, alpha=.20,
                   color=_SECTOR_COLORS.get(sec, "#999"))
    # labelled sector centroids for orientation
    for sec, gs in bg.groupby("sector"):
        cx, cy = gs["x"].mean(), gs["y"].mean()
        ax.scatter(cx, cy, s=260, alpha=.85, color=_SECTOR_COLORS.get(sec, "#999"),
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(sec, (cx, cy), fontsize=9, fontweight="bold",
                    ha="center", va="center", color="white", zorder=4)
    # the trajectory, coloured by time (dark = earlier)
    pts = path[["x", "y"]].values
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="inferno",
                        array=np.linspace(0, 1, len(segs)), linewidth=2.2, alpha=.9, zorder=5)
    ax.add_collection(lc)
    ax.scatter(pts[:, 0], pts[:, 1], c=np.linspace(0, 1, len(pts)),
               cmap="inferno", s=22, zorder=6, edgecolor="white", linewidth=.4)
    ax.scatter(*pts[0], color="black", s=110, marker="o", zorder=7)
    ax.scatter(*pts[-1], color="black", s=170, marker="*", zorder=7)
    ax.annotate(f"{ticker} start", pts[0], textcoords="offset points", xytext=(8, 8), fontsize=9, fontweight="bold")
    ax.annotate(f"{ticker} end", pts[-1], textcoords="offset points", xytext=(8, -12), fontsize=9, fontweight="bold")
    ax.set_title(f"{ticker}: path through correlation space (darker = earlier)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    return fig, save_fig(fig, fname)


def plot_distance_over_time(dist: pd.Series, ticker: str, fname: str, breaks=None):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    dist.plot(ax=ax, color="#b5495b", linewidth=1.8)
    ax.fill_between(dist.index, dist.values, dist.min(), color="#b5495b", alpha=.10)
    if breaks is not None:
        ax.axvspan(breaks[0], breaks[1], color="#9aa7b4", alpha=.18, label="planted regime break")
        ax.legend()
    ax.set_title(f"{ticker}: distance from its sector centroid over time")
    ax.set_ylabel("distance (PC units)"); ax.set_xlabel("")
    fig.tight_layout()
    return fig, save_fig(fig, fname)
