"""Select tradable pairs.

Two-gate funnel:
  1. return correlation above a threshold (cheap, removes obvious non-pairs)
  2. Engle-Granger cointegration p-value below a threshold on the formation
     window (the real statistical test that the spread is stationary)

For pairs that pass, we record the hedge ratio, cointegration p-value and
mean-reversion half-life estimated on the formation window only.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import coint

from .config import CONFIG
from .features import log_returns, hedge_ratio, spread_series, half_life


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    return log_returns(prices).corr().clip(-1, 1)


def cluster_labels(corr: pd.DataFrame, n_clusters: int = 6, method: str = "average"):
    """Hierarchical clusters on correlation distance. Returns (labels, linkage Z)."""
    dist = 1 - corr.to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    z = linkage(squareform(dist, checks=False), method=method)
    labels = pd.Series(fcluster(z, t=n_clusters, criterion="maxclust"),
                       index=corr.index, name="cluster")
    return labels, z


def select_pairs(prices: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Return a table of pairs that pass the correlation and cointegration gates.

    Only the formation window (most recent ``formation_days``) is used so the
    selection is causal: it never peeks at the period we later score events on.
    """
    cfg = cfg or CONFIG.pairs
    # The caller passes exactly the formation window, so selection is causal:
    # it never sees the period on which events are later scored.
    formation = prices
    rets = log_returns(formation)
    corr = rets.corr().clip(-1, 1)
    logp = np.log(formation)

    rows = []
    for a, b in itertools.combinations(formation.columns, 2):
        c = corr.loc[a, b]
        if c < cfg.min_correlation:
            continue
        # Engle-Granger: regress log_a on log_b, test residual for unit root
        try:
            _, pval, _ = coint(logp[a], logp[b])
        except Exception:
            continue
        if pval > cfg.coint_pvalue_max:
            continue
        beta = hedge_ratio(logp[a], logp[b])
        if beta <= 0:           # require a sensible positive hedge ratio
            continue
        spr = spread_series(formation, a, b, beta)
        hl = half_life(spr)
        rows.append({
            "a": a, "b": b, "corr": round(float(c), 3),
            "coint_pvalue": round(float(pval), 4),
            "beta": round(float(beta), 3),
            "half_life": round(float(hl), 1) if np.isfinite(hl) else np.nan,
        })

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    # prefer strong, fast-reverting, clearly-cointegrated pairs
    pairs = pairs[pairs["half_life"].between(1, 120)]
    pairs = pairs.sort_values(["coint_pvalue", "half_life"]).head(cfg.max_pairs)
    return pairs.reset_index(drop=True)
