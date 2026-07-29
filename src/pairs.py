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
    """Return a table of pairs that pass the correlation and cointegration gates."""
    cfg = cfg or CONFIG.pairs
    formation = prices
    rets = log_returns(formation)
    corr = rets.corr().clip(-1, 1)
    logp = np.log(formation)

    # OPTIMIZATION: Use your clustering logic to drastically narrow down pairs
    # If the universe is large, we only check pairs inside the same cluster.
    labels, _ = cluster_labels(corr, n_clusters=getattr(cfg, "n_clusters", 10))
    
    rows = []
    # Loop over combinations, but filter out pairs not in the same cluster 
    # to avoid the O(N^2) cointegration bottleneck on a massive universe.
    for a, b in itertools.combinations(formation.columns, 2):
        if labels[a] != labels[b]:
            continue  # Skip pairs that are far apart asset-structure wise
            
        c = corr.loc[a, b]
        if c < cfg.min_correlation:
            continue

        # Engle-Granger is asymmetric. Test both or handle fallback safely.
        try:
            _, pval, _ = coint(logp[a], logp[b])
            # If the first direction fails, check the inverse order
            if pval > cfg.coint_pvalue_max:
                _, pval_alt, _ = coint(logp[b], logp[a])
                if pval_alt < pval:
                    pval = pval_alt
                    a, b = b, a  # Flip direction because b on a is stronger
        except Exception:
            continue

        if pval > cfg.coint_pvalue_max:
            continue

        beta = hedge_ratio(logp[a], logp[b])
        if beta <= 0:
            continue

        # FIX: Pass log prices here because beta was derived from log prices!
        spr = spread_series(logp, a, b, beta)
        hl = half_life(spr)

        # Inline filter: don't append trash half-lives to dataframes
        if not (1 <= hl <= 120) or not np.isfinite(hl):
            continue

        rows.append({
            "a": a, "b": b, "corr": round(float(c), 3),
            "coint_pvalue": round(float(pval), 4),
            "beta": round(float(beta), 3),
            "half_life": round(float(hl), 1),
        })

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs

    # Sort to surface the absolute strongest pairs
    pairs = pairs.sort_values(["coint_pvalue", "half_life"]).head(cfg.max_pairs)
    return pairs.reset_index(drop=True)


