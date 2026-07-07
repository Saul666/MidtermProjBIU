"""Spread mechanics and feature primitives used across the project."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices).diff().dropna(how="all")


def hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> float:
    """OLS slope beta in  log_a = alpha + beta * log_b.

    The spread  log_a - beta * log_b  is what we test for stationarity.
    """
    x = sm.add_constant(log_b.values)
    model = sm.OLS(log_a.values, x).fit()
    return float(model.params[1])


def spread_series(prices: pd.DataFrame, a: str, b: str, beta: float) -> pd.Series:
    """log(P_a) - beta * log(P_b)."""
    return np.log(prices[a]) - beta * np.log(prices[b])


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """Z-score of the spread using a trailing window only (no look-ahead)."""
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life from an AR(1) fit on the spread.

    Regress d(spread)_t on spread_{t-1}; half-life = -ln(2) / lambda.
    Returns np.inf when the series is not mean-reverting (lambda >= 0).
    """
    s = spread.dropna()
    if len(s) < 30:
        return float("inf")
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    lag, delta = lag.align(delta, join="inner")
    x = sm.add_constant(lag.values)
    lam = sm.OLS(delta.values, x).fit().params[1]
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / lam)


def rolling_corr(ret_a: pd.Series, ret_b: pd.Series, window: int) -> pd.Series:
    """Trailing return correlation between two assets."""
    return ret_a.rolling(window).corr(ret_b)


def hurst(series, max_lag: int = 20) -> float:
    """Hurst exponent of a series. H < 0.5 mean-reverting, ~0.5 random walk,
    > 0.5 trending. Estimated from the scaling of lagged-difference dispersion.
    """
    ts = np.asarray(series, dtype=float)
    ts = ts[~np.isnan(ts)]
    if len(ts) < max_lag + 5 or np.allclose(ts, ts[0]):
        return 0.5
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        d = ts[lag:] - ts[:-lag]
        s = np.std(d)
        tau.append(s if s > 0 else 1e-9)
    slope = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    return float(slope)


def adf_stat(series) -> float:
    """Augmented Dickey-Fuller test statistic on a series (more negative = more
    stationary / mean-reverting). Returns 0.0 on failure.
    """
    from statsmodels.tsa.stattools import adfuller
    ts = np.asarray(series, dtype=float)
    ts = ts[~np.isnan(ts)]
    if len(ts) < 30 or np.allclose(ts, ts[0]):
        return 0.0
    try:
        return float(adfuller(ts, maxlag=1, regression="c", autolag=None)[0])
    except Exception:
        return 0.0
