"""Load the price panel from Yahoo Finance.

``load_prices`` returns a tidy wide DataFrame: index = trading dates,
columns = tickers, values = adjusted close. Data is downloaded live via
``yfinance`` and cached locally so repeat runs are fast.

``load_context`` optionally adds exogenous signals (the VIX volatility index and
per-ticker volume) used by the enrichment features.
"""
from __future__ import annotations
import pandas as pd

from .config import CONFIG, DATA_RAW


# ----------------------------------------------------------------------------
# Price panel
# ----------------------------------------------------------------------------
def load_prices(source: str | None = None, use_cache: bool = True) -> tuple[pd.DataFrame, str]:
    """Return (prices, 'yfinance'). Live download with a local CSV cache.

    The ``source`` argument is accepted for backward compatibility but the only
    source is Yahoo Finance via yfinance.
    """
    cache = DATA_RAW / "prices.csv"
    if use_cache and cache.exists():
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        return prices, "cache"
    prices = _load_yfinance()
    prices.to_csv(cache)
    return prices, "yfinance"


def _load_yfinance() -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(
        CONFIG.tickers, start=CONFIG.start, end=CONFIG.end,
        auto_adjust=True, progress=False, threads=True,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            "yfinance returned no data. Check your internet connection / that "
            "Yahoo Finance is reachable, then re-run."
        )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.dropna(axis=1, how="all").sort_index()
    keep = close.columns[close.isna().mean() < 0.02]   # near-complete history only
    close = close[keep].ffill().dropna()
    if close.shape[1] < 4:
        raise RuntimeError("too few tickers returned from yfinance")
    return close


# ----------------------------------------------------------------------------
# Optional exogenous context: VIX volatility index + per-ticker volume
# ----------------------------------------------------------------------------
def load_context(source: str | None = None) -> dict:
    """Return {'vix': Series|None, 'volume': DataFrame|None} from Yahoo Finance.

    On failure both are None and the labeller falls back to features derived from
    the price returns (a realized-volatility VIX proxy and an absolute-return
    activity spike), so every feature is always computable from real prices.
    """
    out = {"vix": None, "volume": None}
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=CONFIG.start, end=CONFIG.end,
                          auto_adjust=False, progress=False)["Close"]
        out["vix"] = vix.reindex(pd.bdate_range(CONFIG.start, CONFIG.end)).ffill()
        vol = yf.download(CONFIG.tickers, start=CONFIG.start, end=CONFIG.end,
                          auto_adjust=True, progress=False)["Volume"]
        out["volume"] = vol.ffill()
    except Exception:
        pass
    return out
