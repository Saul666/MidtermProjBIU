"""Central configuration for the pairs mean-reversion project.

Every knob lives here so the notebooks stay clean and runs are reproducible.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
for _p in (DATA_RAW, DATA_PROCESSED, FIGURES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Data source: live Yahoo Finance via yfinance (requires internet)
# ----------------------------------------------------------------------------
SOURCE = "yfinance"

START_DATE = "2018-01-01"
END_DATE = "2026-05-31"

# Universe grouped by sector. The sector label is used for sector-relative
# features and for colouring the regime-tracking plots.
SECTORS = {
    # Technology
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech",
    "META": "Tech", "ORCL": "Tech", "CRM": "Tech", "ADBE": "Tech",
    "AMD": "Tech", "INTC": "Tech",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "AXP": "Financials", "MS": "Financials", "SCHW": "Financials",
    "C": "Financials", "WFC": "Financials",
    # Health care
    "JNJ": "Health", "UNH": "Health", "PFE": "Health", "ABBV": "Health",
    "LLY": "Health", "MRK": "Health", "BMY": "Health",
    # Consumer discretionary
    "AMZN": "Consumer", "HD": "Consumer", "LOW": "Consumer", "TGT": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer",
    # Consumer staples
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "CL": "Staples",
    "MDLZ": "Staples", "KMB": "Staples", "WMT": "Staples", "COST": "Staples",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "OXY": "Energy", "SLB": "Energy",
    "COP": "Energy", "EOG": "Energy",
}
TICKERS = list(SECTORS.keys())

SEED = 7  # used by every model for reproducibility


def sector_of(ticker: str) -> str:
    return SECTORS.get(ticker, "Other")


# ----------------------------------------------------------------------------
# Pair selection
# ----------------------------------------------------------------------------
@dataclass
class PairConfig:
    formation_days: int = 504          # ~2 trading years used to select a pair
    min_correlation: float = 0.60      # return-correlation gate before cointegration
    coint_pvalue_max: float = 0.05     # Engle-Granger significance gate
    max_pairs: int = 40                # cap to keep the study tractable


# ----------------------------------------------------------------------------
# Spread / signal definition  (the supervised "event" is an extreme z-score)
# ----------------------------------------------------------------------------
@dataclass
class SignalConfig:
    z_window: int = 60                 # rolling window for spread mean/std (past only)
    entry_z: float = 2.0               # |z| at which an extreme "event" is logged
    exit_z: float = 0.5                # |z| considered "reverted"
    horizon: int = 10                  # trading days allowed for reversion -> the label
    min_gap: int = 5                   # min days between events on the same pair (dedupe)
    baseline_window: int = 252         # long window for correlation / cointegration health
    momentum_window: int = 252         # ~1-year change feature


# ----------------------------------------------------------------------------
# Modeling
# ----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    test_fraction: float = 0.25        # most-recent fraction of events -> test (time split)
    decision_threshold: float = 0.50   # probability cut for the "trade" decision
    feature_cols: tuple = (
        # spread / signal state
        "abs_z", "z_velocity", "spread_vol", "half_life", "spread_slope",
        # pair relationship
        "corr_recent", "beta", "coint_pvalue", "market_vol",
        "ret_a_5d", "ret_b_5d",
        # regime / cross-sectional features (the "detachment" idea)
        "mom_a_252", "mom_b_252", "mom_rel",
        "corr_baseline", "corr_change", "comembership",
        "coint_recent_pvalue", "vol_ratio", "sector_dispersion", "detach_max",
        # spread stationarity (is the spread still mean-reverting now?)
        "adf_stat", "hurst",
        # market context (regime + cross-sectional stress)
        "vix_level", "vix_change", "vol_spike_max", "n_pairs_extreme", "z_rank",
    )


@dataclass
class Config:
    source: str = SOURCE
    start: str = START_DATE
    end: str = END_DATE
    tickers: list = field(default_factory=lambda: list(TICKERS))
    seed: int = SEED
    pairs: PairConfig = field(default_factory=PairConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


CONFIG = Config()
