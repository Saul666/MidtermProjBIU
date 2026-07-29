"""Central configuration for the pairs mean-reversion project.

Every setting lives here so the notebooks stay clean and runs are reproducible.
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

for _path in (DATA_RAW, DATA_PROCESSED, FIGURES, MODELS):
    _path.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Data source
# ----------------------------------------------------------------------------
SOURCE = "yfinance"

START_DATE = "2018-01-01"
END_DATE = "2026-05-31"


# ----------------------------------------------------------------------------
# Stock universe: 200 tickers grouped by sector
# ----------------------------------------------------------------------------

TECH_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "META",
    "ORCL",
    "CRM",
    "ADBE",
    "AMD",
    "INTC",
    "AVGO",
    "CSCO",
    "QCOM",
    "TXN",
    "AMAT",
    "MU",
    "NOW",
    "INTU",
    "IBM",
    "PANW",
    "SNPS",
    "CDNS",
    "KLAC",
    "LRCX",
    "ADI",
    "ANET",
    "FTNT",
    "ROP",
    "APH",
    "TEL",
]

FINANCIAL_TICKERS = [
    "JPM",
    "BAC",
    "GS",
    "AXP",
    "MS",
    "SCHW",
    "C",
    "WFC",
    "BLK",
    "USB",
    "PNC",
    "BK",
    "COF",
    "AIG",
    "MET",
    "PRU",
    "TFC",
    "AFL",
    "ALL",
    "CB",
    "MMC",
    "ICE",
    "CME",
    "SPGI",
    "MCO",
]

HEALTH_TICKERS = [
    "JNJ",
    "UNH",
    "PFE",
    "ABBV",
    "LLY",
    "MRK",
    "BMY",
    "TMO",
    "ABT",
    "DHR",
    "AMGN",
    "GILD",
    "CVS",
    "CI",
    "ELV",
    "MDT",
    "ISRG",
    "SYK",
    "BSX",
    "ZTS",
    "REGN",
    "VRTX",
    "BDX",
    "HUM",
    "MCK",
]

CONSUMER_TICKERS = [
    "AMZN",
    "HD",
    "LOW",
    "TGT",
    "NKE",
    "SBUX",
    "MCD",
    "BKNG",
    "TJX",
    "ROST",
    "MAR",
    "CMG",
    "ORLY",
    "AZO",
    "GM",
    "F",
    "TSLA",
    "DHI",
    "LEN",
    "YUM",
]

STAPLES_TICKERS = [
    "PG",
    "KO",
    "PEP",
    "CL",
    "MDLZ",
    "KMB",
    "WMT",
    "COST",
    "PM",
    "MO",
    "EL",
    "GIS",
    "KHC",
    "SYY",
    "KR",
]

ENERGY_TICKERS = [
    "XOM",
    "CVX",
    "OXY",
    "SLB",
    "COP",
    "EOG",
    "MPC",
    "PSX",
    "VLO",
    "WMB",
    "OKE",
    "KMI",
    "HAL",
    "DVN",
    "FANG",
]

INDUSTRIAL_TICKERS = [
    "CAT",
    "DE",
    "HON",
    "GE",
    "RTX",
    "BA",
    "UPS",
    "FDX",
    "LMT",
    "NOC",
    "GD",
    "ETN",
    "EMR",
    "ITW",
    "PH",
    "CSX",
    "NSC",
    "UNP",
    "WM",
    "RSG",
    "MMM",
    "JCI",
    "PCAR",
    "FAST",
    "URI",
]

COMMUNICATION_TICKERS = [
    "NFLX",
    "DIS",
    "CMCSA",
    "TMUS",
    "T",
    "VZ",
    "CHTR",
    "EA",
    "TTWO",
    "WBD",
]

UTILITY_TICKERS = [
    "NEE",
    "DUK",
    "SO",
    "D",
    "AEP",
    "EXC",
    "SRE",
    "XEL",
    "ED",
    "PEG",
]

REAL_ESTATE_TICKERS = [
    "PLD",
    "AMT",
    "EQIX",
    "CCI",
    "PSA",
    "O",
    "WELL",
    "SPG",
    "DLR",
    "VICI",
]

MATERIAL_TICKERS = [
    "LIN",
    "APD",
    "SHW",
    "ECL",
    "FCX",
    "NEM",
    "NUE",
    "DOW",
    "DD",
    "PPG",
    "VMC",
    "MLM",
    "IFF",
    "CTVA",
    "CF",
]


# Create one ticker-to-sector dictionary.
SECTORS = {
    **{ticker: "Tech" for ticker in TECH_TICKERS},
    **{ticker: "Financials" for ticker in FINANCIAL_TICKERS},
    **{ticker: "Health" for ticker in HEALTH_TICKERS},
    **{ticker: "Consumer" for ticker in CONSUMER_TICKERS},
    **{ticker: "Staples" for ticker in STAPLES_TICKERS},
    **{ticker: "Energy" for ticker in ENERGY_TICKERS},
    **{ticker: "Industrials" for ticker in INDUSTRIAL_TICKERS},
    **{ticker: "Communication" for ticker in COMMUNICATION_TICKERS},
    **{ticker: "Utilities" for ticker in UTILITY_TICKERS},
    **{ticker: "RealEstate" for ticker in REAL_ESTATE_TICKERS},
    **{ticker: "Materials" for ticker in MATERIAL_TICKERS},
}

TICKERS = list(SECTORS.keys())

# Check that the universe contains exactly 200 unique tickers.
if len(TICKERS) != 200:
    raise ValueError(
        f"Expected 200 unique tickers, but found {len(TICKERS)}."
    )


SEED = 7


def sector_of(ticker: str) -> str:
    """Return the sector assigned to a ticker."""
    return SECTORS.get(ticker, "Other")


# ----------------------------------------------------------------------------
# Pair selection
# ----------------------------------------------------------------------------
@dataclass
class PairConfig:
    # Approximately two trading years used to select pairs.
    formation_days: int = 504

    # Minimum return correlation before performing cointegration testing.
    min_correlation: float = 0.60

    # Maximum Engle-Granger cointegration p-value.
    coint_pvalue_max: float = 0.05

    # Maximum number of selected pairs kept for modeling.
    max_pairs: int = 100

    # Clustering reduces the number of pair combinations tested.
    # With 200 tickers, 20 clusters gives roughly 10 stocks per cluster.
    n_clusters: int = 20


# ----------------------------------------------------------------------------
# Spread and signal definition
# ----------------------------------------------------------------------------
@dataclass
class SignalConfig:
    # Rolling window used for spread mean and standard deviation.
    z_window: int = 60

    # Absolute z-score at which an extreme event is logged.
    entry_z: float = 2.0

    # Absolute z-score considered reverted.
    exit_z: float = 0.5

    # Number of trading days allowed for reversion.
    horizon: int = 10

    # Minimum days between events for the same pair.
    min_gap: int = 5

    # Long-term window for relationship-health features.
    baseline_window: int = 252

    # Approximately one year of trading days.
    momentum_window: int = 252


# ----------------------------------------------------------------------------
# Modeling
# ----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # Most recent portion of the events used as the test set.
    test_fraction: float = 0.25

    # Probability threshold used for the final prediction.
    decision_threshold: float = 0.50

    feature_cols: tuple[str, ...] = (
        # Spread and signal state
        "abs_z",
        "z_velocity",
        "spread_vol",
        "half_life",
        "spread_slope",

        # Pair relationship
        "corr_recent",
        "beta",
        "coint_pvalue",
        "market_vol",
        "ret_a_5d",
        "ret_b_5d",

        # Regime and cross-sectional features
        "mom_a_252",
        "mom_b_252",
        "mom_rel",
        "corr_baseline",
        "corr_change",
        "comembership",
        "coint_recent_pvalue",
        "vol_ratio",
        "sector_dispersion",
        "detach_max",

        # Spread stationarity
        "adf_stat",
        "hurst",

        # Market context
        "vix_level",
        "vix_change",
        "vol_spike_max",
        "n_pairs_extreme",
        "z_rank",
    )


# ----------------------------------------------------------------------------
# Main project configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    source: str = SOURCE
    start: str = START_DATE
    end: str = END_DATE

    tickers: list[str] = field(
        default_factory=lambda: list(TICKERS)
    )

    seed: int = SEED

    pairs: PairConfig = field(
        default_factory=PairConfig
    )

    signal: SignalConfig = field(
        default_factory=SignalConfig
    )

    model: ModelConfig = field(
        default_factory=ModelConfig
    )


CONFIG = Config()