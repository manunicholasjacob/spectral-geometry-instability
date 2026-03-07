"""
Constants and default values for the SGI project.
"""

from typing import Dict, List

# Default ticker lists for different universes
SECTOR_ETF_TICKERS: List[str] = [
    "XLF",   # Financials
    "XLK",   # Technology
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLI",   # Industrials
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
]

MULTIASSET_ETF_TICKERS: List[str] = [
    "SPY",   # US Large Cap Equity
    "EFA",   # International Developed Equity
    "EEM",   # Emerging Markets Equity
    "TLT",   # Long-Term Treasury Bonds
    "IEF",   # Intermediate Treasury Bonds
    "LQD",   # Investment Grade Corporate Bonds
    "HYG",   # High Yield Corporate Bonds
    "GLD",   # Gold
    "DBC",   # Commodities
    "VNQ",   # Real Estate
]

LARGECAP_50_TICKERS: List[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "UNH", "XOM", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY", "PEP",
    "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ACN", "ABT", "DHR",
    "NEE", "LIN", "ADBE", "NKE", "TXN", "PM", "WFC", "CRM", "BMY", "UPS",
    "RTX", "ORCL", "MS", "QCOM", "HON", "INTC", "IBM", "CAT", "GE", "BA",
]

# Crisis event dates for event studies
CRISIS_DATES: Dict[str, Dict[str, str]] = {
    "gfc_2008": {
        "name": "Global Financial Crisis",
        "peak_date": "2008-10-10",
        "start": "2008-09-01",
        "end": "2009-03-31",
    },
    "flash_crash_2010": {
        "name": "Flash Crash",
        "peak_date": "2010-05-06",
        "start": "2010-05-01",
        "end": "2010-05-31",
    },
    "euro_crisis_2011": {
        "name": "European Debt Crisis",
        "peak_date": "2011-08-08",
        "start": "2011-07-01",
        "end": "2011-10-31",
    },
    "china_deval_2015": {
        "name": "China Devaluation",
        "peak_date": "2015-08-24",
        "start": "2015-08-01",
        "end": "2015-09-30",
    },
    "volmageddon_2018": {
        "name": "Volmageddon",
        "peak_date": "2018-02-05",
        "start": "2018-01-26",
        "end": "2018-02-28",
    },
    "covid_2020": {
        "name": "COVID-19 Crash",
        "peak_date": "2020-03-16",
        "start": "2020-02-15",
        "end": "2020-04-30",
    },
    "inflation_2022": {
        "name": "Inflation/Rate Shock",
        "peak_date": "2022-06-13",
        "start": "2022-01-01",
        "end": "2022-10-31",
    },
}

# Supported covariance estimators
COVARIANCE_ESTIMATORS: List[str] = [
    "sample",
    "ewma",
    "ledoit_wolf",
    "oas",  # Oracle Approximating Shrinkage
]

# Supported target names
TARGET_NAMES: List[str] = [
    "forward_drawdown",
    "forward_realized_vol",
    "forward_correlation",
    "cov_forecast_error",
    "drawdown_event",
    "correlation_spike_event",
    "vol_spike_event",
]

# SGI metric variants
SGI_VARIANTS: List[str] = [
    "sgi",
    "weighted_sgi",
    "max_angle",
    "mean_angle",
    "sgi_zscore",
    "sgi_percentile",
    "sgi_ema",
]

# Default transaction costs (basis points)
DEFAULT_TRANSACTION_COST_BPS: float = 10.0

# Default rolling windows
DEFAULT_ROLLING_WINDOWS: List[int] = [60, 126, 252, 504]

# Default top-k values
DEFAULT_TOP_K_VALUES: List[int] = [2, 3, 5, 8]

# Default horizons for forward targets
DEFAULT_HORIZONS: List[int] = [5, 10, 20, 60]

# Rebalance frequency mappings
REBALANCE_FREQ_DAYS: Dict[str, int] = {
    "daily": 1,
    "weekly": 5,
    "biweekly": 10,
    "monthly": 21,
    "quarterly": 63,
}

# Numerical constants
EPSILON: float = 1e-10
MIN_EIGENVALUE: float = 1e-8
MAX_CONDITION_NUMBER: float = 1e10
