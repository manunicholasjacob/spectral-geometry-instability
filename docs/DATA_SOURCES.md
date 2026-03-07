# Data Sources Documentation

## Overview

The SGI project uses **real-world financial data** from multiple sources to ensure robust, production-ready analysis. All data is downloaded automatically at runtime - no manual data collection required.

---

## Primary Data Sources

### 1. **Yahoo Finance (via yfinance)**

**What**: Historical price data for stocks, ETFs, and indices  
**Provider**: Yahoo Finance API  
**Library**: `yfinance` Python package  
**Coverage**: 1990s - present (varies by ticker)  
**Frequency**: Daily adjusted close prices  
**Cost**: FREE

**Assets Covered**:
- **Sector ETFs**: 10 US sector ETFs (XLF, XLK, XLE, etc.)
- **Multi-Asset ETFs**: 10 asset class ETFs (SPY, EFA, TLT, GLD, etc.)
- **Large-Cap Stocks**: 50 largest US stocks by market cap
- **Crypto**: Major cryptocurrencies (BTC, ETH, etc.)

**Data Fields**:
- Adjusted Close (primary)
- Open, High, Low, Close
- Volume

---

### 2. **CBOE VIX Index**

**What**: Volatility Index (market fear gauge)  
**Provider**: CBOE via Yahoo Finance  
**Ticker**: `^VIX`  
**Coverage**: 1990 - present  
**Frequency**: Daily  
**Cost**: FREE

**Usage in Project**:
- Control variable in predictive models
- Crisis detection
- Risk regime identification
- VIX integration module (`src/vix_integration.py`)

---

### 3. **Fama-French Factors**

**What**: Academic risk factors (Mkt-RF, SMB, HML, RMW, CMA, Mom)  
**Provider**: Kenneth French Data Library  
**Library**: `pandas-datareader`  
**Coverage**: 1926 - present  
**Frequency**: Daily  
**Cost**: FREE

**Factors Available**:
- **Mkt-RF**: Market excess return
- **SMB**: Small Minus Big (size factor)
- **HML**: High Minus Low (value factor)
- **RMW**: Robust Minus Weak (profitability)
- **CMA**: Conservative Minus Aggressive (investment)
- **Mom**: Momentum factor

**Usage in Project**:
- Factor model integration (`src/factor_model.py`)
- Benchmark comparisons
- Risk decomposition

---

### 4. **Treasury Rates**

**What**: US Treasury yields (risk-free rates)  
**Provider**: Federal Reserve via FRED  
**Tickers**: `^IRX` (13-week), `^FVX` (5-year), `^TNX` (10-year), `^TYX` (30-year)  
**Coverage**: 1960s - present  
**Frequency**: Daily  
**Cost**: FREE

**Usage in Project**:
- Risk-free rate calculations
- Sharpe ratio computations
- Yield curve analysis
- Macro regime detection

---

### 5. **Cryptocurrency Data**

**What**: Major cryptocurrency prices  
**Provider**: Yahoo Finance  
**Tickers**: BTC-USD, ETH-USD, BNB-USD, etc.  
**Coverage**: 2014 - present (varies by coin)  
**Frequency**: Daily  
**Cost**: FREE

**Usage in Project**:
- Crypto universe analysis (`src/crypto_universe.py`)
- Cross-asset spillover studies
- Alternative asset integration

---

## Data Download Process

### Automatic Download

All data is downloaded automatically when you run experiments:

```python
from src.data_loader import load_or_download_prices

# Downloads data automatically if not cached
prices = load_or_download_prices(
    tickers=['SPY', 'QQQ', 'IWM'],
    start_date='2020-01-01',
    end_date='2024-01-01'
)
```

### Caching

Downloaded data is cached in `data/raw/` to avoid repeated downloads:
- First run: Downloads from Yahoo Finance (~10-30 seconds)
- Subsequent runs: Loads from cache (~1 second)

### Data Quality

The pipeline includes automatic data quality checks:
- ✅ Missing data handling (forward fill, interpolation)
- ✅ Outlier detection
- ✅ Alignment across tickers
- ✅ Corporate action adjustments (via adjusted close)

---

## Data Coverage by Universe

| Universe | Assets | Start Date | Typical Coverage |
|----------|--------|------------|------------------|
| **Sector ETFs** | 10 | 2005-01-01 | 20 years |
| **Multi-Asset** | 10 | 2005-01-01 | 20 years |
| **Large-Cap 50** | 50 | 2000-01-01 | 24 years |
| **Crypto** | 10+ | 2017-01-01 | 7 years |

---

## Additional Data Sources (Future Extensions)

### Planned Additions

1. **FRED Economic Data**
   - GDP growth
   - Unemployment rate
   - Inflation (CPI, PCE)
   - Industrial production

2. **Sentiment Data**
   - AAII Sentiment Survey
   - Put/Call ratios
   - News sentiment indices

3. **Alternative Data**
   - Google Trends
   - Social media sentiment
   - Satellite imagery (economic activity)

4. **High-Frequency Data**
   - Intraday prices (1-minute, 5-minute)
   - Order book data
   - Trade and quote data

---

## Data Governance

### Privacy & Compliance
- ✅ All data is publicly available
- ✅ No proprietary data
- ✅ No personal information
- ✅ Compliant with data provider terms of service

### Reproducibility
- ✅ All data sources are documented
- ✅ Download scripts are version-controlled
- ✅ Data transformations are logged
- ✅ Random seeds are fixed

### Data Retention
- Raw data: Cached locally, not pushed to Git
- Processed data: Regenerated on each run
- Results: Saved to `outputs/`, not pushed to Git

---

## Troubleshooting

### Issue: "No data found for ticker XYZ"
**Solution**: Ticker may be delisted or have limited history. Check Yahoo Finance directly.

### Issue: "Connection timeout"
**Solution**: Check internet connection. Yahoo Finance may be temporarily down.

### Issue: "Data has too many missing values"
**Solution**: Adjust `max_missing_frac` in config or choose different date range.

---

## References

- **Yahoo Finance**: https://finance.yahoo.com
- **Kenneth French Data Library**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **CBOE VIX**: https://www.cboe.com/tradable_products/vix/
- **FRED**: https://fred.stlouisfed.org/

---

**Authors**: Manu Nicholas Jacob, Ronit Ghai  
**Last Updated**: 2024  
**Version**: 0.2.0
