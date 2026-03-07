# Examples

## Practical Examples for Common Use Cases

---

## Example 1: Basic SGI Computation

```python
import pandas as pd
import numpy as np
from src.data_loader import load_or_download_prices
from src.preprocessing import preprocess_data
from src.covariance import rolling_covariance_matrices
from src.spectral_geometry import compute_rolling_geometry_features

# Load data
tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD', 'VNQ']
prices = load_or_download_prices(tickers, '2010-01-01', '2024-01-01')

# Preprocess
prices, returns, _ = preprocess_data(prices, save_outputs=False)

# Compute covariance
cov_matrices = rolling_covariance_matrices(returns, window=126)

# Compute SGI
geometry_df = compute_rolling_geometry_features(cov_matrices, top_k=3)

# View results
print(geometry_df[['sgi', 'weighted_sgi', 'absorption_ratio']].tail())
```

---

## Example 2: Compare SGI with VIX

```python
from src.vix_integration import (
    download_vix_data, compute_vix_features, compute_sgi_vix_relationship
)

# Download VIX
vix_data = download_vix_data('2010-01-01', '2024-01-01')
vix_features = compute_vix_features(vix_data)

# Analyze relationship
relationship = compute_sgi_vix_relationship(geometry_df, vix_features)

print(f"SGI-VIX correlation: {relationship['correlation']['sgi_vix_level']:.3f}")
print(f"Lead-lag analysis: {relationship['lead_lag_correlation']}")
```

---

## Example 3: Predictive Modeling

```python
from src.features import build_feature_dataframe, get_feature_columns
from src.targets import build_target_dataframe
from src.predictive_models import run_walk_forward_experiment, summarize_walk_forward_results

# Build features and targets
config = {'covariance': {'rolling_window': 126}, 'universe': {}}
features_df = build_feature_dataframe(returns, geometry_df, config, prices)
targets_df = build_target_dataframe(prices, returns, cov_matrices, config)

# Prepare dataset
df = features_df.copy()
df['target'] = targets_df['forward_drawdown_20d']
df = df.dropna()

# Define feature sets
feature_sets = {
    'baseline': ['realized_vol', 'avg_correlation'],
    'with_sgi': ['realized_vol', 'avg_correlation', 'sgi', 'weighted_sgi'],
}

# Run walk-forward
results = run_walk_forward_experiment(df, 'target', feature_sets, 'linear')
summary = summarize_walk_forward_results(results)
print(summary)
```

---

## Example 4: Portfolio Backtest

```python
from src.portfolio import generate_weight_schedule
from src.backtest import run_rebalancing_backtest, compare_strategies

# Generate weights for different strategies
strategies = ['equal_weight', 'min_variance', 'sgi_conditioned_minvar']
weight_schedules = {}

config = {'portfolio': {'sgi_threshold_percentile': 75}}

for strategy in strategies:
    weights = generate_weight_schedule(
        returns, cov_matrices, geometry_df,
        strategy, config, 'weekly'
    )
    weight_schedules[strategy] = weights

# Compare strategies
comparison = compare_strategies(prices, weight_schedules, transaction_cost_bps=10)
print(comparison[['strategy', 'cagr', 'sharpe', 'max_drawdown']])
```

---

## Example 5: Multi-scale Analysis

```python
from src.multiscale_sgi import (
    compute_multiscale_sgi, compute_regime_from_multiscale
)

# Compute at multiple scales
windows = [60, 126, 252]
multiscale_df = compute_multiscale_sgi(returns, windows, top_k=3)

# Detect regime
regime = compute_regime_from_multiscale(multiscale_df, windows)
print(regime.value_counts())

# View term structure
print(multiscale_df[['sgi_term_spread', 'sgi_scale_agreement']].describe())
```

---

## Example 6: Crypto Analysis

```python
from src.crypto_universe import compute_crypto_sgi, CRYPTO_EVENTS

# Compute crypto SGI
crypto_geometry, crypto_features, _ = compute_crypto_sgi(
    start_date='2021-01-01',
    end_date='2024-01-01',
    window=60
)

# Event study
from src.event_study import compute_event_statistics

for event_name, event_info in CRYPTO_EVENTS.items():
    try:
        stats = compute_event_statistics(
            crypto_geometry['sgi'],
            event_info['peak_date'],
            pre_days=30, post_days=30
        )
        print(f"{event_name}: z-score = {stats['event_zscore']:.2f}")
    except:
        pass
```

---

## Example 7: RMT Denoising

```python
from src.rmt_denoising import (
    rolling_rmt_denoised_covariance, compute_rmt_diagnostics
)

# Compute denoised covariance
denoised_cov = rolling_rmt_denoised_covariance(returns, window=126)

# Compute SGI on denoised
denoised_geometry = compute_rolling_geometry_features(denoised_cov, top_k=3)

# Compare
correlation = geometry_df['sgi'].corr(denoised_geometry['sgi'])
print(f"Standard vs Denoised SGI correlation: {correlation:.3f}")
```

---

## Example 8: Cross-market Spillovers

```python
from src.cross_market_spillovers import (
    compute_multi_universe_sgi, compute_spillover_index
)

# Define universes
universes = {
    'us_sectors': ['XLF', 'XLK', 'XLE', 'XLV'],
    'international': ['EFA', 'EEM', 'VGK', 'VPL'],
}

# Compute SGI for each
universe_sgi = compute_multi_universe_sgi(
    universes, '2015-01-01', '2024-01-01'
)

# Spillover analysis
spillover = compute_spillover_index(universe_sgi)
print(f"Total spillover index: {spillover['total_spillover_index']:.1f}%")
```

---

## Example 9: Quantile Regression for VaR

```python
from src.quantile_regression import (
    walk_forward_quantile_regression, compute_var_exceedance_test
)
import numpy as np

# Prepare data
df = features_df.copy()
df['target'] = targets_df['forward_drawdown_20d']
df = df.dropna()

feature_cols = ['sgi', 'realized_vol', 'avg_correlation']

# Walk-forward at 5% quantile
results = walk_forward_quantile_regression(
    df, 'target', feature_cols, quantile=0.05
)

# Test VaR exceedance
if results['predictions']:
    test = compute_var_exceedance_test(
        np.array(results['predictions']),
        np.array(results['actuals']),
        0.05
    )
    print(f"Exceedance rate: {test['exceedance_rate']:.3f} (target: 0.05)")
```

---

## Example 10: Custom Universe

```python
# Create custom config
custom_config = {
    'universe': {
        'name': 'tech_focus',
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
        'market_proxy': 'QQQ',
    },
    'data': {
        'start_date': '2015-01-01',
        'end_date': '2024-01-01',
    },
    'covariance': {
        'estimator': 'ewma',
        'rolling_window': 60,
        'ewma_span': 30,
    },
    'geometry': {
        'top_k': 3,
    },
}

from src.experiments import run_signal_pipeline
results = run_signal_pipeline(custom_config)
```

---

## Authors

**Manu Nicholas Jacob** (manunicholasjacob@gmail.com)  
**Ronit Ghai** (ronitghai@hotmail.com)

---

*See notebooks for more detailed examples with visualizations.*
