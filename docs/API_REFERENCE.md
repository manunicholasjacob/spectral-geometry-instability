# API Reference

## Spectral Geometry Instability (SGI) - Complete API Documentation

---

## Table of Contents

1. [Core Modules](#core-modules)
2. [Data Modules](#data-modules)
3. [Analysis Modules](#analysis-modules)
4. [Advanced Modules](#advanced-modules)
5. [Utility Modules](#utility-modules)

---

## Core Modules

### `src/spectral_geometry.py`

The core module for SGI computation.

#### `eigendecompose_covariance(cov)`
Eigendecompose a covariance matrix, sorted by descending eigenvalues.

**Parameters:**
- `cov` (np.ndarray): Covariance matrix (n x n)

**Returns:**
- Tuple of (eigenvalues, eigenvectors) sorted descending

---

#### `compute_principal_angles(v_current, v_previous)`
Compute principal angles between two subspaces.

**Parameters:**
- `v_current` (np.ndarray): Current eigenvector matrix (n x k)
- `v_previous` (np.ndarray): Previous eigenvector matrix (n x k)

**Returns:**
- np.ndarray: Array of principal angles in radians

---

#### `compute_sgi_from_angles(angles)`
Compute unweighted SGI from principal angles.

**Parameters:**
- `angles` (np.ndarray): Array of principal angles

**Returns:**
- float: SGI value

---

#### `compute_rolling_geometry_features(cov_matrices, top_k)`
Compute rolling geometry features from covariance matrices.

**Parameters:**
- `cov_matrices` (Dict[pd.Timestamp, np.ndarray]): Dictionary of covariance matrices
- `top_k` (int): Number of top eigenvectors

**Returns:**
- pd.DataFrame: DataFrame with geometry features

---

### `src/covariance.py`

Covariance estimation module.

#### `rolling_covariance_matrices(returns, window, method, **kwargs)`
Compute rolling covariance matrices.

**Parameters:**
- `returns` (pd.DataFrame): Returns DataFrame
- `window` (int): Rolling window size
- `method` (str): Estimation method ('sample', 'ewma', 'ledoit_wolf', 'oas')

**Returns:**
- Dict[pd.Timestamp, np.ndarray]: Dictionary of covariance matrices

---

#### `compute_sample_cov(window_returns)`
Compute sample covariance matrix.

---

#### `compute_ewma_cov(window_returns, span, min_periods)`
Compute EWMA covariance matrix.

---

#### `compute_ledoit_wolf_cov(window_returns)`
Compute Ledoit-Wolf shrinkage covariance.

---

### `src/predictive_models.py`

Predictive modeling module.

#### `run_walk_forward_experiment(df, target_col, feature_sets, model_type, config)`
Run walk-forward experiment comparing feature sets.

**Parameters:**
- `df` (pd.DataFrame): Dataset with features and target
- `target_col` (str): Target column name
- `feature_sets` (Dict[str, List[str]]): Feature set definitions
- `model_type` (str): 'linear' or 'logistic'
- `config` (Dict): Configuration

**Returns:**
- Dict: Walk-forward results

---

#### `summarize_walk_forward_results(results)`
Summarize walk-forward results into comparison table.

---

#### `compute_incremental_r2(results, baseline_set, full_set)`
Compute incremental R² from adding features.

---

### `src/portfolio.py`

Portfolio construction module.

#### `generate_weight_schedule(returns, cov_matrices, geometry_df, strategy, config, rebalance_freq)`
Generate portfolio weight schedule.

**Parameters:**
- `returns` (pd.DataFrame): Returns DataFrame
- `cov_matrices` (Dict): Covariance matrices
- `geometry_df` (pd.DataFrame): Geometry features
- `strategy` (str): Strategy name
- `config` (Dict): Configuration
- `rebalance_freq` (str): Rebalance frequency

**Returns:**
- pd.DataFrame: Weight schedule

**Available Strategies:**
- `equal_weight`
- `inverse_vol`
- `min_variance`
- `shrinkage_minvar`
- `sgi_conditioned_minvar`

---

### `src/backtest.py`

Backtesting module.

#### `run_rebalancing_backtest(prices, weight_schedule, transaction_cost_bps)`
Run a full rebalancing backtest.

**Parameters:**
- `prices` (pd.DataFrame): Price DataFrame
- `weight_schedule` (pd.DataFrame): Weight schedule
- `transaction_cost_bps` (float): Transaction cost in basis points

**Returns:**
- Dict: Backtest results with returns and metrics

---

#### `compute_backtest_metrics(portfolio_returns)`
Compute comprehensive backtest metrics.

**Returns:**
- Dict with: total_return, cagr, volatility, sharpe, sortino, max_drawdown, calmar

---

## Data Modules

### `src/data_loader.py`

#### `load_or_download_prices(tickers, start_date, end_date, cache)`
Load prices from cache or download.

---

#### `download_price_data(tickers, start_date, end_date)`
Download price data from yfinance.

---

### `src/preprocessing.py`

#### `preprocess_data(prices, return_method, max_missing_frac, save_outputs)`
Full preprocessing pipeline.

**Returns:**
- Tuple of (cleaned_prices, returns, metadata)

---

#### `compute_returns(prices, method)`
Compute returns from prices.

**Parameters:**
- `method` (str): 'log' or 'simple'

---

## Analysis Modules

### `src/features.py`

#### `build_feature_dataframe(returns, geometry_df, config, prices)`
Build comprehensive feature DataFrame.

---

#### `get_feature_columns(feature_set, config)`
Get feature columns for named feature set.

**Available Sets:**
- `baseline_only`
- `sgi_only`
- `baseline_plus_sgi`
- `full`

---

### `src/targets.py`

#### `build_target_dataframe(prices, returns, cov_matrices, config)`
Build comprehensive target DataFrame.

**Targets Include:**
- Forward drawdown
- Forward volatility
- Forward correlation
- Drawdown events (binary)
- Correlation spikes (binary)
- Volatility spikes (binary)

---

### `src/event_study.py`

#### `run_event_study(geometry_df, features_df, crisis_dates, pre_days, post_days)`
Run comprehensive event study.

---

#### `extract_event_window(series, event_date, pre_days, post_days)`
Extract data around an event date.

---

## Advanced Modules

### `src/vix_integration.py`

#### `download_vix_data(start_date, end_date)`
Download VIX index data.

---

#### `compute_vix_features(vix_data)`
Compute VIX-derived features.

**Features:**
- vix_level, vix_log, vix_change
- vix_ma_5, vix_ma_20, vix_ma_60
- vix_percentile, vix_zscore
- vix_regime_high, vix_regime_very_high

---

#### `compute_sgi_vix_relationship(geometry_df, vix_features)`
Analyze SGI-VIX relationship.

---

#### `compute_sgi_orthogonal_to_vix(sgi, vix)`
Compute SGI component orthogonal to VIX.

---

### `src/rmt_denoising.py`

#### `denoise_covariance_rmt(cov, n_observations, method)`
Denoise covariance matrix using RMT.

**Methods:**
- `constant_residual`: Replace noise eigenvalues with average
- `shrink_noise`: Shrink noise eigenvalues
- `targeted_shrinkage`: Position-based shrinkage

---

#### `compute_rmt_diagnostics(cov, n_observations)`
Compute RMT-based diagnostics.

---

#### `rolling_rmt_denoised_covariance(returns, window, method)`
Compute rolling RMT-denoised covariance matrices.

---

### `src/quantile_regression.py`

#### `fit_quantile_regression(X, y, quantile)`
Fit quantile regression model.

---

#### `walk_forward_quantile_regression(df, target_col, feature_cols, quantile)`
Walk-forward quantile regression evaluation.

---

#### `compute_var_exceedance_test(predictions, actuals, quantile)`
Test VaR exceedance (Kupiec test).

---

### `src/multiscale_sgi.py`

#### `compute_multiscale_sgi(returns, windows, top_k, estimator)`
Compute SGI at multiple time scales.

---

#### `compute_scale_lead_lag(multiscale_df, windows, max_lag)`
Compute lead-lag relationships between scales.

---

#### `compute_regime_from_multiscale(multiscale_df, windows, threshold_percentile)`
Compute market regime from multi-scale SGI.

**Regimes:**
- `stable`: No scales elevated
- `transitioning`: Some scales elevated
- `elevated`: Most scales elevated
- `crisis`: All scales elevated

---

### `src/cross_market_spillovers.py`

#### `compute_multi_universe_sgi(universes, start_date, end_date)`
Compute SGI for multiple universes.

---

#### `compute_spillover_index(universe_sgi, metric, forecast_horizon)`
Compute Diebold-Yilmaz style spillover index.

---

#### `compute_granger_causality(universe_sgi, metric, max_lag)`
Compute Granger causality between universes.

---

#### `compute_contagion_events(universe_sgi, metric, threshold_percentile)`
Identify contagion events.

---

### `src/factor_model.py`

#### `compute_rolling_factor_loadings(returns, factors, window)`
Compute rolling factor loadings (betas).

---

#### `compute_factor_loading_instability(loadings)`
Compute instability of factor loadings.

---

#### `compute_factor_adjusted_sgi(returns, factors, window, top_k)`
Compute SGI on factor-adjusted (residual) returns.

---

### `src/crypto_universe.py`

#### `compute_crypto_sgi(tickers, start_date, end_date, window, top_k)`
Compute SGI for crypto universe.

---

#### `compare_crypto_vs_traditional_sgi(crypto_geometry, traditional_geometry)`
Compare crypto and traditional market SGI.

---

#### `run_crypto_event_study(geometry_df, events)`
Run event study for crypto-specific events.

---

## Utility Modules

### `src/config.py`

#### `load_experiment_config(config_path)`
Load and merge configuration files.

---

### `src/paths.py`

#### `get_project_root()`
Get project root directory.

---

#### `get_outputs_dir()`, `get_figures_dir()`, etc.
Get various output directories.

---

### `src/utils.py`

#### `save_json(data, path)`, `load_json(path)`
Save/load JSON files.

---

#### `compute_sharpe_ratio(returns, rf, periods_per_year)`
Compute Sharpe ratio.

---

#### `compute_max_drawdown(returns)`
Compute maximum drawdown.

---

### `src/plots.py`

#### `plot_sgi_time_series(geometry_df, crisis_dates, save_path)`
Plot SGI time series with crisis shading.

---

#### `plot_sgi_vs_volatility(features_df, save_path)`
Plot SGI vs realized volatility comparison.

---

#### `plot_portfolio_comparison(portfolio_returns, save_path)`
Plot portfolio performance comparison.

---

### `src/diagnostics.py`

#### `check_price_data_quality(prices)`
Comprehensive price data quality check.

---

#### `check_covariance_matrix(cov, name)`
Check covariance matrix validity.

---

#### `run_all_diagnostics(prices, returns, geometry_df)`
Run all diagnostic checks.

---

## Constants

### `src/constants.py`

```python
SECTOR_ETF_TICKERS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

CRISIS_DATES = {
    'gfc_2008': {'start': '2008-09-01', 'peak_date': '2008-10-10', 'end': '2009-03-31'},
    'covid_2020': {'start': '2020-02-19', 'peak_date': '2020-03-23', 'end': '2020-04-30'},
    ...
}

SUPPORTED_ESTIMATORS = ['sample', 'ewma', 'ledoit_wolf', 'oas']
```

---

## Authors

**Manu Nicholas Jacob** (manunicholasjacob@gmail.com)  
**Ronit Ghai** (ronitghai@hotmail.com)

---

*For detailed docstrings, see the source code.*
