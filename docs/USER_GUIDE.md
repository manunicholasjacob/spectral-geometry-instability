# Spectral Geometry Instability (SGI) - User Guide

## Complete Guide to Using the SGI Research Pipeline

**Version**: 0.2.0  
**Last Updated**: 2024

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Configuration System](#configuration-system)
6. [Running Experiments](#running-experiments)
7. [Module Reference](#module-reference)
8. [Advanced Features](#advanced-features)
9. [Notebooks Guide](#notebooks-guide)
10. [Interpreting Results](#interpreting-results)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)

---

## Introduction

### What is SGI?

**Spectral Geometry Instability (SGI)** is a novel measure of structural market risk based on the rotation of covariance eigenspaces over time. Unlike traditional risk metrics that focus on volatility or correlation levels, SGI captures the *geometric instability* of the covariance structure.

### Key Insight

When the dominant eigenvectors of a covariance matrix rotate significantly from one period to the next, it indicates:
- Breakdown of historical diversification assumptions
- Structural regime change in market dynamics
- Elevated fragility and potential for cascading losses

### Mathematical Foundation

```
SGI = √(Σ θᵢ²)
```

Where θᵢ are the principal angles between consecutive top-k eigenspaces.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/manunicholasjacob/spectral-geometry-instability.git
cd spectral-geometry-instability
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "from src.spectral_geometry import compute_rolling_geometry_features; print('Installation successful!')"
```

### Google Colab Setup

```python
!git clone https://github.com/manunicholasjacob/spectral-geometry-instability.git
%cd spectral-geometry-instability
!pip install -r requirements.txt
```

---

## Quick Start

### Run Your First Experiment

```bash
# Signal exploration (computes SGI and visualizations)
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode signal

# Predictive modeling (tests if SGI predicts future stress)
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode predictive

# Portfolio backtesting (tests SGI-aware strategies)
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode portfolio
```

### Python API Quick Start

```python
from src.config import load_experiment_config
from src.experiments import run_signal_pipeline

# Load configuration
config = load_experiment_config('configs/universe_sector_etfs.yaml')

# Run pipeline
results = run_signal_pipeline(config)

# Access results
geometry_df = results['geometry_df']
features_df = results['features_df']

# View SGI time series
print(geometry_df[['sgi', 'weighted_sgi', 'absorption_ratio']].describe())
```

---

## Core Concepts

### 1. Covariance Eigenspace

The covariance matrix Σ can be decomposed as:
```
Σ = V Λ Vᵀ
```
Where V contains eigenvectors and Λ contains eigenvalues. The top-k eigenvectors span the "dominant eigenspace."

### 2. Principal Angles

Principal angles measure the rotation between two subspaces, independent of:
- Sign flips in eigenvectors
- Basis choice
- Ordering

### 3. SGI Variants

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **SGI** | √(Σ θᵢ²) | Unweighted rotation magnitude |
| **Weighted SGI** | √(Σ wᵢθᵢ²) | Eigenvalue-weighted rotation |
| **Max Angle** | max(θᵢ) | Largest single rotation |
| **Mean Angle** | mean(θᵢ) | Average rotation |

### 4. Absorption Ratio

Fraction of total variance explained by top-k eigenvalues:
```
AR = Σᵢ₌₁ᵏ λᵢ / Σⱼ λⱼ
```

High AR indicates concentrated risk.

---

## Configuration System

### Config File Structure

```yaml
# configs/my_experiment.yaml

inherits: base.yaml  # Inherit from base config

universe:
  name: "my_universe"
  tickers:
    - "SPY"
    - "QQQ"
    - "IWM"
  market_proxy: "SPY"

data:
  start_date: "2005-01-01"
  end_date: "2024-12-31"
  return_type: "log"

covariance:
  estimator: "sample"  # sample, ewma, ledoit_wolf, oas
  rolling_window: 126

geometry:
  top_k: 3

outputs:
  experiment_tag: "my_experiment"
```

### Available Universes

| Config File | Description | Assets |
|-------------|-------------|--------|
| `universe_sector_etfs.yaml` | S&P 500 Sector ETFs | 11 |
| `universe_multiasset.yaml` | Multi-asset ETFs | 8 |
| `universe_largecap50.yaml` | Large-cap stocks | 50 |
| `universe_crypto.yaml` | Cryptocurrencies | 15 |

### Covariance Estimators

| Estimator | Description | Best For |
|-----------|-------------|----------|
| `sample` | Standard sample covariance | Large T/N ratio |
| `ewma` | Exponentially weighted | Time-varying dynamics |
| `ledoit_wolf` | Shrinkage estimator | Small samples |
| `oas` | Oracle Approximating Shrinkage | General use |

---

## Running Experiments

### Command Line Interface

```bash
# Basic usage
python run_experiment.py --config <config_file> --mode <mode>

# Available modes
--mode signal      # Signal exploration
--mode predictive  # Predictive modeling
--mode portfolio   # Portfolio backtesting
--mode event       # Event studies
--mode full        # Run all modes

# Optional arguments
--output-tag <tag>  # Custom output tag
```

### Experiment Scripts

```bash
# VIX Integration
python experiments/run_vix_analysis.py

# RMT Denoising
python experiments/run_rmt_analysis.py

# Quantile Regression
python experiments/run_quantile_analysis.py

# Multi-scale SGI
python experiments/run_multiscale_analysis.py

# Cross-market Spillovers
python experiments/run_spillover_analysis.py

# Factor Model Integration
python experiments/run_factor_analysis.py

# Crypto Analysis
python experiments/run_crypto_analysis.py
```

### Robustness Sweeps

```bash
# Sweep over rolling windows
python experiments/run_robustness_sweep.py --sweep windows

# Sweep over top-k values
python experiments/run_robustness_sweep.py --sweep top_k

# Sweep over covariance estimators
python experiments/run_robustness_sweep.py --sweep estimators

# Run all sweeps
python experiments/run_robustness_sweep.py --sweep all
```

---

## Module Reference

### Core Modules

#### `src/spectral_geometry.py`
The heart of the project - computes SGI metrics.

```python
from src.spectral_geometry import (
    compute_rolling_geometry_features,
    compute_sgi_derived_features,
    compute_principal_angles,
    eigendecompose_covariance
)

# Compute SGI from covariance matrices
geometry_df = compute_rolling_geometry_features(cov_matrices, top_k=3)

# Add derived features (z-scores, percentiles, etc.)
geometry_df = compute_sgi_derived_features(geometry_df)
```

#### `src/covariance.py`
Rolling covariance estimation.

```python
from src.covariance import rolling_covariance_matrices

# Compute rolling covariance
cov_matrices = rolling_covariance_matrices(
    returns, 
    window=126, 
    method='sample'
)
```

#### `src/predictive_models.py`
Walk-forward predictive modeling.

```python
from src.predictive_models import run_walk_forward_experiment

results = run_walk_forward_experiment(
    df, 
    target_col='target',
    feature_sets={'baseline': [...], 'with_sgi': [...]},
    model_type='linear'
)
```

#### `src/portfolio.py`
Portfolio construction strategies.

```python
from src.portfolio import generate_weight_schedule

weights = generate_weight_schedule(
    returns, cov_matrices, geometry_df,
    strategy='sgi_conditioned_minvar',
    config=config
)
```

### Advanced Modules

#### `src/vix_integration.py`
VIX as control variable.

```python
from src.vix_integration import (
    download_vix_data,
    compute_vix_features,
    compute_sgi_vix_relationship
)
```

#### `src/rmt_denoising.py`
Random Matrix Theory denoising.

```python
from src.rmt_denoising import (
    denoise_covariance_rmt,
    rolling_rmt_denoised_covariance
)
```

#### `src/quantile_regression.py`
Quantile regression for tail risk.

```python
from src.quantile_regression import (
    walk_forward_quantile_regression,
    fit_multiple_quantiles
)
```

#### `src/multiscale_sgi.py`
Multi-scale SGI analysis.

```python
from src.multiscale_sgi import compute_multiscale_sgi

multiscale_df = compute_multiscale_sgi(
    returns, 
    windows=[60, 126, 252, 504]
)
```

#### `src/cross_market_spillovers.py`
Cross-market contagion analysis.

```python
from src.cross_market_spillovers import (
    compute_multi_universe_sgi,
    compute_spillover_index
)
```

#### `src/crypto_universe.py`
Cryptocurrency analysis.

```python
from src.crypto_universe import compute_crypto_sgi

geometry_df, features_df, metadata = compute_crypto_sgi()
```

---

## Advanced Features

### 1. VIX Integration

Control for market-wide fear:

```python
from src.vix_integration import *

# Download VIX
vix_data = download_vix_data("2005-01-01", "2024-12-31")

# Compute VIX features
vix_features = compute_vix_features(vix_data)

# Analyze SGI-VIX relationship
relationship = compute_sgi_vix_relationship(geometry_df, vix_features)

# Orthogonalize SGI to VIX
sgi_orthogonal = compute_sgi_orthogonal_to_vix(sgi, vix)
```

### 2. RMT Denoising

Filter noise from covariance:

```python
from src.rmt_denoising import *

# Denoise single matrix
denoised_cov = denoise_covariance_rmt(
    cov, n_observations, method='constant_residual'
)

# Rolling denoised covariance
denoised_matrices = rolling_rmt_denoised_covariance(
    returns, window=126
)
```

### 3. Quantile Regression

Predict tail risk:

```python
from src.quantile_regression import *

# Walk-forward quantile regression
results = walk_forward_quantile_regression(
    df, 'target', feature_cols, quantile=0.05
)

# VaR exceedance test
test = compute_var_exceedance_test(predictions, actuals, 0.05)
```

### 4. Multi-scale Analysis

Analyze SGI at multiple horizons:

```python
from src.multiscale_sgi import *

# Compute at multiple windows
multiscale_df = compute_multiscale_sgi(
    returns, windows=[60, 126, 252, 504]
)

# Analyze scale relationships
lead_lag = compute_scale_lead_lag(multiscale_df, windows)

# Detect regime from multi-scale
regime = compute_regime_from_multiscale(multiscale_df, windows)
```

### 5. Cross-market Spillovers

Analyze contagion:

```python
from src.cross_market_spillovers import *

# Compute SGI for multiple universes
universe_sgi = compute_multi_universe_sgi(universes)

# Spillover index
spillover = compute_spillover_index(universe_sgi)

# Granger causality
granger = compute_granger_causality(universe_sgi)
```

---

## Notebooks Guide

### Available Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_validation.ipynb` | Validate price data, compute returns |
| `02_sgi_exploration.ipynb` | Explore SGI signal, visualize |
| `03_predictive_tests.ipynb` | Test predictive power of SGI |
| `04_portfolio_tests.ipynb` | Backtest SGI-aware portfolios |
| `05_event_studies.ipynb` | Analyze SGI around crises |
| `06_robustness.ipynb` | Test parameter sensitivity |

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or JupyterLab
jupyter lab
```

---

## Interpreting Results

### SGI Time Series

- **Spikes**: Indicate structural instability
- **Sustained elevation**: Regime change in progress
- **Low values**: Stable covariance structure

### Predictive Results

| Metric | Interpretation |
|--------|----------------|
| Incremental R² > 0 | SGI adds predictive value |
| Positive in most folds | Robust signal |
| Coefficient significant | Statistical evidence |

### Portfolio Results

| Metric | Good Value |
|--------|------------|
| Sharpe > 0.5 | Reasonable risk-adjusted return |
| Max DD < 20% | Controlled drawdowns |
| SGI strategy > baseline | SGI conditioning helps |

### Event Study Results

- **Pre-event spike**: SGI has leading indicator properties
- **Event spike**: SGI captures crisis
- **Post-event elevation**: Structural damage persists

---

## Troubleshooting

### Common Issues

#### "No data downloaded"
```python
# Check ticker validity
import yfinance as yf
data = yf.download("SPY", start="2020-01-01", end="2024-01-01")
print(data.head())
```

#### "Singular covariance matrix"
- Increase rolling window
- Use shrinkage estimator (`ledoit_wolf`)
- Remove assets with missing data

#### "Insufficient data for walk-forward"
- Extend date range
- Reduce train/test periods
- Use smaller universe

#### "Memory error"
- Reduce number of assets
- Use smaller windows
- Process in chunks

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.experiments import run_signal_pipeline
results = run_signal_pipeline(config)
```

---

## FAQ

### Q: How do I choose top_k?

**A**: Start with k=3 (captures ~60-80% of variance for typical portfolios). Increase for larger universes. Use robustness sweeps to test sensitivity.

### Q: Which covariance estimator should I use?

**A**: 
- `sample`: When T >> N (many observations, few assets)
- `ewma`: When dynamics change over time
- `ledoit_wolf`: When T ≈ N (small samples)

### Q: How do I add my own assets?

**A**: Create a new config file:
```yaml
inherits: base.yaml
universe:
  name: "my_assets"
  tickers:
    - "AAPL"
    - "GOOGL"
    - "MSFT"
```

### Q: Can I use intraday data?

**A**: Yes, but adjust:
- Rolling window (e.g., 60 bars instead of 126 days)
- Annualization factors
- Transaction cost assumptions

### Q: How do I cite this work?

**A**: See the citation section in README.md.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/manunicholasjacob/spectral-geometry-instability/issues)
- **Documentation**: This guide + inline docstrings
- **Examples**: `notebooks/` directory

---

## Authors

**Manu Nicholas Jacob** (manunicholasjacob@gmail.com)  
**Ronit Ghai** (ronitghai@hotmail.com)

---

*Happy researching!*
