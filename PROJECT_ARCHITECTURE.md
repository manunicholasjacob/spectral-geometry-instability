# Project Architecture

## Spectral Geometry Instability (SGI) Research Pipeline

**Authors**: Manu Nicholas Jacob, Ronit Ghai  
**Version**: 0.2.0  
**Last Updated**: 2026

---

## Overview

This document describes the architecture of the SGI research codebase, designed for studying covariance eigenspace rotation as a structural market risk signal.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ run_experiment│  │  run_all.py  │  │  Jupyter Notebooks   │   │
│  │     .py      │  │              │  │                      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
└─────────┼─────────────────┼─────────────────────┼───────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT ORCHESTRATION                      │
│                      src/experiments.py                          │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐   │
│  │ Signal Pipeline│ │Predictive Pipe │ │ Portfolio Pipeline │   │
│  └────────────────┘ └────────────────┘ └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CORE MODULES                               │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ data_loader │ │ covariance  │ │  spectral_  │ │  features   │ │
│ │             │ │             │ │  geometry   │ │             │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │   targets   │ │ predictive_ │ │  portfolio  │ │  backtest   │ │
│ │             │ │   models    │ │             │ │             │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SUPPORT MODULES                             │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │   config    │ │    paths    │ │    utils    │ │   plots     │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │  constants  │ │logging_utils│ │ diagnostics │ │ event_study │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Descriptions

### Core Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `data_loader.py` | Download and cache price data | `load_or_download_prices()` |
| `preprocessing.py` | Compute returns, clean data | `preprocess_data()` |
| `covariance.py` | Rolling covariance estimation | `rolling_covariance_matrices()` |
| `spectral_geometry.py` | **SGI computation** | `compute_rolling_geometry_features()` |
| `features.py` | Feature engineering | `build_feature_dataframe()` |
| `targets.py` | Forward-looking targets | `build_target_dataframe()` |
| `predictive_models.py` | Walk-forward modeling | `run_walk_forward_experiment()` |
| `portfolio.py` | Portfolio construction | `generate_weight_schedule()` |
| `backtest.py` | Backtesting engine | `run_rebalancing_backtest()` |

### Support Modules

| Module | Purpose |
|--------|---------|
| `config.py` | YAML config loading and merging |
| `paths.py` | Centralized path management |
| `constants.py` | Default values, crisis dates |
| `utils.py` | Helper functions |
| `logging_utils.py` | Experiment logging |
| `diagnostics.py` | Data quality checks |
| `plots.py` | Visualization |
| `event_study.py` | Crisis event analysis |
| `bootstrap.py` | Statistical inference |

---

## Data Flow

```
1. CONFIG LOADING
   configs/*.yaml → config.py → Experiment Config Dict

2. DATA PIPELINE
   yfinance → data_loader.py → preprocessing.py → Returns DataFrame

3. COVARIANCE PIPELINE
   Returns → covariance.py → Dict[Timestamp, np.ndarray]

4. GEOMETRY PIPELINE
   Covariance Matrices → spectral_geometry.py → Geometry DataFrame
   (eigendecomposition → principal angles → SGI metrics)

5. FEATURE PIPELINE
   Returns + Geometry → features.py → Full Feature DataFrame

6. TARGET PIPELINE
   Prices + Returns → targets.py → Target DataFrame

7. MODELING PIPELINE
   Features + Targets → predictive_models.py → Walk-Forward Results

8. PORTFOLIO PIPELINE
   Covariance + Geometry → portfolio.py → Weight Schedule
   Weights + Prices → backtest.py → Performance Metrics
```

---

## Configuration System

### Config Hierarchy

```
base.yaml (defaults)
    ↓
universe_*.yaml (universe-specific overrides)
    ↓
Command-line overrides (--output-tag)
```

### Config Sections

- **project**: Name, version, seed
- **data**: Date range, return type
- **universe**: Tickers, market proxy
- **covariance**: Estimator, window
- **geometry**: Top-k
- **targets**: Horizons, thresholds
- **modeling**: Model type, feature sets
- **portfolio**: Strategies, constraints
- **backtest**: Rebalance frequency, costs
- **outputs**: Save paths, formats

---

## Key Algorithms

### SGI Computation

```python
# For each pair of consecutive covariance matrices:
1. Eigendecompose: Σ_t → (λ, V)
2. Extract top-k: V_t[:, :k]
3. Compute principal angles: θ = subspace_angles(V_t, V_{t-1})
4. Aggregate: SGI = sqrt(sum(θ²))
```

### Walk-Forward Evaluation

```python
# Rolling train/test splits:
for train_window, test_window in walk_forward_splits(data):
    model.fit(train_window)
    predictions = model.predict(test_window)
    metrics.append(evaluate(predictions, actuals))
```

### SGI-Conditioned Portfolio

```python
# Adaptive shrinkage based on SGI:
if sgi > threshold:
    alpha = high_shrinkage
else:
    alpha = low_shrinkage
    
shrunk_cov = (1 - alpha) * cov + alpha * diagonal(cov)
weights = min_variance(shrunk_cov)
```

---

## Output Structure

```
outputs/
├── signal_<tag>_<timestamp>/
│   ├── geometry_features.csv
│   ├── all_features.csv
│   └── config.json
├── predictive_<tag>_<timestamp>/
│   ├── walk_forward_results.json
│   └── full_summary.csv
├── portfolio_<tag>_<timestamp>/
│   ├── strategy_comparison.csv
│   └── <strategy>_metrics.json
├── figures/
│   ├── sgi_timeseries_*.png
│   └── portfolio_comparison_*.png
└── logs/
    └── <experiment>_<timestamp>.log
```

---

## Extension Points

1. **New Covariance Estimators**: Add to `covariance.py`
2. **New SGI Variants**: Add to `spectral_geometry.py`
3. **New Features**: Add to `features.py`
4. **New Targets**: Add to `targets.py`
5. **New Portfolio Strategies**: Add to `portfolio.py`
6. **New Universes**: Create new `configs/universe_*.yaml`

---

## Dependencies

- **Data**: pandas, numpy, yfinance
- **Math**: scipy, scikit-learn, statsmodels
- **Optimization**: cvxpy
- **Visualization**: matplotlib
- **Config**: pyyaml
- **Utilities**: tqdm, joblib, pathlib
