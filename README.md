# Spectral Geometry Instability (SGI)

## Eigenvector Instability as a Structural Risk Signal in Portfolio Allocation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manunicholasjacob/spectral-geometry-instability)

---

## Overview

This repository implements a **research-grade computational finance pipeline** for studying **Spectral Geometry Instability (SGI)** — a novel approach to measuring structural market risk through the rotation of dominant covariance eigenspaces.

Traditional risk models focus on:
- Volatility magnitude
- Correlation levels
- Eigenvalue concentration (absorption ratio)

**SGI adds a new dimension**: measuring whether the market's internal risk geometry is reorganizing, independent of volatility magnitude.

---

## Research Motivation

> **Core Hypothesis**: Large rotations of the dominant covariance eigenspace indicate structural market instability, degradation of diversification assumptions, elevated fragility, and potentially worse out-of-sample portfolio behavior.

The key insight is:
- A covariance matrix can remain high-variance yet structurally stable
- Or moderate-variance yet geometrically unstable
- SGI isolates the latter phenomenon

---

## Key Idea: Spectral Geometry Instability

Given rolling covariance matrices Σ_t, we:
1. Extract top-k eigenvectors V_t
2. Compute principal angles between consecutive eigenspaces
3. Aggregate angles into SGI metrics

**Unweighted SGI**:
```
SGI_t = sqrt(Σ θ_i²)
```

**Weighted SGI**:
```
WSGI_t = sqrt(Σ λ̃_i · θ_i²)
```

Where θ_i are principal angles and λ̃_i are normalized eigenvalue weights.

---

## Repository Structure

```
spectral_geometry_instability/
│
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── setup.sh                  # Setup script
├── run_experiment.py         # Main experiment runner
├── run_all.py                # Run all experiments
│
├── configs/                  # YAML configuration files
│   ├── base.yaml
│   ├── universe_sector_etfs.yaml
│   ├── universe_multiasset.yaml
│   └── universe_largecap50.yaml
│
├── src/                      # Core Python modules
│   ├── config.py             # Config loading
│   ├── constants.py          # Default values
│   ├── paths.py              # Path management
│   ├── utils.py              # Utilities
│   ├── logging_utils.py      # Logging
│   ├── data_loader.py        # Data download/caching
│   ├── preprocessing.py      # Return computation
│   ├── covariance.py         # Covariance estimators
│   ├── spectral_geometry.py  # SGI computation
│   ├── features.py           # Feature engineering
│   ├── targets.py            # Target variables
│   ├── predictive_models.py  # Walk-forward models
│   ├── portfolio.py          # Portfolio construction
│   ├── backtest.py           # Backtesting engine
│   ├── plots.py              # Visualization
│   ├── event_study.py        # Event analysis
│   ├── experiments.py        # Pipeline orchestration
│   ├── diagnostics.py        # Data quality checks
│   ├── bootstrap.py          # Statistical inference
│   ├── vix_integration.py    # VIX control variable
│   ├── rmt_denoising.py      # RMT eigenvalue filtering
│   ├── quantile_regression.py # Tail risk prediction
│   ├── multiscale_sgi.py     # Multi-scale analysis
│   ├── cross_market_spillovers.py # Contagion analysis
│   ├── factor_model.py       # Factor integration
│   └── crypto_universe.py    # Crypto extension
│
├── experiments/              # Experiment scripts
│   ├── run_signal_experiment.py
│   ├── run_predictive_experiment.py
│   ├── run_portfolio_experiment.py
│   ├── run_event_study.py
│   ├── run_vix_analysis.py
│   ├── run_rmt_analysis.py
│   ├── run_quantile_analysis.py
│   ├── run_multiscale_analysis.py
│   ├── run_spillover_analysis.py
│   ├── run_factor_analysis.py
│   ├── run_crypto_analysis.py
│   └── sweep_*.py            # Robustness sweeps
│
├── docs/                     # Documentation
│   ├── USER_GUIDE.md         # Complete user guide
│   ├── QUICKSTART.md         # Quick start guide
│   ├── API_REFERENCE.md      # API documentation
│   └── EXAMPLES.md           # Code examples
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_validation.ipynb
│   ├── 02_sgi_exploration.ipynb
│   ├── 03_predictive_tests.ipynb
│   ├── 04_portfolio_tests.ipynb
│   ├── 05_event_studies.ipynb
│   └── 06_robustness.ipynb
│
├── data/                     # Data storage (gitignored)
│   ├── raw/
│   └── processed/
│
├── outputs/                  # Experiment outputs (gitignored)
│   ├── figures/
│   ├── tables/
│   ├── logs/
│   └── diagnostics/
│
└── paper_assets/             # Publication materials
    ├── figure_manifest.md
    └── results_summary.md
```

---

## Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/manunicholasjacob/spectral-geometry-instability.git
cd spectral-geometry-instability

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run setup
bash setup.sh
```

### Google Colab

```python
!git clone https://github.com/manunicholasjacob/spectral-geometry-instability.git
%cd spectral-geometry-instability
!pip install -r requirements.txt
```

---

## Running Experiments

### Signal Exploration
```bash
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode signal
```

### Predictive Tests
```bash
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode predictive
```

### Portfolio Backtest
```bash
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode portfolio
```

### Event Studies
```bash
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode event
```

### Run All Experiments
```bash
python run_all.py
```

---

## Research Questions

1. **Predictive Fragility**: Does high SGI predict future market stress?
2. **Incremental Value**: Does SGI add information beyond volatility and correlation?
3. **Covariance Failure**: Does SGI predict when covariance forecasts degrade?
4. **Allocation Relevance**: Can SGI-aware portfolios improve robustness?
5. **Generality**: Does SGI work across different asset universes?

---

## Data Universes

| Universe | Assets | Purpose |
|----------|--------|---------|
| Sector ETFs | XLF, XLK, XLE, XLV, XLY, XLP, XLI, XLU, XLB, XLRE | Baseline interpretable universe |
| Multi-Asset | SPY, EFA, EEM, TLT, IEF, LQD, HYG, GLD, DBC, VNQ | Cross-asset class testing |
| Large-Cap 50 | Top 50 S&P 500 stocks | Rich cross-sectional structure |

---

## Current Status

- [x] Repository scaffold
- [x] Config system
- [x] Data loading and preprocessing
- [x] Covariance estimation (sample, EWMA, Ledoit-Wolf)
- [x] SGI computation
- [x] Baseline features and targets
- [x] Predictive modeling layer
- [x] Portfolio construction
- [x] Backtesting engine
- [x] Event studies
- [x] Plotting utilities
- [ ] Full robustness sweeps (in progress)
- [ ] Publication-ready figures
- [ ] Paper draft

---

## Advanced Features (NEW)

All previously planned extensions are now implemented:

| Feature | Module | Description |
|---------|--------|-------------|
| **VIX Integration** | `src/vix_integration.py` | VIX as control variable, SGI-VIX relationship analysis |
| **RMT Denoising** | `src/rmt_denoising.py` | Marcenko-Pastur eigenvalue filtering |
| **Quantile Regression** | `src/quantile_regression.py` | Tail risk prediction, VaR exceedance tests |
| **Multi-scale SGI** | `src/multiscale_sgi.py` | SGI at multiple windows, term structure |
| **Cross-market Spillovers** | `src/cross_market_spillovers.py` | Diebold-Yilmaz spillover index, Granger causality |
| **Factor Model Integration** | `src/factor_model.py` | Factor loading instability, residual SGI |
| **Crypto Universe** | `src/crypto_universe.py` | Cryptocurrency SGI analysis |

### Running Advanced Experiments

```bash
# VIX Analysis
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

---

## Documentation

Comprehensive documentation is available in the `docs/` folder:

| Document | Description |
|----------|-------------|
| [User Guide](docs/USER_GUIDE.md) | Complete guide to using the SGI pipeline |
| [Quick Start](docs/QUICKSTART.md) | Get running in 5 minutes |
| [API Reference](docs/API_REFERENCE.md) | Full API documentation |
| [Examples](docs/EXAMPLES.md) | Practical code examples |

---

## Authors

**Manu Nicholas Jacob**  
Email: manunicholasjacob@gmail.com

**Ronit Ghai**  
Email: ronitghai@hotmail.com

---

## Citation

If you use this code in your research, please cite:

```
@software{sgi2026,
  title={Spectral Geometry Instability: Subspace Rotation as a Structural Risk Signal},
  author={Jacob, Manu Nicholas and Ghai, Ronit},
  year={2026},
  url={https://github.com/manunicholasjacob/spectral-geometry-instability}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Manu Nicholas Jacob and Ronit Ghai
