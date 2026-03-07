# Table Manifest

This document tracks all tables generated for the SGI research paper.

## Main Tables

| Table | Description | Script/Notebook | Status |
|-------|-------------|-----------------|--------|
| Tab 1 | Data summary statistics | `01_data_validation.ipynb` | Pending |
| Tab 2 | SGI summary statistics | `02_sgi_exploration.ipynb` | Pending |
| Tab 3 | Feature correlations | `02_sgi_exploration.ipynb` | Pending |
| Tab 4 | Predictive model comparison (R²) | `03_predictive_tests.ipynb` | Pending |
| Tab 5 | Incremental R² from SGI | `03_predictive_tests.ipynb` | Pending |
| Tab 6 | Portfolio performance comparison | `04_portfolio_tests.ipynb` | Pending |
| Tab 7 | Crisis period performance | `04_portfolio_tests.ipynb` | Pending |

## Supplementary Tables

| Table | Description | Script/Notebook | Status |
|-------|-------------|-----------------|--------|
| S1 | Window sensitivity results | `experiments/sweep_windows.py` | Pending |
| S2 | Top-k sensitivity results | `experiments/sweep_topk.py` | Pending |
| S3 | Estimator comparison | `experiments/sweep_estimators.py` | Pending |
| S4 | Universe comparison | `experiments/sweep_universes.py` | Pending |
| S5 | Transaction cost sweep | `experiments/sweep_thresholds.py` | Pending |
| S6 | Event study statistics | `05_event_studies.ipynb` | Pending |

## Table Specifications

- **Format**: CSV (raw) + LaTeX (formatted)
- **Decimal places**: 3-4 for coefficients, 2 for percentages
- **Significance**: * p<0.1, ** p<0.05, *** p<0.01

## Generation Commands

```bash
# Generate predictive tables
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode predictive

# Generate portfolio tables
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode portfolio

# Generate robustness tables
python experiments/run_robustness_sweep.py --sweep all
```
