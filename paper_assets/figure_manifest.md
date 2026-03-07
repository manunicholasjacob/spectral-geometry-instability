# Figure Manifest

This document tracks all figures generated for the SGI research paper.

## Main Figures

| Figure | Description | Script/Notebook | Status |
|--------|-------------|-----------------|--------|
| Fig 1 | SGI time series with crisis overlay | `02_sgi_exploration.ipynb` | Pending |
| Fig 2 | SGI vs realized volatility | `02_sgi_exploration.ipynb` | Pending |
| Fig 3 | SGI vs average correlation | `02_sgi_exploration.ipynb` | Pending |
| Fig 4 | Event study: SGI around crises | `05_event_studies.ipynb` | Pending |
| Fig 5 | Predictive model comparison | `03_predictive_tests.ipynb` | Pending |
| Fig 6 | Portfolio cumulative returns | `04_portfolio_tests.ipynb` | Pending |
| Fig 7 | Robustness: window sensitivity | `06_robustness.ipynb` | Pending |
| Fig 8 | Robustness: estimator comparison | `06_robustness.ipynb` | Pending |

## Supplementary Figures

| Figure | Description | Script/Notebook | Status |
|--------|-------------|-----------------|--------|
| S1 | Eigenvalue concentration over time | `02_sgi_exploration.ipynb` | Pending |
| S2 | Principal angle distributions | `02_sgi_exploration.ipynb` | Pending |
| S3 | Rolling Sharpe ratio comparison | `04_portfolio_tests.ipynb` | Pending |
| S4 | Transaction cost sensitivity | `experiments/sweep_thresholds.py` | Pending |
| S5 | Universe comparison | `experiments/sweep_universes.py` | Pending |

## Figure Specifications

- **Format**: PNG (300 DPI) + PDF
- **Size**: 14x6 inches (main), 10x6 inches (supplementary)
- **Font**: 11pt base, 14pt titles
- **Style**: seaborn-whitegrid

## Generation Commands

```bash
# Generate all main figures
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode signal
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode event

# Generate robustness figures
python experiments/run_robustness_sweep.py --sweep all
```
