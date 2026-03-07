# Quick Start Guide

## Get Running in 5 Minutes

---

## Step 1: Install

```bash
git clone https://github.com/manunicholasjacob/spectral-geometry-instability.git
cd spectral-geometry-instability
pip install -r requirements.txt
```

## Step 2: Run Your First Experiment

```bash
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode signal
```

This will:
1. Download sector ETF price data
2. Compute rolling covariance matrices
3. Calculate SGI metrics
4. Generate visualizations
5. Save results to `outputs/`

## Step 3: View Results

Check the `outputs/` directory for:
- `geometry_features.csv` - SGI time series
- `all_features.csv` - Full feature set
- `outputs/figures/` - Visualizations

## Step 4: Run Predictive Tests

```bash
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode predictive
```

## Step 5: Run Portfolio Backtest

```bash
python run_experiment.py --config configs/universe_sector_etfs.yaml --mode portfolio
```

---

## Python API

```python
from src.config import load_experiment_config
from src.experiments import run_signal_pipeline

# Load config
config = load_experiment_config('configs/universe_sector_etfs.yaml')

# Run pipeline
results = run_signal_pipeline(config)

# Access SGI
sgi = results['geometry_df']['sgi']
print(f"Mean SGI: {sgi.mean():.4f}")
print(f"Max SGI: {sgi.max():.4f}")
```

---

## Available Experiments

| Command | Description |
|---------|-------------|
| `--mode signal` | Compute SGI, visualize |
| `--mode predictive` | Test predictive power |
| `--mode portfolio` | Backtest strategies |
| `--mode event` | Crisis event studies |
| `--mode full` | Run everything |

---

## Available Universes

| Config | Assets |
|--------|--------|
| `universe_sector_etfs.yaml` | 11 sector ETFs |
| `universe_multiasset.yaml` | 8 multi-asset ETFs |
| `universe_largecap50.yaml` | 50 large-cap stocks |
| `universe_crypto.yaml` | 15 cryptocurrencies |

---

## Next Steps

1. Read the full [User Guide](USER_GUIDE.md)
2. Explore the [Notebooks](../notebooks/)
3. Check the [API Reference](API_REFERENCE.md)

---

*Happy researching!*
