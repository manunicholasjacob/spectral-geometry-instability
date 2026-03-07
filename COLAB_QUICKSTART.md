# Google Colab Quick Start

## Running SGI Project in Google Colab

This guide helps you run the Spectral Geometry Instability project in Google Colab.

---

## Step 1: Clone the Repository

```python
!git clone https://github.com/manunicholasjacob/spectral-geometry-instability.git
%cd spectral-geometry-instability
```

---

## Step 2: Install Dependencies

```python
!pip install -q -r requirements.txt
```

Or use the automated setup script:

```python
!python colab_setup.py
```

---

## Step 3: Test Imports

```python
!python test_imports.py
```

This will verify all modules are working correctly.

---

## Step 4: Run Your First Experiment

### Option A: Signal Exploration

```python
!python run_experiment.py --config configs/universe_sector_etfs.yaml --mode signal
```

### Option B: Use Python API

```python
import sys
sys.path.insert(0, '/content/spectral-geometry-instability')

from src.config import load_experiment_config
from src.experiments import run_signal_pipeline

# Load config
config = load_experiment_config('configs/universe_sector_etfs.yaml')

# Run pipeline
results = run_signal_pipeline(config)

# View results
print(f"SGI computed for {len(results['geometry_df'])} dates")
print(results['geometry_df'][['sgi', 'weighted_sgi', 'absorption_ratio']].describe())
```

---

## Step 5: Explore Notebooks

```python
# List available notebooks
!ls notebooks/

# Open a notebook
# Click on the Files icon (left sidebar) → notebooks/ → double-click any .ipynb file
```

---

## Common Issues & Solutions

### Issue 1: Import Errors

**Solution**: Make sure you've run the setup:
```python
!python colab_setup.py
```

### Issue 2: Module Not Found

**Solution**: Add project to path:
```python
import sys
sys.path.insert(0, '/content/spectral-geometry-instability')
```

### Issue 3: Data Download Fails

**Solution**: Check internet connection and yfinance status:
```python
import yfinance as yf
data = yf.download("SPY", start="2020-01-01", end="2020-12-31")
print(data.head())
```

### Issue 4: Out of Memory

**Solution**: Use smaller universe or shorter date range:
```python
# Edit config to use fewer assets or shorter period
config['universe']['tickers'] = ['SPY', 'QQQ', 'IWM']  # Smaller universe
config['data']['start_date'] = '2020-01-01'  # Shorter period
```

---

## Example: Quick SGI Computation

```python
import sys
sys.path.insert(0, '/content/spectral-geometry-instability')

from src.data_loader import load_or_download_prices
from src.preprocessing import preprocess_data
from src.covariance import rolling_covariance_matrices
from src.spectral_geometry import compute_rolling_geometry_features

# Load data
tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']
prices = load_or_download_prices(tickers, '2020-01-01', '2024-01-01')

# Preprocess
prices, returns, _ = preprocess_data(prices, save_outputs=False)

# Compute covariance
cov_matrices = rolling_covariance_matrices(returns, window=60)

# Compute SGI
geometry_df = compute_rolling_geometry_features(cov_matrices, top_k=3)

# View results
print(geometry_df[['sgi', 'weighted_sgi']].tail(10))

# Plot
import matplotlib.pyplot as plt
geometry_df['sgi'].plot(figsize=(12, 6), title='SGI Over Time')
plt.ylabel('SGI')
plt.grid(True)
plt.show()
```

---

## Running Advanced Experiments

```python
# VIX Analysis
!python experiments/run_vix_analysis.py

# Multi-scale SGI
!python experiments/run_multiscale_analysis.py

# Crypto Analysis
!python experiments/run_crypto_analysis.py
```

---

## Saving Results

```python
# Results are saved to outputs/ directory
!ls outputs/

# Download results to your local machine
from google.colab import files

# Download a specific file
files.download('outputs/signal/geometry_features.csv')

# Or zip and download all outputs
!zip -r outputs.zip outputs/
files.download('outputs.zip')
```

---

## Tips for Colab

1. **Runtime**: Use GPU runtime for faster computation (Runtime → Change runtime type → GPU)
2. **Session timeout**: Colab sessions timeout after inactivity. Save important results frequently.
3. **Storage**: Outputs are temporary. Download important results before session ends.
4. **Memory**: Monitor memory usage (top-right corner). Restart runtime if needed.

---

## Next Steps

1. Read the [User Guide](docs/USER_GUIDE.md)
2. Explore the [Examples](docs/EXAMPLES.md)
3. Check the [API Reference](docs/API_REFERENCE.md)

---

**Authors**: Manu Nicholas Jacob, Ronit Ghai  
**Repository**: https://github.com/manunicholasjacob/spectral-geometry-instability
