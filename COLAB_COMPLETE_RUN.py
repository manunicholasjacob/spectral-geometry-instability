"""
COMPLETE SGI ANALYSIS - ONE-CLICK EXECUTION FOR GOOGLE COLAB
=============================================================

This script runs the entire SGI research pipeline and generates all results.
Simply paste this entire code block into a Google Colab cell and run.

Authors: Manu Nicholas Jacob, Ronit Ghai
Repository: https://github.com/manunicholasjacob/spectral-geometry-instability
"""

# ============================================================
# STEP 1: SETUP
# ============================================================

print("=" * 70)
print("SPECTRAL GEOMETRY INSTABILITY (SGI) - COMPLETE ANALYSIS")
print("=" * 70)
print("\nStep 1: Cloning repository and installing dependencies...")

import subprocess
import sys
from pathlib import Path

# Clone repository
subprocess.run(["git", "clone", "https://github.com/manunicholasjacob/spectral-geometry-instability.git"], check=True)

# Change to project directory
import os
os.chdir("spectral-geometry-instability")

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

print("✓ Setup complete!")

# ============================================================
# STEP 2: IMPORT MODULES
# ============================================================

print("\nStep 2: Importing modules...")

sys.path.insert(0, str(Path.cwd()))

from src.config import load_experiment_config
from src.data_loader import load_or_download_prices
from src.preprocessing import preprocess_data
from src.covariance import rolling_covariance_matrices
from src.spectral_geometry import compute_rolling_geometry_features, compute_sgi_derived_features
from src.features import build_feature_dataframe
from src.targets import build_target_dataframe
from src.predictive_models import run_walk_forward_experiment, summarize_walk_forward_results
from src.portfolio import generate_weight_schedule
from src.backtest import run_rebalancing_backtest, compare_strategies
from src.plots import plot_sgi_time_series, plot_sgi_vs_volatility
from src.paths import ensure_all_directories
from src.utils import save_json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image

print("✓ All modules imported!")

# ============================================================
# STEP 3: LOAD DATA AND COMPUTE SGI
# ============================================================

print("\nStep 3: Loading data and computing SGI...")

# Ensure directories exist
ensure_all_directories()

# Load configuration
config = load_experiment_config('configs/universe_sector_etfs.yaml')

# Load price data
tickers = config['universe']['tickers']
start_date = config['data']['start_date']
end_date = config['data']['end_date']

print(f"  Loading {len(tickers)} sector ETFs from {start_date} to {end_date}...")
prices = load_or_download_prices(tickers, start_date, end_date)

# Preprocess
prices, returns, metadata = preprocess_data(prices, save_outputs=False)
print(f"  ✓ Returns shape: {returns.shape}")

# Compute covariance matrices
window = config['covariance']['rolling_window']
print(f"  Computing rolling covariance (window={window})...")
cov_matrices = rolling_covariance_matrices(returns, window, method='sample')
print(f"  ✓ Computed {len(cov_matrices)} covariance matrices")

# Compute SGI
top_k = config['geometry']['top_k']
print(f"  Computing SGI (top_k={top_k})...")
geometry_df = compute_rolling_geometry_features(cov_matrices, top_k)
geometry_df = compute_sgi_derived_features(geometry_df)
print(f"  ✓ SGI computed for {len(geometry_df)} dates")

# ============================================================
# STEP 4: BUILD FEATURES AND TARGETS
# ============================================================

print("\nStep 4: Building features and targets...")

features_df = build_feature_dataframe(returns, geometry_df, config, prices)
targets_df = build_target_dataframe(prices, returns, cov_matrices, config)

print(f"  ✓ Features shape: {features_df.shape}")
print(f"  ✓ Targets shape: {targets_df.shape}")

# ============================================================
# STEP 5: PREDICTIVE MODELING
# ============================================================

print("\nStep 5: Running predictive modeling (walk-forward)...")

# Check available target columns
print(f"  Available target columns: {list(targets_df.columns)[:10]}...")

# Find the best available target column (prefer 20d, fallback to others)
possible_targets = ['forward_vol_20d', 'forward_vol_10d', 'forward_vol_5d', 
                   'forward_drawdown_20d', 'forward_drawdown_10d']
target_col = None
for col in possible_targets:
    if col in targets_df.columns:
        target_col = col
        break

if target_col is None:
    # Use the first available forward target
    forward_cols = [c for c in targets_df.columns if c.startswith('forward_')]
    if forward_cols:
        target_col = forward_cols[0]
    else:
        raise ValueError("No suitable target columns found!")

print(f"  Using target column: {target_col}")

# Merge features and targets
common_idx = features_df.index.intersection(targets_df.index)
df = features_df.loc[common_idx].copy()
df['target'] = targets_df.loc[common_idx, target_col]
df = df.dropna()

print(f"  Dataset shape after merge: {df.shape}")

# Define feature sets - use only features that exist
available_features = df.columns.tolist()
feature_sets = {}

# Baseline features
baseline_features = [f for f in ['realized_vol', 'avg_correlation', 'market_return'] if f in available_features]
if baseline_features:
    feature_sets['baseline_only'] = baseline_features

# SGI features
sgi_features = [f for f in ['sgi', 'weighted_sgi'] if f in available_features]
if sgi_features:
    feature_sets['sgi_only'] = sgi_features

# Combined
if baseline_features and sgi_features:
    feature_sets['baseline_plus_sgi'] = baseline_features + sgi_features

print(f"  Feature sets: {list(feature_sets.keys())}")

# Run walk-forward
print("  Running walk-forward validation...")
wf_results = run_walk_forward_experiment(
    df, target_col, feature_sets, model_type='linear', config=config
)

# Summarize results
summary = summarize_walk_forward_results(wf_results)
print("\n  Predictive Results:")
print(summary[['feature_set', 'mean_r2', 'mean_mse', 'n_folds']])

# ============================================================
# STEP 6: PORTFOLIO BACKTESTING
# ============================================================

print("\nStep 6: Running portfolio backtests...")

strategies = ['equal_weight', 'min_variance', 'sgi_conditioned_minvar']
weight_schedules = {}

for strategy in strategies:
    print(f"  Generating weights for {strategy}...")
    weights = generate_weight_schedule(
        returns, cov_matrices, geometry_df,
        strategy, config, rebalance_freq='weekly'
    )
    weight_schedules[strategy] = weights

# Run backtests
print("  Running backtests...")
comparison = compare_strategies(prices, weight_schedules, transaction_cost_bps=10)

print("\n  Portfolio Results:")
print(comparison[['strategy', 'cagr', 'sharpe', 'max_drawdown', 'calmar']])

# ============================================================
# STEP 7: VISUALIZATIONS
# ============================================================

print("\nStep 7: Generating visualizations...")

# Create figures directory
fig_dir = Path('outputs/figures')
fig_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: SGI Time Series
print("  Creating SGI time series plot...")
from src.constants import CRISIS_DATES
fig1 = plot_sgi_time_series(geometry_df, CRISIS_DATES, save_path=fig_dir / 'sgi_timeseries.png')
plt.close()

# Plot 2: SGI vs Volatility
print("  Creating SGI vs volatility plot...")
fig2 = plot_sgi_vs_volatility(features_df, save_path=fig_dir / 'sgi_vs_vol.png')
plt.close()

# Plot 3: Portfolio Performance
print("  Creating portfolio performance plot...")
fig, ax = plt.subplots(figsize=(14, 6))
for strategy in strategies:
    backtest_results = run_rebalancing_backtest(
        prices, weight_schedules[strategy], transaction_cost_bps=10
    )
    cumulative = (1 + backtest_results['portfolio_returns']).cumprod()
    ax.plot(cumulative.index, cumulative.values, label=strategy, linewidth=2)

ax.set_title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'portfolio_performance.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Figures saved to outputs/figures/")

# ============================================================
# STEP 8: DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n1. SGI STATISTICS")
print("-" * 70)
sgi_stats = geometry_df[['sgi', 'weighted_sgi', 'absorption_ratio']].describe()
print(sgi_stats)

print("\n2. PREDICTIVE MODELING RESULTS")
print("-" * 70)
print(summary)

print("\n3. PORTFOLIO BACKTEST RESULTS")
print("-" * 70)
print(comparison)

print("\n4. KEY FINDINGS")
print("-" * 70)

# Calculate incremental value of SGI (with error handling)
try:
    baseline_r2 = summary[summary['feature_set'] == 'baseline_only']['mean_r2'].values[0]
    with_sgi_r2 = summary[summary['feature_set'] == 'baseline_plus_sgi']['mean_r2'].values[0]
    incremental_r2 = with_sgi_r2 - baseline_r2
    print(f"  • SGI adds {incremental_r2:.4f} incremental R² to baseline model")
except (IndexError, KeyError):
    incremental_r2 = 0.0
    print(f"  • Incremental R² calculation skipped (insufficient data)")

# Portfolio improvement (with error handling)
try:
    baseline_sharpe = comparison[comparison['strategy'] == 'equal_weight']['sharpe'].values[0]
    sgi_sharpe = comparison[comparison['strategy'] == 'sgi_conditioned_minvar']['sharpe'].values[0]
    sharpe_improvement = ((sgi_sharpe / baseline_sharpe) - 1) * 100
    print(f"  • SGI-conditioned portfolio improves Sharpe by {sharpe_improvement:.1f}%")
except (IndexError, KeyError):
    sharpe_improvement = 0.0
    print(f"  • Sharpe improvement calculation skipped (insufficient data)")

# SGI characteristics
sgi_mean = geometry_df['sgi'].mean()
sgi_std = geometry_df['sgi'].std()
print(f"  • Average SGI: {sgi_mean:.4f} (std: {sgi_std:.4f})")

print("\n5. VISUALIZATIONS")
print("-" * 70)
print("  Displaying figures...")

# Display figures
display(Image('outputs/figures/sgi_timeseries.png'))
display(Image('outputs/figures/sgi_vs_vol.png'))
display(Image('outputs/figures/portfolio_performance.png'))

# ============================================================
# STEP 9: SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results_dir = Path('outputs/complete_analysis')
results_dir.mkdir(parents=True, exist_ok=True)

# Save DataFrames
geometry_df.to_csv(results_dir / 'geometry_features.csv')
features_df.to_csv(results_dir / 'all_features.csv')
targets_df.to_csv(results_dir / 'targets.csv')
summary.to_csv(results_dir / 'predictive_summary.csv', index=False)
comparison.to_csv(results_dir / 'portfolio_comparison.csv', index=False)

# Save summary statistics
results_summary = {
    'sgi_statistics': sgi_stats.to_dict(),
    'predictive_results': summary.to_dict('records'),
    'portfolio_results': comparison.to_dict('records'),
    'key_metrics': {
        'incremental_r2': float(incremental_r2),
        'sharpe_improvement_pct': float(sharpe_improvement),
        'mean_sgi': float(sgi_mean),
        'std_sgi': float(sgi_std),
        'n_observations': len(geometry_df),
        'date_range': {
            'start': str(geometry_df.index.min()),
            'end': str(geometry_df.index.max()),
        }
    }
}

save_json(results_summary, results_dir / 'results_summary.json')

print(f"\n✓ All results saved to: {results_dir}")

# ============================================================
# STEP 10: DOWNLOAD RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOAD RESULTS")
print("=" * 70)

# Zip all results
print("\nZipping all results...")
import zipfile
import shutil

zip_path = 'sgi_complete_results.zip'
shutil.make_archive('sgi_complete_results', 'zip', 'outputs')

print(f"✓ Created {zip_path}")

# Auto-download in Colab
try:
    from google.colab import files
    print("\nDownloading results to your computer...")
    files.download(zip_path)
    print("✓ Download started! Check your browser's download folder.")
except ImportError:
    print("\nNot running in Colab. Results saved locally to outputs/")
    print(f"To download manually, use: files.download('{zip_path}')")

print("\n" + "=" * 70)
print("✓ ANALYSIS COMPLETE!")
print("=" * 70)
print("\nYou now have:")
print("  • SGI time series and derived features")
print("  • Predictive modeling results")
print("  • Portfolio backtest results")
print("  • Publication-quality visualizations")
print("  • All data saved to outputs/complete_analysis/")
print("\nNext steps:")
print("  1. Review the visualizations above")
print("  2. Examine the CSV files in outputs/complete_analysis/")
print("  3. Run advanced experiments (VIX, RMT, multi-scale, etc.)")
print("  4. Start writing your research paper!")
print("=" * 70)
