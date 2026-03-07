#!/usr/bin/env python3
"""
Run multi-scale SGI analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import load_experiment_config
from src.data_loader import load_or_download_prices
from src.preprocessing import preprocess_data
from src.multiscale_sgi import (
    compute_multiscale_sgi, compute_scale_correlation_matrix,
    compute_scale_lead_lag, compute_regime_from_multiscale,
    compute_scale_divergence
)
from src.paths import get_outputs_dir
from src.utils import save_json


def run_multiscale_analysis(config_path: str = "configs/universe_sector_etfs.yaml"):
    """Run multi-scale SGI analysis."""
    
    print("Running Multi-scale SGI Analysis")
    print("=" * 50)
    
    # Load config and data
    config = load_experiment_config(config_path)
    
    tickers = config['universe']['tickers']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    prices = load_or_download_prices(tickers, start_date, end_date)
    prices, returns, _ = preprocess_data(prices, save_outputs=False)
    
    # Define windows
    windows = [60, 126, 252, 504]
    top_k = config['geometry']['top_k']
    
    # Compute multi-scale SGI
    print(f"\nComputing SGI at windows: {windows}")
    multiscale_df = compute_multiscale_sgi(returns, windows, top_k)
    
    print(f"Multi-scale features shape: {multiscale_df.shape}")
    
    # Compute scale correlation
    print("\nComputing scale correlations...")
    scale_corr = compute_scale_correlation_matrix(multiscale_df, windows)
    print(scale_corr)
    
    # Compute lead-lag
    print("\nComputing lead-lag relationships...")
    lead_lag = compute_scale_lead_lag(multiscale_df, windows)
    
    for pair, results in lead_lag.items():
        print(f"  {pair}: optimal lag = {results['optimal_lag']}, max corr = {results['max_correlation']:.3f}")
    
    # Compute regime
    print("\nComputing market regime...")
    regime = compute_regime_from_multiscale(multiscale_df, windows)
    regime_counts = regime.value_counts()
    print(regime_counts)
    
    # Compute scale divergence
    print("\nComputing scale divergence...")
    divergence = compute_scale_divergence(multiscale_df, windows)
    multiscale_df['scale_divergence'] = divergence
    
    # Save results
    output_dir = get_outputs_dir() / "multiscale_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    multiscale_df.to_csv(output_dir / "multiscale_sgi.csv")
    scale_corr.to_csv(output_dir / "scale_correlations.csv")
    save_json(lead_lag, output_dir / "lead_lag_analysis.json")
    regime.to_csv(output_dir / "market_regime.csv")
    
    print(f"\nResults saved to: {output_dir}")
    
    return multiscale_df, lead_lag


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/universe_sector_etfs.yaml")
    args = parser.parse_args()
    
    run_multiscale_analysis(args.config)
