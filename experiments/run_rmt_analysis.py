#!/usr/bin/env python3
"""
Run RMT denoising analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.config import load_experiment_config
from src.data_loader import load_or_download_prices
from src.preprocessing import preprocess_data
from src.rmt_denoising import (
    rolling_rmt_denoised_covariance, compute_rmt_diagnostics,
    denoise_covariance_rmt
)
from src.spectral_geometry import compute_rolling_geometry_features
from src.covariance import rolling_covariance_matrices
from src.paths import get_outputs_dir
from src.utils import save_json


def run_rmt_analysis(config_path: str = "configs/universe_sector_etfs.yaml"):
    """Run RMT denoising analysis."""
    
    print("Running RMT Denoising Analysis")
    print("=" * 50)
    
    # Load config and data
    config = load_experiment_config(config_path)
    
    tickers = config['universe']['tickers']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    prices = load_or_download_prices(tickers, start_date, end_date)
    prices, returns, _ = preprocess_data(prices, save_outputs=False)
    
    window = config['covariance']['rolling_window']
    top_k = config['geometry']['top_k']
    
    # Compute standard covariance matrices
    print("\nComputing standard covariance matrices...")
    standard_cov = rolling_covariance_matrices(returns, window, method='sample')
    
    # Compute RMT-denoised covariance matrices
    print("Computing RMT-denoised covariance matrices...")
    denoised_cov = rolling_rmt_denoised_covariance(returns, window, method='constant_residual')
    
    # Compute SGI for both
    print("Computing SGI for standard covariance...")
    standard_geometry = compute_rolling_geometry_features(standard_cov, top_k)
    
    print("Computing SGI for denoised covariance...")
    denoised_geometry = compute_rolling_geometry_features(denoised_cov, top_k)
    
    # Compare
    common_idx = standard_geometry.index.intersection(denoised_geometry.index)
    
    comparison = pd.DataFrame({
        'sgi_standard': standard_geometry.loc[common_idx, 'sgi'],
        'sgi_denoised': denoised_geometry.loc[common_idx, 'sgi'],
    })
    comparison['sgi_difference'] = comparison['sgi_standard'] - comparison['sgi_denoised']
    
    # Compute RMT diagnostics for sample covariance
    print("\nComputing RMT diagnostics...")
    sample_cov = returns.iloc[-window:].cov().values
    diagnostics = compute_rmt_diagnostics(sample_cov, window)
    
    # Save results
    output_dir = get_outputs_dir() / "rmt_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(output_dir / "sgi_comparison.csv")
    save_json(diagnostics, output_dir / "rmt_diagnostics.json")
    standard_geometry.to_csv(output_dir / "geometry_standard.csv")
    denoised_geometry.to_csv(output_dir / "geometry_denoised.csv")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey findings:")
    print(f"  Signal eigenvalues: {diagnostics['n_signal_eigenvalues']}")
    print(f"  Noise eigenvalues: {diagnostics['n_noise_eigenvalues']}")
    print(f"  Signal variance fraction: {diagnostics['signal_variance_fraction']:.3f}")
    print(f"  SGI correlation (standard vs denoised): {comparison['sgi_standard'].corr(comparison['sgi_denoised']):.3f}")
    
    return comparison, diagnostics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/universe_sector_etfs.yaml")
    args = parser.parse_args()
    
    run_rmt_analysis(args.config)
