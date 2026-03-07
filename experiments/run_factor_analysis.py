#!/usr/bin/env python3
"""
Run factor model integration analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import load_experiment_config
from src.experiments import run_signal_pipeline
from src.factor_model import (
    download_fama_french_factors, compute_rolling_factor_loadings,
    compute_factor_loading_instability, compute_factor_covariance_instability,
    compare_sgi_with_factor_instability, compute_factor_adjusted_sgi,
    build_factor_enhanced_features
)
from src.paths import get_outputs_dir
from src.utils import save_json


def run_factor_analysis(config_path: str = "configs/universe_sector_etfs.yaml"):
    """Run factor model integration analysis."""
    
    print("Running Factor Model Integration Analysis")
    print("=" * 50)
    
    # Run signal pipeline
    config = load_experiment_config(config_path)
    signal_results = run_signal_pipeline(config)
    
    geometry_df = signal_results['geometry_df']
    returns = signal_results['returns']
    
    # Download Fama-French factors
    print("\nDownloading Fama-French factors...")
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    factors = download_fama_french_factors(start_date, end_date)
    
    if factors.empty:
        print("Warning: Could not download Fama-French factors.")
        print("Please provide factor data manually.")
        return None
    
    print(f"Factor data shape: {factors.shape}")
    
    window = config['covariance']['rolling_window']
    
    # Compute rolling factor loadings
    print("\nComputing rolling factor loadings...")
    loadings = compute_rolling_factor_loadings(returns, factors, window)
    
    # Compute factor loading instability
    print("Computing factor loading instability...")
    loading_instability = compute_factor_loading_instability(loadings)
    
    # Compute factor covariance instability
    print("Computing factor covariance instability...")
    factor_cov_instability = compute_factor_covariance_instability(factors, window)
    
    # Combine instability metrics
    factor_instability = pd.concat([loading_instability, factor_cov_instability], axis=1)
    
    # Compare with SGI
    print("\nComparing SGI with factor instability...")
    comparison = compare_sgi_with_factor_instability(geometry_df, factor_instability)
    
    print(f"\nCorrelations with SGI:")
    for metric, corr in comparison['correlations'].items():
        print(f"  {metric}: {corr:.3f}")
    
    # Compute factor-adjusted SGI
    print("\nComputing factor-adjusted SGI...")
    try:
        residual_geometry = compute_factor_adjusted_sgi(returns, factors, window)
        print(f"Residual geometry shape: {residual_geometry.shape}")
    except Exception as e:
        print(f"Warning: Could not compute factor-adjusted SGI: {e}")
        residual_geometry = None
    
    # Build enhanced features
    print("\nBuilding factor-enhanced features...")
    enhanced_features = build_factor_enhanced_features(
        geometry_df, factor_instability, residual_geometry
    )
    
    # Save results
    output_dir = get_outputs_dir() / "factor_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    factor_instability.to_csv(output_dir / "factor_instability.csv")
    save_json(comparison, output_dir / "sgi_factor_comparison.json")
    enhanced_features.to_csv(output_dir / "factor_enhanced_features.csv")
    
    if residual_geometry is not None:
        residual_geometry.to_csv(output_dir / "residual_geometry.csv")
    
    print(f"\nResults saved to: {output_dir}")
    
    return comparison, enhanced_features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/universe_sector_etfs.yaml")
    args = parser.parse_args()
    
    run_factor_analysis(args.config)
