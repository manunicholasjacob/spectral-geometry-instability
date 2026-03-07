#!/usr/bin/env python3
"""
Run quantile regression analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import load_experiment_config
from src.experiments import run_signal_pipeline
from src.targets import build_target_dataframe
from src.features import get_feature_columns
from src.quantile_regression import (
    walk_forward_quantile_regression, compare_quantile_feature_sets,
    fit_multiple_quantiles, compute_var_exceedance_test
)
from src.paths import get_outputs_dir
from src.utils import save_json


def run_quantile_analysis(config_path: str = "configs/universe_sector_etfs.yaml"):
    """Run quantile regression analysis."""
    
    print("Running Quantile Regression Analysis")
    print("=" * 50)
    
    # Run signal pipeline
    config = load_experiment_config(config_path)
    signal_results = run_signal_pipeline(config)
    
    features_df = signal_results['features_df']
    prices = signal_results['prices']
    returns = signal_results['returns']
    cov_matrices = signal_results['cov_matrices']
    
    # Build targets
    print("\nBuilding target variables...")
    targets_df = build_target_dataframe(prices, returns, cov_matrices, config)
    
    # Prepare dataset
    target_col = 'forward_drawdown_20d'
    feature_cols = get_feature_columns('full', config)
    
    # Align data
    common_idx = features_df.index.intersection(targets_df.index)
    df = features_df.loc[common_idx].copy()
    df['target'] = targets_df.loc[common_idx, target_col]
    df = df.dropna()
    
    print(f"Dataset shape: {df.shape}")
    
    # Define feature sets
    feature_sets = {
        'baseline_only': get_feature_columns('baseline_only', config),
        'sgi_only': get_feature_columns('sgi_only', config),
        'baseline_plus_sgi': get_feature_columns('baseline_plus_sgi', config),
    }
    
    # Run quantile regression at multiple quantiles
    quantiles = [0.05, 0.10, 0.25]
    
    all_results = {}
    for q in quantiles:
        print(f"\nRunning quantile regression at q={q}...")
        
        comparison = compare_quantile_feature_sets(
            df, 'target', feature_sets, quantile=q
        )
        all_results[f'q_{q}'] = comparison.to_dict('records')
        
        print(comparison)
    
    # Walk-forward for main quantile
    print("\nRunning walk-forward quantile regression (q=0.05)...")
    wf_results = walk_forward_quantile_regression(
        df, 'target', get_feature_columns('baseline_plus_sgi', config),
        quantile=0.05
    )
    
    # VaR exceedance test
    if wf_results['predictions'] and wf_results['actuals']:
        import numpy as np
        exceedance_test = compute_var_exceedance_test(
            np.array(wf_results['predictions']),
            np.array(wf_results['actuals']),
            0.05
        )
        print(f"\nVaR Exceedance Test:")
        print(f"  Exceedance rate: {exceedance_test['exceedance_rate']:.3f}")
        print(f"  Target rate: {exceedance_test['target_rate']:.3f}")
        print(f"  P-value: {exceedance_test['p_value']:.3f}")
    
    # Save results
    output_dir = get_outputs_dir() / "quantile_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_json(all_results, output_dir / "quantile_comparison.json")
    save_json(wf_results['summary'], output_dir / "walk_forward_summary.json")
    if wf_results['predictions']:
        save_json(exceedance_test, output_dir / "var_exceedance_test.json")
    
    print(f"\nResults saved to: {output_dir}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/universe_sector_etfs.yaml")
    args = parser.parse_args()
    
    run_quantile_analysis(args.config)
