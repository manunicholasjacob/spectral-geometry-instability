#!/usr/bin/env python3
"""Sweep over SGI threshold values for portfolio conditioning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import load_experiment_config
from src.experiments import run_signal_pipeline
from src.portfolio import generate_weight_schedule
from src.backtest import run_rebalancing_backtest
from src.paths import get_outputs_dir


def sweep_thresholds(config_path: str = "configs/universe_sector_etfs.yaml"):
    """Sweep over SGI threshold percentiles."""
    
    config = load_experiment_config(config_path)
    
    # Run signal pipeline once
    signal_results = run_signal_pipeline(config)
    
    prices = signal_results["prices"]
    returns = signal_results["returns"]
    cov_matrices = signal_results["cov_matrices"]
    geometry_df = signal_results["geometry_df"]
    
    # Threshold percentiles to test
    thresholds = [50, 60, 70, 75, 80, 90]
    
    results = []
    
    for threshold in thresholds:
        print(f"\nTesting threshold percentile: {threshold}")
        
        # Update config
        config["portfolio"]["sgi_threshold_percentile"] = threshold
        
        try:
            # Generate weights
            weights = generate_weight_schedule(
                returns, cov_matrices, geometry_df,
                "sgi_conditioned_minvar", config, "weekly"
            )
            
            # Run backtest
            backtest = run_rebalancing_backtest(prices, weights, 10)
            
            result = {
                "threshold_percentile": threshold,
                **backtest["net_metrics"],
                "turnover": backtest["turnover"]["annualized"],
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error with threshold {threshold}: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = get_outputs_dir() / "threshold_sweep_results.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\nThreshold sweep complete. Results saved to: {output_path}")
    return results_df


if __name__ == "__main__":
    sweep_thresholds()
