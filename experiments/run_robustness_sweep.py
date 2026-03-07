#!/usr/bin/env python3
"""
Run robustness sweep experiments.
"""

import sys
from pathlib import Path
import itertools

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import load_experiment_config, merge_configs
from src.experiments import run_signal_pipeline
from src.paths import get_outputs_dir, get_configs_dir
from src.utils import save_json


def run_robustness_sweep(base_config_path: str, sweep_type: str = "windows"):
    """Run robustness sweep over specified parameter."""
    
    base_config = load_experiment_config(base_config_path)
    
    # Define sweep parameters
    sweeps = {
        "windows": [60, 126, 252, 504],
        "top_k": [2, 3, 5, 8],
        "estimators": ["sample", "ewma", "ledoit_wolf"],
    }
    
    if sweep_type not in sweeps:
        raise ValueError(f"Unknown sweep type: {sweep_type}")
    
    results = []
    
    for param_value in sweeps[sweep_type]:
        print(f"\nRunning sweep: {sweep_type} = {param_value}")
        
        # Create modified config
        config = base_config.copy()
        
        if sweep_type == "windows":
            config["covariance"]["rolling_window"] = param_value
        elif sweep_type == "top_k":
            config["geometry"]["top_k"] = param_value
        elif sweep_type == "estimators":
            config["covariance"]["estimator"] = param_value
        
        config["outputs"]["experiment_tag"] = f"sweep_{sweep_type}_{param_value}"
        
        try:
            pipeline_results = run_signal_pipeline(config)
            
            # Extract summary statistics
            geometry_df = pipeline_results["geometry_df"]
            
            result = {
                "sweep_type": sweep_type,
                "param_value": param_value,
                "sgi_mean": geometry_df["sgi"].mean(),
                "sgi_std": geometry_df["sgi"].std(),
                "sgi_max": geometry_df["sgi"].max(),
                "absorption_ratio_mean": geometry_df["absorption_ratio"].mean(),
                "n_observations": len(geometry_df),
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error with {sweep_type}={param_value}: {e}")
    
    # Save sweep results
    results_df = pd.DataFrame(results)
    output_path = get_outputs_dir() / f"robustness_sweep_{sweep_type}.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\nSweep complete. Results saved to: {output_path}")
    return results_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run robustness sweep")
    parser.add_argument("--config", "-c", type=str, default="configs/universe_sector_etfs.yaml")
    parser.add_argument("--sweep", "-s", type=str, 
                       choices=["windows", "top_k", "estimators", "all"],
                       default="windows")
    args = parser.parse_args()
    
    if args.sweep == "all":
        for sweep_type in ["windows", "top_k", "estimators"]:
            run_robustness_sweep(args.config, sweep_type)
    else:
        run_robustness_sweep(args.config, args.sweep)


if __name__ == "__main__":
    main()
