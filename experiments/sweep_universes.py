#!/usr/bin/env python3
"""Sweep over different asset universes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import load_experiment_config
from src.experiments import run_signal_pipeline
from src.paths import get_outputs_dir, get_configs_dir


def sweep_universes():
    """Run signal pipeline across all universes."""
    
    configs_dir = get_configs_dir()
    
    universe_configs = [
        "universe_sector_etfs.yaml",
        "universe_multiasset.yaml",
        "universe_largecap50.yaml",
    ]
    
    results = []
    
    for config_file in universe_configs:
        config_path = configs_dir / config_file
        
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            continue
        
        print(f"\nProcessing universe: {config_file}")
        
        try:
            config = load_experiment_config(config_path)
            pipeline_results = run_signal_pipeline(config)
            
            geometry_df = pipeline_results["geometry_df"]
            universe_name = config.get("universe", {}).get("name", config_file)
            
            result = {
                "universe": universe_name,
                "n_assets": len(pipeline_results["returns"].columns),
                "n_observations": len(geometry_df),
                "sgi_mean": geometry_df["sgi"].mean(),
                "sgi_std": geometry_df["sgi"].std(),
                "sgi_max": geometry_df["sgi"].max(),
                "absorption_ratio_mean": geometry_df["absorption_ratio"].mean(),
                "weighted_sgi_mean": geometry_df["weighted_sgi"].mean(),
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error with {config_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = get_outputs_dir() / "universe_sweep_results.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\nUniverse sweep complete. Results saved to: {output_path}")
    return results_df


if __name__ == "__main__":
    sweep_universes()
