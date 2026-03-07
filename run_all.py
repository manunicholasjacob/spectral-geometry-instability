#!/usr/bin/env python3
"""
Run all experiments for the SGI project.

This script runs the full experiment suite across all universes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_experiment_config
from src.experiments import run_full_pipeline
from src.paths import ensure_all_directories, get_configs_dir


def main():
    print("=" * 60)
    print("Spectral Geometry Instability - Full Experiment Suite")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_all_directories()
    
    # Universe configs to run
    universe_configs = [
        "universe_sector_etfs.yaml",
        "universe_multiasset.yaml",
        # "universe_largecap50.yaml",  # Uncomment for larger universe
    ]
    
    configs_dir = get_configs_dir()
    
    all_results = {}
    
    for config_file in universe_configs:
        config_path = configs_dir / config_file
        
        if not config_path.exists():
            print(f"Warning: Config not found: {config_path}")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Running experiments for: {config_file}")
        print("=" * 60)
        
        try:
            config = load_experiment_config(config_path)
            results = run_full_pipeline(config)
            all_results[config_file] = results
            print(f"Completed: {config_file}")
        except Exception as e:
            print(f"Error running {config_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("=" * 60)
    
    # Print summary
    print("\nResults summary:")
    for config_name, results in all_results.items():
        print(f"  - {config_name}: {len(results)} pipelines completed")


if __name__ == "__main__":
    main()
