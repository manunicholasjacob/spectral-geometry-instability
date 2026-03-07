#!/usr/bin/env python3
"""
Run VIX integration analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import load_experiment_config
from src.experiments import run_signal_pipeline
from src.vix_integration import (
    download_vix_data, compute_vix_features,
    compute_sgi_vix_relationship, add_vix_controls_to_features,
    compute_vix_adjusted_sgi_features
)
from src.paths import get_outputs_dir
from src.utils import save_json


def run_vix_analysis(config_path: str = "configs/universe_sector_etfs.yaml"):
    """Run VIX integration analysis."""
    
    print("Running VIX Integration Analysis")
    print("=" * 50)
    
    # Run signal pipeline
    config = load_experiment_config(config_path)
    signal_results = run_signal_pipeline(config)
    
    geometry_df = signal_results['geometry_df']
    features_df = signal_results['features_df']
    
    # Download VIX data
    print("\nDownloading VIX data...")
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    vix_data = download_vix_data(start_date, end_date)
    
    # Compute VIX features
    print("Computing VIX features...")
    vix_features = compute_vix_features(vix_data)
    
    # Analyze SGI-VIX relationship
    print("Analyzing SGI-VIX relationship...")
    relationship = compute_sgi_vix_relationship(geometry_df, vix_features)
    
    # Add VIX controls to features
    print("Adding VIX controls to features...")
    enhanced_features = add_vix_controls_to_features(features_df, vix_features)
    
    # Compute VIX-adjusted SGI
    print("Computing VIX-adjusted SGI...")
    vix_adjusted = compute_vix_adjusted_sgi_features(geometry_df, vix_features)
    
    # Save results
    output_dir = get_outputs_dir() / "vix_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_json(relationship, output_dir / "sgi_vix_relationship.json")
    enhanced_features.to_csv(output_dir / "features_with_vix.csv")
    vix_adjusted.to_csv(output_dir / "vix_adjusted_sgi.csv")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey findings:")
    print(f"  SGI-VIX correlation: {relationship['correlation']['sgi_vix_level']:.3f}")
    print(f"  Rolling correlation mean: {relationship['rolling_correlation']['mean']:.3f}")
    
    return relationship


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/universe_sector_etfs.yaml")
    args = parser.parse_args()
    
    run_vix_analysis(args.config)
