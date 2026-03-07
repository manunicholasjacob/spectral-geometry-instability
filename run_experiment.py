#!/usr/bin/env python3
"""
Main experiment runner for the SGI project.

Usage:
    python run_experiment.py --config configs/universe_sector_etfs.yaml --mode signal
    python run_experiment.py --config configs/universe_sector_etfs.yaml --mode predictive
    python run_experiment.py --config configs/universe_sector_etfs.yaml --mode portfolio
    python run_experiment.py --config configs/universe_sector_etfs.yaml --mode event
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_experiment_config
from src.experiments import (
    run_signal_pipeline,
    run_predictive_pipeline,
    run_portfolio_pipeline,
    run_event_study_pipeline,
    run_full_pipeline
)
from src.paths import ensure_all_directories


def main():
    parser = argparse.ArgumentParser(
        description="Run SGI experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_experiment.py --config configs/universe_sector_etfs.yaml --mode signal
    python run_experiment.py --config configs/universe_multiasset.yaml --mode predictive
    python run_experiment.py --config configs/universe_largecap50.yaml --mode portfolio
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["signal", "predictive", "portfolio", "event", "full"],
        default="signal",
        help="Experiment mode to run"
    )
    
    parser.add_argument(
        "--output-tag", "-t",
        type=str,
        default="",
        help="Optional tag for output files"
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_all_directories()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Loading configuration from: {config_path}")
    config = load_experiment_config(config_path)
    
    # Override output tag if provided
    if args.output_tag:
        if "outputs" not in config:
            config["outputs"] = {}
        config["outputs"]["experiment_tag"] = args.output_tag
    
    # Run appropriate pipeline
    print(f"Running {args.mode} pipeline...")
    
    if args.mode == "signal":
        results = run_signal_pipeline(config)
    elif args.mode == "predictive":
        results = run_predictive_pipeline(config)
    elif args.mode == "portfolio":
        results = run_portfolio_pipeline(config)
    elif args.mode == "event":
        results = run_event_study_pipeline(config)
    elif args.mode == "full":
        results = run_full_pipeline(config)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    print(f"\nExperiment complete!")
    print(f"Output directory: {results.get('output_dir', 'See logs')}")


if __name__ == "__main__":
    main()
