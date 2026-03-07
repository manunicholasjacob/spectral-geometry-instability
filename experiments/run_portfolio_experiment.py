#!/usr/bin/env python3
"""
Run portfolio backtesting experiment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_experiment_config
from src.experiments import run_portfolio_pipeline


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run portfolio experiment")
    parser.add_argument("--config", "-c", type=str, default="configs/universe_sector_etfs.yaml")
    args = parser.parse_args()
    
    config = load_experiment_config(args.config)
    results = run_portfolio_pipeline(config)
    
    print(f"Portfolio experiment complete. Output: {results['output_dir']}")


if __name__ == "__main__":
    main()
