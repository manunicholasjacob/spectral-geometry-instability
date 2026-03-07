#!/usr/bin/env python3
"""
Run signal exploration experiment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_experiment_config
from src.experiments import run_signal_pipeline
from src.paths import get_configs_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run signal experiment")
    parser.add_argument("--config", "-c", type=str, default="configs/universe_sector_etfs.yaml")
    args = parser.parse_args()
    
    config = load_experiment_config(args.config)
    results = run_signal_pipeline(config)
    
    print(f"Signal experiment complete. Output: {results['output_dir']}")


if __name__ == "__main__":
    main()
