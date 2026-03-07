#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
Run this to check for import errors before running experiments.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")
print("=" * 50)

try:
    print("✓ Testing core modules...")
    from src import config
    from src import constants
    from src import paths
    from src import utils
    from src import logging_utils
    print("  ✓ Core modules OK")
except Exception as e:
    print(f"  ✗ Core modules FAILED: {e}")
    sys.exit(1)

try:
    print("✓ Testing data modules...")
    from src import data_loader
    from src import preprocessing
    print("  ✓ Data modules OK")
except Exception as e:
    print(f"  ✗ Data modules FAILED: {e}")
    sys.exit(1)

try:
    print("✓ Testing computation modules...")
    from src import covariance
    from src import spectral_geometry
    from src import features
    from src import targets
    print("  ✓ Computation modules OK")
except Exception as e:
    print(f"  ✗ Computation modules FAILED: {e}")
    sys.exit(1)

try:
    print("✓ Testing analysis modules...")
    from src import predictive_models
    from src import portfolio
    from src import backtest
    from src import event_study
    from src import bootstrap
    print("  ✓ Analysis modules OK")
except Exception as e:
    print(f"  ✗ Analysis modules FAILED: {e}")
    sys.exit(1)

try:
    print("✓ Testing advanced modules...")
    from src import vix_integration
    from src import rmt_denoising
    from src import quantile_regression
    from src import multiscale_sgi
    from src import cross_market_spillovers
    from src import factor_model
    from src import crypto_universe
    print("  ✓ Advanced modules OK")
except Exception as e:
    print(f"  ✗ Advanced modules FAILED: {e}")
    sys.exit(1)

try:
    print("✓ Testing visualization modules...")
    from src import plots
    from src import diagnostics
    print("  ✓ Visualization modules OK")
except Exception as e:
    print(f"  ✗ Visualization modules FAILED: {e}")
    sys.exit(1)

try:
    print("✓ Testing experiment orchestration...")
    from src import experiments
    print("  ✓ Experiment orchestration OK")
except Exception as e:
    print(f"  ✗ Experiment orchestration FAILED: {e}")
    sys.exit(1)

print("=" * 50)
print("✓ All imports successful!")
print("\nTesting core functionality...")

try:
    from src.config import load_experiment_config
    from src.paths import ensure_all_directories
    
    # Ensure directories exist
    ensure_all_directories()
    print("  ✓ Directory structure created")
    
    # Test loading a config
    config_path = Path("configs/universe_sector_etfs.yaml")
    if config_path.exists():
        config = load_experiment_config(config_path)
        print(f"  ✓ Config loaded: {config.get('universe', {}).get('name', 'unknown')}")
    else:
        print(f"  ⚠ Config file not found: {config_path}")
    
except Exception as e:
    print(f"  ✗ Functionality test FAILED: {e}")
    sys.exit(1)

print("=" * 50)
print("✓ All tests passed!")
print("\nYou can now run experiments:")
print("  python run_experiment.py --config configs/universe_sector_etfs.yaml --mode signal")
