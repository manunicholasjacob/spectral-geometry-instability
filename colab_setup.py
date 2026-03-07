#!/usr/bin/env python3
"""
Google Colab setup script for the SGI project.
Run this first in Colab to set up the environment.

Usage in Colab:
    !git clone https://github.com/manunicholasjacob/spectral-geometry-instability.git
    %cd spectral-geometry-instability
    !python colab_setup.py
"""

import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install all required packages."""
    print("Installing required packages...")
    print("=" * 50)
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("ERROR: requirements.txt not found!")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"
        ])
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    print("\nSetting up directories...")
    print("=" * 50)
    
    try:
        from src.paths import ensure_all_directories
        ensure_all_directories()
        print("✓ All directories created")
        return True
    except Exception as e:
        print(f"✗ Directory setup failed: {e}")
        return False

def test_imports():
    """Test that all imports work."""
    print("\nTesting imports...")
    print("=" * 50)
    
    try:
        # Test core imports
        from src import (
            config, constants, paths, utils, logging_utils,
            data_loader, preprocessing, covariance, spectral_geometry,
            features, targets, predictive_models, portfolio, backtest,
            plots, diagnostics, experiments
        )
        print("✓ All core modules imported successfully")
        
        # Test advanced imports
        from src import (
            vix_integration, rmt_denoising, quantile_regression,
            multiscale_sgi, cross_market_spillovers, factor_model,
            crypto_universe
        )
        print("✓ All advanced modules imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup function."""
    print("=" * 50)
    print("SGI Project - Google Colab Setup")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("\n✗ Setup failed at package installation")
        return False
    
    # Step 2: Setup directories
    if not setup_directories():
        print("\n✗ Setup failed at directory creation")
        return False
    
    # Step 3: Test imports
    if not test_imports():
        print("\n✗ Setup failed at import testing")
        return False
    
    print("\n" + "=" * 50)
    print("✓ Setup complete!")
    print("=" * 50)
    print("\nYou can now run experiments:")
    print("  !python run_experiment.py --config configs/universe_sector_etfs.yaml --mode signal")
    print("\nOr use the notebooks:")
    print("  Navigate to notebooks/ and open any .ipynb file")
    print("\nOr run the test script:")
    print("  !python test_imports.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
