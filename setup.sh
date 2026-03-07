#!/bin/bash
# Setup script for Spectral Geometry Instability project

echo "Setting up Spectral Geometry Instability project..."

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/metadata
mkdir -p outputs/figures
mkdir -p outputs/tables
mkdir -p outputs/logs
mkdir -p outputs/diagnostics
mkdir -p outputs/predictions
mkdir -p outputs/portfolio
mkdir -p outputs/event_studies

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: python run_experiment.py --config configs/universe_sector_etfs.yaml --mode signal"
echo "2. Or open notebooks/01_data_validation.ipynb"
