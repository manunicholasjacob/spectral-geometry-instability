#!/usr/bin/env python3
"""
Run crypto universe analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.crypto_universe import (
    compute_crypto_sgi, compute_crypto_specific_features,
    compare_crypto_vs_traditional_sgi, run_crypto_event_study,
    CRYPTO_TICKERS, CRYPTO_EVENTS, get_crypto_config
)
from src.config import load_experiment_config
from src.experiments import run_signal_pipeline
from src.paths import get_outputs_dir
from src.utils import save_json


def run_crypto_analysis():
    """Run crypto universe analysis."""
    
    print("Running Crypto Universe Analysis")
    print("=" * 50)
    
    # Compute crypto SGI
    print("\nComputing crypto SGI...")
    crypto_geometry, crypto_features, crypto_metadata = compute_crypto_sgi(
        start_date="2020-01-01",
        end_date="2024-12-31",
        window=60,
        top_k=3
    )
    
    print(f"Crypto geometry shape: {crypto_geometry.shape}")
    print(f"Crypto features shape: {crypto_features.shape}")
    
    # Compute crypto-specific features
    print("\nComputing crypto-specific features...")
    # Need returns for this
    from src.data_loader import load_or_download_prices
    from src.preprocessing import preprocess_data
    
    prices = load_or_download_prices(CRYPTO_TICKERS, "2020-01-01", "2024-12-31")
    prices, returns, _ = preprocess_data(prices, save_outputs=False)
    
    crypto_specific = compute_crypto_specific_features(returns, crypto_geometry)
    
    # Run crypto event study
    print("\nRunning crypto event study...")
    event_results = run_crypto_event_study(crypto_geometry, CRYPTO_EVENTS)
    
    print("\nEvent study results:")
    for event_name, results in event_results.items():
        if 'statistics' in results:
            stats = results['statistics']
            print(f"  {event_name}:")
            print(f"    Event value: {stats.get('event_value', 'N/A')}")
            print(f"    Z-score: {stats.get('event_zscore', 'N/A')}")
    
    # Compare with traditional markets
    print("\nComparing with traditional markets...")
    try:
        # Load traditional market SGI
        trad_config = load_experiment_config("configs/universe_sector_etfs.yaml")
        trad_results = run_signal_pipeline(trad_config)
        trad_geometry = trad_results['geometry_df']
        
        comparison = compare_crypto_vs_traditional_sgi(crypto_geometry, trad_geometry)
        
        print(f"\nCrypto vs Traditional comparison:")
        print(f"  Crypto SGI mean: {comparison['crypto_sgi_stats']['mean']:.4f}")
        print(f"  Traditional SGI mean: {comparison['traditional_sgi_stats']['mean']:.4f}")
        print(f"  Correlation: {comparison['correlation']:.3f}")
        
    except Exception as e:
        print(f"Warning: Could not compare with traditional markets: {e}")
        comparison = None
    
    # Save results
    output_dir = get_outputs_dir() / "crypto_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    crypto_geometry.to_csv(output_dir / "crypto_geometry.csv")
    crypto_features.to_csv(output_dir / "crypto_features.csv")
    crypto_specific.to_csv(output_dir / "crypto_specific_features.csv")
    save_json(event_results, output_dir / "crypto_event_study.json")
    
    if comparison:
        save_json(comparison, output_dir / "crypto_vs_traditional.json")
    
    print(f"\nResults saved to: {output_dir}")
    
    return crypto_geometry, event_results


if __name__ == "__main__":
    run_crypto_analysis()
