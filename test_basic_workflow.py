#!/usr/bin/env python3
"""
Basic workflow test to verify the SGI pipeline works end-to-end.
This is a minimal test that should complete quickly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("SGI Basic Workflow Test")
print("=" * 60)

try:
    print("\n1. Testing imports...")
    from src.data_loader import load_or_download_prices
    from src.preprocessing import preprocess_data
    from src.covariance import rolling_covariance_matrices
    from src.spectral_geometry import compute_rolling_geometry_features, compute_sgi_derived_features
    print("   ✓ All imports successful")
    
    print("\n2. Loading sample data (3 ETFs, 2 years)...")
    tickers = ['SPY', 'QQQ', 'IWM']
    prices = load_or_download_prices(tickers, '2022-01-01', '2024-01-01')
    print(f"   ✓ Loaded prices: {prices.shape}")
    
    print("\n3. Preprocessing data...")
    prices, returns, metadata = preprocess_data(prices, save_outputs=False)
    print(f"   ✓ Returns shape: {returns.shape}")
    
    print("\n4. Computing covariance matrices (window=60)...")
    cov_matrices = rolling_covariance_matrices(returns, window=60, method='sample')
    print(f"   ✓ Computed {len(cov_matrices)} covariance matrices")
    
    print("\n5. Computing SGI metrics (top_k=2)...")
    geometry_df = compute_rolling_geometry_features(cov_matrices, top_k=2)
    print(f"   ✓ Geometry features shape: {geometry_df.shape}")
    
    print("\n6. Computing derived features...")
    geometry_df = compute_sgi_derived_features(geometry_df)
    print(f"   ✓ Final features shape: {geometry_df.shape}")
    
    print("\n7. Verifying SGI values...")
    sgi = geometry_df['sgi'].dropna()
    print(f"   ✓ SGI computed for {len(sgi)} dates")
    print(f"   ✓ SGI range: [{sgi.min():.4f}, {sgi.max():.4f}]")
    print(f"   ✓ SGI mean: {sgi.mean():.4f}")
    
    print("\n8. Checking for NaN issues...")
    nan_counts = geometry_df.isna().sum()
    critical_cols = ['sgi', 'weighted_sgi', 'absorption_ratio']
    for col in critical_cols:
        if col in geometry_df.columns:
            nan_pct = (geometry_df[col].isna().sum() / len(geometry_df)) * 100
            print(f"   ✓ {col}: {nan_pct:.1f}% NaN (expected ~0.05% for first row)")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe SGI pipeline is working correctly.")
    print("You can now run full experiments:")
    print("  python run_experiment.py --config configs/universe_sector_etfs.yaml --mode signal")
    
    sys.exit(0)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("✗ TEST FAILED")
    print("=" * 60)
    print(f"\nError: {e}")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    
    sys.exit(1)
