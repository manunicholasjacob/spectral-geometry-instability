#!/usr/bin/env python3
"""
Run cross-market spillover analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.cross_market_spillovers import (
    compute_multi_universe_sgi, compute_cross_universe_correlation,
    compute_spillover_index, compute_granger_causality,
    compute_lead_lag_analysis, compute_contagion_events,
    get_default_universes
)
from src.paths import get_outputs_dir
from src.utils import save_json


def run_spillover_analysis(
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31"
):
    """Run cross-market spillover analysis."""
    
    print("Running Cross-Market Spillover Analysis")
    print("=" * 50)
    
    # Get default universes
    universes = get_default_universes()
    
    # Compute SGI for each universe
    print("\nComputing SGI for each universe...")
    universe_sgi = compute_multi_universe_sgi(
        universes, start_date, end_date, window=126, top_k=3
    )
    
    print(f"Computed SGI for {len(universe_sgi)} universes")
    
    # Cross-universe correlation
    print("\nComputing cross-universe correlations...")
    corr_matrix = compute_cross_universe_correlation(universe_sgi)
    print(corr_matrix)
    
    # Spillover index
    print("\nComputing spillover index...")
    spillover = compute_spillover_index(universe_sgi)
    
    if 'error' not in spillover:
        print(f"  Total spillover index: {spillover['total_spillover_index']:.1f}%")
        print(f"  Net spillovers: {spillover['net_spillover']}")
    else:
        print(f"  Error: {spillover['error']}")
    
    # Granger causality
    print("\nComputing Granger causality...")
    granger = compute_granger_causality(universe_sgi)
    print(granger)
    
    # Lead-lag analysis
    print("\nComputing lead-lag relationships...")
    lead_lag = compute_lead_lag_analysis(universe_sgi)
    
    for pair, results in lead_lag.items():
        print(f"  {pair}: {results['interpretation']}, max corr = {results['max_correlation']:.3f}")
    
    # Contagion events
    print("\nIdentifying contagion events...")
    contagion = compute_contagion_events(universe_sgi)
    print(f"Found {len(contagion)} contagion events")
    
    # Save results
    output_dir = get_outputs_dir() / "spillover_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    corr_matrix.to_csv(output_dir / "cross_universe_correlation.csv")
    save_json(spillover, output_dir / "spillover_index.json")
    granger.to_csv(output_dir / "granger_causality.csv")
    save_json(lead_lag, output_dir / "lead_lag_analysis.json")
    contagion.to_csv(output_dir / "contagion_events.csv", index=False)
    
    # Save individual universe SGI
    for name, df in universe_sgi.items():
        df.to_csv(output_dir / f"sgi_{name}.csv")
    
    print(f"\nResults saved to: {output_dir}")
    
    return universe_sgi, spillover


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    
    run_spillover_analysis(args.start, args.end)
