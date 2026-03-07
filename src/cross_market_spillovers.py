"""
Cross-market Spillovers module for the SGI project.
Analyzes SGI contagion and lead-lag relationships across asset universes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from .data_loader import load_or_download_prices
from .preprocessing import preprocess_data
from .covariance import rolling_covariance_matrices
from .spectral_geometry import compute_rolling_geometry_features
from .constants import (
    SECTOR_ETF_TICKERS, MULTIASSET_ETF_TICKERS, LARGECAP_50_TICKERS, EPSILON
)


def compute_sgi_for_universe(
    tickers: List[str],
    start_date: str,
    end_date: str,
    window: int = 126,
    top_k: int = 3,
    estimator: str = "sample"
) -> pd.DataFrame:
    """
    Compute SGI for a specific universe.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        window: Rolling window size
        top_k: Number of top eigenvectors
        estimator: Covariance estimator
        
    Returns:
        DataFrame with SGI metrics
    """
    # Load and preprocess
    prices = load_or_download_prices(tickers, start_date, end_date)
    prices, returns, _ = preprocess_data(prices, save_outputs=False)
    
    # Compute covariance
    cov_matrices = rolling_covariance_matrices(returns, window, method=estimator)
    
    # Compute geometry
    geometry_df = compute_rolling_geometry_features(cov_matrices, top_k)
    
    return geometry_df


def compute_multi_universe_sgi(
    universes: Dict[str, List[str]],
    start_date: str = "2005-01-01",
    end_date: str = "2024-12-31",
    window: int = 126,
    top_k: int = 3
) -> Dict[str, pd.DataFrame]:
    """
    Compute SGI for multiple universes.
    
    Args:
        universes: Dictionary of universe name -> tickers
        start_date: Start date
        end_date: End date
        window: Rolling window
        top_k: Number of top eigenvectors
        
    Returns:
        Dictionary of universe name -> geometry DataFrame
    """
    results = {}
    
    for name, tickers in universes.items():
        try:
            geometry_df = compute_sgi_for_universe(
                tickers, start_date, end_date, window, top_k
            )
            results[name] = geometry_df
        except Exception as e:
            print(f"Error computing SGI for {name}: {e}")
    
    return results


def compute_cross_universe_correlation(
    universe_sgi: Dict[str, pd.DataFrame],
    metric: str = "sgi"
) -> pd.DataFrame:
    """
    Compute correlation matrix of SGI across universes.
    
    Args:
        universe_sgi: Dictionary of universe name -> geometry DataFrame
        metric: Metric to correlate
        
    Returns:
        Correlation matrix DataFrame
    """
    # Extract metric from each universe
    sgi_series = {}
    for name, df in universe_sgi.items():
        if metric in df.columns:
            sgi_series[name] = df[metric]
    
    # Combine into DataFrame
    combined = pd.DataFrame(sgi_series)
    
    return combined.corr()


def compute_spillover_index(
    universe_sgi: Dict[str, pd.DataFrame],
    metric: str = "sgi",
    forecast_horizon: int = 10,
    var_lags: int = 5
) -> Dict:
    """
    Compute Diebold-Yilmaz style spillover index.
    
    Args:
        universe_sgi: Dictionary of universe name -> geometry DataFrame
        metric: Metric to analyze
        forecast_horizon: Forecast horizon for variance decomposition
        var_lags: Number of VAR lags
        
    Returns:
        Dictionary with spillover analysis
    """
    from statsmodels.tsa.api import VAR
    
    # Extract metric from each universe
    sgi_series = {}
    for name, df in universe_sgi.items():
        if metric in df.columns:
            sgi_series[name] = df[metric]
    
    # Combine and align
    combined = pd.DataFrame(sgi_series).dropna()
    
    if len(combined) < var_lags + forecast_horizon + 100:
        return {"error": "Insufficient data for spillover analysis"}
    
    # Fit VAR model
    try:
        model = VAR(combined)
        results = model.fit(var_lags)
        
        # Forecast error variance decomposition
        fevd = results.fevd(forecast_horizon)
        
        # Compute spillover table
        n_vars = len(combined.columns)
        spillover_table = np.zeros((n_vars, n_vars))
        
        for i, name in enumerate(combined.columns):
            decomp = fevd.decomp[i]
            # Take the last horizon
            spillover_table[i, :] = decomp[-1, :]
        
        # Normalize rows to sum to 100
        spillover_table = spillover_table / spillover_table.sum(axis=1, keepdims=True) * 100
        
        # Create DataFrame
        spillover_df = pd.DataFrame(
            spillover_table,
            index=combined.columns,
            columns=combined.columns
        )
        
        # Compute summary statistics
        own_contribution = np.diag(spillover_table).mean()
        cross_contribution = 100 - own_contribution
        
        # Directional spillovers
        to_others = spillover_table.sum(axis=0) - np.diag(spillover_table)
        from_others = spillover_table.sum(axis=1) - np.diag(spillover_table)
        net_spillover = to_others - from_others
        
        return {
            "spillover_table": spillover_df,
            "total_spillover_index": float(cross_contribution),
            "own_contribution": float(own_contribution),
            "to_others": dict(zip(combined.columns, to_others)),
            "from_others": dict(zip(combined.columns, from_others)),
            "net_spillover": dict(zip(combined.columns, net_spillover)),
            "var_lags": var_lags,
            "forecast_horizon": forecast_horizon,
        }
        
    except Exception as e:
        return {"error": str(e)}


def compute_granger_causality(
    universe_sgi: Dict[str, pd.DataFrame],
    metric: str = "sgi",
    max_lag: int = 10
) -> pd.DataFrame:
    """
    Compute Granger causality between universes.
    
    Args:
        universe_sgi: Dictionary of universe name -> geometry DataFrame
        metric: Metric to analyze
        max_lag: Maximum lag to test
        
    Returns:
        DataFrame with Granger causality p-values
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # Extract metric from each universe
    sgi_series = {}
    for name, df in universe_sgi.items():
        if metric in df.columns:
            sgi_series[name] = df[metric]
    
    # Combine and align
    combined = pd.DataFrame(sgi_series).dropna()
    
    universes = list(combined.columns)
    n = len(universes)
    
    # Initialize p-value matrix
    pvalues = np.ones((n, n))
    
    for i, u1 in enumerate(universes):
        for j, u2 in enumerate(universes):
            if i == j:
                continue
            
            # Test if u2 Granger-causes u1
            data = combined[[u1, u2]].values
            
            try:
                result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                # Get minimum p-value across lags
                min_pvalue = min(
                    result[lag][0]['ssr_ftest'][1] 
                    for lag in range(1, max_lag + 1)
                )
                pvalues[i, j] = min_pvalue
            except Exception:
                pvalues[i, j] = np.nan
    
    return pd.DataFrame(pvalues, index=universes, columns=universes)


def compute_lead_lag_analysis(
    universe_sgi: Dict[str, pd.DataFrame],
    metric: str = "sgi",
    max_lag: int = 20
) -> Dict:
    """
    Compute lead-lag relationships between universes.
    
    Args:
        universe_sgi: Dictionary of universe name -> geometry DataFrame
        metric: Metric to analyze
        max_lag: Maximum lag to test
        
    Returns:
        Dictionary with lead-lag analysis
    """
    # Extract metric from each universe
    sgi_series = {}
    for name, df in universe_sgi.items():
        if metric in df.columns:
            sgi_series[name] = df[metric]
    
    # Combine and align
    combined = pd.DataFrame(sgi_series).dropna()
    
    universes = list(combined.columns)
    results = {}
    
    for i, u1 in enumerate(universes):
        for j, u2 in enumerate(universes):
            if i >= j:
                continue
            
            s1 = combined[u1]
            s2 = combined[u2]
            
            # Compute cross-correlation at different lags
            correlations = {}
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    corr = s1.iloc[:lag].corr(s2.iloc[-lag:])
                elif lag > 0:
                    corr = s1.iloc[lag:].corr(s2.iloc[:-lag])
                else:
                    corr = s1.corr(s2)
                
                if not np.isnan(corr):
                    correlations[lag] = float(corr)
            
            # Find optimal lag
            if correlations:
                optimal_lag = max(correlations, key=correlations.get)
                
                results[f'{u1}_vs_{u2}'] = {
                    'correlations': correlations,
                    'optimal_lag': optimal_lag,
                    'max_correlation': correlations[optimal_lag],
                    'interpretation': f'{u1} leads {u2}' if optimal_lag > 0 else (
                        f'{u2} leads {u1}' if optimal_lag < 0 else 'contemporaneous'
                    ),
                }
    
    return results


def compute_contagion_events(
    universe_sgi: Dict[str, pd.DataFrame],
    metric: str = "sgi",
    threshold_percentile: float = 90,
    window: int = 5
) -> pd.DataFrame:
    """
    Identify contagion events (simultaneous SGI spikes across universes).
    
    Args:
        universe_sgi: Dictionary of universe name -> geometry DataFrame
        metric: Metric to analyze
        threshold_percentile: Percentile for spike detection
        window: Window for simultaneous detection
        
    Returns:
        DataFrame with contagion events
    """
    # Extract metric from each universe
    sgi_series = {}
    for name, df in universe_sgi.items():
        if metric in df.columns:
            sgi_series[name] = df[metric]
    
    # Combine and align
    combined = pd.DataFrame(sgi_series).dropna()
    
    # Compute thresholds
    thresholds = combined.quantile(threshold_percentile / 100)
    
    # Identify spikes
    spikes = combined > thresholds
    
    # Count simultaneous spikes
    spike_count = spikes.sum(axis=1)
    
    # Identify contagion events (more than half of universes spiking)
    n_universes = len(combined.columns)
    contagion_mask = spike_count >= n_universes / 2
    
    # Extract events
    events = []
    in_event = False
    event_start = None
    
    for date, is_contagion in contagion_mask.items():
        if is_contagion and not in_event:
            in_event = True
            event_start = date
        elif not is_contagion and in_event:
            in_event = False
            events.append({
                'start': event_start,
                'end': date,
                'duration': (date - event_start).days,
                'max_spike_count': int(spike_count.loc[event_start:date].max()),
                'affected_universes': list(combined.columns[spikes.loc[event_start:date].any()]),
            })
    
    return pd.DataFrame(events)


def get_default_universes() -> Dict[str, List[str]]:
    """
    Get default universe definitions.
    
    Returns:
        Dictionary of universe name -> tickers
    """
    return {
        'sector_etfs': SECTOR_ETF_TICKERS,
        'multiasset': MULTIASSET_ETF_TICKERS,
        'largecap50': LARGECAP_50_TICKERS,
    }
