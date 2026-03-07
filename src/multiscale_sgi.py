"""
Multi-scale SGI module for the SGI project.
Computes SGI at multiple time scales simultaneously.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from .covariance import rolling_covariance_matrices
from .spectral_geometry import compute_rolling_geometry_features, compute_principal_angles
from .constants import EPSILON


def compute_multiscale_sgi(
    returns: pd.DataFrame,
    windows: List[int] = [60, 126, 252, 504],
    top_k: int = 3,
    estimator: str = "sample"
) -> pd.DataFrame:
    """
    Compute SGI at multiple time scales.
    
    Args:
        returns: Returns DataFrame
        windows: List of rolling window sizes
        top_k: Number of top eigenvectors
        estimator: Covariance estimator
        
    Returns:
        DataFrame with SGI at each scale
    """
    all_sgi = {}
    
    for window in windows:
        # Compute covariance matrices
        cov_matrices = rolling_covariance_matrices(
            returns, window, method=estimator
        )
        
        # Compute geometry features
        geometry_df = compute_rolling_geometry_features(cov_matrices, top_k)
        
        # Store SGI with window suffix
        all_sgi[f'sgi_w{window}'] = geometry_df['sgi']
        all_sgi[f'weighted_sgi_w{window}'] = geometry_df['weighted_sgi']
        all_sgi[f'absorption_ratio_w{window}'] = geometry_df['absorption_ratio']
    
    # Combine into single DataFrame
    result = pd.DataFrame(all_sgi)
    
    # Add derived features
    result = add_multiscale_derived_features(result, windows)
    
    return result


def add_multiscale_derived_features(
    multiscale_df: pd.DataFrame,
    windows: List[int]
) -> pd.DataFrame:
    """
    Add derived features from multi-scale SGI.
    
    Args:
        multiscale_df: DataFrame with SGI at multiple scales
        windows: List of window sizes
        
    Returns:
        DataFrame with additional derived features
    """
    df = multiscale_df.copy()
    
    # Average SGI across scales
    sgi_cols = [f'sgi_w{w}' for w in windows if f'sgi_w{w}' in df.columns]
    if sgi_cols:
        df['sgi_avg'] = df[sgi_cols].mean(axis=1)
        df['sgi_std'] = df[sgi_cols].std(axis=1)
        df['sgi_max'] = df[sgi_cols].max(axis=1)
        df['sgi_min'] = df[sgi_cols].min(axis=1)
    
    # SGI term structure (short vs long)
    if len(windows) >= 2:
        short_window = min(windows)
        long_window = max(windows)
        
        short_col = f'sgi_w{short_window}'
        long_col = f'sgi_w{long_window}'
        
        if short_col in df.columns and long_col in df.columns:
            df['sgi_term_spread'] = df[short_col] - df[long_col]
            df['sgi_term_ratio'] = df[short_col] / (df[long_col] + EPSILON)
    
    # SGI momentum (change in short-term SGI)
    if f'sgi_w{min(windows)}' in df.columns:
        short_sgi = df[f'sgi_w{min(windows)}']
        df['sgi_momentum_5d'] = short_sgi.diff(5)
        df['sgi_momentum_20d'] = short_sgi.diff(20)
    
    # Scale agreement (do all scales agree on direction?)
    if len(sgi_cols) >= 2:
        sgi_changes = df[sgi_cols].diff()
        df['sgi_scale_agreement'] = (sgi_changes > 0).sum(axis=1) / len(sgi_cols)
    
    return df


def compute_scale_correlation_matrix(
    multiscale_df: pd.DataFrame,
    windows: List[int]
) -> pd.DataFrame:
    """
    Compute correlation matrix between SGI at different scales.
    
    Args:
        multiscale_df: DataFrame with multi-scale SGI
        windows: List of window sizes
        
    Returns:
        Correlation matrix DataFrame
    """
    sgi_cols = [f'sgi_w{w}' for w in windows if f'sgi_w{w}' in multiscale_df.columns]
    return multiscale_df[sgi_cols].corr()


def compute_scale_lead_lag(
    multiscale_df: pd.DataFrame,
    windows: List[int],
    max_lag: int = 20
) -> Dict:
    """
    Compute lead-lag relationships between scales.
    
    Args:
        multiscale_df: DataFrame with multi-scale SGI
        windows: List of window sizes
        max_lag: Maximum lag to test
        
    Returns:
        Dictionary with lead-lag analysis
    """
    results = {}
    
    sorted_windows = sorted(windows)
    
    for i, w1 in enumerate(sorted_windows):
        for w2 in sorted_windows[i+1:]:
            col1 = f'sgi_w{w1}'
            col2 = f'sgi_w{w2}'
            
            if col1 not in multiscale_df.columns or col2 not in multiscale_df.columns:
                continue
            
            sgi1 = multiscale_df[col1].dropna()
            sgi2 = multiscale_df[col2].dropna()
            
            # Align
            common_idx = sgi1.index.intersection(sgi2.index)
            sgi1 = sgi1.loc[common_idx]
            sgi2 = sgi2.loc[common_idx]
            
            lead_lag_corrs = {}
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    corr = sgi1.iloc[:lag].corr(sgi2.iloc[-lag:])
                elif lag > 0:
                    corr = sgi1.iloc[lag:].corr(sgi2.iloc[:-lag])
                else:
                    corr = sgi1.corr(sgi2)
                
                if not np.isnan(corr):
                    lead_lag_corrs[lag] = float(corr)
            
            # Find optimal lag
            if lead_lag_corrs:
                optimal_lag = max(lead_lag_corrs, key=lead_lag_corrs.get)
                results[f'{w1}_vs_{w2}'] = {
                    'correlations': lead_lag_corrs,
                    'optimal_lag': optimal_lag,
                    'max_correlation': lead_lag_corrs[optimal_lag],
                }
    
    return results


def compute_regime_from_multiscale(
    multiscale_df: pd.DataFrame,
    windows: List[int],
    threshold_percentile: float = 75
) -> pd.Series:
    """
    Compute market regime from multi-scale SGI.
    
    Args:
        multiscale_df: DataFrame with multi-scale SGI
        windows: List of window sizes
        threshold_percentile: Percentile threshold for high SGI
        
    Returns:
        Series with regime labels
    """
    sgi_cols = [f'sgi_w{w}' for w in windows if f'sgi_w{w}' in multiscale_df.columns]
    
    if not sgi_cols:
        return pd.Series(index=multiscale_df.index, data='unknown')
    
    # Compute thresholds
    thresholds = {}
    for col in sgi_cols:
        thresholds[col] = multiscale_df[col].quantile(threshold_percentile / 100)
    
    # Count how many scales show high SGI
    high_count = pd.Series(0, index=multiscale_df.index)
    for col in sgi_cols:
        high_count += (multiscale_df[col] > thresholds[col]).astype(int)
    
    # Define regimes
    n_scales = len(sgi_cols)
    
    def classify_regime(count):
        if count == 0:
            return 'stable'
        elif count < n_scales / 2:
            return 'transitioning'
        elif count < n_scales:
            return 'elevated'
        else:
            return 'crisis'
    
    return high_count.apply(classify_regime)


def compute_scale_divergence(
    multiscale_df: pd.DataFrame,
    windows: List[int]
) -> pd.Series:
    """
    Compute divergence between scales (disagreement indicator).
    
    Args:
        multiscale_df: DataFrame with multi-scale SGI
        windows: List of window sizes
        
    Returns:
        Series with divergence values
    """
    sgi_cols = [f'sgi_w{w}' for w in windows if f'sgi_w{w}' in multiscale_df.columns]
    
    if len(sgi_cols) < 2:
        return pd.Series(index=multiscale_df.index, data=0.0)
    
    # Normalize each scale to z-scores
    normalized = pd.DataFrame(index=multiscale_df.index)
    for col in sgi_cols:
        mean = multiscale_df[col].rolling(252, min_periods=60).mean()
        std = multiscale_df[col].rolling(252, min_periods=60).std()
        normalized[col] = (multiscale_df[col] - mean) / (std + EPSILON)
    
    # Divergence = standard deviation across scales
    return normalized.std(axis=1)


def get_multiscale_feature_columns(windows: List[int]) -> List[str]:
    """
    Get list of all multi-scale feature column names.
    
    Args:
        windows: List of window sizes
        
    Returns:
        List of feature column names
    """
    cols = []
    
    for w in windows:
        cols.extend([
            f'sgi_w{w}',
            f'weighted_sgi_w{w}',
            f'absorption_ratio_w{w}',
        ])
    
    # Derived features
    cols.extend([
        'sgi_avg',
        'sgi_std',
        'sgi_max',
        'sgi_min',
        'sgi_term_spread',
        'sgi_term_ratio',
        'sgi_momentum_5d',
        'sgi_momentum_20d',
        'sgi_scale_agreement',
    ])
    
    return cols
