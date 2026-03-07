"""
VIX Integration module for the SGI project.
Adds VIX as a control variable and analyzes SGI-VIX relationships.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yfinance as yf

from .paths import get_raw_data_dir
from .constants import EPSILON


def download_vix_data(
    start_date: str = "2005-01-01",
    end_date: str = "2024-12-31",
    cache: bool = True
) -> pd.DataFrame:
    """
    Download VIX index data from Yahoo Finance.
    
    Args:
        start_date: Start date string
        end_date: End date string
        cache: Whether to cache data locally
        
    Returns:
        DataFrame with VIX data
    """
    cache_file = get_raw_data_dir() / f"vix_{start_date}_{end_date}.csv"
    
    if cache and cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    # Download VIX
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    
    if cache:
        vix.to_csv(cache_file)
    
    return vix


def compute_vix_features(vix_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIX-derived features.
    
    Args:
        vix_data: Raw VIX DataFrame
        
    Returns:
        DataFrame with VIX features
    """
    # Extract close price
    if 'Close' in vix_data.columns:
        vix = vix_data['Close']
    elif isinstance(vix_data, pd.Series):
        vix = vix_data
    else:
        vix = vix_data.iloc[:, 0]
    
    features = pd.DataFrame(index=vix.index)
    
    # Level
    features['vix_level'] = vix
    
    # Log VIX
    features['vix_log'] = np.log(vix + EPSILON)
    
    # VIX change
    features['vix_change'] = vix.diff()
    features['vix_pct_change'] = vix.pct_change()
    
    # VIX momentum
    features['vix_ma_5'] = vix.rolling(5).mean()
    features['vix_ma_20'] = vix.rolling(20).mean()
    features['vix_ma_60'] = vix.rolling(60).mean()
    
    # VIX relative to moving averages
    features['vix_vs_ma20'] = vix / (features['vix_ma_20'] + EPSILON)
    features['vix_vs_ma60'] = vix / (features['vix_ma_60'] + EPSILON)
    
    # VIX percentile (expanding)
    features['vix_percentile'] = vix.expanding(min_periods=252).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) > 1 else 50
    )
    
    # VIX z-score (rolling)
    rolling_mean = vix.rolling(252, min_periods=60).mean()
    rolling_std = vix.rolling(252, min_periods=60).std()
    features['vix_zscore'] = (vix - rolling_mean) / (rolling_std + EPSILON)
    
    # VIX term structure proxy (if available, use VIX9D vs VIX)
    # For now, use VIX slope approximation
    features['vix_slope'] = features['vix_ma_5'] - features['vix_ma_20']
    
    # VIX regime (high/low)
    features['vix_regime_high'] = (vix > 20).astype(int)
    features['vix_regime_very_high'] = (vix > 30).astype(int)
    
    return features


def compute_sgi_vix_relationship(
    geometry_df: pd.DataFrame,
    vix_features: pd.DataFrame
) -> Dict:
    """
    Analyze relationship between SGI and VIX.
    
    Args:
        geometry_df: DataFrame with SGI metrics
        vix_features: DataFrame with VIX features
        
    Returns:
        Dictionary with relationship statistics
    """
    # Align indices
    common_idx = geometry_df.index.intersection(vix_features.index)
    sgi = geometry_df.loc[common_idx, 'sgi']
    vix = vix_features.loc[common_idx, 'vix_level']
    
    results = {
        'n_observations': len(common_idx),
        'date_range': {
            'start': str(common_idx.min()),
            'end': str(common_idx.max()),
        }
    }
    
    # Correlation analysis
    results['correlation'] = {
        'sgi_vix_level': float(sgi.corr(vix)),
        'sgi_vix_change': float(sgi.corr(vix_features.loc[common_idx, 'vix_change'])),
        'sgi_vix_zscore': float(sgi.corr(vix_features.loc[common_idx, 'vix_zscore'])),
    }
    
    # Rolling correlation
    rolling_corr = sgi.rolling(252).corr(vix)
    results['rolling_correlation'] = {
        'mean': float(rolling_corr.mean()),
        'std': float(rolling_corr.std()),
        'min': float(rolling_corr.min()),
        'max': float(rolling_corr.max()),
    }
    
    # Lead-lag analysis
    lead_lag_corrs = {}
    for lag in [-20, -10, -5, -1, 0, 1, 5, 10, 20]:
        if lag < 0:
            # SGI leads VIX
            corr = sgi.iloc[:lag].corr(vix.iloc[-lag:])
        elif lag > 0:
            # VIX leads SGI
            corr = sgi.iloc[lag:].corr(vix.iloc[:-lag])
        else:
            corr = sgi.corr(vix)
        lead_lag_corrs[f'lag_{lag}'] = float(corr) if not np.isnan(corr) else None
    
    results['lead_lag_correlation'] = lead_lag_corrs
    
    # Conditional analysis
    high_vix_mask = vix > vix.quantile(0.75)
    low_vix_mask = vix < vix.quantile(0.25)
    
    results['conditional_sgi'] = {
        'sgi_when_vix_high': float(sgi[high_vix_mask].mean()),
        'sgi_when_vix_low': float(sgi[low_vix_mask].mean()),
        'sgi_when_vix_normal': float(sgi[~high_vix_mask & ~low_vix_mask].mean()),
    }
    
    return results


def add_vix_controls_to_features(
    features_df: pd.DataFrame,
    vix_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Add VIX control variables to feature DataFrame.
    
    Args:
        features_df: Existing features DataFrame
        vix_features: VIX features DataFrame
        
    Returns:
        Combined features DataFrame
    """
    # Align indices
    common_idx = features_df.index.intersection(vix_features.index)
    
    result = features_df.loc[common_idx].copy()
    
    # Add key VIX features
    vix_cols = ['vix_level', 'vix_log', 'vix_change', 'vix_zscore', 
                'vix_percentile', 'vix_regime_high']
    
    for col in vix_cols:
        if col in vix_features.columns:
            result[col] = vix_features.loc[common_idx, col]
    
    # Add interaction terms
    if 'sgi' in result.columns and 'vix_level' in result.columns:
        result['sgi_x_vix'] = result['sgi'] * result['vix_level']
        result['sgi_x_vix_zscore'] = result['sgi'] * result['vix_zscore']
    
    return result


def compute_sgi_orthogonal_to_vix(
    sgi: pd.Series,
    vix: pd.Series
) -> pd.Series:
    """
    Compute SGI component orthogonal to VIX (residual after regressing on VIX).
    
    Args:
        sgi: SGI series
        vix: VIX series
        
    Returns:
        Orthogonalized SGI series
    """
    from sklearn.linear_model import LinearRegression
    
    # Align
    common_idx = sgi.index.intersection(vix.index)
    sgi_aligned = sgi.loc[common_idx].dropna()
    vix_aligned = vix.loc[sgi_aligned.index].dropna()
    
    # Further align after dropna
    common_idx = sgi_aligned.index.intersection(vix_aligned.index)
    sgi_aligned = sgi_aligned.loc[common_idx]
    vix_aligned = vix_aligned.loc[common_idx]
    
    # Regress SGI on VIX
    X = vix_aligned.values.reshape(-1, 1)
    y = sgi_aligned.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Residual is orthogonal component
    predicted = model.predict(X)
    residual = y - predicted
    
    return pd.Series(residual, index=common_idx, name='sgi_orthogonal')


def compute_vix_adjusted_sgi_features(
    geometry_df: pd.DataFrame,
    vix_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute VIX-adjusted SGI features.
    
    Args:
        geometry_df: Geometry DataFrame
        vix_features: VIX features DataFrame
        
    Returns:
        DataFrame with VIX-adjusted features
    """
    common_idx = geometry_df.index.intersection(vix_features.index)
    
    result = pd.DataFrame(index=common_idx)
    
    # Original SGI
    result['sgi'] = geometry_df.loc[common_idx, 'sgi']
    
    # VIX level
    result['vix_level'] = vix_features.loc[common_idx, 'vix_level']
    
    # Orthogonalized SGI
    result['sgi_orthogonal'] = compute_sgi_orthogonal_to_vix(
        geometry_df.loc[common_idx, 'sgi'],
        vix_features.loc[common_idx, 'vix_level']
    )
    
    # SGI / VIX ratio
    result['sgi_vix_ratio'] = result['sgi'] / (result['vix_level'] / 100 + EPSILON)
    
    # SGI conditional on VIX regime
    vix_high = result['vix_level'] > 20
    result['sgi_vix_high'] = result['sgi'].where(vix_high, np.nan)
    result['sgi_vix_low'] = result['sgi'].where(~vix_high, np.nan)
    
    return result
