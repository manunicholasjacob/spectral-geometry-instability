"""
Factor Model Integration module for the SGI project.
Compares SGI with factor loading instability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression

from .constants import EPSILON


def download_fama_french_factors(
    start_date: str = "2005-01-01",
    end_date: str = "2024-12-31"
) -> pd.DataFrame:
    """
    Download Fama-French factor data.
    
    Note: This uses a simplified approach. For production, use
    pandas_datareader or download directly from Ken French's website.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with factor returns
    """
    try:
        import pandas_datareader.data as web
        
        ff_factors = web.DataReader(
            'F-F_Research_Data_Factors_daily',
            'famafrench',
            start=start_date,
            end=end_date
        )[0]
        
        # Convert to decimal
        ff_factors = ff_factors / 100
        
        return ff_factors
        
    except Exception:
        # Return empty DataFrame if download fails
        # User should provide their own factor data
        return pd.DataFrame()


def compute_rolling_factor_loadings(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    window: int = 126
) -> Dict[str, pd.DataFrame]:
    """
    Compute rolling factor loadings (betas) for each asset.
    
    Args:
        returns: Asset returns DataFrame
        factors: Factor returns DataFrame
        window: Rolling window size
        
    Returns:
        Dictionary mapping factor name to DataFrame of loadings
    """
    # Align indices
    common_idx = returns.index.intersection(factors.index)
    returns = returns.loc[common_idx]
    factors = factors.loc[common_idx]
    
    # Get factor names (exclude RF if present)
    factor_names = [c for c in factors.columns if c != 'RF']
    
    # Initialize results
    loadings = {f: pd.DataFrame(index=returns.index, columns=returns.columns) 
                for f in factor_names}
    
    # Rolling regression
    for i in range(window, len(returns)):
        start_idx = i - window
        end_idx = i
        
        window_returns = returns.iloc[start_idx:end_idx]
        window_factors = factors.iloc[start_idx:end_idx]
        
        # Excess returns (subtract RF if available)
        if 'RF' in factors.columns:
            excess_returns = window_returns.sub(window_factors['RF'], axis=0)
        else:
            excess_returns = window_returns
        
        # Regress each asset on factors
        X = window_factors[factor_names].values
        
        for asset in returns.columns:
            y = excess_returns[asset].values
            
            # Skip if NaN
            mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            if mask.sum() < window // 2:
                continue
            
            try:
                model = LinearRegression()
                model.fit(X[mask], y[mask])
                
                date = returns.index[end_idx - 1]
                for j, factor in enumerate(factor_names):
                    loadings[factor].loc[date, asset] = model.coef_[j]
                    
            except Exception:
                continue
    
    return loadings


def compute_factor_loading_instability(
    loadings: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compute instability of factor loadings over time.
    
    Args:
        loadings: Dictionary of factor name -> loadings DataFrame
        
    Returns:
        DataFrame with loading instability metrics
    """
    results = pd.DataFrame()
    
    for factor, loading_df in loadings.items():
        # Cross-sectional standard deviation of loadings
        results[f'{factor}_loading_std'] = loading_df.std(axis=1)
        
        # Change in loadings
        loading_change = loading_df.diff()
        results[f'{factor}_loading_change_std'] = loading_change.std(axis=1)
        
        # Average absolute change
        results[f'{factor}_loading_abs_change'] = loading_change.abs().mean(axis=1)
    
    # Aggregate instability
    change_cols = [c for c in results.columns if 'change_std' in c]
    if change_cols:
        results['total_loading_instability'] = results[change_cols].mean(axis=1)
    
    return results


def compute_factor_covariance_instability(
    factors: pd.DataFrame,
    window: int = 126
) -> pd.DataFrame:
    """
    Compute instability of factor covariance matrix.
    
    Args:
        factors: Factor returns DataFrame
        window: Rolling window size
        
    Returns:
        DataFrame with factor covariance instability
    """
    from .spectral_geometry import compute_principal_angles, eigendecompose_covariance
    
    factor_names = [c for c in factors.columns if c != 'RF']
    factor_returns = factors[factor_names]
    
    results = pd.DataFrame(index=factors.index)
    
    prev_eigenvectors = None
    
    for i in range(window, len(factors)):
        start_idx = i - window
        end_idx = i
        
        window_factors = factor_returns.iloc[start_idx:end_idx]
        
        # Compute factor covariance
        cov = window_factors.cov().values
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eigendecompose_covariance(cov)
        
        date = factors.index[end_idx - 1]
        
        # Absorption ratio
        results.loc[date, 'factor_absorption_ratio'] = eigenvalues[0] / eigenvalues.sum()
        
        # Factor SGI
        if prev_eigenvectors is not None:
            angles = compute_principal_angles(eigenvectors, prev_eigenvectors)
            results.loc[date, 'factor_sgi'] = np.sqrt(np.sum(angles ** 2))
        
        prev_eigenvectors = eigenvectors
    
    return results


def compare_sgi_with_factor_instability(
    geometry_df: pd.DataFrame,
    factor_instability: pd.DataFrame
) -> Dict:
    """
    Compare asset SGI with factor loading instability.
    
    Args:
        geometry_df: Asset geometry DataFrame
        factor_instability: Factor instability DataFrame
        
    Returns:
        Dictionary with comparison results
    """
    # Align indices
    common_idx = geometry_df.index.intersection(factor_instability.index)
    
    sgi = geometry_df.loc[common_idx, 'sgi']
    
    results = {
        'n_observations': len(common_idx),
        'correlations': {},
        'lead_lag': {},
    }
    
    # Correlations with each instability metric
    for col in factor_instability.columns:
        metric = factor_instability.loc[common_idx, col]
        corr = sgi.corr(metric)
        if not np.isnan(corr):
            results['correlations'][col] = float(corr)
    
    # Lead-lag with total loading instability
    if 'total_loading_instability' in factor_instability.columns:
        instability = factor_instability.loc[common_idx, 'total_loading_instability']
        
        for lag in [-10, -5, -1, 0, 1, 5, 10]:
            if lag < 0:
                corr = sgi.iloc[:lag].corr(instability.iloc[-lag:])
            elif lag > 0:
                corr = sgi.iloc[lag:].corr(instability.iloc[:-lag])
            else:
                corr = sgi.corr(instability)
            
            if not np.isnan(corr):
                results['lead_lag'][f'lag_{lag}'] = float(corr)
    
    return results


def compute_factor_adjusted_sgi(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    window: int = 126,
    top_k: int = 3
) -> pd.DataFrame:
    """
    Compute SGI on factor-adjusted (residual) returns.
    
    Args:
        returns: Asset returns DataFrame
        factors: Factor returns DataFrame
        window: Rolling window size
        top_k: Number of top eigenvectors
        
    Returns:
        DataFrame with factor-adjusted SGI
    """
    from .covariance import rolling_covariance_matrices
    from .spectral_geometry import compute_rolling_geometry_features
    
    # Align indices
    common_idx = returns.index.intersection(factors.index)
    returns = returns.loc[common_idx]
    factors = factors.loc[common_idx]
    
    # Get factor names
    factor_names = [c for c in factors.columns if c != 'RF']
    
    # Compute residual returns
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    for i in range(window, len(returns)):
        start_idx = i - window
        end_idx = i
        
        window_returns = returns.iloc[start_idx:end_idx]
        window_factors = factors.iloc[start_idx:end_idx][factor_names]
        
        # Regress each asset on factors
        X = window_factors.values
        
        for asset in returns.columns:
            y = window_returns[asset].values
            
            mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            if mask.sum() < window // 2:
                continue
            
            try:
                model = LinearRegression()
                model.fit(X[mask], y[mask])
                
                # Compute residual for last observation
                date = returns.index[end_idx - 1]
                predicted = model.predict(factors.loc[date, factor_names].values.reshape(1, -1))[0]
                residuals.loc[date, asset] = returns.loc[date, asset] - predicted
                
            except Exception:
                continue
    
    # Drop NaN rows
    residuals = residuals.dropna()
    
    # Compute SGI on residuals
    cov_matrices = rolling_covariance_matrices(residuals, window)
    geometry_df = compute_rolling_geometry_features(cov_matrices, top_k)
    
    # Rename columns
    geometry_df = geometry_df.add_prefix('residual_')
    
    return geometry_df


def build_factor_enhanced_features(
    geometry_df: pd.DataFrame,
    factor_instability: pd.DataFrame,
    residual_geometry: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build feature DataFrame with factor-related features.
    
    Args:
        geometry_df: Asset geometry DataFrame
        factor_instability: Factor instability DataFrame
        residual_geometry: Optional residual geometry DataFrame
        
    Returns:
        Combined feature DataFrame
    """
    # Align indices
    common_idx = geometry_df.index.intersection(factor_instability.index)
    
    result = geometry_df.loc[common_idx].copy()
    
    # Add factor instability features
    for col in factor_instability.columns:
        result[col] = factor_instability.loc[common_idx, col]
    
    # Add residual geometry if available
    if residual_geometry is not None:
        common_idx = result.index.intersection(residual_geometry.index)
        result = result.loc[common_idx]
        
        for col in residual_geometry.columns:
            result[col] = residual_geometry.loc[common_idx, col]
    
    # Interaction terms
    if 'total_loading_instability' in result.columns:
        result['sgi_x_loading_instability'] = (
            result['sgi'] * result['total_loading_instability']
        )
    
    if 'factor_sgi' in result.columns:
        result['sgi_x_factor_sgi'] = result['sgi'] * result['factor_sgi']
    
    return result
