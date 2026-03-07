"""
Covariance estimation module for the SGI project.
Implements multiple rolling covariance estimators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Callable
from sklearn.covariance import LedoitWolf, OAS

from .utils import is_psd, make_psd
from .constants import EPSILON


def compute_sample_cov(window_returns: pd.DataFrame) -> np.ndarray:
    """
    Compute sample covariance matrix.
    
    Args:
        window_returns: DataFrame of returns for the window
        
    Returns:
        Sample covariance matrix
    """
    return window_returns.cov().values


def compute_ewma_cov(
    window_returns: pd.DataFrame,
    span: int = 60,
    min_periods: int = 20
) -> np.ndarray:
    """
    Compute EWMA (Exponentially Weighted Moving Average) covariance matrix.
    
    Args:
        window_returns: DataFrame of returns for the window
        span: EWMA span parameter
        min_periods: Minimum periods required
        
    Returns:
        EWMA covariance matrix
    """
    # Compute EWMA covariance using pandas
    ewma_cov = window_returns.ewm(span=span, min_periods=min_periods).cov()
    
    # Get the last covariance matrix
    last_date = window_returns.index[-1]
    cov_matrix = ewma_cov.loc[last_date].values
    
    return cov_matrix


def compute_ledoit_wolf_cov(window_returns: pd.DataFrame) -> np.ndarray:
    """
    Compute Ledoit-Wolf shrinkage covariance matrix.
    
    Args:
        window_returns: DataFrame of returns for the window
        
    Returns:
        Ledoit-Wolf covariance matrix
    """
    lw = LedoitWolf()
    lw.fit(window_returns.values)
    return lw.covariance_


def compute_oas_cov(window_returns: pd.DataFrame) -> np.ndarray:
    """
    Compute Oracle Approximating Shrinkage covariance matrix.
    
    Args:
        window_returns: DataFrame of returns for the window
        
    Returns:
        OAS covariance matrix
    """
    oas = OAS()
    oas.fit(window_returns.values)
    return oas.covariance_


def compute_covariance(
    window_returns: pd.DataFrame,
    method: str = "sample",
    **kwargs
) -> np.ndarray:
    """
    Compute covariance matrix using specified method.
    
    Args:
        window_returns: DataFrame of returns for the window
        method: Covariance estimation method
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Covariance matrix
    """
    methods = {
        "sample": compute_sample_cov,
        "ewma": lambda r: compute_ewma_cov(r, **kwargs),
        "ledoit_wolf": compute_ledoit_wolf_cov,
        "oas": compute_oas_cov,
    }
    
    if method not in methods:
        raise ValueError(f"Unknown covariance method: {method}. Available: {list(methods.keys())}")
    
    cov = methods[method](window_returns)
    
    # Ensure PSD
    if not is_psd(cov):
        cov = make_psd(cov)
    
    return cov


def rolling_covariance_matrices(
    returns: pd.DataFrame,
    window: int,
    method: str = "sample",
    min_periods: Optional[int] = None,
    **kwargs
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Compute rolling covariance matrices.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window size
        method: Covariance estimation method
        min_periods: Minimum periods required (defaults to window)
        **kwargs: Additional arguments for covariance method
        
    Returns:
        Dictionary mapping dates to covariance matrices
    """
    if min_periods is None:
        min_periods = window
    
    cov_matrices = {}
    dates = returns.index
    n_obs = len(dates)
    
    for i in range(min_periods, n_obs + 1):
        start_idx = max(0, i - window)
        end_idx = i
        
        window_returns = returns.iloc[start_idx:end_idx]
        
        # Skip if insufficient data
        if len(window_returns) < min_periods:
            continue
        
        # Skip if any column has all NaN
        if window_returns.isnull().all().any():
            continue
        
        try:
            cov = compute_covariance(window_returns, method=method, **kwargs)
            end_date = dates[end_idx - 1]
            cov_matrices[end_date] = cov
        except Exception as e:
            # Skip problematic windows
            continue
    
    return cov_matrices


def compute_correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix from covariance matrix.
    
    Args:
        cov: Covariance matrix
        
    Returns:
        Correlation matrix
    """
    std = np.sqrt(np.diag(cov))
    std[std == 0] = EPSILON  # Avoid division by zero
    
    outer_std = np.outer(std, std)
    corr = cov / outer_std
    
    # Ensure diagonal is exactly 1
    np.fill_diagonal(corr, 1.0)
    
    return corr


def shrink_covariance(
    cov: np.ndarray,
    alpha: float,
    target: str = "diagonal"
) -> np.ndarray:
    """
    Shrink covariance matrix toward a target.
    
    Args:
        cov: Original covariance matrix
        alpha: Shrinkage intensity (0 = no shrinkage, 1 = full shrinkage)
        target: Shrinkage target ("diagonal", "identity", "constant_corr")
        
    Returns:
        Shrunk covariance matrix
    """
    n = cov.shape[0]
    
    if target == "diagonal":
        target_matrix = np.diag(np.diag(cov))
    elif target == "identity":
        avg_var = np.trace(cov) / n
        target_matrix = avg_var * np.eye(n)
    elif target == "constant_corr":
        # Shrink toward constant correlation matrix
        std = np.sqrt(np.diag(cov))
        corr = compute_correlation_from_cov(cov)
        avg_corr = (corr.sum() - n) / (n * (n - 1))
        target_corr = np.full((n, n), avg_corr)
        np.fill_diagonal(target_corr, 1.0)
        target_matrix = np.outer(std, std) * target_corr
    else:
        raise ValueError(f"Unknown shrinkage target: {target}")
    
    return (1 - alpha) * cov + alpha * target_matrix


def compute_rolling_average_correlation(
    returns: pd.DataFrame,
    window: int
) -> pd.Series:
    """
    Compute rolling average pairwise correlation.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window size
        
    Returns:
        Series of average correlations
    """
    def avg_corr(window_data):
        corr = window_data.corr()
        n = len(corr)
        if n < 2:
            return np.nan
        # Average off-diagonal correlation
        mask = ~np.eye(n, dtype=bool)
        return corr.values[mask].mean()
    
    return returns.rolling(window).apply(
        lambda x: avg_corr(returns.loc[x.index]),
        raw=False
    ).iloc[:, 0]
