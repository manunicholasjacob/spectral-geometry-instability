"""
Spectral Geometry module for the SGI project.
This is the heart of the project - computes SGI and related metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.linalg import subspace_angles

from .constants import EPSILON, MIN_EIGENVALUE


def eigendecompose_covariance(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecompose a covariance matrix, sorted by descending eigenvalues.
    
    Args:
        cov: Covariance matrix (n x n)
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted descending by eigenvalue
        - eigenvalues: shape (n,)
        - eigenvectors: shape (n, n), columns are eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def get_topk_eigenspace(
    cov: np.ndarray,
    top_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract top-k eigenvalues and eigenvectors from covariance matrix.
    
    Args:
        cov: Covariance matrix
        top_k: Number of top eigenvectors to extract
        
    Returns:
        Tuple of (top_k eigenvalues, top_k eigenvectors)
    """
    eigenvalues, eigenvectors = eigendecompose_covariance(cov)
    
    # Ensure top_k doesn't exceed matrix dimension
    top_k = min(top_k, len(eigenvalues))
    
    return eigenvalues[:top_k], eigenvectors[:, :top_k]


def compute_principal_angles(
    v_current: np.ndarray,
    v_previous: np.ndarray
) -> np.ndarray:
    """
    Compute principal angles between two subspaces.
    
    Principal angles measure the "rotation" between subspaces,
    independent of sign flips or basis choice.
    
    Args:
        v_current: Current eigenvector matrix (n x k)
        v_previous: Previous eigenvector matrix (n x k)
        
    Returns:
        Array of principal angles in radians, sorted ascending
    """
    # Use scipy's subspace_angles which handles numerical issues
    angles = subspace_angles(v_current, v_previous)
    
    # Ensure angles are in [0, pi/2]
    angles = np.clip(angles, 0, np.pi / 2)
    
    return angles


def compute_sgi_from_angles(angles: np.ndarray) -> float:
    """
    Compute unweighted SGI from principal angles.
    
    SGI = sqrt(sum(theta_i^2))
    
    Args:
        angles: Array of principal angles in radians
        
    Returns:
        SGI value
    """
    return np.sqrt(np.sum(angles ** 2))


def compute_weighted_sgi_from_angles(
    angles: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Compute weighted SGI from principal angles.
    
    WSGI = sqrt(sum(w_i * theta_i^2))
    
    Args:
        angles: Array of principal angles in radians
        weights: Normalized weights (should sum to 1)
        
    Returns:
        Weighted SGI value
    """
    # Normalize weights
    weights = weights / (weights.sum() + EPSILON)
    
    return np.sqrt(np.sum(weights * angles ** 2))


def compute_max_angle(angles: np.ndarray) -> float:
    """
    Compute maximum principal angle.
    
    Args:
        angles: Array of principal angles
        
    Returns:
        Maximum angle in radians
    """
    return float(np.max(angles))


def compute_mean_angle(angles: np.ndarray) -> float:
    """
    Compute mean principal angle.
    
    Args:
        angles: Array of principal angles
        
    Returns:
        Mean angle in radians
    """
    return float(np.mean(angles))


def compute_absorption_ratio(
    eigenvalues: np.ndarray,
    top_k: int
) -> float:
    """
    Compute absorption ratio (fraction of variance explained by top-k).
    
    Args:
        eigenvalues: All eigenvalues (sorted descending)
        top_k: Number of top eigenvalues
        
    Returns:
        Absorption ratio in [0, 1]
    """
    total_var = np.sum(eigenvalues)
    if total_var <= EPSILON:
        return 0.0
    
    top_k = min(top_k, len(eigenvalues))
    top_var = np.sum(eigenvalues[:top_k])
    
    return float(top_var / total_var)


def compute_eigenvalue_concentration(
    eigenvalues: np.ndarray,
    top_k: int
) -> float:
    """
    Compute eigenvalue concentration (Herfindahl-like measure).
    
    Args:
        eigenvalues: All eigenvalues (sorted descending)
        top_k: Number of top eigenvalues to consider
        
    Returns:
        Concentration measure
    """
    top_k = min(top_k, len(eigenvalues))
    top_eigenvalues = eigenvalues[:top_k]
    
    total = np.sum(top_eigenvalues)
    if total <= EPSILON:
        return 0.0
    
    normalized = top_eigenvalues / total
    return float(np.sum(normalized ** 2))


def compute_eigengap(eigenvalues: np.ndarray, k: int) -> float:
    """
    Compute eigengap between k-th and (k+1)-th eigenvalue.
    
    Args:
        eigenvalues: Eigenvalues sorted descending
        k: Index of eigenvalue (1-indexed, so k=3 means gap after 3rd)
        
    Returns:
        Eigengap (difference between consecutive eigenvalues)
    """
    if k >= len(eigenvalues):
        return 0.0
    
    return float(eigenvalues[k-1] - eigenvalues[k])


def compute_relative_eigengap(eigenvalues: np.ndarray, k: int) -> float:
    """
    Compute relative eigengap (normalized by eigenvalue magnitude).
    
    Args:
        eigenvalues: Eigenvalues sorted descending
        k: Index of eigenvalue
        
    Returns:
        Relative eigengap
    """
    if k >= len(eigenvalues):
        return 0.0
    
    gap = eigenvalues[k-1] - eigenvalues[k]
    avg = (eigenvalues[k-1] + eigenvalues[k]) / 2
    
    if avg <= EPSILON:
        return 0.0
    
    return float(gap / avg)


def compute_rolling_geometry_features(
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    top_k: int,
    compute_eigengap: bool = True
) -> pd.DataFrame:
    """
    Compute rolling geometry features from covariance matrices.
    
    This is the main function that produces the SGI time series.
    
    Args:
        cov_matrices: Dictionary mapping dates to covariance matrices
        top_k: Number of top eigenvectors to use
        compute_eigengap: Whether to compute eigengap metrics
        
    Returns:
        DataFrame with geometry features indexed by date
    """
    dates = sorted(cov_matrices.keys())
    
    records = []
    prev_eigenvectors = None
    
    for date in dates:
        cov = cov_matrices[date]
        
        # Get eigendecomposition
        eigenvalues, eigenvectors = eigendecompose_covariance(cov)
        top_eigenvalues = eigenvalues[:top_k]
        top_eigenvectors = eigenvectors[:, :top_k]
        
        record = {
            "date": date,
            "top_k": top_k,
            "absorption_ratio": compute_absorption_ratio(eigenvalues, top_k),
            "eigen_concentration": compute_eigenvalue_concentration(eigenvalues, top_k),
        }
        
        # Compute eigengap if requested
        if compute_eigengap:
            record["eigengap"] = compute_eigengap(eigenvalues, top_k)
            record["relative_eigengap"] = compute_relative_eigengap(eigenvalues, top_k)
        
        # Compute SGI metrics (requires previous eigenspace)
        if prev_eigenvectors is not None:
            angles = compute_principal_angles(top_eigenvectors, prev_eigenvectors)
            
            # Normalize eigenvalues for weighting
            weights = top_eigenvalues / (top_eigenvalues.sum() + EPSILON)
            
            record["sgi"] = compute_sgi_from_angles(angles)
            record["weighted_sgi"] = compute_weighted_sgi_from_angles(angles, weights)
            record["max_angle"] = compute_max_angle(angles)
            record["mean_angle"] = compute_mean_angle(angles)
            
            # Store individual angles
            for i, angle in enumerate(angles):
                record[f"angle_{i+1}"] = angle
        else:
            # First observation - no previous eigenspace
            record["sgi"] = np.nan
            record["weighted_sgi"] = np.nan
            record["max_angle"] = np.nan
            record["mean_angle"] = np.nan
        
        records.append(record)
        prev_eigenvectors = top_eigenvectors
    
    df = pd.DataFrame(records)
    df.set_index("date", inplace=True)
    
    return df


def compute_sgi_derived_features(geometry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived SGI features (z-scores, percentiles, EMA, etc.).
    
    Args:
        geometry_df: DataFrame with base geometry features
        
    Returns:
        DataFrame with additional derived features
    """
    df = geometry_df.copy()
    
    # Z-score of SGI (rolling)
    sgi = df["sgi"]
    rolling_mean = sgi.rolling(252, min_periods=60).mean()
    rolling_std = sgi.rolling(252, min_periods=60).std()
    df["sgi_zscore"] = (sgi - rolling_mean) / (rolling_std + EPSILON)
    
    # Percentile of SGI (expanding)
    df["sgi_percentile"] = sgi.expanding(min_periods=60).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) > 1 else 50
    )
    
    # EMA-smoothed SGI
    df["sgi_ema"] = sgi.ewm(span=20, min_periods=10).mean()
    
    # SGI jump (day-over-day change)
    df["sgi_jump"] = sgi.diff()
    
    # SGI persistence (rolling mean)
    df["sgi_persistence"] = sgi.rolling(20, min_periods=10).mean()
    
    # SGI / volatility ratio (if realized_vol exists)
    if "realized_vol" in df.columns:
        df["sgi_vol_ratio"] = sgi / (df["realized_vol"] + EPSILON)
    
    return df


def compute_subspace_drift_speed(
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    top_k: int,
    window: int = 20
) -> pd.Series:
    """
    Compute cumulative subspace rotation over a rolling window.
    
    Args:
        cov_matrices: Dictionary of covariance matrices
        top_k: Number of top eigenvectors
        window: Rolling window for cumulative rotation
        
    Returns:
        Series of cumulative rotation values
    """
    geometry_df = compute_rolling_geometry_features(cov_matrices, top_k)
    
    # Cumulative SGI over window
    drift_speed = geometry_df["sgi"].rolling(window, min_periods=1).sum()
    
    return drift_speed
