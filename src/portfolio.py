"""
Portfolio construction module for the SGI project.
Implements baseline and SGI-aware portfolio strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import cvxpy as cp

from .constants import EPSILON
from .covariance import shrink_covariance


def compute_equal_weight(n_assets: int) -> np.ndarray:
    """
    Compute equal weight portfolio.
    
    Args:
        n_assets: Number of assets
        
    Returns:
        Weight array
    """
    return np.ones(n_assets) / n_assets


def compute_inverse_vol_weights(
    window_returns: pd.DataFrame,
    min_vol: float = 0.001
) -> np.ndarray:
    """
    Compute inverse volatility weights.
    
    Args:
        window_returns: Returns DataFrame for estimation window
        min_vol: Minimum volatility to avoid division issues
        
    Returns:
        Weight array
    """
    vols = window_returns.std().values
    vols = np.maximum(vols, min_vol)
    
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    
    return weights


def compute_min_variance_weights(
    cov: np.ndarray,
    long_only: bool = True,
    max_weight: float = 1.0,
    min_weight: float = 0.0
) -> np.ndarray:
    """
    Compute minimum variance portfolio weights.
    
    Args:
        cov: Covariance matrix
        long_only: Whether to enforce long-only constraint
        max_weight: Maximum weight per asset
        min_weight: Minimum weight per asset
        
    Returns:
        Weight array
    """
    n = cov.shape[0]
    
    # Define optimization problem
    w = cp.Variable(n)
    
    # Objective: minimize portfolio variance
    objective = cp.Minimize(cp.quad_form(w, cov))
    
    # Constraints
    constraints = [cp.sum(w) == 1]
    
    if long_only:
        constraints.append(w >= min_weight)
    
    if max_weight < 1.0:
        constraints.append(w <= max_weight)
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if w.value is None:
            # Fallback to equal weight
            return compute_equal_weight(n)
        
        weights = w.value
        
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        return weights
    
    except Exception:
        return compute_equal_weight(n)


def compute_shrinkage_minvar_weights(
    cov: np.ndarray,
    shrinkage_alpha: float = 0.5,
    long_only: bool = True
) -> np.ndarray:
    """
    Compute minimum variance weights with shrinkage.
    
    Args:
        cov: Covariance matrix
        shrinkage_alpha: Shrinkage intensity
        long_only: Whether to enforce long-only
        
    Returns:
        Weight array
    """
    shrunk_cov = shrink_covariance(cov, shrinkage_alpha, target="diagonal")
    return compute_min_variance_weights(shrunk_cov, long_only=long_only)


def compute_sgi_conditioned_shrinkage_cov(
    cov: np.ndarray,
    sgi_value: float,
    sgi_threshold: float,
    alpha_low: float = 0.0,
    alpha_high: float = 0.5
) -> np.ndarray:
    """
    Compute SGI-conditioned shrinkage covariance.
    
    When SGI is high, apply more shrinkage toward diagonal.
    
    Args:
        cov: Original covariance matrix
        sgi_value: Current SGI value
        sgi_threshold: Threshold for high SGI
        alpha_low: Shrinkage when SGI is low
        alpha_high: Shrinkage when SGI is high
        
    Returns:
        Shrunk covariance matrix
    """
    if np.isnan(sgi_value):
        alpha = alpha_low
    elif sgi_value > sgi_threshold:
        # Linear interpolation above threshold
        alpha = alpha_high
    else:
        # Linear interpolation below threshold
        alpha = alpha_low + (alpha_high - alpha_low) * (sgi_value / sgi_threshold)
    
    return shrink_covariance(cov, alpha, target="diagonal")


def compute_sgi_conditioned_minvar_weights(
    cov: np.ndarray,
    sgi_value: float,
    config: Dict
) -> np.ndarray:
    """
    Compute SGI-conditioned minimum variance weights.
    
    Args:
        cov: Covariance matrix
        sgi_value: Current SGI value
        config: Configuration dictionary
        
    Returns:
        Weight array
    """
    portfolio_config = config.get("portfolio", {})
    
    sgi_threshold = portfolio_config.get("sgi_threshold_percentile", 75) / 100 * np.pi / 2
    alpha_low = portfolio_config.get("shrinkage_alpha_low", 0.0)
    alpha_high = portfolio_config.get("shrinkage_alpha_high", 0.5)
    long_only = portfolio_config.get("long_only", True)
    
    # Apply SGI-conditioned shrinkage
    shrunk_cov = compute_sgi_conditioned_shrinkage_cov(
        cov, sgi_value, sgi_threshold, alpha_low, alpha_high
    )
    
    return compute_min_variance_weights(shrunk_cov, long_only=long_only)


def compute_sgi_leverage_scaling(
    sgi_value: float,
    sgi_percentile: float,
    threshold_percentile: float = 75,
    scale_low: float = 1.0,
    scale_high: float = 0.5
) -> float:
    """
    Compute leverage scaling factor based on SGI.
    
    Args:
        sgi_value: Current SGI value
        sgi_percentile: SGI percentile (0-100)
        threshold_percentile: Threshold for scaling
        scale_low: Scale when SGI is low
        scale_high: Scale when SGI is high
        
    Returns:
        Leverage scaling factor
    """
    if np.isnan(sgi_percentile):
        return scale_low
    
    if sgi_percentile > threshold_percentile:
        # Reduce leverage when SGI is high
        excess = (sgi_percentile - threshold_percentile) / (100 - threshold_percentile)
        return scale_low - (scale_low - scale_high) * excess
    
    return scale_low


def generate_weight_schedule(
    returns: pd.DataFrame,
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    geometry_df: pd.DataFrame,
    strategy: str,
    config: Dict,
    rebalance_freq: str = "weekly"
) -> pd.DataFrame:
    """
    Generate portfolio weight schedule over time.
    
    Args:
        returns: Returns DataFrame
        cov_matrices: Dictionary of covariance matrices
        geometry_df: Geometry features DataFrame
        strategy: Strategy name
        config: Configuration dictionary
        rebalance_freq: Rebalance frequency
        
    Returns:
        DataFrame of weights indexed by date
    """
    from .constants import REBALANCE_FREQ_DAYS
    
    rebal_days = REBALANCE_FREQ_DAYS.get(rebalance_freq, 5)
    
    dates = sorted(cov_matrices.keys())
    assets = returns.columns.tolist()
    n_assets = len(assets)
    
    weight_records = []
    
    for i, date in enumerate(dates):
        # Only rebalance at specified frequency
        if i % rebal_days != 0:
            continue
        
        cov = cov_matrices[date]
        
        # Get SGI value if needed
        sgi_value = np.nan
        if date in geometry_df.index:
            sgi_value = geometry_df.loc[date, "sgi"]
        
        # Compute weights based on strategy
        if strategy == "equal_weight":
            weights = compute_equal_weight(n_assets)
        
        elif strategy == "inverse_vol":
            # Get recent returns for vol estimation
            date_idx = returns.index.get_loc(date)
            window = min(60, date_idx)
            window_returns = returns.iloc[date_idx - window:date_idx]
            weights = compute_inverse_vol_weights(window_returns)
        
        elif strategy == "min_variance":
            weights = compute_min_variance_weights(cov, long_only=True)
        
        elif strategy == "shrinkage_minvar":
            weights = compute_shrinkage_minvar_weights(cov, shrinkage_alpha=0.3)
        
        elif strategy == "sgi_conditioned_minvar":
            weights = compute_sgi_conditioned_minvar_weights(cov, sgi_value, config)
        
        elif strategy == "sgi_conditioned_shrinkage":
            sgi_threshold = config.get("portfolio", {}).get("sgi_threshold_percentile", 75) / 100 * np.pi / 2
            shrunk_cov = compute_sgi_conditioned_shrinkage_cov(
                cov, sgi_value, sgi_threshold
            )
            weights = compute_min_variance_weights(shrunk_cov, long_only=True)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        record = {"date": date}
        for j, asset in enumerate(assets):
            record[asset] = weights[j]
        
        weight_records.append(record)
    
    weight_df = pd.DataFrame(weight_records)
    weight_df.set_index("date", inplace=True)
    
    return weight_df


def apply_leverage_scaling(
    weight_schedule: pd.DataFrame,
    geometry_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Apply SGI-based leverage scaling to weights.
    
    Args:
        weight_schedule: Weight DataFrame
        geometry_df: Geometry features DataFrame
        config: Configuration dictionary
        
    Returns:
        Scaled weight DataFrame
    """
    scaled_weights = weight_schedule.copy()
    
    threshold = config.get("portfolio", {}).get("sgi_conditioning", {}).get("threshold_percentile", 75)
    scale_low = config.get("portfolio", {}).get("sgi_conditioning", {}).get("leverage_scale_low", 1.0)
    scale_high = config.get("portfolio", {}).get("sgi_conditioning", {}).get("leverage_scale_high", 0.5)
    
    for date in scaled_weights.index:
        if date in geometry_df.index and "sgi_percentile" in geometry_df.columns:
            sgi_pct = geometry_df.loc[date, "sgi_percentile"]
            scale = compute_sgi_leverage_scaling(
                geometry_df.loc[date, "sgi"],
                sgi_pct,
                threshold,
                scale_low,
                scale_high
            )
            scaled_weights.loc[date] *= scale
    
    return scaled_weights
