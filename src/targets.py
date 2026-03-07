"""
Target variable computation for the SGI project.
Computes forward-looking targets for predictive modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from .constants import EPSILON


def compute_forward_drawdown_target(
    prices: pd.DataFrame,
    horizon: int,
    threshold: Optional[float] = None
) -> pd.Series:
    """
    Compute forward maximum drawdown over horizon.
    
    Args:
        prices: DataFrame of prices
        horizon: Forward horizon in days
        threshold: Optional threshold for binary target
        
    Returns:
        Series of forward drawdown values (or binary if threshold given)
    """
    # Equal-weight portfolio for aggregate drawdown
    portfolio_value = prices.mean(axis=1)
    
    result = []
    for i in range(len(portfolio_value)):
        if i + horizon >= len(portfolio_value):
            result.append(np.nan)
            continue
        
        forward_window = portfolio_value.iloc[i:i + horizon + 1]
        cummax = forward_window.cummax()
        drawdown = (forward_window - cummax) / cummax
        max_dd = abs(drawdown.min())
        
        if threshold is not None:
            result.append(1 if max_dd > threshold else 0)
        else:
            result.append(max_dd)
    
    name = f"forward_drawdown_{horizon}d"
    if threshold:
        name = f"drawdown_event_{horizon}d"
    
    return pd.Series(result, index=prices.index, name=name)


def compute_forward_realized_vol_target(
    returns: pd.DataFrame,
    horizon: int
) -> pd.Series:
    """
    Compute forward realized volatility over horizon.
    
    Args:
        returns: DataFrame of returns
        horizon: Forward horizon in days
        
    Returns:
        Series of forward realized volatility (annualized)
    """
    # Cross-sectional average returns
    avg_returns = returns.mean(axis=1)
    
    result = []
    for i in range(len(avg_returns)):
        if i + horizon >= len(avg_returns):
            result.append(np.nan)
            continue
        
        forward_returns = avg_returns.iloc[i + 1:i + horizon + 1]
        vol = forward_returns.std() * np.sqrt(252)
        result.append(vol)
    
    return pd.Series(result, index=returns.index, name=f"forward_vol_{horizon}d")


def compute_forward_correlation_target(
    returns: pd.DataFrame,
    horizon: int
) -> pd.Series:
    """
    Compute forward average correlation over horizon.
    
    Args:
        returns: DataFrame of returns
        horizon: Forward horizon in days
        
    Returns:
        Series of forward average correlation
    """
    n_assets = returns.shape[1]
    
    result = []
    for i in range(len(returns)):
        if i + horizon >= len(returns):
            result.append(np.nan)
            continue
        
        forward_returns = returns.iloc[i + 1:i + horizon + 1]
        if len(forward_returns) < 5:
            result.append(np.nan)
            continue
        
        corr_matrix = forward_returns.corr().values
        # Upper triangle excluding diagonal
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_corr = corr_matrix[mask].mean()
        result.append(avg_corr)
    
    return pd.Series(result, index=returns.index, name=f"forward_correlation_{horizon}d")


def compute_covariance_forecast_error_target(
    returns: pd.DataFrame,
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    horizon: int,
    method: str = "fro"
) -> pd.Series:
    """
    Compute covariance forecast error (difference between predicted and realized).
    
    Args:
        returns: DataFrame of returns
        cov_matrices: Dictionary of historical covariance matrices
        horizon: Forward horizon for realized covariance
        method: Error metric ("fro" for Frobenius norm, "eigenspace" for subspace angle)
        
    Returns:
        Series of forecast error values
    """
    dates = sorted(cov_matrices.keys())
    
    result = {}
    for date in dates:
        # Get predicted covariance (current estimate)
        predicted_cov = cov_matrices[date]
        
        # Find forward window
        date_idx = returns.index.get_loc(date)
        if date_idx + horizon >= len(returns):
            result[date] = np.nan
            continue
        
        # Compute realized covariance
        forward_returns = returns.iloc[date_idx + 1:date_idx + horizon + 1]
        if len(forward_returns) < horizon // 2:
            result[date] = np.nan
            continue
        
        realized_cov = forward_returns.cov().values
        
        if method == "fro":
            # Frobenius norm of difference
            error = np.linalg.norm(predicted_cov - realized_cov, 'fro')
            # Normalize by scale
            scale = np.linalg.norm(predicted_cov, 'fro') + EPSILON
            result[date] = error / scale
        elif method == "eigenspace":
            # Subspace angle between top eigenspaces
            from scipy.linalg import subspace_angles
            
            _, v_pred = np.linalg.eigh(predicted_cov)
            _, v_real = np.linalg.eigh(realized_cov)
            
            # Top 3 eigenvectors
            k = min(3, v_pred.shape[1])
            angles = subspace_angles(v_pred[:, -k:], v_real[:, -k:])
            result[date] = np.mean(angles)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    series = pd.Series(result, name=f"cov_forecast_error_{horizon}d")
    return series.reindex(returns.index)


def compute_correlation_spike_target(
    returns: pd.DataFrame,
    horizon: int,
    threshold: float = 0.7,
    baseline_window: int = 60
) -> pd.Series:
    """
    Compute binary target for correlation spike events.
    
    Args:
        returns: DataFrame of returns
        horizon: Forward horizon
        threshold: Correlation threshold for spike
        baseline_window: Window for baseline correlation
        
    Returns:
        Binary series (1 = spike, 0 = no spike)
    """
    forward_corr = compute_forward_correlation_target(returns, horizon)
    
    # Spike if forward correlation exceeds threshold
    spike = (forward_corr > threshold).astype(int)
    
    return spike.rename(f"correlation_spike_{horizon}d")


def compute_vol_spike_target(
    returns: pd.DataFrame,
    horizon: int,
    multiplier: float = 1.5,
    baseline_window: int = 60
) -> pd.Series:
    """
    Compute binary target for volatility spike events.
    
    Args:
        returns: DataFrame of returns
        horizon: Forward horizon
        multiplier: Multiplier over baseline for spike
        baseline_window: Window for baseline volatility
        
    Returns:
        Binary series (1 = spike, 0 = no spike)
    """
    # Compute baseline volatility
    avg_returns = returns.mean(axis=1)
    baseline_vol = avg_returns.rolling(baseline_window).std() * np.sqrt(252)
    
    # Compute forward volatility
    forward_vol = compute_forward_realized_vol_target(returns, horizon)
    
    # Spike if forward vol exceeds multiplier * baseline
    spike = (forward_vol > multiplier * baseline_vol).astype(int)
    
    return spike.rename(f"vol_spike_{horizon}d")


def build_target_dataframe(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    config: Dict
) -> pd.DataFrame:
    """
    Build comprehensive target DataFrame.
    
    Args:
        prices: Price DataFrame
        returns: Returns DataFrame
        cov_matrices: Dictionary of covariance matrices
        config: Configuration dictionary
        
    Returns:
        DataFrame with all target variables
    """
    horizons = config.get("targets", {}).get("horizons", [5, 10, 20, 60])
    dd_threshold = config.get("targets", {}).get("drawdown_threshold", 0.05)
    corr_threshold = config.get("targets", {}).get("correlation_spike_threshold", 0.7)
    vol_multiplier = config.get("targets", {}).get("vol_spike_multiplier", 1.5)
    
    targets = pd.DataFrame(index=returns.index)
    
    for horizon in horizons:
        # Continuous targets
        targets[f"forward_drawdown_{horizon}d"] = compute_forward_drawdown_target(
            prices, horizon
        )
        targets[f"forward_vol_{horizon}d"] = compute_forward_realized_vol_target(
            returns, horizon
        )
        targets[f"forward_correlation_{horizon}d"] = compute_forward_correlation_target(
            returns, horizon
        )
        
        # Binary targets
        targets[f"drawdown_event_{horizon}d"] = compute_forward_drawdown_target(
            prices, horizon, threshold=dd_threshold
        )
        targets[f"correlation_spike_{horizon}d"] = compute_correlation_spike_target(
            returns, horizon, threshold=corr_threshold
        )
        targets[f"vol_spike_{horizon}d"] = compute_vol_spike_target(
            returns, horizon, multiplier=vol_multiplier
        )
    
    # Covariance forecast error (use medium horizon)
    medium_horizon = horizons[len(horizons) // 2] if horizons else 20
    targets[f"cov_forecast_error_{medium_horizon}d"] = compute_covariance_forecast_error_target(
        returns, cov_matrices, medium_horizon
    )
    
    return targets
