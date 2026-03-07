"""
Feature engineering module for the SGI project.
Computes baseline features and combines with SGI metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .constants import EPSILON


def compute_realized_volatility(
    returns: pd.DataFrame,
    window: int = 20
) -> pd.Series:
    """
    Compute rolling realized volatility (annualized).
    
    Args:
        returns: DataFrame of returns
        window: Rolling window size
        
    Returns:
        Series of realized volatility values
    """
    # Cross-sectional average volatility
    vol = returns.rolling(window).std().mean(axis=1)
    
    # Annualize
    return vol * np.sqrt(252)


def compute_average_correlation(
    returns: pd.DataFrame,
    window: int = 60
) -> pd.Series:
    """
    Compute rolling average pairwise correlation.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window size
        
    Returns:
        Series of average correlation values
    """
    n_assets = returns.shape[1]
    
    def avg_corr_window(window_data):
        if len(window_data) < 10:
            return np.nan
        corr_matrix = window_data.corr().values
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        return corr_matrix[mask].mean()
    
    result = []
    for i in range(len(returns)):
        if i < window - 1:
            result.append(np.nan)
        else:
            window_data = returns.iloc[i - window + 1:i + 1]
            result.append(avg_corr_window(window_data))
    
    return pd.Series(result, index=returns.index, name="avg_correlation")


def compute_market_return(
    returns: pd.DataFrame,
    market_proxy: Optional[str] = None,
    window: int = 20
) -> pd.Series:
    """
    Compute rolling market return.
    
    Args:
        returns: DataFrame of returns
        market_proxy: Column name for market proxy (if None, use equal-weight average)
        window: Rolling window size
        
    Returns:
        Series of market returns
    """
    if market_proxy and market_proxy in returns.columns:
        market_ret = returns[market_proxy]
    else:
        # Equal-weight average
        market_ret = returns.mean(axis=1)
    
    return market_ret.rolling(window).sum()


def compute_cross_sectional_dispersion(
    returns: pd.DataFrame,
    window: int = 20
) -> pd.Series:
    """
    Compute rolling cross-sectional return dispersion.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window size
        
    Returns:
        Series of dispersion values
    """
    # Daily cross-sectional standard deviation
    daily_dispersion = returns.std(axis=1)
    
    # Rolling average
    return daily_dispersion.rolling(window).mean()


def compute_rolling_skewness(
    returns: pd.DataFrame,
    window: int = 60
) -> pd.Series:
    """
    Compute rolling average skewness.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window size
        
    Returns:
        Series of skewness values
    """
    return returns.rolling(window).skew().mean(axis=1)


def compute_rolling_kurtosis(
    returns: pd.DataFrame,
    window: int = 60
) -> pd.Series:
    """
    Compute rolling average kurtosis.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window size
        
    Returns:
        Series of kurtosis values
    """
    return returns.rolling(window).kurt().mean(axis=1)


def compute_drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute drawdown from prices.
    
    Args:
        prices: DataFrame of prices
        
    Returns:
        DataFrame of drawdowns (negative values)
    """
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown


def compute_max_drawdown_rolling(
    prices: pd.DataFrame,
    window: int = 60
) -> pd.Series:
    """
    Compute rolling maximum drawdown.
    
    Args:
        prices: DataFrame of prices
        window: Rolling window size
        
    Returns:
        Series of max drawdown values
    """
    def max_dd(window_prices):
        cummax = window_prices.cummax()
        dd = (window_prices - cummax) / cummax
        return dd.min()
    
    # Average max drawdown across assets
    result = []
    for i in range(len(prices)):
        if i < window - 1:
            result.append(np.nan)
        else:
            window_data = prices.iloc[i - window + 1:i + 1]
            mdd = window_data.apply(max_dd).mean()
            result.append(abs(mdd))
    
    return pd.Series(result, index=prices.index, name="max_drawdown")


def build_feature_dataframe(
    returns: pd.DataFrame,
    geometry_df: pd.DataFrame,
    config: Dict,
    prices: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build comprehensive feature DataFrame.
    
    Args:
        returns: Returns DataFrame
        geometry_df: Geometry features DataFrame (from spectral_geometry)
        config: Configuration dictionary
        prices: Optional prices DataFrame for drawdown features
        
    Returns:
        Combined feature DataFrame
    """
    window = config.get("covariance", {}).get("rolling_window", 126)
    short_window = min(20, window // 6)
    
    # Start with geometry features
    features = geometry_df.copy()
    
    # Add baseline features
    features["realized_vol"] = compute_realized_volatility(returns, short_window)
    features["avg_correlation"] = compute_average_correlation(returns, window)
    features["cross_sectional_dispersion"] = compute_cross_sectional_dispersion(returns, short_window)
    features["rolling_skewness"] = compute_rolling_skewness(returns, window)
    features["rolling_kurtosis"] = compute_rolling_kurtosis(returns, window)
    
    # Market return
    market_proxy = config.get("universe", {}).get("market_proxy")
    features["market_return"] = compute_market_return(returns, market_proxy, short_window)
    
    # Drawdown features if prices available
    if prices is not None:
        features["max_drawdown"] = compute_max_drawdown_rolling(prices, window)
    
    # Lagged features
    for lag in [1, 5, 20]:
        features[f"sgi_lag{lag}"] = features["sgi"].shift(lag)
        features[f"realized_vol_lag{lag}"] = features["realized_vol"].shift(lag)
    
    # Interaction features
    features["sgi_x_vol"] = features["sgi"] * features["realized_vol"]
    features["sgi_x_corr"] = features["sgi"] * features["avg_correlation"]
    
    return features


def get_feature_columns(feature_set: str, config: Dict) -> List[str]:
    """
    Get list of feature columns for a named feature set.
    
    Args:
        feature_set: Name of feature set
        config: Configuration dictionary
        
    Returns:
        List of feature column names
    """
    feature_sets = config.get("modeling", {}).get("feature_sets", {})
    
    if feature_set in feature_sets:
        return feature_sets[feature_set]
    
    # Default feature sets
    defaults = {
        "baseline_only": [
            "realized_vol",
            "avg_correlation",
            "market_return",
        ],
        "sgi_only": [
            "sgi",
            "weighted_sgi",
        ],
        "baseline_plus_sgi": [
            "realized_vol",
            "avg_correlation",
            "market_return",
            "sgi",
            "weighted_sgi",
        ],
        "spectral_baseline": [
            "realized_vol",
            "avg_correlation",
            "absorption_ratio",
            "eigen_concentration",
        ],
        "full": [
            "realized_vol",
            "avg_correlation",
            "market_return",
            "cross_sectional_dispersion",
            "absorption_ratio",
            "eigen_concentration",
            "sgi",
            "weighted_sgi",
            "max_angle",
            "mean_angle",
        ],
    }
    
    return defaults.get(feature_set, defaults["full"])
