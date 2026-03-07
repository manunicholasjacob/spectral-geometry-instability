"""
Data preprocessing for the SGI project.
Handles return computation, alignment, and cleaning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .paths import get_processed_data_dir, get_metadata_dir
from .utils import save_json


def compute_returns(
    prices: pd.DataFrame,
    method: str = "log"
) -> pd.DataFrame:
    """
    Compute returns from prices.
    
    Args:
        prices: DataFrame of prices
        method: "log" for log returns, "simple" for simple returns
        
    Returns:
        DataFrame of returns
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method}. Use 'log' or 'simple'.")
    
    return returns.dropna()


def filter_assets_by_missingness(
    returns: pd.DataFrame,
    max_missing_frac: float = 0.05
) -> pd.DataFrame:
    """
    Filter out assets with too many missing values.
    
    Args:
        returns: DataFrame of returns
        max_missing_frac: Maximum fraction of missing values allowed
        
    Returns:
        Filtered DataFrame
    """
    missing_frac = returns.isnull().mean()
    valid_assets = missing_frac[missing_frac <= max_missing_frac].index
    return returns[valid_assets]


def align_and_clean_prices(
    prices: pd.DataFrame,
    dropna: bool = True,
    ffill_limit: int = 5
) -> pd.DataFrame:
    """
    Align dates and clean price data.
    
    Args:
        prices: DataFrame of prices
        dropna: Whether to drop rows with any NaN
        ffill_limit: Maximum consecutive NaNs to forward fill
        
    Returns:
        Cleaned price DataFrame
    """
    # Forward fill small gaps
    prices = prices.ffill(limit=ffill_limit)
    
    # Drop remaining NaN rows if requested
    if dropna:
        prices = prices.dropna()
    
    return prices


def validate_prices(prices: pd.DataFrame) -> Dict[str, any]:
    """
    Validate price data and return diagnostics.
    
    Args:
        prices: DataFrame of prices
        
    Returns:
        Dictionary of validation results
    """
    diagnostics = {
        "n_assets": len(prices.columns),
        "n_observations": len(prices),
        "start_date": str(prices.index.min().date()),
        "end_date": str(prices.index.max().date()),
        "assets": list(prices.columns),
        "missing_values": int(prices.isnull().sum().sum()),
        "missing_by_asset": prices.isnull().sum().to_dict(),
        "zero_prices": int((prices == 0).sum().sum()),
        "negative_prices": int((prices < 0).sum().sum()),
    }
    
    # Check for constant prices (zero variance)
    price_std = prices.std()
    diagnostics["zero_variance_assets"] = list(price_std[price_std == 0].index)
    
    return diagnostics


def validate_returns(returns: pd.DataFrame) -> Dict[str, any]:
    """
    Validate return data and return diagnostics.
    
    Args:
        returns: DataFrame of returns
        
    Returns:
        Dictionary of validation results
    """
    diagnostics = {
        "n_assets": len(returns.columns),
        "n_observations": len(returns),
        "start_date": str(returns.index.min().date()),
        "end_date": str(returns.index.max().date()),
        "assets": list(returns.columns),
        "missing_values": int(returns.isnull().sum().sum()),
        "inf_values": int(np.isinf(returns.values).sum()),
    }
    
    # Summary statistics
    diagnostics["mean_return"] = returns.mean().mean()
    diagnostics["mean_volatility"] = returns.std().mean()
    diagnostics["min_return"] = float(returns.min().min())
    diagnostics["max_return"] = float(returns.max().max())
    
    # Check for extreme returns (potential data errors)
    extreme_threshold = 0.5  # 50% daily return is suspicious
    extreme_returns = (returns.abs() > extreme_threshold).sum().sum()
    diagnostics["extreme_returns_count"] = int(extreme_returns)
    
    return diagnostics


def build_data_metadata(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: Optional[Dict] = None
) -> Dict:
    """
    Build comprehensive metadata about the data.
    
    Args:
        prices: Price DataFrame
        returns: Returns DataFrame
        config: Optional configuration dictionary
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        "price_diagnostics": validate_prices(prices),
        "return_diagnostics": validate_returns(returns),
        "processing_timestamp": pd.Timestamp.now().isoformat(),
    }
    
    if config:
        metadata["config"] = {
            "universe": config.get("universe", {}).get("name", "unknown"),
            "start_date": config.get("data", {}).get("start_date"),
            "end_date": config.get("data", {}).get("end_date"),
            "return_type": config.get("data", {}).get("return_type", "log"),
        }
    
    return metadata


def preprocess_data(
    prices: pd.DataFrame,
    return_method: str = "log",
    max_missing_frac: float = 0.05,
    save_outputs: bool = True,
    output_tag: str = ""
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Full preprocessing pipeline.
    
    Args:
        prices: Raw price DataFrame
        return_method: Method for computing returns
        max_missing_frac: Maximum missing fraction per asset
        save_outputs: Whether to save processed data
        output_tag: Tag for output filenames
        
    Returns:
        Tuple of (cleaned_prices, returns, metadata)
    """
    # Clean prices
    cleaned_prices = align_and_clean_prices(prices)
    
    # Compute returns
    returns = compute_returns(cleaned_prices, method=return_method)
    
    # Filter by missingness
    returns = filter_assets_by_missingness(returns, max_missing_frac)
    
    # Align prices to returns
    cleaned_prices = cleaned_prices.loc[returns.index, returns.columns]
    
    # Build metadata
    metadata = build_data_metadata(cleaned_prices, returns)
    
    # Save outputs if requested
    if save_outputs:
        tag = f"_{output_tag}" if output_tag else ""
        
        processed_dir = get_processed_data_dir()
        cleaned_prices.to_csv(processed_dir / f"prices{tag}.csv")
        returns.to_csv(processed_dir / f"returns{tag}.csv")
        
        metadata_dir = get_metadata_dir()
        save_json(metadata, metadata_dir / f"data_metadata{tag}.json")
    
    return cleaned_prices, returns, metadata


def load_processed_data(output_tag: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously processed data.
    
    Args:
        output_tag: Tag used when saving
        
    Returns:
        Tuple of (prices, returns)
    """
    tag = f"_{output_tag}" if output_tag else ""
    processed_dir = get_processed_data_dir()
    
    prices = pd.read_csv(processed_dir / f"prices{tag}.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(processed_dir / f"returns{tag}.csv", index_col=0, parse_dates=True)
    
    return prices, returns
