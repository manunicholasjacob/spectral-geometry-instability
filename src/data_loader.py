"""
Data loading and caching for the SGI project.
Handles downloading data from yfinance and caching locally.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import yfinance as yf

from .paths import get_raw_data_dir, get_processed_data_dir
from .constants import SECTOR_ETF_TICKERS, MULTIASSET_ETF_TICKERS, LARGECAP_50_TICKERS


def download_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    auto_adjust: bool = True,
    progress: bool = True
) -> pd.DataFrame:
    """
    Download price data from yfinance.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        auto_adjust: Whether to use adjusted prices
        progress: Whether to show download progress
        
    Returns:
        DataFrame with adjusted close prices, indexed by date
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=progress
    )
    
    # Handle single ticker case
    if len(tickers) == 1:
        prices = data[['Close']].copy()
        prices.columns = tickers
    else:
        # Extract Close prices
        if 'Close' in data.columns.get_level_values(0):
            prices = data['Close'].copy()
        elif 'Adj Close' in data.columns.get_level_values(0):
            prices = data['Adj Close'].copy()
        else:
            prices = data.copy()
    
    # Ensure column order matches input tickers
    available_tickers = [t for t in tickers if t in prices.columns]
    prices = prices[available_tickers]
    
    return prices


def get_cache_filename(tickers: List[str], start_date: str, end_date: str) -> str:
    """
    Generate a cache filename based on tickers and date range.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        
    Returns:
        Cache filename
    """
    ticker_hash = "_".join(sorted(tickers)[:5])  # Use first 5 tickers for name
    n_tickers = len(tickers)
    return f"prices_{ticker_hash}_{n_tickers}_{start_date}_{end_date}.csv"


def save_raw_prices(df: pd.DataFrame, path: Path) -> None:
    """
    Save raw price data to CSV.
    
    Args:
        df: Price DataFrame
        path: Path to save to
    """
    df.to_csv(path)


def load_raw_prices(path: Path) -> pd.DataFrame:
    """
    Load raw price data from CSV.
    
    Args:
        path: Path to load from
        
    Returns:
        Price DataFrame
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_or_download_prices(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache: bool = True,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load prices from cache or download if not available.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date string
        end_date: End date string
        cache: Whether to use caching
        force_download: Force re-download even if cached
        
    Returns:
        Price DataFrame
    """
    if cache and not force_download:
        cache_file = get_raw_data_dir() / get_cache_filename(tickers, start_date, end_date)
        if cache_file.exists():
            return load_raw_prices(cache_file)
    
    # Download fresh data
    prices = download_price_data(tickers, start_date, end_date)
    
    # Cache if requested
    if cache:
        cache_file = get_raw_data_dir() / get_cache_filename(tickers, start_date, end_date)
        save_raw_prices(prices, cache_file)
    
    return prices


def load_prices_from_config(config: Dict) -> pd.DataFrame:
    """
    Load prices based on configuration dictionary.
    
    Args:
        config: Configuration dictionary with data and universe sections
        
    Returns:
        Price DataFrame
    """
    # Get tickers
    if "universe" in config and "tickers" in config["universe"]:
        tickers = config["universe"]["tickers"]
    else:
        tickers = SECTOR_ETF_TICKERS
    
    # Get date range
    start_date = config.get("data", {}).get("start_date", "2005-01-01")
    end_date = config.get("data", {}).get("end_date", "2024-12-31")
    
    # Get cache setting
    cache = config.get("data", {}).get("cache_data", True)
    
    return load_or_download_prices(tickers, start_date, end_date, cache=cache)


def get_universe_tickers(universe_name: str) -> List[str]:
    """
    Get ticker list for a named universe.
    
    Args:
        universe_name: Name of universe ("sector_etfs", "multiasset", "largecap50")
        
    Returns:
        List of ticker symbols
    """
    universes = {
        "sector_etfs": SECTOR_ETF_TICKERS,
        "multiasset": MULTIASSET_ETF_TICKERS,
        "largecap50": LARGECAP_50_TICKERS,
    }
    
    if universe_name not in universes:
        raise ValueError(f"Unknown universe: {universe_name}. Available: {list(universes.keys())}")
    
    return universes[universe_name]


def download_market_data(
    market_proxy: str = "SPY",
    start_date: str = "2005-01-01",
    end_date: str = "2024-12-31"
) -> pd.DataFrame:
    """
    Download market proxy data.
    
    Args:
        market_proxy: Market proxy ticker
        start_date: Start date
        end_date: End date
        
    Returns:
        Market price DataFrame
    """
    return download_price_data([market_proxy], start_date, end_date, progress=False)


def download_vix_data(
    start_date: str = "2005-01-01",
    end_date: str = "2024-12-31"
) -> pd.DataFrame:
    """
    Download VIX data.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        VIX DataFrame
    """
    return download_price_data(["^VIX"], start_date, end_date, progress=False)
