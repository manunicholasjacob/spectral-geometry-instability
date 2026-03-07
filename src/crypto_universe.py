"""
Crypto Universe module for the SGI project.
Extends SGI analysis to cryptocurrency markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .data_loader import download_price_data
from .preprocessing import preprocess_data
from .covariance import rolling_covariance_matrices
from .spectral_geometry import compute_rolling_geometry_features, compute_sgi_derived_features
from .features import build_feature_dataframe
from .paths import get_raw_data_dir


# Default crypto tickers (Yahoo Finance format)
CRYPTO_TICKERS = [
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
    "BNB-USD",   # Binance Coin
    "XRP-USD",   # Ripple
    "ADA-USD",   # Cardano
    "SOL-USD",   # Solana
    "DOGE-USD",  # Dogecoin
    "DOT-USD",   # Polkadot
    "AVAX-USD",  # Avalanche
    "MATIC-USD", # Polygon
    "LINK-USD",  # Chainlink
    "UNI-USD",   # Uniswap
    "ATOM-USD",  # Cosmos
    "LTC-USD",   # Litecoin
    "ETC-USD",   # Ethereum Classic
]

# Stablecoin tickers (for reference/exclusion)
STABLECOIN_TICKERS = [
    "USDT-USD",  # Tether
    "USDC-USD",  # USD Coin
    "BUSD-USD",  # Binance USD
    "DAI-USD",   # Dai
]

# Major crypto events for event studies
CRYPTO_EVENTS = {
    "btc_halving_2020": {
        "name": "Bitcoin Halving 2020",
        "start": "2020-05-01",
        "peak_date": "2020-05-11",
        "end": "2020-05-31",
    },
    "defi_summer_2020": {
        "name": "DeFi Summer 2020",
        "start": "2020-06-01",
        "peak_date": "2020-09-01",
        "end": "2020-10-31",
    },
    "crypto_crash_may_2021": {
        "name": "Crypto Crash May 2021",
        "start": "2021-05-01",
        "peak_date": "2021-05-19",
        "end": "2021-06-30",
    },
    "terra_luna_collapse": {
        "name": "Terra/Luna Collapse",
        "start": "2022-05-01",
        "peak_date": "2022-05-12",
        "end": "2022-05-31",
    },
    "ftx_collapse": {
        "name": "FTX Collapse",
        "start": "2022-11-01",
        "peak_date": "2022-11-11",
        "end": "2022-11-30",
    },
    "btc_halving_2024": {
        "name": "Bitcoin Halving 2024",
        "start": "2024-04-01",
        "peak_date": "2024-04-20",
        "end": "2024-05-15",
    },
}


def download_crypto_prices(
    tickers: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    cache: bool = True
) -> pd.DataFrame:
    """
    Download cryptocurrency price data.
    
    Args:
        tickers: List of crypto tickers (uses default if None)
        start_date: Start date
        end_date: End date
        cache: Whether to cache data
        
    Returns:
        DataFrame with crypto prices
    """
    if tickers is None:
        tickers = CRYPTO_TICKERS
    
    cache_file = get_raw_data_dir() / f"crypto_prices_{start_date}_{end_date}.csv"
    
    if cache and cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    # Download
    prices = download_price_data(tickers, start_date, end_date)
    
    if cache:
        prices.to_csv(cache_file)
    
    return prices


def compute_crypto_sgi(
    tickers: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    window: int = 60,  # Shorter window for crypto
    top_k: int = 3,
    estimator: str = "ewma"  # EWMA better for volatile crypto
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Compute SGI for crypto universe.
    
    Args:
        tickers: List of crypto tickers
        start_date: Start date
        end_date: End date
        window: Rolling window (shorter for crypto)
        top_k: Number of top eigenvectors
        estimator: Covariance estimator
        
    Returns:
        Tuple of (geometry_df, features_df, metadata)
    """
    if tickers is None:
        tickers = CRYPTO_TICKERS
    
    # Download and preprocess
    prices = download_crypto_prices(tickers, start_date, end_date)
    prices, returns, metadata = preprocess_data(
        prices, return_method='log', save_outputs=False
    )
    
    # Compute covariance
    cov_matrices = rolling_covariance_matrices(
        returns, window, method=estimator, span=window // 2
    )
    
    # Compute geometry
    geometry_df = compute_rolling_geometry_features(cov_matrices, top_k)
    geometry_df = compute_sgi_derived_features(geometry_df)
    
    # Build features
    config = {
        'covariance': {'rolling_window': window},
        'universe': {'market_proxy': 'BTC-USD'},
    }
    features_df = build_feature_dataframe(returns, geometry_df, config, prices)
    
    return geometry_df, features_df, metadata


def compute_crypto_specific_features(
    returns: pd.DataFrame,
    geometry_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute crypto-specific features.
    
    Args:
        returns: Crypto returns DataFrame
        geometry_df: Geometry DataFrame
        
    Returns:
        DataFrame with crypto-specific features
    """
    features = geometry_df.copy()
    
    # Bitcoin dominance proxy (BTC correlation with others)
    if 'BTC-USD' in returns.columns:
        btc_returns = returns['BTC-USD']
        other_returns = returns.drop(columns=['BTC-USD'], errors='ignore')
        
        # Rolling correlation with BTC
        btc_corr = other_returns.rolling(20).apply(
            lambda x: x.corr(btc_returns.loc[x.index])
        ).mean(axis=1)
        features['btc_correlation'] = btc_corr
        
        # BTC relative strength
        btc_vol = btc_returns.rolling(20).std()
        other_vol = other_returns.rolling(20).std().mean(axis=1)
        features['btc_relative_vol'] = btc_vol / (other_vol + 1e-8)
    
    # Altcoin season indicator (alts outperforming BTC)
    if 'BTC-USD' in returns.columns:
        btc_cum = (1 + returns['BTC-USD']).rolling(30).apply(
            lambda x: x.prod() - 1
        )
        alt_cum = (1 + returns.drop(columns=['BTC-USD'], errors='ignore')).rolling(30).apply(
            lambda x: x.prod() - 1
        ).mean(axis=1)
        features['altcoin_season'] = (alt_cum > btc_cum).astype(int)
    
    # Extreme return frequency
    extreme_threshold = 0.1  # 10% daily move
    extreme_count = (returns.abs() > extreme_threshold).sum(axis=1)
    features['extreme_return_count'] = extreme_count
    
    # Cross-sectional momentum dispersion
    momentum_20d = returns.rolling(20).sum()
    features['momentum_dispersion'] = momentum_20d.std(axis=1)
    
    return features


def compare_crypto_vs_traditional_sgi(
    crypto_geometry: pd.DataFrame,
    traditional_geometry: pd.DataFrame
) -> Dict:
    """
    Compare SGI characteristics between crypto and traditional markets.
    
    Args:
        crypto_geometry: Crypto geometry DataFrame
        traditional_geometry: Traditional market geometry DataFrame
        
    Returns:
        Dictionary with comparison results
    """
    # Align indices
    common_idx = crypto_geometry.index.intersection(traditional_geometry.index)
    
    crypto_sgi = crypto_geometry.loc[common_idx, 'sgi']
    trad_sgi = traditional_geometry.loc[common_idx, 'sgi']
    
    results = {
        'n_observations': len(common_idx),
        'date_range': {
            'start': str(common_idx.min()),
            'end': str(common_idx.max()),
        },
        'crypto_sgi_stats': {
            'mean': float(crypto_sgi.mean()),
            'std': float(crypto_sgi.std()),
            'max': float(crypto_sgi.max()),
            'median': float(crypto_sgi.median()),
        },
        'traditional_sgi_stats': {
            'mean': float(trad_sgi.mean()),
            'std': float(trad_sgi.std()),
            'max': float(trad_sgi.max()),
            'median': float(trad_sgi.median()),
        },
        'correlation': float(crypto_sgi.corr(trad_sgi)),
    }
    
    # Lead-lag analysis
    lead_lag = {}
    for lag in [-10, -5, -1, 0, 1, 5, 10]:
        if lag < 0:
            corr = crypto_sgi.iloc[:lag].corr(trad_sgi.iloc[-lag:])
        elif lag > 0:
            corr = crypto_sgi.iloc[lag:].corr(trad_sgi.iloc[:-lag])
        else:
            corr = crypto_sgi.corr(trad_sgi)
        
        if not np.isnan(corr):
            lead_lag[f'lag_{lag}'] = float(corr)
    
    results['lead_lag'] = lead_lag
    
    # Ratio analysis
    results['sgi_ratio'] = {
        'mean_ratio': float(crypto_sgi.mean() / (trad_sgi.mean() + 1e-8)),
        'volatility_ratio': float(crypto_sgi.std() / (trad_sgi.std() + 1e-8)),
    }
    
    return results


def run_crypto_event_study(
    geometry_df: pd.DataFrame,
    events: Optional[Dict] = None,
    pre_days: int = 30,
    post_days: int = 30
) -> Dict:
    """
    Run event study for crypto-specific events.
    
    Args:
        geometry_df: Geometry DataFrame
        events: Event dictionary (uses default if None)
        pre_days: Days before event
        post_days: Days after event
        
    Returns:
        Dictionary with event study results
    """
    from .event_study import extract_event_window, compute_event_statistics
    
    if events is None:
        events = CRYPTO_EVENTS
    
    results = {}
    
    for event_name, event_info in events.items():
        try:
            stats = compute_event_statistics(
                geometry_df['sgi'],
                event_info['peak_date'],
                pre_days,
                post_days
            )
            results[event_name] = {
                'event_info': event_info,
                'statistics': stats,
            }
        except Exception as e:
            results[event_name] = {'error': str(e)}
    
    return results


def get_crypto_config() -> Dict:
    """
    Get default configuration for crypto analysis.
    
    Returns:
        Configuration dictionary
    """
    return {
        'project': {
            'name': 'SGI Crypto Analysis',
            'version': '0.1.0',
        },
        'data': {
            'start_date': '2020-01-01',
            'end_date': '2024-12-31',
            'return_type': 'log',
        },
        'universe': {
            'name': 'crypto',
            'tickers': CRYPTO_TICKERS,
            'market_proxy': 'BTC-USD',
        },
        'covariance': {
            'estimator': 'ewma',
            'rolling_window': 60,
            'ewma_span': 30,
        },
        'geometry': {
            'top_k': 3,
        },
        'targets': {
            'horizons': [1, 5, 10, 20],
            'drawdown_threshold': 0.10,  # Higher for crypto
        },
        'outputs': {
            'experiment_tag': 'crypto',
        },
    }
