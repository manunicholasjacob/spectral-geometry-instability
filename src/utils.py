"""
Utility functions for the SGI project.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Dictionary containing YAML contents
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save data to a YAML file.
    
    Args:
        data: Dictionary to save
        path: Path to save to
    """
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dictionary containing JSON contents
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        path: Path to save to
        indent: JSON indentation level
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator between keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def hash_config(config: Dict[str, Any], length: int = 8) -> str:
    """
    Generate a hash from a config dictionary for unique identification.
    
    Args:
        config: Configuration dictionary
        length: Length of hash to return
        
    Returns:
        Hash string
    """
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:length]


def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_psd(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if a matrix is positive semi-definite.
    
    Args:
        matrix: Square matrix to check
        tol: Tolerance for eigenvalue check
        
    Returns:
        True if matrix is PSD
    """
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return np.all(eigenvalues >= -tol)
    except np.linalg.LinAlgError:
        return False


def is_symmetric(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is symmetric.
    
    Args:
        matrix: Square matrix to check
        tol: Tolerance for symmetry check
        
    Returns:
        True if matrix is symmetric
    """
    return np.allclose(matrix, matrix.T, atol=tol)


def make_psd(matrix: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
    """
    Make a matrix positive semi-definite by clipping negative eigenvalues.
    
    Args:
        matrix: Square matrix
        min_eigenvalue: Minimum eigenvalue to enforce
        
    Returns:
        PSD matrix
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def compute_condition_number(matrix: np.ndarray) -> float:
    """
    Compute the condition number of a matrix.
    
    Args:
        matrix: Square matrix
        
    Returns:
        Condition number
    """
    try:
        return np.linalg.cond(matrix)
    except np.linalg.LinAlgError:
        return np.inf


def rolling_window_indices(
    n_obs: int,
    window_size: int,
    min_periods: Optional[int] = None
) -> List[tuple]:
    """
    Generate rolling window start/end indices.
    
    Args:
        n_obs: Total number of observations
        window_size: Size of rolling window
        min_periods: Minimum periods required (defaults to window_size)
        
    Returns:
        List of (start, end) index tuples
    """
    if min_periods is None:
        min_periods = window_size
    
    indices = []
    for end in range(min_periods, n_obs + 1):
        start = max(0, end - window_size)
        indices.append((start, end))
    
    return indices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from prices.
    
    Args:
        prices: DataFrame of prices
        
    Returns:
        DataFrame of log returns
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple returns from prices.
    
    Args:
        prices: DataFrame of prices
        
    Returns:
        DataFrame of simple returns
    """
    return prices.pct_change().dropna()


def annualize_volatility(vol: float, periods_per_year: int = 252) -> float:
    """
    Annualize volatility.
    
    Args:
        vol: Daily volatility
        periods_per_year: Trading periods per year
        
    Returns:
        Annualized volatility
    """
    return vol * np.sqrt(periods_per_year)


def annualize_return(ret: float, periods_per_year: int = 252) -> float:
    """
    Annualize return.
    
    Args:
        ret: Daily return
        periods_per_year: Trading periods per year
        
    Returns:
        Annualized return
    """
    return ret * periods_per_year


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sortino ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0
    
    downside_std = np.sqrt((downside_returns ** 2).mean())
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown from returns.
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown (positive number)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())


def compute_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Series of returns
        periods_per_year: Trading periods per year
        
    Returns:
        Calmar ratio
    """
    max_dd = compute_max_drawdown(returns)
    if max_dd == 0:
        return np.inf if returns.mean() > 0 else 0.0
    
    ann_return = annualize_return(returns.mean(), periods_per_year)
    return ann_return / max_dd
