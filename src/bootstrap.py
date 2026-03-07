"""
Bootstrap and statistical inference module for the SGI project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable


def stationary_bootstrap(
    data: np.ndarray,
    block_size: float = 10.0,
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Perform stationary bootstrap resampling.
    
    Args:
        data: Original data array
        block_size: Expected block size (geometric distribution parameter)
        n_samples: Number of bootstrap samples
        seed: Random seed
        
    Returns:
        Array of shape (n_samples, len(data))
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    p = 1.0 / block_size  # Probability of starting new block
    
    samples = np.zeros((n_samples, n))
    
    for i in range(n_samples):
        idx = np.random.randint(0, n)
        for j in range(n):
            samples[i, j] = data[idx]
            
            # Decide whether to continue block or start new
            if np.random.random() < p:
                idx = np.random.randint(0, n)
            else:
                idx = (idx + 1) % n
    
    return samples


def bootstrap_statistic(
    data: np.ndarray,
    statistic_func: Callable,
    n_bootstrap: int = 1000,
    block_size: float = 10.0,
    confidence_level: float = 0.95,
    seed: Optional[int] = None
) -> Dict:
    """
    Compute bootstrap confidence intervals for a statistic.
    
    Args:
        data: Original data
        statistic_func: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        block_size: Block size for stationary bootstrap
        confidence_level: Confidence level
        seed: Random seed
        
    Returns:
        Dictionary with point estimate and confidence interval
    """
    # Point estimate
    point_estimate = statistic_func(data)
    
    # Bootstrap samples
    bootstrap_samples = stationary_bootstrap(data, block_size, n_bootstrap, seed)
    
    # Compute statistic for each sample
    bootstrap_stats = np.array([statistic_func(sample) for sample in bootstrap_samples])
    
    # Confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return {
        "point_estimate": float(point_estimate),
        "mean": float(bootstrap_stats.mean()),
        "std": float(bootstrap_stats.std()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": confidence_level,
    }


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: float = 10.0,
    seed: Optional[int] = None
) -> Dict:
    """
    Bootstrap confidence interval for correlation.
    
    Args:
        x: First series
        y: Second series
        n_bootstrap: Number of bootstrap samples
        block_size: Block size
        seed: Random seed
        
    Returns:
        Dictionary with correlation and CI
    """
    def corr_func(indices):
        return np.corrcoef(x[indices.astype(int)], y[indices.astype(int)])[0, 1]
    
    n = len(x)
    indices = np.arange(n)
    
    # Bootstrap indices
    bootstrap_indices = stationary_bootstrap(indices, block_size, n_bootstrap, seed)
    
    # Compute correlations
    correlations = []
    for idx in bootstrap_indices:
        idx = idx.astype(int) % n
        correlations.append(np.corrcoef(x[idx], y[idx])[0, 1])
    
    correlations = np.array(correlations)
    
    return {
        "point_estimate": float(np.corrcoef(x, y)[0, 1]),
        "mean": float(correlations.mean()),
        "std": float(correlations.std()),
        "ci_lower": float(np.percentile(correlations, 2.5)),
        "ci_upper": float(np.percentile(correlations, 97.5)),
    }


def bootstrap_regression_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: float = 10.0,
    seed: Optional[int] = None
) -> Dict:
    """
    Bootstrap confidence intervals for regression coefficients.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_bootstrap: Number of bootstrap samples
        block_size: Block size
        seed: Random seed
        
    Returns:
        Dictionary with coefficient CIs
    """
    from sklearn.linear_model import LinearRegression
    
    n = len(y)
    indices = np.arange(n)
    
    # Bootstrap indices
    bootstrap_indices = stationary_bootstrap(indices, block_size, n_bootstrap, seed)
    
    # Fit models
    coefficients = []
    for idx in bootstrap_indices:
        idx = idx.astype(int) % n
        model = LinearRegression()
        model.fit(X[idx], y[idx])
        coefficients.append(model.coef_)
    
    coefficients = np.array(coefficients)
    
    # Point estimate
    model = LinearRegression()
    model.fit(X, y)
    
    results = {
        "point_estimates": model.coef_.tolist(),
        "means": coefficients.mean(axis=0).tolist(),
        "stds": coefficients.std(axis=0).tolist(),
        "ci_lowers": np.percentile(coefficients, 2.5, axis=0).tolist(),
        "ci_uppers": np.percentile(coefficients, 97.5, axis=0).tolist(),
    }
    
    return results


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    statistic_func: Callable,
    n_permutations: int = 1000,
    seed: Optional[int] = None
) -> Dict:
    """
    Perform permutation test for independence.
    
    Args:
        x: First series
        y: Second series
        statistic_func: Function to compute test statistic
        n_permutations: Number of permutations
        seed: Random seed
        
    Returns:
        Dictionary with test results
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Observed statistic
    observed = statistic_func(x, y)
    
    # Permutation distribution
    permuted_stats = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        permuted_stats.append(statistic_func(x, y_perm))
    
    permuted_stats = np.array(permuted_stats)
    
    # P-value (two-sided)
    p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed))
    
    return {
        "observed_statistic": float(observed),
        "p_value": float(p_value),
        "permutation_mean": float(permuted_stats.mean()),
        "permutation_std": float(permuted_stats.std()),
    }
