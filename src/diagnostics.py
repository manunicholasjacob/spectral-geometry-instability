"""
Diagnostics module for the SGI project.
Provides data quality checks and validation utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .paths import get_diagnostics_dir
from .utils import save_json, is_psd, is_symmetric, compute_condition_number


def check_price_data_quality(prices: pd.DataFrame) -> Dict:
    """
    Comprehensive price data quality check.
    
    Args:
        prices: Price DataFrame
        
    Returns:
        Dictionary of quality metrics and issues
    """
    results = {
        "shape": prices.shape,
        "date_range": {
            "start": str(prices.index.min()),
            "end": str(prices.index.max()),
        },
        "assets": list(prices.columns),
        "issues": [],
    }
    
    # Missing values
    missing_total = prices.isnull().sum().sum()
    missing_by_asset = prices.isnull().sum()
    results["missing"] = {
        "total": int(missing_total),
        "by_asset": missing_by_asset.to_dict(),
        "pct_missing": float(missing_total / prices.size * 100),
    }
    
    if missing_total > 0:
        results["issues"].append(f"Found {missing_total} missing values")
    
    # Zero prices
    zero_count = (prices == 0).sum().sum()
    if zero_count > 0:
        results["issues"].append(f"Found {zero_count} zero prices")
        results["zero_prices"] = int(zero_count)
    
    # Negative prices
    neg_count = (prices < 0).sum().sum()
    if neg_count > 0:
        results["issues"].append(f"Found {neg_count} negative prices")
        results["negative_prices"] = int(neg_count)
    
    # Constant prices (zero variance)
    std = prices.std()
    zero_var_assets = list(std[std == 0].index)
    if zero_var_assets:
        results["issues"].append(f"Zero variance assets: {zero_var_assets}")
        results["zero_variance_assets"] = zero_var_assets
    
    # Date gaps
    date_diffs = prices.index.to_series().diff().dropna()
    max_gap = date_diffs.max()
    if max_gap > pd.Timedelta(days=5):
        results["issues"].append(f"Large date gap detected: {max_gap}")
        results["max_date_gap"] = str(max_gap)
    
    results["is_valid"] = len(results["issues"]) == 0
    
    return results


def check_return_data_quality(returns: pd.DataFrame) -> Dict:
    """
    Comprehensive return data quality check.
    
    Args:
        returns: Returns DataFrame
        
    Returns:
        Dictionary of quality metrics and issues
    """
    results = {
        "shape": returns.shape,
        "date_range": {
            "start": str(returns.index.min()),
            "end": str(returns.index.max()),
        },
        "issues": [],
    }
    
    # Missing values
    missing_total = returns.isnull().sum().sum()
    results["missing_total"] = int(missing_total)
    if missing_total > 0:
        results["issues"].append(f"Found {missing_total} missing values")
    
    # Infinite values
    inf_count = np.isinf(returns.values).sum()
    if inf_count > 0:
        results["issues"].append(f"Found {inf_count} infinite values")
        results["inf_count"] = int(inf_count)
    
    # Extreme returns
    extreme_threshold = 0.5  # 50% daily return
    extreme_count = (returns.abs() > extreme_threshold).sum().sum()
    if extreme_count > 0:
        results["issues"].append(f"Found {extreme_count} extreme returns (>{extreme_threshold*100}%)")
        results["extreme_returns"] = int(extreme_count)
    
    # Summary statistics
    results["statistics"] = {
        "mean": float(returns.mean().mean()),
        "std": float(returns.std().mean()),
        "min": float(returns.min().min()),
        "max": float(returns.max().max()),
        "skew": float(returns.skew().mean()),
        "kurtosis": float(returns.kurtosis().mean()),
    }
    
    results["is_valid"] = len(results["issues"]) == 0
    
    return results


def check_covariance_matrix(cov: np.ndarray, name: str = "covariance") -> Dict:
    """
    Check covariance matrix validity.
    
    Args:
        cov: Covariance matrix
        name: Name for reporting
        
    Returns:
        Dictionary of validation results
    """
    results = {
        "name": name,
        "shape": cov.shape,
        "issues": [],
    }
    
    # Check square
    if cov.shape[0] != cov.shape[1]:
        results["issues"].append("Matrix is not square")
        results["is_valid"] = False
        return results
    
    # Check symmetry
    results["is_symmetric"] = is_symmetric(cov)
    if not results["is_symmetric"]:
        results["issues"].append("Matrix is not symmetric")
    
    # Check PSD
    results["is_psd"] = is_psd(cov)
    if not results["is_psd"]:
        results["issues"].append("Matrix is not positive semi-definite")
    
    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvalsh(cov)
    results["eigenvalues"] = {
        "min": float(eigenvalues.min()),
        "max": float(eigenvalues.max()),
        "n_negative": int((eigenvalues < 0).sum()),
        "n_near_zero": int((eigenvalues < 1e-10).sum()),
    }
    
    # Condition number
    results["condition_number"] = float(compute_condition_number(cov))
    if results["condition_number"] > 1e10:
        results["issues"].append(f"High condition number: {results['condition_number']:.2e}")
    
    # Check for NaN/Inf
    if np.isnan(cov).any():
        results["issues"].append("Matrix contains NaN values")
    if np.isinf(cov).any():
        results["issues"].append("Matrix contains infinite values")
    
    results["is_valid"] = len(results["issues"]) == 0
    
    return results


def check_feature_target_alignment(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame
) -> Dict:
    """
    Check alignment between features and targets.
    
    Args:
        features_df: Features DataFrame
        targets_df: Targets DataFrame
        
    Returns:
        Dictionary of alignment checks
    """
    results = {
        "features_shape": features_df.shape,
        "targets_shape": targets_df.shape,
        "issues": [],
    }
    
    # Check index alignment
    common_dates = features_df.index.intersection(targets_df.index)
    results["common_dates"] = len(common_dates)
    results["features_only_dates"] = len(features_df.index.difference(targets_df.index))
    results["targets_only_dates"] = len(targets_df.index.difference(features_df.index))
    
    if len(common_dates) == 0:
        results["issues"].append("No overlapping dates between features and targets")
    
    # Check for potential lookahead
    if len(common_dates) > 0:
        feature_end = features_df.index.max()
        target_end = targets_df.index.max()
        if feature_end > target_end:
            results["issues"].append("Features extend beyond targets - potential lookahead")
    
    # Check for missing values in aligned data
    aligned_features = features_df.loc[common_dates]
    aligned_targets = targets_df.loc[common_dates]
    
    results["aligned_features_missing"] = int(aligned_features.isnull().sum().sum())
    results["aligned_targets_missing"] = int(aligned_targets.isnull().sum().sum())
    
    results["is_valid"] = len(results["issues"]) == 0
    
    return results


def check_sgi_sanity(geometry_df: pd.DataFrame) -> Dict:
    """
    Sanity check SGI values.
    
    Args:
        geometry_df: DataFrame with SGI metrics
        
    Returns:
        Dictionary of sanity checks
    """
    results = {
        "shape": geometry_df.shape,
        "columns": list(geometry_df.columns),
        "issues": [],
    }
    
    # Check SGI column exists
    if "sgi" not in geometry_df.columns:
        results["issues"].append("Missing 'sgi' column")
        results["is_valid"] = False
        return results
    
    sgi = geometry_df["sgi"]
    
    # Basic statistics
    results["sgi_stats"] = {
        "mean": float(sgi.mean()),
        "std": float(sgi.std()),
        "min": float(sgi.min()),
        "max": float(sgi.max()),
        "n_missing": int(sgi.isnull().sum()),
    }
    
    # Check for invalid values
    if (sgi < 0).any():
        results["issues"].append("SGI contains negative values")
    
    if sgi.isnull().all():
        results["issues"].append("SGI is all NaN")
    
    # Check for constant SGI (suspicious)
    if sgi.std() == 0:
        results["issues"].append("SGI has zero variance")
    
    # Check angle bounds (should be in [0, pi/2] radians)
    if "max_angle" in geometry_df.columns:
        max_angle = geometry_df["max_angle"]
        if (max_angle > np.pi/2 + 0.01).any():
            results["issues"].append("max_angle exceeds pi/2")
    
    results["is_valid"] = len(results["issues"]) == 0
    
    return results


def run_all_diagnostics(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    geometry_df: Optional[pd.DataFrame] = None,
    save_results: bool = True,
    output_tag: str = ""
) -> Dict:
    """
    Run all diagnostic checks and optionally save results.
    
    Args:
        prices: Price DataFrame
        returns: Returns DataFrame
        geometry_df: Optional geometry DataFrame
        save_results: Whether to save results
        output_tag: Tag for output filename
        
    Returns:
        Dictionary of all diagnostic results
    """
    results = {
        "price_quality": check_price_data_quality(prices),
        "return_quality": check_return_data_quality(returns),
    }
    
    if geometry_df is not None:
        results["sgi_sanity"] = check_sgi_sanity(geometry_df)
    
    # Overall validity
    results["all_valid"] = all(
        r.get("is_valid", True) for r in results.values()
    )
    
    if save_results:
        tag = f"_{output_tag}" if output_tag else ""
        save_json(results, get_diagnostics_dir() / f"diagnostics{tag}.json")
    
    return results
