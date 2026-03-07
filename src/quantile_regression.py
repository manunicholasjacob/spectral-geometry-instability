"""
Quantile Regression module for the SGI project.
Implements quantile regression for tail risk prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from .constants import EPSILON


def fit_quantile_regression(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.05
) -> Dict:
    """
    Fit quantile regression model.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector
        quantile: Quantile to estimate (e.g., 0.05 for 5th percentile)
        
    Returns:
        Dictionary with model results
    """
    # Add constant
    X_const = sm.add_constant(X)
    
    # Fit quantile regression
    model = QuantReg(y, X_const)
    result = model.fit(q=quantile)
    
    return {
        "quantile": quantile,
        "coefficients": result.params.tolist(),
        "pvalues": result.pvalues.tolist(),
        "conf_int_lower": result.conf_int()[0].tolist(),
        "conf_int_upper": result.conf_int()[1].tolist(),
        "pseudo_r2": float(result.prsquared),
        "n_observations": len(y),
    }


def fit_multiple_quantiles(
    X: np.ndarray,
    y: np.ndarray,
    quantiles: List[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
) -> Dict[float, Dict]:
    """
    Fit quantile regression at multiple quantiles.
    
    Args:
        X: Feature matrix
        y: Target vector
        quantiles: List of quantiles to estimate
        
    Returns:
        Dictionary mapping quantile to results
    """
    results = {}
    
    for q in quantiles:
        try:
            results[q] = fit_quantile_regression(X, y, q)
        except Exception as e:
            results[q] = {"error": str(e)}
    
    return results


def predict_quantile(
    X: np.ndarray,
    coefficients: np.ndarray
) -> np.ndarray:
    """
    Predict quantile values.
    
    Args:
        X: Feature matrix
        coefficients: Fitted coefficients (including intercept)
        
    Returns:
        Predicted quantile values
    """
    X_const = sm.add_constant(X)
    return X_const @ coefficients


def walk_forward_quantile_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    quantile: float = 0.05,
    train_years: int = 3,
    test_years: int = 1
) -> Dict:
    """
    Walk-forward quantile regression evaluation.
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        feature_cols: Feature column names
        quantile: Quantile to estimate
        train_years: Training period in years
        test_years: Test period in years
        
    Returns:
        Dictionary with walk-forward results
    """
    train_days = train_years * 252
    test_days = test_years * 252
    
    n = len(df)
    
    fold_results = []
    all_predictions = []
    all_actuals = []
    
    start = 0
    while start + train_days + test_days <= n:
        train_end = start + train_days
        test_end = train_end + test_days
        
        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]
        
        # Prepare data
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        
        # Drop NaN
        train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        test_mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        if len(X_train) < 100 or len(X_test) < 10:
            start += test_days
            continue
        
        try:
            # Fit model
            result = fit_quantile_regression(X_train, y_train, quantile)
            
            # Predict
            coefficients = np.array(result["coefficients"])
            y_pred = predict_quantile(X_test, coefficients)
            
            # Evaluate: coverage (what fraction of actuals are below predicted quantile)
            coverage = np.mean(y_test < y_pred)
            
            fold_results.append({
                "test_start": str(test_df.index[0]),
                "test_end": str(test_df.index[-1]),
                "coverage": float(coverage),
                "target_coverage": quantile,
                "coverage_error": float(coverage - quantile),
                "n_test": len(y_test),
                "pseudo_r2": result["pseudo_r2"],
            })
            
            all_predictions.extend(y_pred.tolist())
            all_actuals.extend(y_test.tolist())
            
        except Exception as e:
            fold_results.append({
                "test_start": str(test_df.index[0]),
                "error": str(e),
            })
        
        start += test_days
    
    # Summary statistics
    if fold_results:
        coverages = [r["coverage"] for r in fold_results if "coverage" in r]
        
        summary = {
            "quantile": quantile,
            "n_folds": len(fold_results),
            "mean_coverage": float(np.mean(coverages)) if coverages else None,
            "std_coverage": float(np.std(coverages)) if coverages else None,
            "mean_coverage_error": float(np.mean([r["coverage_error"] for r in fold_results if "coverage_error" in r])) if coverages else None,
        }
    else:
        summary = {"error": "No valid folds"}
    
    return {
        "summary": summary,
        "fold_results": fold_results,
        "predictions": all_predictions,
        "actuals": all_actuals,
    }


def compute_var_exceedance_test(
    predictions: np.ndarray,
    actuals: np.ndarray,
    quantile: float
) -> Dict:
    """
    Test VaR exceedance (Kupiec test).
    
    Args:
        predictions: Predicted quantile values
        actuals: Actual values
        quantile: Target quantile
        
    Returns:
        Dictionary with test results
    """
    n = len(actuals)
    exceedances = np.sum(actuals < predictions)
    expected_exceedances = n * quantile
    
    # Kupiec likelihood ratio test
    p_hat = exceedances / n
    
    if p_hat == 0 or p_hat == 1:
        lr_stat = np.inf
    else:
        lr_stat = -2 * (
            exceedances * np.log(quantile / p_hat) +
            (n - exceedances) * np.log((1 - quantile) / (1 - p_hat))
        )
    
    # Chi-squared p-value (1 df)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(lr_stat, 1)
    
    return {
        "n_observations": n,
        "n_exceedances": int(exceedances),
        "expected_exceedances": float(expected_exceedances),
        "exceedance_rate": float(p_hat),
        "target_rate": quantile,
        "lr_statistic": float(lr_stat) if not np.isinf(lr_stat) else None,
        "p_value": float(p_value) if not np.isnan(p_value) else None,
        "reject_at_5pct": p_value < 0.05 if not np.isnan(p_value) else None,
    }


def compare_quantile_feature_sets(
    df: pd.DataFrame,
    target_col: str,
    feature_sets: Dict[str, List[str]],
    quantile: float = 0.05,
    train_years: int = 3,
    test_years: int = 1
) -> pd.DataFrame:
    """
    Compare feature sets for quantile regression.
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        feature_sets: Dictionary of feature set name -> feature columns
        quantile: Quantile to estimate
        train_years: Training period
        test_years: Test period
        
    Returns:
        Comparison DataFrame
    """
    results = []
    
    for set_name, feature_cols in feature_sets.items():
        # Filter to available features
        available = [c for c in feature_cols if c in df.columns]
        
        if len(available) == 0:
            continue
        
        wf_results = walk_forward_quantile_regression(
            df, target_col, available, quantile, train_years, test_years
        )
        
        summary = wf_results["summary"]
        
        results.append({
            "feature_set": set_name,
            "n_features": len(available),
            "quantile": quantile,
            "mean_coverage": summary.get("mean_coverage"),
            "coverage_error": summary.get("mean_coverage_error"),
            "n_folds": summary.get("n_folds"),
        })
    
    return pd.DataFrame(results)


def fit_cvar_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Fit Conditional VaR (Expected Shortfall) regression.
    
    Uses quantile regression at alpha and averages predictions below.
    
    Args:
        X: Feature matrix
        y: Target vector
        alpha: Tail probability
        
    Returns:
        Dictionary with CVaR regression results
    """
    # First fit VaR (quantile regression)
    var_result = fit_quantile_regression(X, y, alpha)
    
    # Predict VaR
    var_predictions = predict_quantile(X, np.array(var_result["coefficients"]))
    
    # Compute CVaR as average of y below VaR
    tail_mask = y < var_predictions
    
    if np.sum(tail_mask) > 0:
        cvar_estimate = np.mean(y[tail_mask])
    else:
        cvar_estimate = np.nan
    
    return {
        "alpha": alpha,
        "var_coefficients": var_result["coefficients"],
        "cvar_estimate": float(cvar_estimate) if not np.isnan(cvar_estimate) else None,
        "n_tail_observations": int(np.sum(tail_mask)),
        "var_pseudo_r2": var_result["pseudo_r2"],
    }
