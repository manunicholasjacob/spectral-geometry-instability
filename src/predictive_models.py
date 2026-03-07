"""
Predictive modeling module for the SGI project.
Implements walk-forward evaluation for interpretable models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, accuracy_score, precision_score, recall_score
)
import statsmodels.api as sm

from .paths import get_predictions_dir
from .utils import save_json


def prepare_model_dataset(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    target_name: str,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Prepare aligned dataset for modeling.
    
    Args:
        features_df: Features DataFrame
        targets_df: Targets DataFrame
        target_name: Name of target column
        feature_cols: List of feature column names
        
    Returns:
        Combined DataFrame with features and target
    """
    # Get common dates
    common_dates = features_df.index.intersection(targets_df.index)
    
    # Select columns
    available_features = [c for c in feature_cols if c in features_df.columns]
    
    df = features_df.loc[common_dates, available_features].copy()
    df["target"] = targets_df.loc[common_dates, target_name]
    
    # Drop rows with any NaN
    df = df.dropna()
    
    return df


def run_linear_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target"
) -> Dict:
    """
    Run linear regression with statsmodels for coefficient analysis.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        
    Returns:
        Dictionary with results
    """
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Add constant for statsmodels
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    
    # Fit model
    model = sm.OLS(y_train, X_train_sm).fit()
    
    # Predictions
    y_pred_train = model.predict(X_train_sm)
    y_pred_test = model.predict(X_test_sm)
    
    # Metrics
    results = {
        "model_type": "linear_regression",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "test_mse": mean_squared_error(y_test, y_pred_test),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "coefficients": dict(zip(["const"] + feature_cols, model.params)),
        "pvalues": dict(zip(["const"] + feature_cols, model.pvalues)),
        "predictions": y_pred_test.tolist(),
        "actuals": y_test.tolist(),
    }
    
    return results


def run_logistic_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target"
) -> Dict:
    """
    Run logistic regression for binary targets.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        
    Returns:
        Dictionary with results
    """
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values.astype(int)
    
    # Fit model
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    results = {
        "model_type": "logistic_regression",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_auroc": roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "coefficients": dict(zip(feature_cols, model.coef_[0])),
        "intercept": float(model.intercept_[0]),
        "predictions_proba": y_pred_proba.tolist(),
        "predictions": y_pred.tolist(),
        "actuals": y_test.tolist(),
    }
    
    return results


def run_ridge_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    alpha: float = 1.0
) -> Dict:
    """
    Run ridge regression.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        alpha: Regularization strength
        
    Returns:
        Dictionary with results
    """
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    
    results = {
        "model_type": "ridge_regression",
        "alpha": alpha,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_r2": r2_score(y_test, y_pred_test),
        "test_mse": mean_squared_error(y_test, y_pred_test),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "coefficients": dict(zip(feature_cols, model.coef_)),
    }
    
    return results


def walk_forward_splits(
    df: pd.DataFrame,
    train_years: int = 3,
    test_years: int = 1,
    min_obs: int = 252
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits.
    
    Args:
        df: Full dataset
        train_years: Years of training data
        test_years: Years of test data
        min_obs: Minimum observations required
        
    Returns:
        List of (train_df, test_df) tuples
    """
    splits = []
    
    train_days = train_years * 252
    test_days = test_years * 252
    
    dates = df.index
    n = len(dates)
    
    start = 0
    while start + train_days + test_days <= n:
        train_end = start + train_days
        test_end = train_end + test_days
        
        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]
        
        if len(train_df) >= min_obs and len(test_df) > 0:
            splits.append((train_df, test_df))
        
        # Move forward by test period
        start += test_days
    
    return splits


def run_walk_forward_experiment(
    df: pd.DataFrame,
    target_col: str,
    feature_sets: Dict[str, List[str]],
    model_type: str = "linear",
    config: Optional[Dict] = None
) -> Dict:
    """
    Run walk-forward experiment comparing feature sets.
    
    Args:
        df: Full dataset with features and target
        target_col: Target column name
        feature_sets: Dictionary of feature set name -> feature columns
        model_type: "linear" or "logistic"
        config: Optional configuration
        
    Returns:
        Dictionary with all results
    """
    config = config or {}
    train_years = config.get("modeling", {}).get("train_years", 3)
    test_years = config.get("modeling", {}).get("test_years", 1)
    min_obs = config.get("modeling", {}).get("min_train_obs", 252)
    
    # Get splits
    splits = walk_forward_splits(df, train_years, test_years, min_obs)
    
    if len(splits) == 0:
        return {"error": "No valid splits found"}
    
    results = {
        "n_splits": len(splits),
        "model_type": model_type,
        "target": target_col,
        "feature_sets": {},
    }
    
    # Run for each feature set
    for set_name, feature_cols in feature_sets.items():
        # Filter to available features
        available = [c for c in feature_cols if c in df.columns]
        if len(available) == 0:
            continue
        
        set_results = []
        
        for train_df, test_df in splits:
            if model_type == "linear":
                fold_result = run_linear_regression(
                    train_df, test_df, available, target_col
                )
            elif model_type == "logistic":
                fold_result = run_logistic_regression(
                    train_df, test_df, available, target_col
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            fold_result["test_start"] = str(test_df.index[0])
            fold_result["test_end"] = str(test_df.index[-1])
            set_results.append(fold_result)
        
        results["feature_sets"][set_name] = set_results
    
    return results


def summarize_walk_forward_results(results: Dict) -> pd.DataFrame:
    """
    Summarize walk-forward results into a comparison table.
    
    Args:
        results: Results from run_walk_forward_experiment
        
    Returns:
        Summary DataFrame
    """
    model_type = results.get("model_type", "linear")
    
    summary_rows = []
    
    for set_name, fold_results in results.get("feature_sets", {}).items():
        if model_type == "linear":
            r2_values = [r["test_r2"] for r in fold_results]
            mse_values = [r["test_mse"] for r in fold_results]
            
            summary_rows.append({
                "feature_set": set_name,
                "mean_r2": np.mean(r2_values),
                "std_r2": np.std(r2_values),
                "mean_mse": np.mean(mse_values),
                "n_folds": len(fold_results),
            })
        else:
            auroc_values = [r["test_auroc"] for r in fold_results if not np.isnan(r.get("test_auroc", np.nan))]
            acc_values = [r["test_accuracy"] for r in fold_results]
            
            summary_rows.append({
                "feature_set": set_name,
                "mean_auroc": np.mean(auroc_values) if auroc_values else np.nan,
                "std_auroc": np.std(auroc_values) if auroc_values else np.nan,
                "mean_accuracy": np.mean(acc_values),
                "n_folds": len(fold_results),
            })
    
    return pd.DataFrame(summary_rows)


def compute_incremental_r2(
    results: Dict,
    baseline_set: str = "baseline_only",
    full_set: str = "baseline_plus_sgi"
) -> Dict:
    """
    Compute incremental R² from adding SGI features.
    
    Args:
        results: Walk-forward results
        baseline_set: Name of baseline feature set
        full_set: Name of feature set with SGI
        
    Returns:
        Dictionary with incremental metrics
    """
    feature_sets = results.get("feature_sets", {})
    
    if baseline_set not in feature_sets or full_set not in feature_sets:
        return {"error": "Feature sets not found"}
    
    baseline_r2 = [r["test_r2"] for r in feature_sets[baseline_set]]
    full_r2 = [r["test_r2"] for r in feature_sets[full_set]]
    
    incremental = [f - b for f, b in zip(full_r2, baseline_r2)]
    
    return {
        "mean_baseline_r2": np.mean(baseline_r2),
        "mean_full_r2": np.mean(full_r2),
        "mean_incremental_r2": np.mean(incremental),
        "std_incremental_r2": np.std(incremental),
        "pct_positive_incremental": np.mean([i > 0 for i in incremental]) * 100,
    }


def save_prediction_outputs(
    results: Dict,
    output_dir: Path,
    experiment_tag: str = ""
) -> None:
    """
    Save prediction outputs to files.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        experiment_tag: Tag for filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tag = f"_{experiment_tag}" if experiment_tag else ""
    
    # Save full results
    save_json(results, output_dir / f"walk_forward_results{tag}.json")
    
    # Save summary
    summary = summarize_walk_forward_results(results)
    summary.to_csv(output_dir / f"walk_forward_summary{tag}.csv", index=False)
    
    # Save incremental analysis
    incremental = compute_incremental_r2(results)
    save_json(incremental, output_dir / f"incremental_analysis{tag}.json")
