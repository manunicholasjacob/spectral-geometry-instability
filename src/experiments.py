"""
Experiment orchestration module for the SGI project.
High-level pipelines that combine all components.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .config import load_experiment_config, ExperimentConfig
from .paths import (
    get_outputs_dir, get_figures_dir, get_tables_dir,
    get_experiment_output_dir, ensure_all_directories
)
from .logging_utils import ExperimentLogger, log_config, log_dataframe_info
from .data_loader import load_prices_from_config, load_or_download_prices
from .preprocessing import preprocess_data
from .covariance import rolling_covariance_matrices
from .spectral_geometry import compute_rolling_geometry_features, compute_sgi_derived_features
from .features import build_feature_dataframe
from .targets import build_target_dataframe
from .diagnostics import run_all_diagnostics
from .utils import save_json, get_timestamp


def run_signal_pipeline(config: Dict) -> Dict:
    """
    Run the signal exploration pipeline.
    
    Computes SGI and related metrics, generates visualizations.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with pipeline results
    """
    timestamp = get_timestamp()
    experiment_tag = config.get("outputs", {}).get("experiment_tag", "signal")
    
    with ExperimentLogger(f"signal_{experiment_tag}") as logger:
        logger.info("Starting signal pipeline")
        log_config(logger, config)
        
        # Ensure directories exist
        ensure_all_directories()
        
        # Load and preprocess data
        logger.info("Loading price data...")
        tickers = config.get("universe", {}).get("tickers", [])
        start_date = config.get("data", {}).get("start_date", "2005-01-01")
        end_date = config.get("data", {}).get("end_date", "2024-12-31")
        
        prices = load_or_download_prices(tickers, start_date, end_date)
        logger.info(f"Loaded prices: {prices.shape}")
        
        # Preprocess
        return_method = config.get("data", {}).get("return_type", "log")
        prices, returns, metadata = preprocess_data(
            prices, return_method=return_method, output_tag=experiment_tag
        )
        log_dataframe_info(logger, returns, "Returns")
        
        # Compute covariance matrices
        logger.info("Computing rolling covariance matrices...")
        cov_estimator = config.get("covariance", {}).get("estimator", "sample")
        rolling_window = config.get("covariance", {}).get("rolling_window", 126)
        ewma_span = config.get("covariance", {}).get("ewma_span", 60)
        
        cov_matrices = rolling_covariance_matrices(
            returns, rolling_window, method=cov_estimator, span=ewma_span
        )
        logger.info(f"Computed {len(cov_matrices)} covariance matrices")
        
        # Compute geometry features
        logger.info("Computing spectral geometry features...")
        top_k = config.get("geometry", {}).get("top_k", 3)
        
        geometry_df = compute_rolling_geometry_features(cov_matrices, top_k)
        geometry_df = compute_sgi_derived_features(geometry_df)
        log_dataframe_info(logger, geometry_df, "Geometry")
        
        # Build full feature DataFrame
        logger.info("Building feature DataFrame...")
        features_df = build_feature_dataframe(returns, geometry_df, config, prices)
        
        # Run diagnostics
        logger.info("Running diagnostics...")
        diagnostics = run_all_diagnostics(
            prices, returns, geometry_df, save_results=True, output_tag=experiment_tag
        )
        
        # Generate plots
        logger.info("Generating plots...")
        from .plots import plot_sgi_time_series, plot_sgi_vs_volatility, plot_sgi_vs_correlation
        
        figures_dir = get_figures_dir()
        plot_sgi_time_series(
            geometry_df, 
            save_path=figures_dir / f"sgi_timeseries_{experiment_tag}.png"
        )
        plot_sgi_vs_volatility(
            features_df,
            save_path=figures_dir / f"sgi_vs_vol_{experiment_tag}.png"
        )
        plot_sgi_vs_correlation(
            features_df,
            save_path=figures_dir / f"sgi_vs_corr_{experiment_tag}.png"
        )
        
        # Save outputs
        output_dir = get_experiment_output_dir(f"signal_{experiment_tag}", timestamp)
        geometry_df.to_csv(output_dir / "geometry_features.csv")
        features_df.to_csv(output_dir / "all_features.csv")
        save_json(config, output_dir / "config.json")
        
        logger.info(f"Signal pipeline complete. Outputs saved to {output_dir}")
        
        return {
            "prices": prices,
            "returns": returns,
            "cov_matrices": cov_matrices,
            "geometry_df": geometry_df,
            "features_df": features_df,
            "diagnostics": diagnostics,
            "output_dir": output_dir,
        }


def run_predictive_pipeline(config: Dict) -> Dict:
    """
    Run the predictive modeling pipeline.
    
    Tests whether SGI predicts future market stress.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with pipeline results
    """
    from .predictive_models import (
        prepare_model_dataset, run_walk_forward_experiment,
        summarize_walk_forward_results, save_prediction_outputs
    )
    from .features import get_feature_columns
    
    timestamp = get_timestamp()
    experiment_tag = config.get("outputs", {}).get("experiment_tag", "predictive")
    
    with ExperimentLogger(f"predictive_{experiment_tag}") as logger:
        logger.info("Starting predictive pipeline")
        
        # First run signal pipeline to get features
        signal_results = run_signal_pipeline(config)
        
        features_df = signal_results["features_df"]
        prices = signal_results["prices"]
        returns = signal_results["returns"]
        cov_matrices = signal_results["cov_matrices"]
        
        # Build targets
        logger.info("Building target variables...")
        targets_df = build_target_dataframe(prices, returns, cov_matrices, config)
        log_dataframe_info(logger, targets_df, "Targets")
        
        # Define feature sets
        feature_sets = {
            "baseline_only": get_feature_columns("baseline_only", config),
            "sgi_only": get_feature_columns("sgi_only", config),
            "baseline_plus_sgi": get_feature_columns("baseline_plus_sgi", config),
            "full": get_feature_columns("full", config),
        }
        
        # Run experiments for each target
        all_results = {}
        target_cols = [c for c in targets_df.columns if not c.endswith("_event")]
        
        for target_col in target_cols[:4]:  # Limit to first 4 targets
            logger.info(f"Running walk-forward for target: {target_col}")
            
            # Prepare dataset
            dataset = prepare_model_dataset(
                features_df, targets_df, target_col, 
                get_feature_columns("full", config)
            )
            
            if len(dataset) < 500:
                logger.warning(f"Insufficient data for {target_col}, skipping")
                continue
            
            # Run walk-forward
            results = run_walk_forward_experiment(
                dataset, "target", feature_sets, 
                model_type="linear", config=config
            )
            
            all_results[target_col] = results
        
        # Save results
        output_dir = get_experiment_output_dir(f"predictive_{experiment_tag}", timestamp)
        
        for target_col, results in all_results.items():
            save_prediction_outputs(results, output_dir, target_col)
        
        # Create summary table
        summary_rows = []
        for target_col, results in all_results.items():
            summary = summarize_walk_forward_results(results)
            summary["target"] = target_col
            summary_rows.append(summary)
        
        if summary_rows:
            full_summary = pd.concat(summary_rows, ignore_index=True)
            full_summary.to_csv(output_dir / "full_summary.csv", index=False)
        
        save_json(config, output_dir / "config.json")
        
        logger.info(f"Predictive pipeline complete. Outputs saved to {output_dir}")
        
        return {
            "signal_results": signal_results,
            "targets_df": targets_df,
            "all_results": all_results,
            "output_dir": output_dir,
        }


def run_portfolio_pipeline(config: Dict) -> Dict:
    """
    Run the portfolio backtesting pipeline.
    
    Tests SGI-aware portfolio strategies.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with pipeline results
    """
    from .portfolio import generate_weight_schedule
    from .backtest import run_rebalancing_backtest, compare_strategies, save_backtest_results
    from .plots import plot_portfolio_comparison
    
    timestamp = get_timestamp()
    experiment_tag = config.get("outputs", {}).get("experiment_tag", "portfolio")
    
    with ExperimentLogger(f"portfolio_{experiment_tag}") as logger:
        logger.info("Starting portfolio pipeline")
        
        # First run signal pipeline
        signal_results = run_signal_pipeline(config)
        
        prices = signal_results["prices"]
        returns = signal_results["returns"]
        cov_matrices = signal_results["cov_matrices"]
        geometry_df = signal_results["geometry_df"]
        
        # Generate weight schedules for each strategy
        strategies = config.get("portfolio", {}).get("strategies", [
            "equal_weight", "inverse_vol", "min_variance", "sgi_conditioned_minvar"
        ])
        rebalance_freq = config.get("backtest", {}).get("rebalance_frequency", "weekly")
        
        weight_schedules = {}
        for strategy in strategies:
            logger.info(f"Generating weights for strategy: {strategy}")
            try:
                weights = generate_weight_schedule(
                    returns, cov_matrices, geometry_df, 
                    strategy, config, rebalance_freq
                )
                weight_schedules[strategy] = weights
            except Exception as e:
                logger.error(f"Failed to generate weights for {strategy}: {e}")
        
        # Run backtests
        transaction_cost = config.get("backtest", {}).get("transaction_cost_bps", 10)
        
        backtest_results = {}
        portfolio_returns = {}
        
        for strategy, weights in weight_schedules.items():
            logger.info(f"Running backtest for: {strategy}")
            results = run_rebalancing_backtest(prices, weights, transaction_cost)
            backtest_results[strategy] = results
            portfolio_returns[strategy] = results["net_returns"]
        
        # Compare strategies
        comparison = compare_strategies(prices, weight_schedules, transaction_cost)
        
        # Generate plots
        figures_dir = get_figures_dir()
        plot_portfolio_comparison(
            portfolio_returns,
            save_path=figures_dir / f"portfolio_comparison_{experiment_tag}.png"
        )
        
        # Save results
        output_dir = get_experiment_output_dir(f"portfolio_{experiment_tag}", timestamp)
        
        comparison.to_csv(output_dir / "strategy_comparison.csv", index=False)
        
        for strategy, results in backtest_results.items():
            save_backtest_results(results, strategy, output_dir)
        
        save_json(config, output_dir / "config.json")
        
        logger.info(f"Portfolio pipeline complete. Outputs saved to {output_dir}")
        
        return {
            "signal_results": signal_results,
            "weight_schedules": weight_schedules,
            "backtest_results": backtest_results,
            "comparison": comparison,
            "output_dir": output_dir,
        }


def run_event_study_pipeline(config: Dict) -> Dict:
    """
    Run the event study pipeline.
    
    Analyzes SGI behavior around crisis events.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with pipeline results
    """
    from .event_study import (
        run_event_study, plot_event_windows, plot_sgi_crisis_overlay,
        save_event_study_results
    )
    from .constants import CRISIS_DATES
    
    timestamp = get_timestamp()
    experiment_tag = config.get("outputs", {}).get("experiment_tag", "event_study")
    
    with ExperimentLogger(f"event_{experiment_tag}") as logger:
        logger.info("Starting event study pipeline")
        
        # First run signal pipeline
        signal_results = run_signal_pipeline(config)
        
        geometry_df = signal_results["geometry_df"]
        features_df = signal_results["features_df"]
        
        # Run event study
        logger.info("Running event study analysis...")
        event_results = run_event_study(
            geometry_df, features_df, CRISIS_DATES,
            pre_days=60, post_days=60
        )
        
        # Generate plots
        figures_dir = get_figures_dir()
        
        plot_sgi_crisis_overlay(
            geometry_df, CRISIS_DATES,
            save_path=figures_dir / f"sgi_crisis_overlay_{experiment_tag}.png"
        )
        
        plot_event_windows(
            event_results["event_data"], "sgi",
            save_path=figures_dir / f"sgi_event_windows_{experiment_tag}.png",
            title="SGI Around Crisis Events"
        )
        
        # Save results
        output_dir = get_experiment_output_dir(f"event_{experiment_tag}", timestamp)
        save_event_study_results(event_results, output_dir, experiment_tag)
        save_json(config, output_dir / "config.json")
        
        logger.info(f"Event study pipeline complete. Outputs saved to {output_dir}")
        
        return {
            "signal_results": signal_results,
            "event_results": event_results,
            "output_dir": output_dir,
        }


def run_full_pipeline(config: Dict) -> Dict:
    """
    Run all pipelines in sequence.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with all results
    """
    results = {}
    
    results["signal"] = run_signal_pipeline(config)
    results["predictive"] = run_predictive_pipeline(config)
    results["portfolio"] = run_portfolio_pipeline(config)
    results["event_study"] = run_event_study_pipeline(config)
    
    return results
