"""
Backtesting module for the SGI project.
Implements walk-forward portfolio backtesting with transaction costs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import (
    compute_sharpe_ratio, compute_sortino_ratio,
    compute_max_drawdown, compute_calmar_ratio,
    annualize_return, annualize_volatility
)
from .paths import get_portfolio_dir
from .utils import save_json


def compute_portfolio_returns(
    prices: pd.DataFrame,
    weight_schedule: pd.DataFrame
) -> pd.Series:
    """
    Compute portfolio returns from prices and weight schedule.
    
    Args:
        prices: Price DataFrame
        weight_schedule: Weight DataFrame (rebalance dates)
        
    Returns:
        Series of daily portfolio returns
    """
    # Compute daily returns
    returns = prices.pct_change().dropna()
    
    # Forward fill weights to all dates
    all_dates = returns.index
    weights_filled = weight_schedule.reindex(all_dates).ffill()
    
    # Align
    common_dates = returns.index.intersection(weights_filled.index)
    returns = returns.loc[common_dates]
    weights = weights_filled.loc[common_dates]
    
    # Drop any dates before first weight
    first_weight_date = weight_schedule.index[0]
    returns = returns.loc[first_weight_date:]
    weights = weights.loc[first_weight_date:]
    
    # Compute portfolio return
    portfolio_returns = (returns * weights.shift(1)).sum(axis=1)
    
    return portfolio_returns.dropna()


def compute_turnover(weight_schedule: pd.DataFrame) -> pd.Series:
    """
    Compute turnover at each rebalance.
    
    Args:
        weight_schedule: Weight DataFrame
        
    Returns:
        Series of turnover values
    """
    weight_changes = weight_schedule.diff().abs()
    turnover = weight_changes.sum(axis=1) / 2  # Divide by 2 for one-way turnover
    
    return turnover


def apply_transaction_costs(
    portfolio_returns: pd.Series,
    weight_schedule: pd.DataFrame,
    cost_bps: float = 10.0
) -> pd.Series:
    """
    Apply transaction costs to portfolio returns.
    
    Args:
        portfolio_returns: Gross portfolio returns
        weight_schedule: Weight DataFrame
        cost_bps: Transaction cost in basis points (one-way)
        
    Returns:
        Net portfolio returns
    """
    # Compute turnover
    turnover = compute_turnover(weight_schedule)
    
    # Reindex turnover to daily dates
    turnover_daily = turnover.reindex(portfolio_returns.index).fillna(0)
    
    # Apply costs (two-way)
    cost_rate = cost_bps / 10000
    costs = turnover_daily * cost_rate * 2
    
    net_returns = portfolio_returns - costs
    
    return net_returns


def compute_backtest_metrics(
    portfolio_returns: pd.Series,
    periods_per_year: int = 252
) -> Dict:
    """
    Compute comprehensive backtest metrics.
    
    Args:
        portfolio_returns: Series of portfolio returns
        periods_per_year: Trading periods per year
        
    Returns:
        Dictionary of metrics
    """
    # Basic statistics
    total_return = (1 + portfolio_returns).prod() - 1
    n_years = len(portfolio_returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
    
    metrics = {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(volatility),
        "sharpe": float(compute_sharpe_ratio(portfolio_returns, 0, periods_per_year)),
        "sortino": float(compute_sortino_ratio(portfolio_returns, 0, periods_per_year)),
        "max_drawdown": float(compute_max_drawdown(portfolio_returns)),
        "calmar": float(compute_calmar_ratio(portfolio_returns, periods_per_year)),
        "skewness": float(portfolio_returns.skew()),
        "kurtosis": float(portfolio_returns.kurtosis()),
        "n_observations": len(portfolio_returns),
        "n_years": float(n_years),
        "start_date": str(portfolio_returns.index[0]),
        "end_date": str(portfolio_returns.index[-1]),
    }
    
    # Win rate
    metrics["win_rate"] = float((portfolio_returns > 0).mean())
    
    # Worst periods
    metrics["worst_day"] = float(portfolio_returns.min())
    metrics["worst_month"] = float(portfolio_returns.resample('ME').sum().min())
    metrics["worst_year"] = float(portfolio_returns.resample('YE').sum().min())
    
    # Best periods
    metrics["best_day"] = float(portfolio_returns.max())
    metrics["best_month"] = float(portfolio_returns.resample('ME').sum().max())
    
    return metrics


def run_rebalancing_backtest(
    prices: pd.DataFrame,
    weight_schedule: pd.DataFrame,
    transaction_cost_bps: float = 0.0
) -> Dict:
    """
    Run a full rebalancing backtest.
    
    Args:
        prices: Price DataFrame
        weight_schedule: Weight DataFrame
        transaction_cost_bps: Transaction cost in basis points
        
    Returns:
        Dictionary with returns and metrics
    """
    # Compute gross returns
    gross_returns = compute_portfolio_returns(prices, weight_schedule)
    
    # Apply transaction costs
    if transaction_cost_bps > 0:
        net_returns = apply_transaction_costs(
            gross_returns, weight_schedule, transaction_cost_bps
        )
    else:
        net_returns = gross_returns
    
    # Compute metrics
    gross_metrics = compute_backtest_metrics(gross_returns)
    net_metrics = compute_backtest_metrics(net_returns)
    
    # Compute turnover statistics
    turnover = compute_turnover(weight_schedule)
    
    results = {
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "gross_metrics": gross_metrics,
        "net_metrics": net_metrics,
        "turnover": {
            "mean": float(turnover.mean()),
            "std": float(turnover.std()),
            "total": float(turnover.sum()),
            "annualized": float(turnover.mean() * 252 / len(weight_schedule) * len(turnover)),
        },
        "transaction_cost_bps": transaction_cost_bps,
    }
    
    return results


def compare_strategies(
    prices: pd.DataFrame,
    weight_schedules: Dict[str, pd.DataFrame],
    transaction_cost_bps: float = 10.0
) -> pd.DataFrame:
    """
    Compare multiple portfolio strategies.
    
    Args:
        prices: Price DataFrame
        weight_schedules: Dictionary of strategy name -> weight schedule
        transaction_cost_bps: Transaction cost
        
    Returns:
        Comparison DataFrame
    """
    comparison_rows = []
    
    for strategy_name, weights in weight_schedules.items():
        results = run_rebalancing_backtest(prices, weights, transaction_cost_bps)
        
        row = {
            "strategy": strategy_name,
            **results["net_metrics"],
            "turnover_annual": results["turnover"]["annualized"],
        }
        comparison_rows.append(row)
    
    return pd.DataFrame(comparison_rows)


def compute_crisis_performance(
    portfolio_returns: pd.Series,
    crisis_periods: Dict[str, Dict[str, str]]
) -> pd.DataFrame:
    """
    Compute performance during crisis periods.
    
    Args:
        portfolio_returns: Portfolio returns series
        crisis_periods: Dictionary of crisis name -> {start, end}
        
    Returns:
        Crisis performance DataFrame
    """
    crisis_rows = []
    
    for crisis_name, dates in crisis_periods.items():
        start = pd.to_datetime(dates["start"])
        end = pd.to_datetime(dates["end"])
        
        # Filter to crisis period
        mask = (portfolio_returns.index >= start) & (portfolio_returns.index <= end)
        crisis_returns = portfolio_returns[mask]
        
        if len(crisis_returns) == 0:
            continue
        
        crisis_rows.append({
            "crisis": crisis_name,
            "start": str(start.date()),
            "end": str(end.date()),
            "total_return": float((1 + crisis_returns).prod() - 1),
            "max_drawdown": float(compute_max_drawdown(crisis_returns)),
            "volatility": float(crisis_returns.std() * np.sqrt(252)),
            "n_days": len(crisis_returns),
        })
    
    return pd.DataFrame(crisis_rows)


def save_backtest_results(
    results: Dict,
    strategy_name: str,
    output_dir: Optional[Path] = None
) -> None:
    """
    Save backtest results to files.
    
    Args:
        results: Backtest results dictionary
        strategy_name: Strategy name for filenames
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = get_portfolio_dir()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save returns
    results["gross_returns"].to_csv(output_dir / f"{strategy_name}_gross_returns.csv")
    results["net_returns"].to_csv(output_dir / f"{strategy_name}_net_returns.csv")
    
    # Save metrics
    metrics = {
        "gross_metrics": results["gross_metrics"],
        "net_metrics": results["net_metrics"],
        "turnover": results["turnover"],
    }
    save_json(metrics, output_dir / f"{strategy_name}_metrics.json")
