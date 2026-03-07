"""
Plotting module for the SGI project.
Provides publication-ready visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import CRISIS_DATES


def setup_plot_style():
    """Set up consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def add_crisis_shading(ax, crisis_dates: Optional[Dict] = None):
    """
    Add shaded regions for crisis periods.
    
    Args:
        ax: Matplotlib axis
        crisis_dates: Dictionary of crisis periods (uses default if None)
    """
    if crisis_dates is None:
        crisis_dates = CRISIS_DATES
    
    colors = plt.cm.Reds(np.linspace(0.2, 0.4, len(crisis_dates)))
    
    for i, (name, dates) in enumerate(crisis_dates.items()):
        start = pd.to_datetime(dates["start"])
        end = pd.to_datetime(dates["end"])
        ax.axvspan(start, end, alpha=0.3, color=colors[i], label=dates.get("name", name))


def plot_sgi_time_series(
    geometry_df: pd.DataFrame,
    crisis_dates: Optional[Dict] = None,
    save_path: Optional[Path] = None,
    title: str = "Spectral Geometry Instability (SGI) Over Time"
):
    """
    Plot SGI time series with crisis shading.
    
    Args:
        geometry_df: DataFrame with SGI metrics
        crisis_dates: Optional crisis dates dictionary
        save_path: Optional path to save figure
        title: Plot title
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot SGI
    ax1 = axes[0]
    ax1.plot(geometry_df.index, geometry_df["sgi"], color='blue', linewidth=1, label='SGI')
    if "sgi_ema" in geometry_df.columns:
        ax1.plot(geometry_df.index, geometry_df["sgi_ema"], color='red', 
                 linewidth=1.5, linestyle='--', label='SGI (EMA)')
    add_crisis_shading(ax1, crisis_dates)
    ax1.set_ylabel("SGI (radians)")
    ax1.set_title(title)
    ax1.legend(loc='upper right')
    
    # Plot weighted SGI
    ax2 = axes[1]
    ax2.plot(geometry_df.index, geometry_df["weighted_sgi"], color='green', linewidth=1)
    add_crisis_shading(ax2, crisis_dates)
    ax2.set_ylabel("Weighted SGI")
    ax2.set_title("Weighted SGI (eigenvalue-weighted)")
    
    # Plot absorption ratio
    ax3 = axes[2]
    ax3.plot(geometry_df.index, geometry_df["absorption_ratio"], color='purple', linewidth=1)
    add_crisis_shading(ax3, crisis_dates)
    ax3.set_ylabel("Absorption Ratio")
    ax3.set_title("Absorption Ratio (top-k variance fraction)")
    ax3.set_xlabel("Date")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.savefig(save_path.with_suffix('.pdf'))
    
    plt.show()
    plt.close()


def plot_sgi_vs_volatility(
    features_df: pd.DataFrame,
    save_path: Optional[Path] = None
):
    """
    Plot SGI vs realized volatility comparison.
    
    Args:
        features_df: DataFrame with features
        save_path: Optional path to save figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time series comparison
    ax1 = axes[0, 0]
    ax1.plot(features_df.index, features_df["sgi"], label='SGI', alpha=0.7)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(features_df.index, features_df["realized_vol"], 
                  color='orange', label='Realized Vol', alpha=0.7)
    ax1.set_ylabel("SGI", color='blue')
    ax1_twin.set_ylabel("Realized Vol", color='orange')
    ax1.set_title("SGI vs Realized Volatility Over Time")
    
    # Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(features_df["realized_vol"], features_df["sgi"], alpha=0.3, s=10)
    ax2.set_xlabel("Realized Volatility")
    ax2.set_ylabel("SGI")
    ax2.set_title("SGI vs Volatility Scatter")
    
    # Correlation
    corr = features_df[["sgi", "realized_vol"]].corr().iloc[0, 1]
    ax2.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax2.transAxes,
             verticalalignment='top', fontsize=12)
    
    # Rolling correlation
    ax3 = axes[1, 0]
    rolling_corr = features_df["sgi"].rolling(252).corr(features_df["realized_vol"])
    ax3.plot(features_df.index, rolling_corr)
    ax3.axhline(y=0, color='gray', linestyle='--')
    ax3.set_ylabel("Rolling Correlation")
    ax3.set_title("Rolling 1-Year Correlation (SGI vs Vol)")
    
    # Distribution comparison
    ax4 = axes[1, 1]
    ax4.hist(features_df["sgi"].dropna(), bins=50, alpha=0.5, label='SGI', density=True)
    ax4.set_xlabel("SGI")
    ax4.set_ylabel("Density")
    ax4.set_title("SGI Distribution")
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    plt.close()


def plot_sgi_vs_correlation(
    features_df: pd.DataFrame,
    save_path: Optional[Path] = None
):
    """
    Plot SGI vs average correlation comparison.
    
    Args:
        features_df: DataFrame with features
        save_path: Optional path to save figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series
    ax1 = axes[0]
    ax1.plot(features_df.index, features_df["sgi"], label='SGI', alpha=0.7)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(features_df.index, features_df["avg_correlation"], 
                  color='green', label='Avg Correlation', alpha=0.7)
    ax1.set_ylabel("SGI", color='blue')
    ax1_twin.set_ylabel("Avg Correlation", color='green')
    ax1.set_title("SGI vs Average Correlation Over Time")
    
    # Scatter
    ax2 = axes[1]
    ax2.scatter(features_df["avg_correlation"], features_df["sgi"], alpha=0.3, s=10)
    ax2.set_xlabel("Average Correlation")
    ax2.set_ylabel("SGI")
    ax2.set_title("SGI vs Correlation Scatter")
    
    corr = features_df[["sgi", "avg_correlation"]].corr().iloc[0, 1]
    ax2.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax2.transAxes,
             verticalalignment='top', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    plt.close()


def plot_drawdown_overlay(
    prices: pd.DataFrame,
    geometry_df: pd.DataFrame,
    save_path: Optional[Path] = None
):
    """
    Plot drawdown with SGI overlay.
    
    Args:
        prices: Price DataFrame
        geometry_df: Geometry DataFrame
        save_path: Optional path to save figure
    """
    setup_plot_style()
    
    # Compute portfolio drawdown
    portfolio = prices.mean(axis=1)
    cummax = portfolio.cummax()
    drawdown = (portfolio - cummax) / cummax
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Drawdown
    ax1 = axes[0]
    ax1.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
    ax1.set_ylabel("Drawdown")
    ax1.set_title("Portfolio Drawdown")
    
    # SGI
    ax2 = axes[1]
    ax2.plot(geometry_df.index, geometry_df["sgi"], color='blue', linewidth=1)
    ax2.set_ylabel("SGI")
    ax2.set_title("SGI")
    ax2.set_xlabel("Date")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    plt.close()


def plot_prediction_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[Path] = None
):
    """
    Plot prediction model comparison.
    
    Args:
        results_df: DataFrame with model comparison results
        save_path: Optional path to save figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results_df))
    width = 0.35
    
    if "r2_baseline" in results_df.columns and "r2_with_sgi" in results_df.columns:
        bars1 = ax.bar(x - width/2, results_df["r2_baseline"], width, label='Baseline')
        bars2 = ax.bar(x + width/2, results_df["r2_with_sgi"], width, label='With SGI')
        ax.set_ylabel("R²")
    elif "auroc_baseline" in results_df.columns:
        bars1 = ax.bar(x - width/2, results_df["auroc_baseline"], width, label='Baseline')
        bars2 = ax.bar(x + width/2, results_df["auroc_with_sgi"], width, label='With SGI')
        ax.set_ylabel("AUROC")
    
    ax.set_xlabel("Target")
    ax.set_title("Model Performance: Baseline vs With SGI")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    plt.close()


def plot_heatmap(
    table: pd.DataFrame,
    title: str,
    save_path: Optional[Path] = None,
    cmap: str = "RdYlGn",
    fmt: str = ".3f"
):
    """
    Plot a heatmap from a DataFrame.
    
    Args:
        table: DataFrame to plot
        title: Plot title
        save_path: Optional path to save figure
        cmap: Colormap name
        fmt: Number format string
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(table.values, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_xticklabels(table.columns, rotation=45, ha='right')
    ax.set_yticklabels(table.index)
    
    # Add text annotations
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            text = ax.text(j, i, format(table.iloc[i, j], fmt),
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    plt.close()


def plot_portfolio_comparison(
    portfolio_returns: Dict[str, pd.Series],
    save_path: Optional[Path] = None
):
    """
    Plot portfolio performance comparison.
    
    Args:
        portfolio_returns: Dictionary of strategy name -> returns series
        save_path: Optional path to save figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Cumulative returns
    ax1 = axes[0]
    for name, returns in portfolio_returns.items():
        cumulative = (1 + returns).cumprod()
        ax1.plot(cumulative.index, cumulative.values, label=name, linewidth=1.5)
    
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Portfolio Cumulative Returns")
    ax1.legend()
    ax1.set_yscale('log')
    
    # Rolling Sharpe
    ax2 = axes[1]
    for name, returns in portfolio_returns.items():
        rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        ax2.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=1)
    
    ax2.set_ylabel("Rolling Sharpe (1Y)")
    ax2.set_title("Rolling Sharpe Ratio")
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    plt.close()
