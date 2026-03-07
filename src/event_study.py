"""
Event study module for the SGI project.
Analyzes SGI behavior around crisis events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import CRISIS_DATES
from .paths import get_event_studies_dir, get_figures_dir


def extract_event_window(
    series: pd.Series,
    event_date: str,
    pre_days: int = 60,
    post_days: int = 60
) -> pd.Series:
    """
    Extract data around an event date.
    
    Args:
        series: Time series data
        event_date: Event date string
        pre_days: Days before event
        post_days: Days after event
        
    Returns:
        Series indexed by days relative to event
    """
    event_dt = pd.to_datetime(event_date)
    
    # Find nearest date in series
    if event_dt not in series.index:
        # Find closest date
        idx = series.index.get_indexer([event_dt], method='nearest')[0]
        event_dt = series.index[idx]
    
    event_idx = series.index.get_loc(event_dt)
    
    start_idx = max(0, event_idx - pre_days)
    end_idx = min(len(series), event_idx + post_days + 1)
    
    window_data = series.iloc[start_idx:end_idx].copy()
    
    # Reindex to days relative to event
    relative_days = np.arange(start_idx - event_idx, end_idx - event_idx)
    window_data.index = relative_days
    
    return window_data


def build_event_study_dataframe(
    series_dict: Dict[str, pd.Series],
    event_dates: Dict[str, str],
    pre_days: int = 60,
    post_days: int = 60
) -> pd.DataFrame:
    """
    Build event study DataFrame for multiple series and events.
    
    Args:
        series_dict: Dictionary of series name -> series
        event_dates: Dictionary of event name -> date
        pre_days: Days before event
        post_days: Days after event
        
    Returns:
        DataFrame with multi-level columns (event, series)
    """
    all_data = {}
    
    for event_name, event_date in event_dates.items():
        for series_name, series in series_dict.items():
            try:
                window = extract_event_window(series, event_date, pre_days, post_days)
                all_data[(event_name, series_name)] = window
            except Exception:
                continue
    
    df = pd.DataFrame(all_data)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['event', 'series'])
    
    return df


def compute_event_statistics(
    series: pd.Series,
    event_date: str,
    pre_days: int = 60,
    post_days: int = 60
) -> Dict:
    """
    Compute statistics for a series around an event.
    
    Args:
        series: Time series
        event_date: Event date
        pre_days: Days before
        post_days: Days after
        
    Returns:
        Dictionary of statistics
    """
    window = extract_event_window(series, event_date, pre_days, post_days)
    
    pre_window = window[window.index < 0]
    post_window = window[window.index > 0]
    event_value = window.get(0, np.nan)
    
    stats = {
        "event_value": float(event_value) if not np.isnan(event_value) else None,
        "pre_mean": float(pre_window.mean()),
        "pre_std": float(pre_window.std()),
        "post_mean": float(post_window.mean()),
        "post_std": float(post_window.std()),
        "pre_max": float(pre_window.max()),
        "post_max": float(post_window.max()),
        "change_mean": float(post_window.mean() - pre_window.mean()),
    }
    
    # Z-score at event
    if stats["pre_std"] > 0:
        stats["event_zscore"] = (event_value - stats["pre_mean"]) / stats["pre_std"]
    else:
        stats["event_zscore"] = None
    
    return stats


def run_event_study(
    geometry_df: pd.DataFrame,
    features_df: pd.DataFrame,
    crisis_dates: Optional[Dict] = None,
    pre_days: int = 60,
    post_days: int = 60
) -> Dict:
    """
    Run comprehensive event study.
    
    Args:
        geometry_df: Geometry features DataFrame
        features_df: Full features DataFrame
        crisis_dates: Crisis dates dictionary
        pre_days: Days before event
        post_days: Days after event
        
    Returns:
        Dictionary with event study results
    """
    if crisis_dates is None:
        crisis_dates = CRISIS_DATES
    
    # Series to analyze
    series_dict = {
        "sgi": geometry_df["sgi"],
        "weighted_sgi": geometry_df["weighted_sgi"],
        "absorption_ratio": geometry_df["absorption_ratio"],
    }
    
    if "realized_vol" in features_df.columns:
        series_dict["realized_vol"] = features_df["realized_vol"]
    if "avg_correlation" in features_df.columns:
        series_dict["avg_correlation"] = features_df["avg_correlation"]
    
    # Extract event dates
    event_dates = {name: info["peak_date"] for name, info in crisis_dates.items()}
    
    # Build event study DataFrame
    event_df = build_event_study_dataframe(
        series_dict, event_dates, pre_days, post_days
    )
    
    # Compute statistics for each event and series
    statistics = {}
    for event_name, event_date in event_dates.items():
        statistics[event_name] = {}
        for series_name, series in series_dict.items():
            try:
                stats = compute_event_statistics(
                    series, event_date, pre_days, post_days
                )
                statistics[event_name][series_name] = stats
            except Exception:
                continue
    
    return {
        "event_data": event_df,
        "statistics": statistics,
        "crisis_dates": crisis_dates,
    }


def plot_event_windows(
    event_df: pd.DataFrame,
    series_name: str = "sgi",
    save_path: Optional[Path] = None,
    title: Optional[str] = None
):
    """
    Plot event windows for a specific series.
    
    Args:
        event_df: Event study DataFrame
        series_name: Series to plot
        save_path: Optional save path
        title: Optional title
    """
    # Get events for this series
    events = event_df.columns.get_level_values('event').unique()
    
    n_events = len(events)
    n_cols = min(3, n_events)
    n_rows = (n_events + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_events == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, event in enumerate(events):
        ax = axes[i]
        
        try:
            data = event_df[(event, series_name)]
            ax.plot(data.index, data.values, linewidth=1.5)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Event')
            ax.axhline(y=data[data.index < 0].mean(), color='gray', 
                      linestyle=':', alpha=0.5, label='Pre-event mean')
            ax.set_xlabel("Days relative to event")
            ax.set_ylabel(series_name)
            ax.set_title(event.replace('_', ' ').title())
            ax.legend(fontsize=8)
        except Exception:
            ax.set_visible(False)
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.savefig(save_path.with_suffix('.pdf'))
    
    plt.show()
    plt.close()


def plot_sgi_crisis_overlay(
    geometry_df: pd.DataFrame,
    crisis_dates: Optional[Dict] = None,
    save_path: Optional[Path] = None
):
    """
    Plot SGI with crisis periods highlighted.
    
    Args:
        geometry_df: Geometry DataFrame
        crisis_dates: Crisis dates dictionary
        save_path: Optional save path
    """
    if crisis_dates is None:
        crisis_dates = CRISIS_DATES
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot SGI
    ax.plot(geometry_df.index, geometry_df["sgi"], 
            color='blue', linewidth=1, label='SGI')
    
    # Add crisis shading
    colors = plt.cm.Reds(np.linspace(0.2, 0.5, len(crisis_dates)))
    
    for i, (name, info) in enumerate(crisis_dates.items()):
        start = pd.to_datetime(info["start"])
        end = pd.to_datetime(info["end"])
        ax.axvspan(start, end, alpha=0.3, color=colors[i], 
                  label=info.get("name", name))
    
    ax.set_xlabel("Date")
    ax.set_ylabel("SGI (radians)")
    ax.set_title("Spectral Geometry Instability with Crisis Periods")
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.savefig(save_path.with_suffix('.pdf'))
    
    plt.show()
    plt.close()


def save_event_study_results(
    results: Dict,
    output_dir: Optional[Path] = None,
    tag: str = ""
) -> None:
    """
    Save event study results.
    
    Args:
        results: Event study results
        output_dir: Output directory
        tag: Filename tag
    """
    if output_dir is None:
        output_dir = get_event_studies_dir()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_{tag}" if tag else ""
    
    # Save event data
    results["event_data"].to_csv(output_dir / f"event_data{suffix}.csv")
    
    # Save statistics
    from .utils import save_json
    save_json(results["statistics"], output_dir / f"event_statistics{suffix}.json")
