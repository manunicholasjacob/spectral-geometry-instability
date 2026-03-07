"""
Path management for the SGI project.
Centralizes all path definitions and ensures directories exist.
"""

from pathlib import Path
from typing import Optional
import hashlib
import json


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """Get the raw data directory."""
    path = get_data_dir() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_processed_data_dir() -> Path:
    """Get the processed data directory."""
    path = get_data_dir() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_metadata_dir() -> Path:
    """Get the metadata directory."""
    path = get_data_dir() / "metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_configs_dir() -> Path:
    """Get the configs directory."""
    return get_project_root() / "configs"


def get_outputs_dir() -> Path:
    """Get the outputs directory."""
    path = get_project_root() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_figures_dir() -> Path:
    """Get the figures output directory."""
    path = get_outputs_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_tables_dir() -> Path:
    """Get the tables output directory."""
    path = get_outputs_dir() / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_dir() -> Path:
    """Get the logs output directory."""
    path = get_outputs_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_diagnostics_dir() -> Path:
    """Get the diagnostics output directory."""
    path = get_outputs_dir() / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_predictions_dir() -> Path:
    """Get the predictions output directory."""
    path = get_outputs_dir() / "predictions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_portfolio_dir() -> Path:
    """Get the portfolio output directory."""
    path = get_outputs_dir() / "portfolio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_event_studies_dir() -> Path:
    """Get the event studies output directory."""
    path = get_outputs_dir() / "event_studies"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_paper_assets_dir() -> Path:
    """Get the paper assets directory."""
    path = get_project_root() / "paper_assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_experiment_output_dir(experiment_tag: str, timestamp: Optional[str] = None) -> Path:
    """
    Get experiment-specific output directory.
    
    Args:
        experiment_tag: Identifier for the experiment
        timestamp: Optional timestamp string
        
    Returns:
        Path to experiment output directory
    """
    if timestamp:
        dir_name = f"{experiment_tag}_{timestamp}"
    else:
        dir_name = experiment_tag
    
    path = get_outputs_dir() / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_to_hash(config: dict) -> str:
    """
    Generate a short hash from config for unique experiment identification.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        8-character hash string
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def ensure_all_directories() -> None:
    """Create all necessary directories if they don't exist."""
    get_raw_data_dir()
    get_processed_data_dir()
    get_metadata_dir()
    get_figures_dir()
    get_tables_dir()
    get_logs_dir()
    get_diagnostics_dir()
    get_predictions_dir()
    get_portfolio_dir()
    get_event_studies_dir()
    get_paper_assets_dir()
