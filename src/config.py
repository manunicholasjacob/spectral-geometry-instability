"""
Configuration management for the SGI project.
Handles loading, merging, and validating configuration files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from .paths import get_configs_dir
from .utils import load_yaml, flatten_dict


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a configuration file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    return load_yaml(config_path)


def load_base_config() -> Dict[str, Any]:
    """
    Load the base configuration file.
    
    Returns:
        Base configuration dictionary
    """
    base_path = get_configs_dir() / "base.yaml"
    if base_path.exists():
        return load_yaml(base_path)
    return {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    Override values take precedence.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_experiment_config(
    config_path: Union[str, Path],
    merge_base: bool = True
) -> Dict[str, Any]:
    """
    Load an experiment configuration, optionally merging with base config.
    
    Args:
        config_path: Path to experiment config YAML file
        merge_base: Whether to merge with base.yaml
        
    Returns:
        Complete configuration dictionary
    """
    experiment_config = load_config(config_path)
    
    if merge_base:
        base_config = load_base_config()
        return merge_configs(base_config, experiment_config)
    
    return experiment_config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required top-level sections
    required_sections = ["data", "covariance", "geometry"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate data section
    if "data" in config:
        data = config["data"]
        if "start_date" not in data:
            errors.append("Missing data.start_date")
        if "end_date" not in data:
            errors.append("Missing data.end_date")
    
    # Validate universe section
    if "universe" in config:
        universe = config["universe"]
        if "tickers" not in universe or not universe["tickers"]:
            errors.append("Missing or empty universe.tickers")
    
    # Validate covariance section
    if "covariance" in config:
        cov = config["covariance"]
        if "rolling_window" in cov:
            if cov["rolling_window"] < 20:
                errors.append("covariance.rolling_window should be >= 20")
        if "estimator" in cov:
            valid_estimators = ["sample", "ewma", "ledoit_wolf", "oas"]
            if cov["estimator"] not in valid_estimators:
                errors.append(f"Invalid covariance.estimator: {cov['estimator']}")
    
    # Validate geometry section
    if "geometry" in config:
        geom = config["geometry"]
        if "top_k" in geom:
            if geom["top_k"] < 1:
                errors.append("geometry.top_k must be >= 1")
    
    return errors


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from a nested config using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "covariance.rolling_window")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


@dataclass
class ExperimentConfig:
    """Structured experiment configuration."""
    
    # Project settings
    name: str = "sgi_experiment"
    seed: int = 42
    
    # Data settings
    start_date: str = "2005-01-01"
    end_date: str = "2024-12-31"
    tickers: List[str] = field(default_factory=list)
    market_proxy: Optional[str] = None
    return_type: str = "log"
    
    # Covariance settings
    cov_estimator: str = "sample"
    rolling_window: int = 126
    ewma_span: int = 60
    
    # Geometry settings
    top_k: int = 3
    
    # Target settings
    horizons: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    # Output settings
    output_dir: str = "outputs"
    experiment_tag: str = ""
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ExperimentConfig":
        """Create ExperimentConfig from dictionary."""
        return cls(
            name=get_config_value(config, "project.name", "sgi_experiment"),
            seed=get_config_value(config, "project.seed", 42),
            start_date=get_config_value(config, "data.start_date", "2005-01-01"),
            end_date=get_config_value(config, "data.end_date", "2024-12-31"),
            tickers=get_config_value(config, "universe.tickers", []),
            market_proxy=get_config_value(config, "universe.market_proxy"),
            return_type=get_config_value(config, "data.return_type", "log"),
            cov_estimator=get_config_value(config, "covariance.estimator", "sample"),
            rolling_window=get_config_value(config, "covariance.rolling_window", 126),
            ewma_span=get_config_value(config, "covariance.ewma_span", 60),
            top_k=get_config_value(config, "geometry.top_k", 3),
            horizons=get_config_value(config, "targets.horizons", [5, 10, 20, 60]),
            output_dir=get_config_value(config, "outputs.base_dir", "outputs"),
            experiment_tag=get_config_value(config, "outputs.experiment_tag", ""),
        )
