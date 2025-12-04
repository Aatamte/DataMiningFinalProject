"""Configuration loader for SLM-RL training."""

import os
import sys
from pathlib import Path

import yaml


def load_config(config_path: str | Path | None = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses TRAIN_CONFIG env var.

    Returns:
        Configuration dictionary

    Raises:
        SystemExit: If config path not set or file doesn't exist
    """
    if config_path is None:
        config_path = os.environ.get("TRAIN_CONFIG")

    if not config_path:
        print("ERROR: TRAIN_CONFIG environment variable not set")
        print("Set it in .env or export TRAIN_CONFIG=configs/default.yaml")
        sys.exit(1)

    config_path = Path(config_path)

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Check your TRAIN_CONFIG path")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Loaded config: {config_path}")
    return config


def get_config_value(config: dict, *keys, default=None):
    """Safely get nested config value.

    Args:
        config: Config dictionary
        *keys: Nested keys (e.g., "model", "name")
        default: Default value if not found

    Returns:
        Config value or default
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
