import logging
import os

import yaml

from src.config import DATASET_CONFIG_DIR

logger = logging.getLogger(__name__)


def read_dataset_config(name: str) -> dict:
    config_file = os.path.join(DATASET_CONFIG_DIR, f"{name}.yaml")
    try:
        logger.info(f"Reading dataset config from '{config_file}'.")
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(
            f"Reading the specified config failed with error: {e}. "
            "Returning empty config."
        )
        return {}


def read_config(config_file: str) -> dict:
    logger.info(f"Reading config from '{config_file}'.")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_nested(d, *keys, default=None):
    value = d
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing key: {' -> '.join(map(str, keys))}")
            return default
        value = value[key]
    return value
