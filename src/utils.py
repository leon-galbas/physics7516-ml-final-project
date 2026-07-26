import logging

import yaml

logger = logging.getLogger(__name__)


def read_config(config_file: str) -> dict:
    logger.info(f"Reading config from '{config_file}'.")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_nested(d, *keys, default=None):
    value = d
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value
