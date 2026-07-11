import logging

import numpy as np
import yaml

logger = logging.getLogger(__name__)

DEFAULT_DATASET_CONFIG = "config/dataset/default.yaml"


def save_dataset(
    outfile: str,
    X: np.ndarray,
    Y: np.ndarray,
    compressed: bool = True,
    allow_pickle: bool = False,
) -> None:
    if compressed:
        logger.info(f"Saving compressed dataset to '{outfile}'.")
        np.savez_compressed(outfile, X=X, Y=Y, allow_pickle=allow_pickle)
    else:
        logger.info(f"Saving uncompressed dataset to '{outfile}'.")
        np.savez(outfile, X=X, Y=Y, allow_pickle=allow_pickle)


def append_dataset(
    outfile: str,
    X: np.ndarray,
    Y: np.ndarray,
    compressed: bool = True,
    allow_pickle: bool = False,
) -> None:
    pass  # TODO


def load_dataset(
    infile: str, allow_pickle: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(infile, allow_pickle=allow_pickle)
    X = data["X"]
    Y = data["Y"]

    return X, Y


def read_dataset_config(config_file: str) -> dict | None:
    try:
        logger.info(f"Reading dataset config from '{config_file}'.")
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.warning(
            f"Reading the specified config failed with error: {e}. "
            "Reading default config instead."
        )
        try:
            with open(DEFAULT_DATASET_CONFIG, "r") as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.warning(
                f"Reading the default config also failed with error: {e}. "
                "Returning None."
            )
            return None
