import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

from src.config import DATA_DIR, DATASET_CONFIG_DIR

logger = logging.getLogger(__name__)


def get_dataset_filename(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.npz")


def get_config_filename(name: str) -> str:
    return os.path.join(DATASET_CONFIG_DIR, f"{name}.yaml")


def save_dataset(
    name: str,
    trajectories: np.ndarray,
    launch_params: np.ndarray,
    trajectory_characteristics: np.ndarray,
    compressed: bool = True,
    allow_pickle: bool = False,
) -> None:
    outfile = get_dataset_filename(name)
    if compressed:
        logger.info(f"Saving compressed dataset to '{outfile}'.")
        np.savez_compressed(
            outfile,
            trajectories=trajectories,
            launch_parameters=launch_params,
            trajectory_characteristics=trajectory_characteristics,
            allow_pickle=allow_pickle,
        )
    else:
        logger.info(f"Saving uncompressed dataset to '{outfile}'.")
        np.savez(
            outfile,
            trajectories=trajectories,
            launch_parameters=launch_params,
            trajectory_characteristics=trajectory_characteristics,
            allow_pickle=allow_pickle,
        )


def append_dataset(
    name: str,
    trajectories: np.ndarray,
    launch_params: np.ndarray,
    trajectory_characteristics: np.ndarray,
    compressed: bool = True,
    allow_pickle: bool = False,
) -> None:
    outfile = get_dataset_filename(name)
    filename = Path(outfile)

    with np.load(filename) as data:
        old_traj = data["trajectories"]
        old_params = data["launch_parameters"]
        old_characs = data["trajectory_characteristics"]

    # validate shapes
    if old_traj.shape[1:] != trajectories.shape[1:]:
        raise ValueError(
            "Trajectory shapes do not match: "
            f"{old_traj.shape[1:]} vs {trajectories.shape[1:]}"
        )
    if old_params.shape[1:] != launch_params.shape[1:]:
        raise ValueError(
            "Launch parameter shapes do not match: "
            f"{old_params.shape[1:]} vs {launch_params.shape[1:]}"
        )
    if old_characs.shape[1:] != trajectory_characteristics.shape[1:]:
        raise ValueError(
            "Characteristic shapes do not match: "
            f"{old_characs.shape[1:]} vs {trajectory_characteristics.shape[1:]}"
        )

    trajectories = np.concatenate([old_traj, trajectories], axis=0)
    launch_params = np.concatenate([old_params, launch_params], axis=0)
    trajectory_characteristics = np.concatenate(
        [old_characs, trajectory_characteristics], axis=0
    )

    # write to temporary file first to prevent data loss in case of error
    with tempfile.NamedTemporaryFile(
        dir=filename.parent, suffix=".npz", delete=False
    ) as tmp:
        tmp_name = tmp.name

    try:
        if compressed:
            logger.info(f"Appending compressed dataset to '{outfile}'.")
            np.savez_compressed(
                tmp_name,
                trajectories=trajectories,
                launch_parameters=launch_params,
                trajectory_characteristics=trajectory_characteristics,
                allow_pickle=allow_pickle,
            )
        else:
            logger.info(f"Appending uncompressed dataset to '{outfile}'.")
            np.savez(
                tmp_name,
                trajectories=trajectories,
                launch_parameters=launch_params,
                trajectory_characteristics=trajectory_characteristics,
                allow_pickle=allow_pickle,
            )
        os.replace(tmp_name, filename)
    except Exception as e:
        logger.error(f"Appending failed with exception: {e}")
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def load_dataset(
    name: str, allow_pickle: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    infile = get_dataset_filename(name)
    logger.info(f"Loading dataset from '{infile}'...")
    data = np.load(infile, allow_pickle=allow_pickle)
    trajectories = data["trajectories"]
    launch_params = data["launch_parameters"]
    trajectory_characteristics = data["trajectory_characteristics"]

    return trajectories, launch_params, trajectory_characteristics


def read_dataset_config(name: str) -> dict:
    config_file = get_config_filename(name)
    try:
        logger.info(f"Reading dataset config from '{config_file}'.")
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.warning(
            f"Reading the specified config failed with error: {e}. "
            "Using default values instead."
        )
        return {}
