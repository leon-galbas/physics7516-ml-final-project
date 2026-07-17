import argparse
import logging
from os import path

import numpy as np

from src.data.generator import TrajectoryGenerator
from src.data.io import (
    append_dataset,
    get_dataset_filename,
    save_dataset,
)
from src.utils import read_dataset_config

logger = logging.getLogger(__name__)


def main(dataset_name: str, overwrite: bool = False) -> None:
    """Generates and saves a dataset of launch parameters and simulated trajectories.

    Args:
        dataset_name (str): Name of the dataset.
        overwrite (bool): Whether to overwrite an existing dataset of the same name.
            Defaults to False.
    """
    # check if dataset already exists
    if path.exists(get_dataset_filename(dataset_name)) and not overwrite:
        logger.warning(
            f"A dataset of the name '{dataset_name}' already exists. Skipping..."
        )
        return

    # read config
    config = read_dataset_config(dataset_name)

    # initialize generator
    gen = TrajectoryGenerator(**config)

    # generate data (periodically save to avoid data loss)
    n_samples = config.get("n_samples", 1000)
    generation_batch_size = 10000
    n_batches = int(np.ceil(n_samples / generation_batch_size))

    logger.info(f"Starting to generate dataset of {n_samples} samples...")
    for i in range(n_batches):
        logger.info(f"Generating batch {i + 1}/{n_batches}.")
        trajectories, launch_params, trajectory_characteristics = gen.generate(
            min(n_samples, generation_batch_size), verbose=True
        )
        if i == 0:
            save_dataset(
                dataset_name, trajectories, launch_params, trajectory_characteristics
            )
        else:
            append_dataset(
                dataset_name, trajectories, launch_params, trajectory_characteristics
            )
        n_samples -= generation_batch_size


if __name__ == "__main__":
    # Set up logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name for the dataset.")
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset of the same name.",
    )
    args = parser.parse_args()

    dataset_name = args.name
    overwrite = args.overwrite

    main(dataset_name, overwrite=overwrite)
