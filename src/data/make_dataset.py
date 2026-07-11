import argparse
import logging
from os import path

from src.config import DATA_DIR
from src.data.generator import TrajectoryGenerator
from src.data.io import read_dataset_config, save_dataset

logger = logging.getLogger(__name__)


def main(outfile: str, config_file: str | None = None) -> None:
    """Generates and saves a dataset of launch parameters and simulated trajectories.

    Args:
        outfile (str): Path for the saved dataset.
        config_file (str | None, optional): Path to the configuration file for the data-
            set to be generated. If None is given or individual values are missing,
            tries to load 'config/dataset/default.yaml'. If this does not exist,
            generates a dataset of 1000 samples with default parameters.
            Defaults to None.
    """
    output_path = path.join(DATA_DIR, outfile)

    if config_file is not None:
        config = read_dataset_config(config_file)
    else:
        config = {}

    n_samples = config.get("n_samples", 1000)
    gen = TrajectoryGenerator()
    trajectories, launch_params, trajectory_characteristics = gen.generate(
        n_samples, verbose=True
    )
    save_dataset(output_path, trajectories, launch_params, trajectory_characteristics)


if __name__ == "__main__":
    # Set up logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help="Filename for the dataset.")
    parser.add_argument("-c", "--config", help="Configuration file.", default=None)
    args = parser.parse_args()

    outfile = args.outfile
    config_file = args.config

    main(outfile, config_file=config_file)
