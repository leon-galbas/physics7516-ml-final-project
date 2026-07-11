import argparse
import logging

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
    if config_file is not None:
        config = read_dataset_config(config_file)
    else:
        config = None

    if config is None:
        logger.info("Running dataset generation with the default configuration.")
        gen = TrajectoryGenerator()
        trajectories, launch_params, trajectory_characteristics = gen.generate(
            1000, verbose=True
        )
        save_dataset(outfile, trajectories, launch_params, trajectory_characteristics)
    else:
        pass  # TODO


if __name__ == "__main__":
    # Set up logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help="Output file for the dataset.")
    parser.add_argument("-c", "--config", help="Configuration file.", default=None)
    args = parser.parse_args()

    outfile = args.outfile
    config_file = args.config

    main(outfile, config_file=config_file)
