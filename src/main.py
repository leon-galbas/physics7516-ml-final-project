import logging
from datetime import datetime
from os import path
from pathlib import Path

from src.config import LOGS_DIR
from src.training.train import main as train

logger = logging.getLogger(__name__)


configs = [
    "experiments/training/01_MLP_velocity/config.yaml",
    "experiments/training/02_MLP_all-params/config.yaml",
    "experiments/training/03_GRU_all-params/config.yaml",
    "experiments/training/04_GRU_with-acceleration/config.yaml",
    "experiments/training/05_GRU_with-cropping/config.yaml",
    "experiments/training/06_GRU_with-cropping_bigger/config.yaml",
    "experiments/training/07_GRU_with-cropping_endpoint/config.yaml",
    "experiments/training/08_GRU_with-cropping_bigger_endpoint/config.yaml",
]
epochs = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]

checkpoint_intervals = [
    50,
    50,
    10,
    10,
    10,
    10,
    20,
    20,
]


def main():
    for i in range(len(configs)):
        try:
            train(configs[i], epochs=epochs[i], check_freq=checkpoint_intervals[i])
        except Exception as e:
            logger.error(
                f"Execution of training run '{configs[i]}' failed with error {e}."
            )


if __name__ == "__main__":
    # logger setup
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    logfile = path.join(LOGS_DIR, f"{timestamp}_{script_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logfile),
        ],
    )

    main()
