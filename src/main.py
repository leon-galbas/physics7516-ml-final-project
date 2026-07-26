import logging
from datetime import datetime
from os import path
from pathlib import Path

from src.config import LOGS_DIR
from src.training.train import main as train

logger = logging.getLogger(__name__)


configs = [
    "experiments/01_MLP_velocity/config.yaml",
    # "experiments/02_MLP_all-params/config.yaml",
    "experiments/03_GRU_all-params/config.yaml",
    "experiments/04_GRU_with-acceleration/config.yaml",
    "experiments/05_GRU_with-cropping/config.yaml",
]
epochs = [
    1000,
    # 1000,
    100,
    100,
    100,
]

checkpoint_intervals = [
    50,
    # 50,
    5,
    5,
    5,
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
