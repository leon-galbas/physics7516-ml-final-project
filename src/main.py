import logging

from src.data.io import load_dataset
from src.data.make_dataset import main as make_dataset

logger = logging.getLogger(__name__)


def main():
    filename = "data/test.npz"
    make_dataset(filename)
    X, Y = load_dataset(filename)
    print("Success!")


if __name__ == "__main__":
    # Set up logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Run main script
    main()
