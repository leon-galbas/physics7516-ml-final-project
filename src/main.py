import logging

from src.data.io import load_dataset
from src.data.make_dataset import main as make_dataset
from src.visualization.trajectory import plot_trajectories_3d, plot_trajectory_3d

logger = logging.getLogger(__name__)


def main():
    datasets = ["example", "example_gauss-noise", "example_random-walk-noise"]
    for dataset in datasets:
        make_dataset(dataset)
        X, Y, Z = load_dataset(dataset)
        plot_trajectories_3d(X[100:110])
        plot_trajectory_3d(X[111], Y[111])
    print("Success!")


if __name__ == "__main__":
    # Set up logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Run main script
    main()
