import logging

from src.data.io import load_dataset
from src.data.make_dataset import main as make_dataset
from src.visualization.trajectory import plot_trajectories_3d, plot_trajectory_3d

logger = logging.getLogger(__name__)


def main():
    filename = "dataset_100000_default.npz"
    make_dataset(filename, config_file="config/dataset/dataset.yaml")
    X, Y, Z = load_dataset(filename)
    plot_trajectories_3d(X[:10], outfile="multiple_trajectories.pdf")
    plot_trajectory_3d(X[11], Y[11], outfile="single_trajectory.pdf")
    print("Success!")


if __name__ == "__main__":
    # Set up logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Run main script
    main()
