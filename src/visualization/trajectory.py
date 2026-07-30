import matplotlib as mpl

from src.data.repository import DataRepository
from src.data.sample import RawSample

mpl.use("Qt5Agg")
import logging
from os import path

import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURE_DIR

logger = logging.getLogger(__name__)


def plot_trajectory_3d(
    sample: RawSample,
    plot_launch_params: bool = False,
    outfile: str | None = None,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # plot trajectory
    xs = sample.position[:, 0]
    ys = sample.position[:, 1]
    zs = sample.position[:, 2]

    ax.plot(xs, ys, zs, linestyle="-", color="blue", label="Trajectory")
    ax.plot(
        xs[0],
        ys[0],
        zs[0],
        marker="x",
        ms=8,
        mew=1.5,
        linestyle="",
        color="green",
        label="Launch point",
    )
    ax.plot(
        xs[-1],
        ys[-1],
        zs[-1],
        marker="x",
        ms=8,
        mew=1.5,
        linestyle="",
        color="red",
        label="Hit point",
    )

    # plot ground surface
    x_plane = np.array([np.min(xs), np.max(xs)])
    y_plane = np.array([np.min(ys), np.max(ys)])
    grid_x, grid_y = np.meshgrid(x_plane, y_plane)
    grid_z = np.full_like(grid_x, 0)

    ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.5, color="grey", label="Ground")

    # plot launch parameters if given
    if plot_launch_params:
        lp = sample.launch_params
        params = (
            f"$(v_0,\\theta,\\phi) = ({lp['v0']:.2f},{lp['theta']:.2f},{lp['phi']:.2f})$\n"
            f"$(u_x,u_y,u_z) = ({lp['ux']:.2f},{lp['uy']:.2f},{lp['uz']:.2f})$\n"
            f"$\\beta_D = {lp['beta_D']:.2f}$, $\\beta_M = {lp['beta_M']:.2f}$"
        )
        fig.text(
            0.02,
            0.95,
            params,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # format plot
    ax.set_xlabel("$x$[m]")
    ax.set_ylabel("$y$[m]")
    ax.set_zlabel("$z$[m]")
    plt.legend()

    if outfile is None:
        plt.show()
        plt.close()
    else:
        plot_path = path.join(FIGURE_DIR, outfile)
        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info(f"Plot saved to '{plot_path}'.")
        plt.close()


def plot_trajectories_3d(samples: list[RawSample], outfile: str | None = None) -> None:
    # check values
    if isinstance(samples, list):
        trajectories = [s.position for s in samples]
    else:
        raise TypeError("Input must be a list of 'RawSample'!")

    # initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # plot trajectories
    ax.plot(
        0,
        0,
        0,
        marker="x",
        ms=8,
        mew=1.5,
        linestyle="",
        color="green",
        label="Launch point",
    )
    for i, trajectory in enumerate(trajectories):
        xs = trajectory[:, 0]
        ys = trajectory[:, 1]
        zs = trajectory[:, 2]

        ax.plot(
            xs,
            ys,
            zs,
            linestyle="-",
            color="blue",
            label="Trajectories" if i == 0 else "_nolegend_",
        )
        ax.plot(
            xs[-1],
            ys[-1],
            zs[-1],
            marker="x",
            ms=8,
            mew=1.5,
            linestyle="",
            color="red",
            label="Hit points" if i == 0 else "_nolegend_",
        )

    # plot ground surface
    xmin = min([np.min(traj[:, 0]) for traj in trajectories])
    xmax = max([np.max(traj[:, 0]) for traj in trajectories])
    ymin = min([np.min(traj[:, 1]) for traj in trajectories])
    ymax = max([np.max(traj[:, 1]) for traj in trajectories])
    x_plane = np.array([xmin, xmax])
    y_plane = np.array([ymin, ymax])
    grid_x, grid_y = np.meshgrid(x_plane, y_plane)
    grid_z = np.full_like(grid_x, 0)

    ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.5, color="grey", label="Ground")

    # format plot
    ax.set_xlabel("$x$[m]")
    ax.set_ylabel("$y$[m]")
    ax.set_zlabel("$z$[m]")
    plt.legend()

    if outfile is None:
        plt.show()
        plt.close()
    else:
        plot_path = path.join(FIGURE_DIR, outfile)
        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info(f"Plot saved to '{plot_path}'.")
        plt.close()


def main() -> None:
    repo_file = "data/raw/10000p_default_ranges.h5"
    outfile = "sample_trajectories_3d.pdf"
    with DataRepository(repo_file, "r") as repo:
        samples = repo[:10]
    plot_trajectories_3d(samples, outfile=outfile)


if __name__ == "__main__":
    main()
