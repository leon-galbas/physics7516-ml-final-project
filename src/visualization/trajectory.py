import matplotlib as mpl

mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_3d(
    trajectory: np.ndarray, launch_parameters: np.ndarray | None = None
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # plot trajectory
    xs = trajectory[:, 1]
    ys = trajectory[:, 2]
    zs = trajectory[:, 3]

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
    if launch_parameters is not None:
        v0, theta, phi, beta_D, beta_M, ux, uy, uz = launch_parameters
        params = (
            f"$(v_0,\\theta,\\phi) = ({v0:.2f},{theta:.2f},{phi:.2f})$\n"
            f"$(u_x,u_y,u_z) = ({ux:.2f},{uy:.2f},{uz:.2f})$\n"
            f"$\\beta_D = {beta_D:.2f}$, $\\beta_M = {beta_M:.2f}$"
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
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    plt.legend()

    plt.show()
    plt.close()


def plot_trajectories_3d(
    trajectories: np.ndarray | list[np.ndarray],
) -> None:
    # check values
    if isinstance(trajectories, list):
        trajectories = np.stack(trajectories)
    if isinstance(trajectories, np.ndarray):
        if trajectories.ndim != 3:
            raise ValueError(
                "Input must have shape (n_trajectories, n_points, 4).\n"
                "For a single trajectory try 'plot_trajectory_3d'."
            )
        if trajectories.shape[2] != 4:
            raise ValueError("Last dimension must contain t, x, y, z coordinates.")
    else:
        raise TypeError("Input must either be a numpy array or a list of numpy arrays!")

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
        xs = trajectory[:, 1]
        ys = trajectory[:, 2]
        zs = trajectory[:, 3]

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
    xmin, xmax = (np.min(trajectories[:, :, 1]), np.max(trajectories[:, :, 1]))
    ymin, ymax = (np.min(trajectories[:, :, 2]), np.max(trajectories[:, :, 2]))
    x_plane = np.array([xmin, xmax])
    y_plane = np.array([ymin, ymax])
    grid_x, grid_y = np.meshgrid(x_plane, y_plane)
    grid_z = np.full_like(grid_x, 0)

    ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.5, color="grey", label="Ground")

    # format plot
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    plt.legend()

    plt.show()
    plt.close()
