import logging
from os import path
from typing import Mapping

import numpy as np
from scipy.integrate import OdeSolution, solve_ivp
from tqdm import tqdm

from src.config import SIM_REPO_DIR
from src.data.repository import DataRepository
from src.data.sample import RawSample

logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """Generate simulated projectile trajectories with drag, Magnus force, and wind.

    The generator samples random launch and environmental parameters, numerically
    integrates the equations of motion, and returns trajectories resampled to a
    fixed number of time points together with the sampled parameters and several
    trajectory characteristics.

    Instances are iterable and produce one trajectory sample per iteration.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Initialize the trajectory generator.

        Args:
            seed: Seed used to initialize the random number generator.
            **kwargs: Optional configuration parameters. Supported keys are:

                * ``seed``: Random seed for data generation.
                * ``t_max``: Maximum integration time.
                * ``n_timepoints``: Number of uniformly resampled trajectory points.
                * ``eps``: Small numerical offset used to avoid singularities.
                * ``v0_range``: Initial speed range.
                * ``theta_range``: Launch polar angle range.
                * ``phi_range``: Launch azimuth angle range.
                * ``beta_D_range``: Drag coefficient range.
                * ``beta_M_range``: Magnus coefficient range.
                * ``u_mag_range``: Wind speed range.
                * ``u_theta_range``: Wind polar angle range.
                * ``u_phi_range``: Wind azimuth angle range.
                * ``g``: Gravitational acceleration.

        Raises:
            ValueError: If ``omega`` is not a three-dimensional vector.
        """
        # initialize general parameters
        self.t_max: float = kwargs.get("t_max", 100.0)
        self.n_timepoints: int = kwargs.get("n_timepoints", 1000)
        self.eps: float = kwargs.get("eps", 1e-6)

        # initialize launch parameter ranges
        self.v0_range: tuple[float, float] = kwargs.get("v0_range", (0.0, 75.0))
        self.theta_range: tuple[float, float] = kwargs.get(
            "theta_range", (0.0, np.pi / 2 - self.eps)
        )
        self.phi_range: tuple[float, float] = kwargs.get(
            "phi_range", (0.0, 2 * np.pi - self.eps)
        )
        self.beta_D_range: tuple[float, float] = kwargs.get(
            "beta_D_range", (0.003, 0.04)
        )
        self.beta_M_range: tuple[float, float] = kwargs.get("beta_M_range", (0.0, 0.3))
        self.u_mag_range: tuple[float, float] = kwargs.get("u_mag_range", (0.0, 25))
        self.u_theta_range: tuple[float, float] = kwargs.get(
            "u_theta_range", (0.0, np.pi)
        )
        self.u_phi_range: tuple[float, float] = kwargs.get(
            "u_phi_range", (0.0, 2 * np.pi - self.eps)
        )
        self.g: float = kwargs.get("g", 9.81)

        # initialize random number generator
        self.seed: int = kwargs.get("seed", 42)
        self.rng: np.random.Generator = np.random.default_rng(seed=self.seed)

        # Track how many generations yielded trajectories that did not reach the ground
        self._run_count: int = 0
        self._fail_count: int = 0

    def __iter__(self) -> TrajectoryGenerator:
        """Return the generator itself.

        Returns:
            TrajectoryGenerator: The current generator instance.
        """
        return self

    def __next__(self) -> RawSample:
        """Generate the next trajectory sample.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing the trajectory, launch parameters, and trajectory
                characteristics.
        """
        return self._generate_sample()

    def generate(self, n_samples: int, verbose: bool = False) -> list[RawSample] | None:
        """Generate multiple trajectory samples.

        Args:
            n_samples: Number of trajectories to generate.
            verbose: If ``True``, display a progress bar during generation.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                * trajectories of shape ``(n_samples, n_timepoints, 4)``,
                * launch parameters of shape ``(n_samples, n_launch_params)``,
                * trajectory characteristics of shape
                    ``(n_samples, n_trajectory_characteristics)``.
        """
        logger.info(
            f"Generating {n_samples} trajectories with "
            f"{self.n_timepoints} timepoints each."
        )
        samples = []
        iterator = range(n_samples)
        if verbose:
            iterator = tqdm(iterator)
        for i in iterator:
            sample = next(self)
            samples.append(sample)

        logger.info(f"Done! Fail ratio: {self.fail_ratio:.4f}")

        return samples

    def generate_to_repo(
        self, repo_name: str, n_samples: int, verbose: bool = False
    ) -> list[RawSample] | None:
        """Generate multiple trajectory samples.

        Args:
            n_samples: Number of trajectories to generate.
            verbose: If ``True``, display a progress bar during generation.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                * trajectories of shape ``(n_samples, n_timepoints, 4)``,
                * launch parameters of shape ``(n_samples, n_launch_params)``,
                * trajectory characteristics of shape
                    ``(n_samples, n_trajectory_characteristics)``.
        """
        repository: str = path.join(SIM_REPO_DIR, f"{repo_name}.h5")

        if not path.exists(repository):
            logger.info(
                f"The repository '{repository}' does not exists. Creating new repo."
            )
            with DataRepository(repository, "a") as repo:
                pass
        else:
            with DataRepository(repository, "r") as repo:
                repo_len = len(repo)
                if repo_len >= n_samples:
                    logger.info(
                        f"The repository '{repository}' already contains {repo_len} >= {n_samples} samples. Skipping generation..."
                    )
                    return
                elif repo_len > 0 and repo_len < n_samples:
                    logger.info(
                        f"The repository '{repository}' already contains {repo_len} < {n_samples} samples. Generating the remaining {n_samples - repo_len} samples..."
                    )
                    n_samples = n_samples - repo_len
                    last_sample = repo[-1]
                    rand_state = last_sample.metadata.get("rng_state")  # pyright: ignore[reportAttributeAccessIssue]
                    if rand_state is not None:
                        self.random_state = rand_state
                    else:
                        raise ValueError(
                            "Could not infer the random state from the latest sample in the repository!"
                        )
                else:
                    logger.info(
                        f"The repository '{repository}' is empty. Generating {n_samples} samples..."
                    )

        logger.info(
            f"Generating {n_samples} trajectories with "
            f"{self.n_timepoints} timepoints each."
        )
        iterator = range(n_samples)
        if verbose:
            iterator = tqdm(iterator)
        for i in iterator:
            sample = next(self)
            with DataRepository(repository, "a") as repo:
                repo.append(sample)

        logger.info(f"Done! Fail ratio: {self.fail_ratio:.4f}")

    @property
    def fail_ratio(self) -> float:
        """Fraction of attempted trajectory generations that failed.

        Returns:
            float: Ratio of failed trajectory generation attempts to total attempts.
        """
        return self._fail_count / self._run_count

    @property
    def random_state(self) -> Mapping:
        return self.rng.bit_generator.state

    @random_state.setter
    def random_state(self, state: Mapping) -> None:
        self.rng.bit_generator.state = state

    ####################################################################################
    # Internal methods
    ####################################################################################
    def _get_launch_parameters(self) -> dict[str, float]:
        v0 = self.rng.uniform(*self.v0_range)
        theta = np.arccos(
            self.rng.uniform(np.cos(self.theta_range[1]), np.cos(self.theta_range[0]))
        )
        phi = self.rng.uniform(*self.phi_range)
        beta_D = self.rng.uniform(*self.beta_D_range)
        beta_M = self.rng.uniform(*self.beta_M_range)
        u_mag = self.rng.uniform(*self.u_mag_range)
        u_theta = np.arccos(
            self.rng.uniform(
                np.cos(self.u_theta_range[1]), np.cos(self.u_theta_range[0])
            )
        )
        u_phi = self.rng.uniform(*self.u_phi_range)
        omega = self.rng.normal(size=3)
        omega /= np.linalg.norm(omega)

        # convert to cartesian
        ux = u_mag * np.sin(u_theta) * np.cos(u_phi)
        uy = u_mag * np.sin(u_theta) * np.sin(u_phi)
        uz = u_mag * np.cos(u_theta)
        omega_x, omega_y, omega_z = omega

        return {
            "v0": v0,
            "theta": theta,
            "phi": phi,
            "ux": ux,
            "uy": uy,
            "uz": uz,
            "omega_x": omega_x,
            "omega_y": omega_y,
            "omega_z": omega_z,
            "beta_D": beta_D,
            "beta_M": beta_M,
        }

    def _compute_trajectory(self, launch_parameters: dict[str, float]) -> OdeSolution:
        # fetch launch parameters from dictionary
        v0, theta, phi = (launch_parameters[k] for k in ["v0", "theta", "phi"])
        ux, uy, uz = (launch_parameters[k] for k in ["ux", "uy", "uz"])
        omega_x, omega_y, omega_z = (
            launch_parameters[k] for k in ["omega_x", "omega_y", "omega_z"]
        )
        beta_D, beta_M = (launch_parameters[k] for k in ["beta_D", "beta_M"])

        # wind vector
        wind = np.array([ux, uy, uz], dtype=float)

        # spin axis.
        spin_axis = np.array([omega_x, omega_y, omega_z], dtype=float)

        # initial position
        x0 = 0.0
        y0 = 0.0
        z0 = self.eps  # avoids immediate ground event at t=0

        # initial velocity from spherical coordinates
        vx0 = v0 * np.sin(theta) * np.cos(phi)
        vy0 = v0 * np.sin(theta) * np.sin(phi)
        vz0 = v0 * np.cos(theta)

        y_initial = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=float)

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            """Right hand side of the initial value problem to solve.

            We have a six-dimensional state vector y = (x, y, z, vx, vy, vz) want to
            solve the system
                dr/dt = v
                dv/dt = gravity - drag + magnus

            Args:
                t (float): Time.
                state (np.ndarray): State vector (x, y, z, vx, vy, vz).

            Returns:
                np.ndarray: Right hand side dy/dt of the IVP.
            """
            x, y, z, vx, vy, vz = state

            v = np.array([vx, vy, vz])
            q = v - wind
            q_norm = np.linalg.norm(q)

            gravity_acc = np.array([0.0, 0.0, -self.g])
            drag_acc = -beta_D * q_norm * q
            magnus_acc = beta_M * np.cross(spin_axis, q)
            acc = gravity_acc + drag_acc + magnus_acc

            return np.array([vx, vy, vz, acc[0], acc[1], acc[2]])

        # Stop the integration when z=0, i.e. when the ball hits the ground
        def event_hit_ground(t: float, state: np.ndarray) -> np.ndarray:
            return state[2]

        setattr(event_hit_ground, "terminal", True)
        setattr(event_hit_ground, "direction", -1)

        sol: OdeSolution = solve_ivp(
            rhs,
            t_span=(0.0, self.t_max),
            y0=y_initial,
            events=event_hit_ground,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10,
        )

        return sol

    def _resample(self, sol: OdeSolution) -> tuple[np.ndarray, np.ndarray]:
        # Reject trajectories that did not hit the ground
        if len(sol.t_events[0]) == 0 or sol.status != 1:  # pyright: ignore[reportAttributeAccessIssue]
            raise RuntimeError("Trajectory did not hit the ground within t_max.")
        t_hit = sol.t_events[0][0]  # pyright: ignore[reportAttributeAccessIssue]

        # Resample to fixed length
        tau = np.linspace(0.0, 1.0, self.n_timepoints)
        t_eval = tau * t_hit
        states = sol.sol(t_eval).T  # pyright: ignore[reportAttributeAccessIssue]
        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]
        positions = np.stack([x, y, z], axis=1)

        return t_eval, positions

    def _generate_sample(self, max_tries: int = 100) -> RawSample:
        success = False
        try_count = 0

        while not success:
            if try_count >= max_tries:
                raise RuntimeError(
                    f"Sample generation was unsuccessful after {max_tries} tries."
                )
            try:
                self._run_count += 1
                try_count += 1
                launch_params = self._get_launch_parameters()
                trajectory_sol = self._compute_trajectory(launch_params)
                t, pos = self._resample(trajectory_sol)
                metadata = {
                    "seed": self.seed,
                    "rng_state": self.rng.bit_generator.state,
                }
                sample = RawSample(t, pos, launch_params, metadata)
                success = True
            except Exception:
                self._fail_count += 1

        return sample
