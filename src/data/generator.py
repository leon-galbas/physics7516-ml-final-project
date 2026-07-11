import logging

import numpy as np
from scipy.integrate import OdeSolution, solve_ivp
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """Generate simulated projectile trajectories with drag, Magnus force, and wind.

    The generator samples random launch and environmental parameters, numerically
    integrates the equations of motion, and returns trajectories resampled to a
    fixed number of time points together with the sampled parameters and several
    trajectory characteristics.

    Instances are iterable and produce one trajectory sample per iteration.

    Attributes:
        n_launch_params: Number of sampled launch and environment parameters.
        n_trajectory_characteristics: Number of computed trajectory summary
            characteristics.
    """

    # constant class variables
    n_launch_params: int = 8
    n_trajectory_characteristics: int = 8

    def __init__(
        self,
        seed: int = 42,
        **kwargs,
    ) -> None:
        """Initialize the trajectory generator.

        Args:
            seed: Seed used to initialize the random number generator.
            **kwargs: Optional configuration parameters. Supported keys are:

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
                * ``omega``: Three-dimensional unit vector defining the spin axis.

        Raises:
            ValueError: If ``omega`` is not a three-dimensional vector.
        """
        # initialize general parameters
        self.t_max: float = kwargs.get("t_max", 100.0)
        self.n_timepoints: int = kwargs.get("n_timepoints", 10000)
        self.eps: float = kwargs.get("eps", 1e-6)

        # initialize launch parameters
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
        omega: list[float] = kwargs.get("omega", [1.0, 1.0, 1.0])
        if len(omega) != 3:
            raise ValueError("Omega must be a 3D vector!")
        self.omega: np.ndarray = np.array(omega) / np.linalg.norm(omega)

        # initialize random number generator
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)

        # Track how many generations yielded trajectories that did not reach the ground
        self._run_count: int = 0
        self._fail_count: int = 0

    def __iter__(self) -> TrajectoryGenerator:
        """Return the generator itself.

        Returns:
            TrajectoryGenerator: The current generator instance.
        """
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the next trajectory sample.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing the trajectory, launch parameters, and trajectory
                characteristics.
        """
        return self._generate_sample()

    def generate(
        self, n_samples: int, verbose: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        trajectories = np.empty((n_samples, self.n_timepoints, 4))
        launch_params = np.empty((n_samples, self.n_launch_params))
        traj_characs = np.empty((n_samples, self.n_trajectory_characteristics))

        logger.info(
            f"Generating {n_samples} trajectories with {self.n_timepoints} timepoints each."
        )

        iterator = range(n_samples)
        if verbose:
            iterator = tqdm(iterator)
        for i in iterator:
            traj, params, characs = next(self)
            trajectories[i] = traj
            launch_params[i] = params
            traj_characs[i] = characs

        logger.info(f"Done! Fail ratio: {self.fail_ratio:.4f}")

        return trajectories, launch_params, traj_characs

    @property
    def fail_ratio(self) -> float:
        """Fraction of attempted trajectory generations that failed.

        Returns:
            float: Ratio of failed trajectory generation attempts to total attempts.
        """
        return self._fail_count / self._run_count

    ####################################################################################
    # Internal methods
    ####################################################################################
    def _get_launch_parameters(self) -> np.ndarray:
        v0 = self.rng.uniform(*self.v0_range)
        theta = self.rng.uniform(*self.theta_range)
        phi = self.rng.uniform(*self.phi_range)
        beta_D = self.rng.uniform(*self.beta_D_range)
        beta_M = self.rng.uniform(*self.beta_M_range)
        u_mag = self.rng.uniform(*self.u_mag_range)
        u_theta = self.rng.uniform(*self.u_theta_range)
        u_phi = self.rng.uniform(*self.u_phi_range)

        return np.array([v0, theta, phi, beta_D, beta_M, u_mag, u_theta, u_phi])

    def _compute_trajectory(self, launch_parameters: np.ndarray) -> OdeSolution:
        v0, theta, phi, beta_D, beta_M, u_mag, u_theta, u_phi = launch_parameters

        # Wind vector
        ux = u_mag * np.sin(u_theta) * np.cos(u_phi)
        uy = u_mag * np.sin(u_theta) * np.sin(u_phi)
        uz = u_mag * np.cos(u_theta)
        wind = np.array([ux, uy, uz], dtype=float)

        # Fixed spin axis. NOTE: Maybe make this a parameter later
        spin_axis = self.omega

        # Initial position
        x0 = 0.0
        y0 = 0.0
        z0 = self.eps  # avoids immediate ground event at t=0

        # Initial velocity from spherical coordinates
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

    def _resample_and_compute_characteristics(
        self, sol: OdeSolution
    ) -> tuple[np.ndarray, np.ndarray]:
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
        trajectory = np.stack([t_eval, x, y, z], axis=1)

        # Compute characteristics
        x_absmax, y_absmax, z_absmax = np.max(np.abs(trajectory), axis=0)[1:]
        x_hit, y_hit = trajectory[-1, 1:3]
        traj_dist = np.linalg.norm(np.array([x_hit, y_hit]))
        traj_length = np.sum(np.linalg.norm(np.diff(trajectory[:, 1:], axis=0), axis=1))
        characteristics = np.array(
            [t_hit, x_hit, y_hit, x_absmax, y_absmax, z_absmax, traj_dist, traj_length]
        )

        return trajectory, characteristics

    def _generate_sample(
        self, max_tries: int = 100, return_raw: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                raw_trajectory = self._compute_trajectory(launch_params)
                trajectory, characteristics = (
                    self._resample_and_compute_characteristics(raw_trajectory)
                )
                success = True
            except Exception:
                self._fail_count += 1

        return trajectory, launch_params, characteristics
