import numpy as np
from scipy.integrate import solve_ivp


class TrajectoryGenerator:
    def __init__(
        self,
        launch_parameter_ranges: dict[str, tuple[float, float]] = {},  # pyright: ignore[reportCallInDefaultInitializer]
        g: float = 9.81,
        omega: list[float] = [1.0, 1.0, 1.0],  # pyright: ignore[reportCallInDefaultInitializer]
        t_max: float = 100.0,
        n_timepoints: int = 1000,
        eps: float = 1e-3,
        seed: int = 42,
    ) -> None:
        # initialize launch parameter ranges
        self.v0_range: tuple[float, float] = launch_parameter_ranges.get(
            "v0", (0.0, 100.0)
        )
        self.theta_range: tuple[float, float] = launch_parameter_ranges.get(
            "theta", (0.0, np.pi / 2 - eps)
        )
        self.phi_range: tuple[float, float] = launch_parameter_ranges.get(
            "phi", (0.0, 2 * np.pi - eps)
        )
        self.beta_D_range: tuple[float, float] = launch_parameter_ranges.get(
            "beta_D", (0.0, 0.1)
        )
        self.beta_M_range: tuple[float, float] = launch_parameter_ranges.get(
            "beta_M", (0.0, 0.1)
        )
        self.ux_range: tuple[float, float] = launch_parameter_ranges.get(
            "ux", (0.0, 10)
        )
        self.uy_range: tuple[float, float] = launch_parameter_ranges.get(
            "uy", (0.0, 10)
        )
        self.uz_range: tuple[float, float] = launch_parameter_ranges.get(
            "uz", (0.0, 10)
        )

        # initialize additional parameters
        self.g: float = g
        if len(omega) != 3:
            raise ValueError("Omega must be a 3D vector!")
        self.omega: np.ndarray = np.array(omega) / np.linalg.norm(omega)
        self.t_max: float = t_max
        self.n_timepoints: int = n_timepoints
        self.eps: float = eps
        self.seed: int = seed

        # initialize random number generator
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)

    def _get_launch_parameters(self) -> np.ndarray:
        v0 = self.rng.uniform(*self.v0_range)
        theta = self.rng.uniform(*self.theta_range)
        phi = self.rng.uniform(*self.phi_range)
        beta_D = self.rng.uniform(*self.beta_D_range)
        beta_M = self.rng.uniform(*self.beta_M_range)
        ux = self.rng.uniform(*self.ux_range)
        uy = self.rng.uniform(*self.uy_range)
        uz = self.rng.uniform(*self.uz_range)

        return np.array([v0, theta, phi, beta_D, beta_M, ux, uy, uz])

    def _compute_trajectory(self, launch_parameters: np.ndarray) -> np.ndarray:
        v0, theta, phi, beta_D, beta_M, ux, uy, uz = launch_parameters

        # Wind vector
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

        sol = solve_ivp(
            rhs,
            t_span=(0.0, self.t_max),
            y0=y_initial,
            events=event_hit_ground,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10,
        )

        # Reject trajectories that did not hit the ground
        if len(sol.t_events[0]) == 0 or sol.status != 1:
            raise RuntimeError("Trajectory did not hit the ground within t_max.")

        t_hit = sol.t_events[0][0]

        # Resample to fixed length
        tau = np.linspace(0.0, 1.0, self.n_timepoints)
        t_eval = tau * t_hit

        states = sol.sol(t_eval).T

        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]

        trajectory = np.stack([t_eval, x, y, z], axis=1)

        return trajectory

    def _generate_sample(self) -> tuple[np.ndarray, np.ndarray]:
        launch_params = self._get_launch_parameters()
        trajectory = self._compute_trajectory(launch_params)

        return launch_params, trajectory

    def __iter__(self):
        while True:
            yield self._generate_sample()
