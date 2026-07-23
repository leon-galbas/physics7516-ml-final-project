import numpy as np
from scipy.interpolate import CubicSpline

from src.data.sample import ProcessedSample


def _even(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.linspace(0, 1, n)


def _log(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.logspace(0, 1, n)


def _uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0, 1, n)


def _gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    u = []
    while len(u) < n:
        x = rng.normal(0.5, 0.15, size=n)
        x = x[(x >= 0) & (x <= 1)]
        u.extend(x)

    return np.array(u[:n])


class Resample:
    DISTRIBUTIONS = {
        "even": _even,
        "log": _log,
        "uniform": _uniform,
        "gaussian": _gaussian,
    }

    def __init__(
        self,
        n_timepoints: int,
        distribution: str = "even",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_timepoints = n_timepoints
        try:
            self.dist_func = self.DISTRIBUTIONS[distribution]
        except Exception as e:
            raise ValueError(
                f"The distribution '{distribution}' is not implemented. Failed with error {e}."
            )
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

    def __call__(self, sample: ProcessedSample) -> ProcessedSample:
        # sample new timepoints
        t_old = sample.trajectory["t"]
        t_min = np.min(t_old)
        t_max = np.max(t_old)
        u = self.dist_func(self.n_timepoints, self.rng)
        t_new = np.sort(t_min + u * (t_max - t_min))

        # interpolate the trajectory for the new times
        keys = [key for key in sample.trajectory.keys() if key != "t"]
        xyz = np.stack(
            [sample.trajectory[key] for key in keys],
            axis=1,
        )
        spline = CubicSpline(t_old, xyz)
        xyz_new = spline(t_new)

        # return resampled sample
        sample.trajectory["t"] = t_new
        for i, key in enumerate(keys):
            sample.trajectory[key] = xyz_new[:, i]

        return sample


class IndexSlice:
    def __init__(self, min_idx: float, max_idx: float) -> None:
        self.min_idx: float = min_idx
        self.max_idx: float = max_idx

    def __call__(self, sample: ProcessedSample) -> ProcessedSample:
        # slice all trajectory arrays
        for key in sample.trajectory.keys():
            arr = sample.trajectory[key]
            arr = arr[self.min_idx : self.max_idx]
            sample.trajectory[key] = arr

        return sample


class TimeSlice:
    def __init__(self, t_min: float, t_max: float) -> None:
        self.t_min: float = t_min
        self.t_max: float = t_max

    def __call__(self, sample: ProcessedSample) -> ProcessedSample:
        # get time mask
        t = sample.trajectory["t"]
        mask = (t >= self.t_min) & (t <= self.t_max)

        # slice all trajectory arrays
        for key in sample.trajectory.keys():
            arr = sample.trajectory[key]
            arr = arr[mask]
            sample.trajectory[key] = arr

        return sample
