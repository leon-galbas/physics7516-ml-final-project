import numpy as np

from src.data.sample import ProcessedSample


class GaussianNoise:
    def __init__(
        self, std: float | list[float], rng: np.random.Generator | None = None
    ) -> None:
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        if isinstance(std, float):
            self.std = np.array([std, std, std])
        elif isinstance(std, list):
            if len(std) != 3:
                raise ValueError(
                    "The standard deviations must be a scalar or a list of length 3. "
                    f"Received list of length {len(std)}."
                )
            self.std = np.array(std)
        else:
            raise TypeError(
                "The standard deviations must be a scalar or a list of length 3. "
                f"Received '{std}'."
            )

    def __call__(self, sample: ProcessedSample) -> ProcessedSample:
        n = len(sample.trajectory["t"])
        noise = self.rng.normal(0.0, self.std, size=(n, 3))
        for i, key in enumerate(["x", "y", "z"]):
            sample.trajectory[key] += noise[:, i]

        return sample


class RandomWalkNoise:
    def __init__(
        self, std: float | list[float], rng: np.random.Generator | None = None
    ) -> None:
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        if isinstance(std, float):
            self.std = np.array([std, std, std])
        elif isinstance(std, list):
            if len(std) != 3:
                raise ValueError(
                    "The standard deviations must be a scalar or a list of length 3. "
                    f"Received list of length {len(std)}."
                )
            self.std = np.array(std)
        else:
            raise TypeError(
                "The standard deviations must be a scalar or a list of length 3. "
                f"Received '{std}'."
            )

    def __call__(self, sample: ProcessedSample) -> ProcessedSample:
        n = len(sample.trajectory["t"])
        increments = self.rng.normal(0.0, self.std, size=(n, 3))
        drift = np.cumsum(increments, axis=0)
        for i, key in enumerate(["x", "y", "z"]):
            sample.trajectory[key] += drift[:, i]

        return sample
