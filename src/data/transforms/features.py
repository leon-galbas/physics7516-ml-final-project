import numpy as np

from src.data.sample import ProcessedSample


class ComputeVelocities:
    def __init__(self, dims: list[str] = ["x", "y", "z"]) -> None:
        self.dims = dims

    def __call__(self, sample: ProcessedSample) -> ProcessedSample:
        t = sample.trajectory["t"]
        for dim in self.dims:
            r = sample.trajectory[dim]
            v = np.gradient(r, t)
            sample.trajectory[f"v{dim}"] = v

        return sample
