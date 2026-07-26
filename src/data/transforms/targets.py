import numpy as np

from src.data.sample import ProcessedSample


class ComputeHitpoint:
    def __call__(self, sample: ProcessedSample) -> ProcessedSample:
        sample.targets["hit_x"] = sample.trajectory["x"][-1]
        sample.targets["hit_y"] = sample.trajectory["y"][-1]

        return sample


class ComputeEuclideanVelocity:
    def __call__(self, sample: ProcessedSample) -> ProcessedSample:
        v0 = sample.targets["v0"]
        theta = sample.targets["theta"]
        phi = sample.targets["phi"]

        vx = v0 * np.sin(theta) * np.cos(phi)
        vy = v0 * np.sin(theta) * np.sin(phi)
        vz = v0 * np.cos(theta)

        targets = {
            "vx": vx,
            "vy": vy,
            "vz": vz,
        }
        sample.targets.update(targets)

        return sample
