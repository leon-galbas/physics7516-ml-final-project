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


class ComputeAccelerations:
    def __init__(self, dims: list[str] = ["x", "y", "z"]) -> None:
        self.dims = dims

    def __call__(self, sample: ProcessedSample) -> ProcessedSample:
        t = sample.trajectory["t"]
        for dim in self.dims:
            v = sample.trajectory[f"v{dim}"]
            a = np.gradient(v, t)
            sample.trajectory[f"a{dim}"] = a

        return sample


class ComputeScalarFeatures:
    def __call__(self, sample: ProcessedSample) -> ProcessedSample:
        _t = sample.trajectory["t"]
        _x = sample.trajectory["x"]
        _y = sample.trajectory["y"]
        _z = sample.trajectory["z"]
        _points = np.column_stack((_x, _y, _z))

        # get start and end point
        x0, y0, z0 = _points[0, :]
        xT, yT, zT = _points[-1, :]
        T = _t[-1]

        # get displacement and trajectory length
        Dx, Dy, Dz = _points[-1, :] - _points[0, :]
        distance = np.linalg.norm(np.array([Dx, Dy, Dz]))
        length = np.linalg.norm(np.diff(_points, axis=0), axis=1).sum()

        # get apex
        _idx = np.argmax(_z)
        x_peak, y_peak, z_peak = _points[_idx, :]
        t_peak = _t[_idx]

        # get initial and final velocities
        vx_init, vy_init, vz_init = (_points[1, :] - _points[0, :]) / (_t[1] - _t[0])
        v_init = np.linalg.norm(np.array([vx_init, vy_init, vz_init]))
        vx_final, vy_final, vz_final = (_points[-1, :] - _points[-2, :]) / (
            _t[-1] - _t[-2]
        )
        v_final = np.linalg.norm(np.array([vx_final, vy_final, vz_final]))

        # add features to sample
        features = {
            "x0": x0,
            "y0": y0,
            "z0": z0,
            "xT": xT,
            "yT": yT,
            "zT": zT,
            "T": T,
            "Dx": Dx,
            "Dy": Dy,
            "Dz": Dz,
            "distance": distance,
            "length": length,
            "x_peak": x_peak,
            "y_peak": y_peak,
            "z_peak": z_peak,
            "t_peak": t_peak,
            "vx_init": vx_init,
            "vy_init": vy_init,
            "vz_init": vz_init,
            "v_init": v_init,
            "vx_final": vx_final,
            "vy_final": vy_final,
            "vz_final": vz_final,
            "v_final": v_final,
        }
        sample.scalars.update(features)

        return sample
