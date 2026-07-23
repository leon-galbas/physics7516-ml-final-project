from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RawSample:
    """Immutable output sample of the simulator."""

    t: np.ndarray
    position: np.ndarray
    launch_params: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        s = (
            "RawSample(\n"
            f"t = {self.t},\n"
            f"position = {self.position},\n"
            f"launch_params = {self.launch_params},\n"
            f"metadata = {self.metadata}\n"
            ")"
        )
        return s


@dataclass
class ProcessedSample:
    """Temporary representation of a sample during preprocessing."""

    trajectory: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls, raw: RawSample, include_metadata: bool = False
    ) -> ProcessedSample:
        # get data from raw sample
        trajectory = {
            "t": raw.t,
            "x": raw.position[:, 0],
            "y": raw.position[:, 1],
            "z": raw.position[:, 2],
        }
        scalars = {}
        targets = raw.launch_params
        if include_metadata:
            metadata = raw.metadata
        else:
            metadata = {}

        # create processed sample
        return cls(
            trajectory=trajectory, scalars=scalars, targets=targets, metadata=metadata
        )

    def __str__(self) -> str:
        s = (
            "ProcessedSample(\n"
            f"trajectory = {self.trajectory},\n"
            f"scalars = {self.scalars},\n"
            f"targets = {self.targets},\n"
            f"metadata = {self.metadata}\n"
            ")"
        )
        return s
