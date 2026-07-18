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


@dataclass
class ProcessedSample:
    """Temporary representation of a sample during preprocessing."""

    trajectory: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
