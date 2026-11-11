"""Measurement records. Coordinates remain in the survey frame, in metres."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Measurement:
    rod: int
    window: int
    ray: int
    screens: list
    source: str

    @property
    def key(self):
        return self.rod, self.window, self.ray

    @property
    def id(self):
        return "_".join(map(str, self.key))

    @property
    def points(self):
        return [np.asarray(p, dtype=float) for p in self.screens
                if p is not None and all(v is not None for v in p)]

    def to_dict(self):
        return {"id": self.id, "rod": self.rod, "window": self.window,
                "ray": self.ray, "screens_m": self.screens, "source": self.source}


@dataclass
class RayFit:
    point: np.ndarray
    direction: np.ndarray
    rms: float
    count: int

    def to_dict(self):
        return {"point_m": self.point.tolist(), "direction": self.direction.tolist(),
                "rms_mm": self.rms * 1000, "point_count": self.count}
