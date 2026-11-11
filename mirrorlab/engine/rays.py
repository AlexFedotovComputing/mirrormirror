from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


from .backend import get_array_module, get_backend, to_numpy, normalize, orthonormal_basis, project_to_transverse


@dataclass
class RayBundle:
    position: Any
    direction: Any
    power: Any
    wavelength_m: Any
    refractive_index: Any
    polarization: Optional[Any] = None
    intensity: Optional[Any] = None
    time_s: Optional[Any] = None
    ray_id: Optional[Any] = None
    parent_id: Optional[Any] = None
    depth: Optional[Any] = None
    bundle_reflected: Optional[Any] = None

    def __post_init__(self) -> None:
        xp = get_array_module(self.position, self.direction, self.power, self.wavelength_m, self.refractive_index)
        self.position = xp.asarray(self.position, dtype=float)
        self.direction = normalize(xp.asarray(self.direction, dtype=float))
        self.power = xp.asarray(self.power, dtype=float).reshape(-1)
        self.wavelength_m = xp.asarray(self.wavelength_m, dtype=float).reshape(-1)
        self.refractive_index = xp.asarray(self.refractive_index, dtype=float).reshape(-1)

        n = self.position.shape[0]
        if self.position.shape != (n, 3):
            raise ValueError(f"position must have shape (N, 3), got {self.position.shape}")
        if self.direction.shape != (n, 3):
            raise ValueError(f"direction must have shape (N, 3), got {self.direction.shape}")
        for name, arr in {
            "power": self.power,
            "wavelength_m": self.wavelength_m,
            "refractive_index": self.refractive_index,
        }.items():
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape (N,), got {arr.shape}")

        if self.polarization is None:
            u0, _, _ = orthonormal_basis(to_numpy(self.direction[0]), xp=xp)
            self.polarization = xp.broadcast_to(u0, (n, 3)).copy()
        else:
            self.polarization = project_to_transverse(xp.asarray(self.polarization, dtype=float), self.direction)

        if self.intensity is None:
            self.intensity = self.power.copy()
        else:
            self.intensity = xp.asarray(self.intensity, dtype=float).reshape(-1)

        if self.time_s is None:
            self.time_s = xp.zeros(n, dtype=float)
        else:
            self.time_s = xp.asarray(self.time_s, dtype=float).reshape(-1)
        if self.time_s.shape != (n,):
            raise ValueError(f"time_s must have shape (N,), got {self.time_s.shape}")

        if self.ray_id is None:
            self.ray_id = xp.arange(n, dtype=np.int64)
        else:
            self.ray_id = xp.asarray(self.ray_id, dtype=np.int64).reshape(-1)
        if self.parent_id is None:
            self.parent_id = xp.full(n, -1, dtype=np.int64)
        else:
            self.parent_id = xp.asarray(self.parent_id, dtype=np.int64).reshape(-1)
        if self.depth is None:
            self.depth = xp.zeros(n, dtype=np.int64)
        else:
            self.depth = xp.asarray(self.depth, dtype=np.int64).reshape(-1)
        if self.bundle_reflected is None:
            self.bundle_reflected = xp.zeros(n, dtype=bool)
        else:
            self.bundle_reflected = xp.asarray(self.bundle_reflected, dtype=bool).reshape(-1)
        if self.bundle_reflected.shape != (n,):
            raise ValueError(f"bundle_reflected must have shape (N,), got {self.bundle_reflected.shape}")

    @property
    def xp(self):
        return get_array_module(self.position)

    @property
    def n_rays(self) -> int:
        return int(self.position.shape[0])

    def subset(self, mask: Any) -> "RayBundle":
        return RayBundle(
            position=self.position[mask],
            direction=self.direction[mask],
            power=self.power[mask],
            wavelength_m=self.wavelength_m[mask],
            refractive_index=self.refractive_index[mask],
            polarization=self.polarization[mask],
            intensity=self.intensity[mask],
            time_s=self.time_s[mask],
            ray_id=self.ray_id[mask],
            parent_id=self.parent_id[mask],
            depth=self.depth[mask],
            bundle_reflected=self.bundle_reflected[mask],
        )

    def subset_preserving_values(self, mask: Any) -> "RayBundle":
        """Select rays without re-normalizing already validated state arrays.

        This is for internal acceleration structures that need an exact view of
        the current numerical state. The public ``subset`` keeps its historical
        validation/normalization behaviour.
        """
        selected = copy.copy(self)
        selected.position = self.position[mask]
        selected.direction = self.direction[mask]
        selected.power = self.power[mask]
        selected.wavelength_m = self.wavelength_m[mask]
        selected.refractive_index = self.refractive_index[mask]
        selected.polarization = self.polarization[mask]
        selected.intensity = self.intensity[mask]
        selected.time_s = self.time_s[mask]
        selected.ray_id = self.ray_id[mask]
        selected.parent_id = self.parent_id[mask]
        selected.depth = self.depth[mask]
        selected.bundle_reflected = self.bundle_reflected[mask]
        return selected

    def replace_ids(self, ray_id: Any, parent_id: Any) -> "RayBundle":
        return RayBundle(
            position=self.position,
            direction=self.direction,
            power=self.power,
            wavelength_m=self.wavelength_m,
            refractive_index=self.refractive_index,
            polarization=self.polarization,
            intensity=self.intensity,
            time_s=self.time_s,
            ray_id=ray_id,
            parent_id=parent_id,
            depth=self.depth,
            bundle_reflected=self.bundle_reflected,
        )

    def propagate(self, new_position: Any, delta_time_s: Any) -> "RayBundle":
        return RayBundle(
            position=new_position,
            direction=self.direction,
            power=self.power,
            wavelength_m=self.wavelength_m,
            refractive_index=self.refractive_index,
            polarization=self.polarization,
            intensity=self.intensity,
            time_s=self.time_s + delta_time_s,
            ray_id=self.ray_id,
            parent_id=self.parent_id,
            depth=self.depth,
            bundle_reflected=self.bundle_reflected,
        )

    def with_bundle_reflected(self) -> "RayBundle":
        xp = self.xp
        return RayBundle(
            position=self.position,
            direction=self.direction,
            power=self.power,
            wavelength_m=self.wavelength_m,
            refractive_index=self.refractive_index,
            polarization=self.polarization,
            intensity=self.intensity,
            time_s=self.time_s,
            ray_id=self.ray_id,
            parent_id=self.parent_id,
            depth=self.depth,
            bundle_reflected=xp.ones(self.n_rays, dtype=bool),
        )

    @classmethod
    def concat(cls, chunks: Sequence["RayBundle"], backend: Optional[str] = None) -> "RayBundle":
        chunks = [c for c in chunks if c is not None and c.n_rays > 0]
        if not chunks:
            xp = get_backend(backend or "numpy")
            empty_vec = xp.zeros((0, 3), dtype=float)
            empty = xp.zeros((0,), dtype=float)
            empty_i = xp.zeros((0,), dtype=np.int64)
            return RayBundle(
                position=empty_vec,
                direction=empty_vec,
                power=empty,
                wavelength_m=empty,
                refractive_index=empty,
                polarization=empty_vec,
                intensity=empty,
                time_s=empty,
                ray_id=empty_i,
                parent_id=empty_i,
                depth=empty_i,
                bundle_reflected=xp.zeros((0,), dtype=bool),
            )
        xp = chunks[0].xp
        return RayBundle(
            position=xp.concatenate([c.position for c in chunks], axis=0),
            direction=xp.concatenate([c.direction for c in chunks], axis=0),
            power=xp.concatenate([c.power for c in chunks], axis=0),
            wavelength_m=xp.concatenate([c.wavelength_m for c in chunks], axis=0),
            refractive_index=xp.concatenate([c.refractive_index for c in chunks], axis=0),
            polarization=xp.concatenate([c.polarization for c in chunks], axis=0),
            intensity=xp.concatenate([c.intensity for c in chunks], axis=0),
            time_s=xp.concatenate([c.time_s for c in chunks], axis=0),
            ray_id=xp.concatenate([c.ray_id for c in chunks], axis=0),
            parent_id=xp.concatenate([c.parent_id for c in chunks], axis=0),
            depth=xp.concatenate([c.depth for c in chunks], axis=0),
            bundle_reflected=xp.concatenate([c.bundle_reflected for c in chunks], axis=0),
        )

@dataclass
class Intersection:
    valid: Any
    distance: Any
    points: Any
    normals: Any
    local_u: Optional[Any] = None
    local_v: Optional[Any] = None

@dataclass
class NearestHit:
    surface_index: Any
    distance: Any
    points: Any
    normals: Any
    local_u: Any
    local_v: Any

    def subset(self, mask: Any) -> "NearestHit":
        return NearestHit(
            surface_index=self.surface_index[mask],
            distance=self.distance[mask],
            points=self.points[mask],
            normals=self.normals[mask],
            local_u=self.local_u[mask],
            local_v=self.local_v[mask],
        )
