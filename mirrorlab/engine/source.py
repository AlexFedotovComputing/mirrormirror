from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np


from .backend import get_backend, asarray1, orthonormal_basis, project_to_transverse


from .rays import RayBundle


@dataclass
class GaussianBeamSource:
    waist_position: Sequence[float]
    axis: Sequence[float]
    waist_radius: float
    peak_intensity: float
    wavelength_m: float
    polarization_reference: Sequence[float] = (0.0, 1.0, 0.0)
    radial_positions: int = 15
    cutoff_ratio: float = 1.0
    n_medium: float = 1.0
    backend: str = "numpy"

    def emit(self) -> RayBundle:
        xp = get_backend(self.backend)
        waist = asarray1(self.waist_position, xp=xp)
        u, v, w = orthonormal_basis(self.axis, self.polarization_reference, xp=xp)
        pol = project_to_transverse(xp.broadcast_to(asarray1(self.polarization_reference, xp=xp), (1, 3)).copy(), xp.broadcast_to(w, (1, 3)))[0]

        n_r = int(self.radial_positions)
        Rc = self.cutoff_ratio * self.waist_radius

        points: List[Any] = []
        dirs: List[Any] = []
        intensities: List[float] = []
        powers: List[float] = []
        pols: List[Any] = []

        radii = [Rc * k / n_r for k in range(n_r + 1)]
        boundaries = [0.0]
        for k in range(1, n_r + 1):
            boundaries.append(0.5 * (radii[k - 1] + radii[k]))
        boundaries.append(Rc)

        for k, r in enumerate(radii):
            if k == 0:
                m = 1
                thetas = np.array([0.0], dtype=float)
            else:
                m = 6 * k
                thetas = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False, dtype=float)

            r_in = boundaries[k]
            r_out = boundaries[k + 1]
            annulus_area = math.pi * (r_out * r_out - r_in * r_in)
            cell_area = annulus_area / m if m > 0 else 0.0

            for theta in thetas:
                cth = float(math.cos(theta))
                sth = float(math.sin(theta))
                p = waist + r * cth * u + r * sth * v
                I = self.peak_intensity * math.exp(-2.0 * (r / self.waist_radius) ** 2)
                P = I * cell_area
                points.append(p)
                dirs.append(w)
                intensities.append(I)
                powers.append(P)
                pols.append(pol)

        position = xp.stack(points, axis=0)
        direction = xp.stack(dirs, axis=0)
        polarization = xp.stack(pols, axis=0)
        n = position.shape[0]
        return RayBundle(
            position=position,
            direction=direction,
            power=xp.asarray(powers, dtype=float),
            wavelength_m=xp.full(n, self.wavelength_m, dtype=float),
            refractive_index=xp.full(n, self.n_medium, dtype=float),
            polarization=polarization,
            intensity=xp.asarray(intensities, dtype=float),
            time_s=xp.zeros(n, dtype=float),
            ray_id=xp.arange(n, dtype=np.int64),
            parent_id=xp.full(n, -1, dtype=np.int64),
            depth=xp.zeros(n, dtype=np.int64),
        )
