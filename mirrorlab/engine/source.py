from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np


from .backend import get_backend, asarray1, orthonormal_basis, project_to_transverse


from .rays import RayBundle


def _stratified_square_samples(count: int, seed: int = 0) -> np.ndarray:
    """One reproducible random sample in each of N equal-area square strata.

    Rows contain either floor(N / rows) or ceil(N / rows) cells. A row's
    height is its cell count / N, so every rectangular cell has area 1 / N,
    including when N is not a perfect square. Cell aspect ratios stay near 1.
    """
    rows = max(1, round(math.sqrt(count)))
    row_ends = np.arange(rows + 1, dtype=np.int64) * count // rows
    rng = np.random.default_rng(seed)
    samples = np.empty((count, 2), dtype=float)
    for start, end in zip(row_ends[:-1], row_ends[1:]):
        columns = end - start
        jitter = rng.random((columns, 2))
        samples[start:end, 0] = (np.arange(columns) + jitter[:, 0]) / columns
        samples[start:end, 1] = (start + columns * jitter[:, 1]) / count
    return samples


def _concentric_disk(samples: np.ndarray) -> np.ndarray:
    """Shirley–Chiu equal-area square-to-disk map (Jacobian pi).

    https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions
    Unlike polar mapping this keeps the strata reasonably compact at the centre.
    """
    a, b = (2 * samples - 1).T
    horizontal = np.abs(a) > np.abs(b)
    origin = (a == 0) & (b == 0)
    radius = np.where(horizontal, a, b)
    angle = np.zeros(len(samples))
    angle[horizontal] = math.pi / 4 * b[horizontal] / a[horizontal]
    vertical = ~horizontal & ~origin
    angle[vertical] = math.pi / 2 - math.pi / 4 * a[vertical] / b[vertical]
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))


def stratified_disk_samples(count: int, seed: int = 0) -> np.ndarray:
    """N equal-area strata on the unit disk, one fixed-seed random point each."""
    if type(count) is not int or count < 1:
        raise ValueError("ray_count must be a positive integer")
    return _concentric_disk(_stratified_square_samples(count, seed))


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

    def emit(self, ray_count: int | None = None) -> RayBundle:
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

        if ray_count is not None:
            if type(ray_count) is not int or ray_count < 1:
                raise ValueError("ray_count must be a positive integer")
            # Equal-area 2D strata avoid both spiral patterns and IID clumps.
            xy = Rc * stratified_disk_samples(ray_count)
            radii = np.linalg.norm(xy, axis=1)
            offsets = xp.asarray(xy[:, 0])[:, None] * u + xp.asarray(xy[:, 1])[:, None] * v
            intensity = self.peak_intensity * np.exp(-2.0 * (radii / self.waist_radius) ** 2)
            power = intensity * (math.pi * Rc * Rc / ray_count)
            # Normalize quadrature to the analytic power inside the aperture.
            total = self.peak_intensity * math.pi * self.waist_radius**2 / 2 * (-math.expm1(-2 * self.cutoff_ratio**2))
            if power.sum() > 0:
                power *= total / power.sum()
            return RayBundle(
                position=waist + offsets, direction=xp.broadcast_to(w, (ray_count, 3)).copy(),
                power=xp.asarray(power), intensity=xp.asarray(intensity),
                wavelength_m=xp.full(ray_count, self.wavelength_m),
                refractive_index=xp.full(ray_count, self.n_medium),
                polarization=xp.broadcast_to(pol, (ray_count, 3)).copy(),
                time_s=xp.zeros(ray_count), ray_id=xp.arange(ray_count, dtype=np.int64),
                parent_id=xp.full(ray_count, -1, dtype=np.int64), depth=xp.zeros(ray_count, dtype=np.int64),
            )

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
