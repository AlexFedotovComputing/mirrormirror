from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np


from .backend import get_array_module, dot, norm, count_nonzero


from .rays import RayBundle, NearestHit


from .surfaces import Surface, PlaneSurface


from .elements import OpticalElement


@dataclass
class SurfaceGroupBounds:
    """Conservative backend-resident bounds for a contiguous surface group."""

    surface_indices: Tuple[int, ...]
    minimum: Any
    maximum: Any
    sphere_center: Any
    sphere_radius: Any

@dataclass
class CompiledScene:
    surfaces: List[Surface]
    use_broad_phase: bool = True
    intersection_tests: int = field(default=0, init=False)
    possible_tests: int = field(default=0, init=False)
    _search_plan: List[Union[int, SurfaceGroupBounds]] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        self._search_plan = self._build_search_plan()

    @staticmethod
    def _bundle_group_name(surface: Surface) -> Optional[str]:
        if not isinstance(surface, PlaneSurface) or not surface.name.startswith("BUNDLE_"):
            return None
        marker = " Mirror "
        if marker not in surface.name:
            return None
        return surface.name.split(marker, 1)[0]

    @classmethod
    def _broad_phase_group_name(cls, surface: Surface) -> Optional[str]:
        bundle_name = cls._bundle_group_name(surface)
        if bundle_name is not None:
            return bundle_name
        if isinstance(surface, PlaneSurface) and surface.shape != "infinite" and ":" in surface.name:
            return surface.name.split(":", 1)[0]
        return None

    @staticmethod
    def _plane_bounds(surface: PlaneSurface) -> Tuple[Any, Any]:
        """Return an AABB guaranteed to contain a finite plane primitive."""
        xp = get_array_module(surface.center)
        center = xp.asarray(surface.center, dtype=float)
        u = xp.asarray(surface._u, dtype=float)
        v = xp.asarray(surface._v, dtype=float)

        if surface.shape == "disk":
            # Projection of a radius-r disk onto coordinate axis i is
            # r*sqrt(u_i**2 + v_i**2); this is tighter than the enclosing square.
            extent = float(surface.radius) * xp.sqrt(u * u + v * v)
            minimum = center - extent
            maximum = center + extent
        elif surface.shape == "rectangle":
            extent = 0.5 * float(surface.width) * xp.abs(u) + 0.5 * float(surface.height) * xp.abs(v)
            minimum = center - extent
            maximum = center + extent
        elif surface.shape == "triangle":
            vertices = xp.asarray(surface.triangle_vertices, dtype=float)
            points = center[None, :] + vertices[:, 0, None] * u[None, :] + vertices[:, 1, None] * v[None, :]
            minimum = xp.min(points, axis=0)
            maximum = xp.max(points, axis=0)
        else:
            raise ValueError("An infinite plane cannot have a finite broad-phase bound.")

        # Enlarge, never shrink, the mathematical bound. The absolute term is
        # larger than intersection epsilons near the origin; the relative term
        # protects large translated coordinates and float round-off.
        scale = xp.maximum(1.0, xp.maximum(xp.abs(minimum), xp.abs(maximum)))
        padding = 1e-10 + 64.0 * np.finfo(float).eps * scale
        return minimum - padding, maximum + padding

    def _build_search_plan(self) -> List[Union[int, SurfaceGroupBounds]]:
        plan: List[Union[int, SurfaceGroupBounds]] = []
        i = 0
        while i < len(self.surfaces):
            group_name = self._broad_phase_group_name(self.surfaces[i])
            if group_name is None:
                plan.append(i)
                i += 1
                continue

            indices: List[int] = []
            while i < len(self.surfaces) and self._broad_phase_group_name(self.surfaces[i]) == group_name:
                indices.append(i)
                i += 1
            bounds = [self._plane_bounds(self.surfaces[index]) for index in indices]
            xp = get_array_module(bounds[0][0])
            minimum = xp.min(xp.stack([item[0] for item in bounds], axis=0), axis=0)
            maximum = xp.max(xp.stack([item[1] for item in bounds], axis=0), axis=0)
            sphere_center = 0.5 * (minimum + maximum)
            sphere_radius = norm(0.5 * (maximum - minimum)) + 1e-10
            plan.append(
                SurfaceGroupBounds(
                    tuple(indices),
                    minimum,
                    maximum,
                    sphere_center,
                    sphere_radius,
                )
            )
        return plan


    @staticmethod
    def _ray_group_candidates(rays: RayBundle, bounds: SurfaceGroupBounds, best_t: Any) -> Any:
        """Test a ray against a conservative sphere enclosing the group's AABB."""
        xp = rays.xp
        center = xp.asarray(bounds.sphere_center, dtype=float)
        radius = xp.asarray(bounds.sphere_radius, dtype=float)
        offset = rays.position - center[None, :]
        projection = dot(rays.direction, offset)
        discriminant = projection * projection - (dot(offset, offset) - radius * radius)
        sqrt_discriminant = xp.sqrt(xp.maximum(discriminant, 0.0))
        entry = -projection - sqrt_discriminant
        exit_ = -projection + sqrt_discriminant
        return (discriminant >= 0.0) & (exit_ > 0.0) & (entry <= best_t)

    def _empty_nearest_arrays(self, rays: RayBundle) -> Tuple[Any, Any, Any, Any, Any, Any]:
        xp = rays.xp
        n = rays.n_rays
        return (
            xp.full(n, xp.inf, dtype=float),
            xp.full(n, -1, dtype=np.int64),
            xp.zeros((n, 3), dtype=float),
            xp.zeros((n, 3), dtype=float),
            xp.zeros(n, dtype=float),
            xp.zeros(n, dtype=float),
        )

    def _update_nearest(
        self,
        rays: RayBundle,
        surface_indices: Iterable[int],
        arrays: Tuple[Any, Any, Any, Any, Any, Any],
        *,
        skip_repeated_bundle_reflections: bool,
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        xp = rays.xp
        best_t, best_idx, best_points, best_normals, best_u, best_v = arrays
        for i in surface_indices:
            surface = self.surfaces[i]
            self.intersection_tests += rays.n_rays
            hit = surface.intersect(rays)
            valid = hit.valid
            if skip_repeated_bundle_reflections and surface.name.startswith("BUNDLE_"):
                valid = valid & ~rays.bundle_reflected
            better = valid & (hit.distance < best_t)
            if count_nonzero(better) == 0:
                continue
            best_t = xp.where(better, hit.distance, best_t)
            best_idx = xp.where(better, i, best_idx)
            best_points = xp.where(better[:, None], hit.points, best_points)
            best_normals = xp.where(better[:, None], hit.normals, best_normals)
            if hit.local_u is not None:
                best_u = xp.where(better, hit.local_u, best_u)
            if hit.local_v is not None:
                best_v = xp.where(better, hit.local_v, best_v)
        return best_t, best_idx, best_points, best_normals, best_u, best_v

    @staticmethod
    def _nearest_from_arrays(arrays: Tuple[Any, Any, Any, Any, Any, Any]) -> NearestHit:
        best_t, best_idx, best_points, best_normals, best_u, best_v = arrays
        return NearestHit(
            surface_index=best_idx,
            distance=best_t,
            points=best_points,
            normals=best_normals,
            local_u=best_u,
            local_v=best_v,
        )

    def find_nearest_exhaustive(
        self,
        rays: RayBundle,
        *,
        skip_repeated_bundle_reflections: bool = False,
    ) -> NearestHit:
        """Reference implementation used for regression and differential tests."""
        arrays = self._empty_nearest_arrays(rays)
        arrays = self._update_nearest(
            rays,
            range(len(self.surfaces)),
            arrays,
            skip_repeated_bundle_reflections=skip_repeated_bundle_reflections,
        )
        return self._nearest_from_arrays(arrays)

    def find_nearest_broad_phase(
        self,
        rays: RayBundle,
        *,
        skip_repeated_bundle_reflections: bool = False,
    ) -> NearestHit:
        xp = rays.xp
        arrays = self._empty_nearest_arrays(rays)

        for entry in self._search_plan:
            if isinstance(entry, int):
                arrays = self._update_nearest(
                    rays,
                    (entry,),
                    arrays,
                    skip_repeated_bundle_reflections=skip_repeated_bundle_reflections,
                )
                continue

            candidate_mask = self._ray_group_candidates(rays, entry, arrays[0])
            is_bundle_group = self.surfaces[entry.surface_indices[0]].name.startswith("BUNDLE_")
            if skip_repeated_bundle_reflections and is_bundle_group:
                candidate_mask = candidate_mask & ~rays.bundle_reflected
            candidate_indices = xp.nonzero(candidate_mask)[0]
            if candidate_indices.size == 0:
                continue
            candidate_rays = rays.subset_preserving_values(candidate_indices)
            candidate_arrays = tuple(array[candidate_indices] for array in arrays)
            candidate_arrays = self._update_nearest(
                candidate_rays,
                entry.surface_indices,
                candidate_arrays,
                skip_repeated_bundle_reflections=skip_repeated_bundle_reflections,
            )
            for destination, values in zip(arrays, candidate_arrays):
                destination[candidate_indices] = values

        return self._nearest_from_arrays(arrays)

    def find_nearest(
        self, rays: RayBundle, *, collision_mode: str = "exhaustive",
        skip_repeated_bundle_reflections: bool = False,
    ) -> NearestHit:
        self.possible_tests += rays.n_rays * len(self.surfaces)
        if collision_mode == "expected":
            return self.find_nearest_broad_phase(
                rays, skip_repeated_bundle_reflections=skip_repeated_bundle_reflections,
            )
        if collision_mode == "exhaustive":
            # Full search must not hide collisions through route assumptions.
            return self.find_nearest_exhaustive(rays, skip_repeated_bundle_reflections=False)
        raise ValueError("collision_mode must be 'exhaustive' or 'expected'")


@dataclass
class Scene:
    elements: List[OpticalElement] = field(default_factory=list)

    def add(self, *elements: OpticalElement) -> "Scene":
        self.elements.extend(elements)
        return self

    def build_surfaces(self) -> List[Surface]:
        out: List[Surface] = []
        for element in self.elements:
            out.extend(element.build_surfaces())
        return out

    def compile(self, backend: str = "numpy", *, use_broad_phase: bool = True) -> CompiledScene:
        return CompiledScene(
            [surface.to_backend(backend) for surface in self.build_surfaces()],
            use_broad_phase=use_broad_phase,
        )
