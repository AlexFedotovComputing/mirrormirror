from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


from .backend import get_array_module, get_backend, to_numpy, count_nonzero, LIGHT_SPEED_M_S


from .rays import RayBundle


from .scene import CompiledScene, Scene


@dataclass
class TraceResult:
    final_rays: RayBundle
    segments: List[Dict[str, Any]]
    detector_hits: List[Dict[str, Any]]
    surface_names: List[str]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def detector_power_summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for item in self.detector_hits:
            name = str(item["surface"])
            power = float(np.sum(item["power"]))
            out[name] = out.get(name, 0.0) + power
        return out

@dataclass
class RayTracer:
    scene: Scene
    backend: str = "numpy"
    max_interactions: int = 16
    max_time_s: Optional[float] = None
    min_power: float = 1e-18
    surface_epsilon: float = 1e-7
    record_segments: bool = True
    max_active_rays: Optional[int] = None
    bundle_clip_inner_radius_m: Optional[float] = None
    bundle_clip_outer_radius_m: Optional[float] = None
    bundle_clip_z_min_m: Optional[float] = None
    bundle_clip_z_max_m: Optional[float] = None
    bundle_clip_epsilon_m: float = 1e-9
    skip_repeated_bundle_reflections: bool = False
    collision_mode: str = "exhaustive"
    compiled_scene: CompiledScene = field(init=False)

    def __post_init__(self) -> None:
        if self.collision_mode not in {"exhaustive", "expected"}:
            raise ValueError("collision_mode must be 'exhaustive' or 'expected'")
        self.compiled_scene = self.scene.compile(self.backend)

    def _distance_to_time(self, distance_m: Any, refractive_index: Any) -> Any:
        xp = get_array_module(distance_m, refractive_index)
        return xp.asarray(distance_m, dtype=float) * xp.asarray(refractive_index, dtype=float) / LIGHT_SPEED_M_S

    def _time_to_distance(self, delta_time_s: Any, refractive_index: Any) -> Any:
        xp = get_array_module(delta_time_s, refractive_index)
        return xp.asarray(delta_time_s, dtype=float) * LIGHT_SPEED_M_S / xp.asarray(refractive_index, dtype=float)

    def _make_segment_block(self, rays: RayBundle, end_points: Any, surface_names: Any, end_time_s: Any) -> Dict[str, Any]:
        return {
            "ray_id": to_numpy(rays.ray_id),
            "parent_id": to_numpy(rays.parent_id),
            "depth": to_numpy(rays.depth),
            "surface": to_numpy(surface_names),
            "x0": to_numpy(rays.position[:, 0]),
            "y0": to_numpy(rays.position[:, 1]),
            "z0": to_numpy(rays.position[:, 2]),
            "x1": to_numpy(end_points[:, 0]),
            "y1": to_numpy(end_points[:, 1]),
            "z1": to_numpy(end_points[:, 2]),
            "power": to_numpy(rays.power),
            "intensity": to_numpy(rays.intensity),
            "bundle_reflected": to_numpy(rays.bundle_reflected),
            "t0_s": to_numpy(rays.time_s),
            "t1_s": to_numpy(end_time_s),
        }

    def _propagate_to_time_limit(self, rays: RayBundle, delta_time_s: Any) -> RayBundle:
        end_distance = self._time_to_distance(delta_time_s, rays.refractive_index)
        end_points = rays.position + end_distance[:, None] * rays.direction
        return rays.propagate(end_points, delta_time_s)

    def _bundle_clip_surface_name(self, kind: int) -> str:
        if kind == 1 and self.bundle_clip_inner_radius_m is not None:
            return f"<bundle_inner_cylinder_{float(self.bundle_clip_inner_radius_m):g}m>"
        if kind == 2 and self.bundle_clip_outer_radius_m is not None:
            return f"<bundle_outer_cylinder_{float(self.bundle_clip_outer_radius_m):g}m>"
        if kind == 3 and self.bundle_clip_z_min_m is not None:
            return f"<bundle_z_min_{float(self.bundle_clip_z_min_m):g}m>"
        if kind == 4 and self.bundle_clip_z_max_m is not None:
            return f"<bundle_z_max_{float(self.bundle_clip_z_max_m):g}m>"
        return "<bundle_system_boundary>"

    def _nearest_bundle_cylinder_clip(self, rays: RayBundle) -> tuple[Any, Any, Any, np.ndarray]:
        xp = rays.xp
        n = rays.n_rays
        clip_mask = xp.zeros(n, dtype=bool)
        clip_distance = xp.full(n, xp.inf, dtype=float)
        clip_points = rays.position.copy()
        clip_kind = xp.zeros(n, dtype=np.int64)

        if (
            self.bundle_clip_inner_radius_m is None
            and self.bundle_clip_outer_radius_m is None
            and self.bundle_clip_z_min_m is None
            and self.bundle_clip_z_max_m is None
        ):
            return clip_mask, clip_distance, clip_points, np.full(n, "", dtype=object)

        bundle_mask = xp.asarray(rays.bundle_reflected, dtype=bool)
        if count_nonzero(bundle_mask) == 0:
            return clip_mask, clip_distance, clip_points, np.full(n, "", dtype=object)

        pxy = rays.position[:, :2]
        dxy = rays.direction[:, :2]
        r0_sq = xp.sum(pxy * pxy, axis=1)
        r0 = xp.sqrt(r0_sq)
        radial_dot = xp.sum(pxy * dxy, axis=1)
        a = xp.sum(dxy * dxy, axis=1)
        eps = float(self.bundle_clip_epsilon_m)
        inf = xp.full(n, xp.inf, dtype=float)

        def update_candidate(candidate_mask: Any, candidate_t: Any, kind_value: int) -> None:
            nonlocal clip_mask, clip_distance, clip_kind
            better = candidate_mask & (candidate_t < clip_distance)
            if count_nonzero(better) == 0:
                return
            clip_mask = clip_mask | better
            clip_distance = xp.where(better, candidate_t, clip_distance)
            clip_kind = xp.where(better, kind_value, clip_kind)

        def consider_boundary(radius_m: Optional[float], is_inner: bool) -> None:
            if radius_m is None:
                return
            radius = float(radius_m)
            kind_value = 1 if is_inner else 2
            if is_inner:
                immediate = bundle_mask & (r0 <= radius + eps) & (radial_dot < 0.0)
            else:
                immediate = bundle_mask & (r0 >= radius - eps) & (radial_dot > 0.0)
            update_candidate(immediate, xp.zeros(n, dtype=float), kind_value)

            c = r0_sq - radius * radius
            disc = radial_dot * radial_dot - a * c
            valid_quad = bundle_mask & (a > 1e-20) & (disc >= 0.0)
            sqrt_disc = xp.sqrt(xp.maximum(disc, 0.0))
            for candidate_t in ((-radial_dot - sqrt_disc) / a, (-radial_dot + sqrt_disc) / a):
                candidate_t = xp.where(candidate_t > eps, candidate_t, inf)
                candidate_xy = pxy + candidate_t[:, None] * dxy
                candidate_radial_dot = xp.sum(candidate_xy * dxy, axis=1)
                if is_inner:
                    crossing = (r0 >= radius - eps) & (candidate_radial_dot < 0.0)
                else:
                    crossing = (r0 <= radius + eps) & (candidate_radial_dot > 0.0)
                update_candidate(valid_quad & crossing, candidate_t, kind_value)

        consider_boundary(self.bundle_clip_inner_radius_m, is_inner=True)
        consider_boundary(self.bundle_clip_outer_radius_m, is_inner=False)

        z0 = rays.position[:, 2]
        dz = rays.direction[:, 2]

        def consider_z_boundary(z_m: Optional[float], is_min: bool) -> None:
            if z_m is None:
                return
            z_boundary = float(z_m)
            kind_value = 3 if is_min else 4
            if is_min:
                immediate = bundle_mask & (z0 <= z_boundary + eps) & (dz < 0.0)
                crossing = z0 >= z_boundary - eps
            else:
                immediate = bundle_mask & (z0 >= z_boundary - eps) & (dz > 0.0)
                crossing = z0 <= z_boundary + eps
            update_candidate(immediate, xp.zeros(n, dtype=float), kind_value)

            valid_linear = bundle_mask & (xp.abs(dz) > 1e-20)
            candidate_t = (z_boundary - z0) / dz
            candidate_t = xp.where(candidate_t > eps, candidate_t, inf)
            update_candidate(valid_linear & crossing, candidate_t, kind_value)

        consider_z_boundary(self.bundle_clip_z_min_m, is_min=True)
        consider_z_boundary(self.bundle_clip_z_max_m, is_min=False)

        if count_nonzero(clip_mask) > 0:
            safe_distance = xp.where(clip_mask, clip_distance, 0.0)
            clip_points = rays.position + safe_distance[:, None] * rays.direction

        kind_host = to_numpy(clip_kind).astype(int)
        names = np.full(n, "", dtype=object)
        for kind_value in (1, 2, 3, 4):
            names[kind_host == kind_value] = self._bundle_clip_surface_name(kind_value)
        return clip_mask, clip_distance, clip_points, names

    def _append_bundle_clip_segments(
        self,
        segments: List[Dict[str, Any]],
        rays: RayBundle,
        clip_mask: Any,
        clip_distance: Any,
        clip_points: Any,
        clip_surface_names: np.ndarray,
    ) -> None:
        if not self.record_segments or count_nonzero(clip_mask) == 0:
            return
        clipped_rays = rays.subset(clip_mask)
        clip_time = self._distance_to_time(clip_distance[clip_mask], clipped_rays.refractive_index)
        mask_host = to_numpy(clip_mask).astype(bool)
        segments.append(
            self._make_segment_block(
                clipped_rays,
                clip_points[clip_mask],
                clip_surface_names[mask_host],
                clipped_rays.time_s + clip_time,
            )
        )

    def trace(self, rays: RayBundle) -> TraceResult:
        self.compiled_scene.intersection_tests = 0
        self.compiled_scene.possible_tests = 0
        xp = get_backend(self.backend)
        active = rays
        finished: List[RayBundle] = []
        segments: List[Dict[str, Any]] = []
        detector_hits: List[Dict[str, Any]] = []

        next_id = int(np.max(to_numpy(active.ray_id))) + 1 if active.n_rays > 0 else 0

        for depth in range(self.max_interactions):
            if active.n_rays == 0:
                break
            if self.max_active_rays is not None and active.n_rays > self.max_active_rays:
                order = np.argsort(-to_numpy(active.power))[: self.max_active_rays]
                active = active.subset(xp.asarray(order, dtype=np.int64))

            if self.max_time_s is not None:
                remaining_time = float(self.max_time_s) - active.time_s
                within_time_mask = remaining_time > 0.0
                if count_nonzero(~within_time_mask) > 0:
                    finished.append(active.subset(~within_time_mask))
                if count_nonzero(within_time_mask) == 0:
                    active = RayBundle.concat([], backend=self.backend)
                    break
                active = active.subset(within_time_mask)
                remaining_time = float(self.max_time_s) - active.time_s

            nearest = self.compiled_scene.find_nearest(
                active,
                collision_mode=self.collision_mode,
                skip_repeated_bundle_reflections=self.skip_repeated_bundle_reflections,
            )
            surface_hit_mask = nearest.surface_index >= 0
            hit_time = self._distance_to_time(nearest.distance, active.refractive_index)
            if self.max_time_s is not None:
                surface_hit_mask = surface_hit_mask & (hit_time <= remaining_time + 1e-15)

            clip_mask, clip_distance, clip_points, clip_surface_names = self._nearest_bundle_cylinder_clip(active)
            clip_time = self._distance_to_time(clip_distance, active.refractive_index)
            if self.max_time_s is not None:
                clip_mask = clip_mask & (clip_time <= remaining_time + 1e-15)
            clip_before_surface = clip_mask & (~surface_hit_mask | (clip_distance <= nearest.distance + self.surface_epsilon))
            surface_hit_mask = surface_hit_mask & ~clip_before_surface
            miss_mask = ~(surface_hit_mask | clip_before_surface)

            self._append_bundle_clip_segments(
                segments,
                active,
                clip_before_surface,
                clip_distance,
                clip_points,
                clip_surface_names,
            )

            if count_nonzero(miss_mask) > 0:
                missed = active.subset(miss_mask)
                if self.max_time_s is not None:
                    miss_remaining_time = float(self.max_time_s) - missed.time_s
                    miss_propagated = self._propagate_to_time_limit(missed, miss_remaining_time)
                    if self.record_segments:
                        surface_names = np.full(miss_propagated.n_rays, "<time_limit>", dtype=object)
                        segments.append(
                            self._make_segment_block(
                                missed,
                                miss_propagated.position,
                                surface_names,
                                miss_propagated.time_s,
                            )
                        )
                    finished.append(miss_propagated)
                else:
                    finished.append(missed)

            if count_nonzero(surface_hit_mask) == 0:
                active = RayBundle.concat([], backend=self.backend)
                break

            if self.record_segments:
                idx_host = to_numpy(nearest.surface_index[surface_hit_mask]).astype(int)
                hit_rays = active.subset(surface_hit_mask)
                hit_end_time = active.time_s[surface_hit_mask] + hit_time[surface_hit_mask]
                surface_names = np.asarray([self.compiled_scene.surfaces[i].name for i in idx_host], dtype=object)
                segments.append(
                    self._make_segment_block(
                        hit_rays,
                        nearest.points[surface_hit_mask],
                        surface_names,
                        hit_end_time,
                    )
                )

            children: List[RayBundle] = []
            for i, surface in enumerate(self.compiled_scene.surfaces):
                surf_mask = surface_hit_mask & (nearest.surface_index == i)
                if count_nonzero(surf_mask) == 0:
                    continue
                rays_sub = active.subset(surf_mask).propagate(nearest.points[surf_mask], hit_time[surf_mask])
                hit_sub = nearest.subset(surf_mask)
                new_children, new_hits = surface.interact(rays_sub, hit_sub, self)

                for record in new_hits:
                    detector_hits.append(record)

                for child in new_children:
                    if child.n_rays == 0:
                        continue
                    new_ids = xp.arange(next_id, next_id + child.n_rays, dtype=np.int64)
                    next_id += child.n_rays
                    child = child.replace_ids(ray_id=new_ids, parent_id=child.parent_id)
                    keep = child.power > self.min_power
                    if count_nonzero(keep) > 0:
                        children.append(child.subset(keep))

            active = RayBundle.concat(children, backend=self.backend)

        # An interaction cap is a computational stop, not free propagation.
        # Keep unfinished rays at the last traced point; otherwise a straight
        # segment could pass through an untested mirror before the time limit.
        unfinished_count = active.n_rays
        if unfinished_count:
            finished.append(active)

        final_rays = RayBundle.concat(finished, backend=self.backend)
        return TraceResult(
            final_rays=final_rays,
            segments=segments,
            detector_hits=detector_hits,
            surface_names=[s.name for s in self.compiled_scene.surfaces],
            diagnostics={
                "terminated_by_interaction_limit": unfinished_count > 0,
                "unfinished_count": unfinished_count,
                "collision_mode": self.collision_mode,
                "intersection_tests": self.compiled_scene.intersection_tests,
                "possible_tests": self.compiled_scene.possible_tests,
            },
        )
