"""Compare conservative broad-phase search plans on the full 20k-ray trace."""

from __future__ import annotations

from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

from benchmark_performance import ROOT, load_application, peak_working_set_bytes
import raytrace as rt


def group_bounds(compiled: rt.CompiledScene, indices: list[int]) -> rt.SurfaceGroupBounds:
    bounds = [compiled._plane_bounds(compiled.surfaces[index]) for index in indices]
    xp = rt.get_array_module(bounds[0][0])
    minimum = xp.min(xp.stack([item[0] for item in bounds], axis=0), axis=0)
    maximum = xp.max(xp.stack([item[1] for item in bounds], axis=0), axis=0)
    sphere_center = 0.5 * (minimum + maximum)
    sphere_radius = rt.norm(0.5 * (maximum - minimum)) + 1e-10
    return rt.SurfaceGroupBounds(
        tuple(indices), minimum, maximum, sphere_center, sphere_radius
    )


def component_plan(compiled: rt.CompiledScene):
    plan = []
    i = 0
    while i < len(compiled.surfaces):
        surface = compiled.surfaces[i]
        bundle_name = compiled._bundle_group_name(surface)
        if bundle_name is not None:
            key = bundle_name
        elif isinstance(surface, rt.PlaneSurface) and ":" in surface.name:
            key = surface.name.split(":", 1)[0]
        else:
            plan.append(i)
            i += 1
            continue
        indices = []
        while i < len(compiled.surfaces):
            candidate = compiled.surfaces[i]
            candidate_key = compiled._bundle_group_name(candidate)
            if candidate_key is None and isinstance(candidate, rt.PlaneSurface) and ":" in candidate.name:
                candidate_key = candidate.name.split(":", 1)[0]
            if candidate_key != key:
                break
            indices.append(i)
            i += 1
        plan.append(group_bounds(compiled, indices))
    return plan


def bundle_plan(compiled: rt.CompiledScene):
    plan = []
    i = 0
    while i < len(compiled.surfaces):
        key = compiled._bundle_group_name(compiled.surfaces[i])
        if key is None:
            plan.append(i)
            i += 1
            continue
        indices = []
        while i < len(compiled.surfaces) and compiled._bundle_group_name(compiled.surfaces[i]) == key:
            indices.append(i)
            i += 1
        plan.append(group_bounds(compiled, indices))
    return plan


def individual_plan(compiled: rt.CompiledScene):
    plan = []
    for index, surface in enumerate(compiled.surfaces):
        if isinstance(surface, rt.PlaneSurface) and surface.shape != "infinite":
            plan.append(group_bounds(compiled, [index]))
        else:
            plan.append(index)
    return plan


def sphere_candidates(rays, bounds, best_t):
    """Conservative sphere enclosing a group's already padded AABB."""
    xp = rays.xp
    minimum = xp.asarray(bounds.minimum, dtype=float)
    maximum = xp.asarray(bounds.maximum, dtype=float)
    center = 0.5 * (minimum + maximum)
    radius = xp.sqrt(xp.sum((0.5 * (maximum - minimum)) ** 2)) + 1e-10
    offset = rays.position - center[None, :]
    projection = rt.dot(rays.direction, offset)
    discriminant = projection * projection - (rt.dot(offset, offset) - radius * radius)
    sqrt_discriminant = xp.sqrt(xp.maximum(discriminant, 0.0))
    entry = -projection - sqrt_discriminant
    exit_ = -projection + sqrt_discriminant
    return (discriminant >= 0.0) & (exit_ > 0.0) & (entry <= best_t)


def run_variant(app, scene, rays, variant: str):
    tracer = rt.RayTracer(
        scene=scene,
        backend="numpy",
        max_interactions=app.MAX_SECONDARY_RAY_GENERATIONS,
        max_time_s=app.INTEGRATION_TIME_S,
        bundle_clip_inner_radius_m=app.BUNDLE_RAY_INNER_CYLINDER_RADIUS_M,
        bundle_clip_outer_radius_m=app.BUNDLE_RAY_OUTER_CYLINDER_RADIUS_M,
        bundle_clip_z_min_m=app.BUNDLE_RAY_Z_MIN_M,
        bundle_clip_z_max_m=app.BUNDLE_RAY_Z_MAX_M,
        use_broad_phase=variant != "exhaustive",
    )
    if variant == "bundle_groups":
        tracer.compiled_scene._search_plan = bundle_plan(tracer.compiled_scene)
    elif variant in {"components_aabb", "components_sphere"}:
        tracer.compiled_scene._search_plan = component_plan(tracer.compiled_scene)
    elif variant == "individual_aabb":
        tracer.compiled_scene._search_plan = individual_plan(tracer.compiled_scene)

    metrics = {"ray_surface_tests": 0, "intersect_calls": 0, "find_seconds": 0.0}

    def wrap_intersect(function):
        def wrapper(surface, selected_rays):
            metrics["intersect_calls"] += 1
            metrics["ray_surface_tests"] += selected_rays.n_rays
            return function(surface, selected_rays)
        return wrapper

    original_find = tracer.compiled_scene.find_nearest

    def find_wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_find(*args, **kwargs)
        finally:
            metrics["find_seconds"] += time.perf_counter() - started

    tracer.compiled_scene.find_nearest = find_wrapper
    with ExitStack() as stack:
        if variant in {"components_aabb", "individual_aabb"}:
            stack.enter_context(
                patch.object(
                    rt.CompiledScene,
                    "_ray_group_candidates",
                    staticmethod(rt.CompiledScene._ray_aabb_candidates),
                )
            )
        for cls in (rt.PlaneSurface, rt.SphericalCapSurface, rt.CylinderSurface):
            stack.enter_context(patch.object(cls, "intersect", wrap_intersect(cls.intersect)))
        started = time.perf_counter()
        result = tracer.trace(rays)
        metrics["trace_seconds"] = time.perf_counter() - started

    metrics.update(
        {
            "variant": variant,
            "search_plan_entries": len(tracer.compiled_scene._search_plan),
            "segment_rows": sum(len(block["ray_id"]) for block in result.segments),
            "detector_rows": sum(len(block["ray_id"]) for block in result.detector_hits),
            "detector_power": result.detector_power_summary(),
            "peak_working_set_bytes_so_far": peak_working_set_bytes(),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=("exhaustive", "bundle_groups", "components_aabb", "components_sphere", "individual_aabb"),
    )
    args = parser.parse_args()
    app, _ = load_application()
    rays = app.emit_initial_rays(app.build_initial_source("numpy"), target_count=20_000)
    scene = app.build_initial_scene()
    results = [run_variant(app, scene, rays, variant) for variant in args.variants]
    output = ROOT / "benchmark_broad_phase_variants.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
