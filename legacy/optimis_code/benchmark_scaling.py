"""Diagnostic scaling measurements; production physics code is not modified."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time

from benchmark_performance import ROOT, load_application
from raytrace import CompiledScene, RayTracer


def main() -> None:
    app, _ = load_application()
    scene = app.build_initial_scene()

    ray_scaling = []
    for target_count in (5_000, 10_000, 20_000):
        source = app.build_initial_source("numpy")
        rays = app.emit_initial_rays(source, target_count=target_count)
        tracer = RayTracer(
            scene=scene,
            backend="numpy",
            max_interactions=app.MAX_SECONDARY_RAY_GENERATIONS,
            max_time_s=app.INTEGRATION_TIME_S,
            bundle_clip_inner_radius_m=app.BUNDLE_RAY_INNER_CYLINDER_RADIUS_M,
            bundle_clip_outer_radius_m=app.BUNDLE_RAY_OUTER_CYLINDER_RADIUS_M,
            bundle_clip_z_min_m=app.BUNDLE_RAY_Z_MIN_M,
            bundle_clip_z_max_m=app.BUNDLE_RAY_Z_MAX_M,
        )
        started = time.perf_counter()
        result = tracer.trace(rays)
        elapsed = time.perf_counter() - started
        ray_scaling.append(
            {
                "initial_rays": target_count,
                "trace_seconds": elapsed,
                "segment_rows": sum(len(block["ray_id"]) for block in result.segments),
                "detector_rows": sum(len(block["ray_id"]) for block in result.detector_hits),
            }
        )
        del result, tracer, rays, source
        gc.collect()

    source = app.build_initial_source("numpy")
    rays = app.emit_initial_rays(source, target_count=20_000)
    compiled = scene.compile("numpy")
    surface_scaling = []
    for surface_count in (46, 92, 138, len(compiled.surfaces)):
        subset_scene = CompiledScene(compiled.surfaces[:surface_count])
        samples = []
        for _ in range(3):
            started = time.perf_counter()
            subset_scene.find_nearest(rays)
            samples.append(time.perf_counter() - started)
        surface_scaling.append(
            {
                "rays": rays.n_rays,
                "surfaces": surface_count,
                "seconds_samples": samples,
                "seconds_min": min(samples),
                "ray_surface_tests": rays.n_rays * surface_count,
            }
        )

    report = {
        "ray_scaling_full_trace": ray_scaling,
        "surface_scaling_single_find_nearest": surface_scaling,
    }
    path = ROOT / "benchmark_scaling.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
