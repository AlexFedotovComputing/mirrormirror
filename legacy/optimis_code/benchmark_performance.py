"""Low-overhead, reproducible performance instrumentation for the TPC ray tracer.

This file intentionally does not change any physical or visualization parameter.
It loads the production entry point, wraps selected functions, and invokes its
normal ``main`` function with explicit command-line arguments.
"""

from __future__ import annotations

import argparse
import cProfile
import ctypes
from ctypes import wintypes
import importlib.util
import json
import os
from pathlib import Path
import platform
import pstats
import sys
import time
from typing import Any, Callable

import numpy as np

import raytrace as rt


ROOT = Path(__file__).resolve().parent
APP_PATH = ROOT / "из+к-_оптимизация.py"


def load_application() -> tuple[Any, float]:
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("tpc_calibration_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, time.perf_counter() - started


def peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        return None

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    handle = get_current_process()
    ok = get_process_memory_info(
        handle, ctypes.byref(counters), counters.cb
    )
    return int(counters.PeakWorkingSetSize) if ok else None


class Instrumentation:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.timings: dict[str, dict[str, float | int]] = {}
        self.active_rays: list[int] = []
        self.produced_rays: list[int] = []
        self.surface_counts: list[int] = []
        self.broad_group_counts: list[int] = []
        self.result: Any = None
        self.subset_calls = 0
        self.subset_input_rays = 0
        self.subset_output_rays = 0
        self.concat_calls = 0
        self.concat_input_chunks = 0
        self.to_numpy_calls = 0
        self.to_numpy_input_bytes = 0
        self.intersection_calls = 0
        self.intersection_ray_tests = 0
        self.bundle_intersection_ray_tests = 0

    def add_timing(self, name: str, elapsed: float) -> None:
        row = self.timings.setdefault(name, {"seconds": 0.0, "calls": 0})
        row["seconds"] = float(row["seconds"]) + elapsed
        row["calls"] = int(row["calls"]) + 1

    def timed(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.add_timing(name, time.perf_counter() - started)

        wrapper.__name__ = getattr(function, "__name__", name)
        wrapper.__doc__ = getattr(function, "__doc__", None)
        return wrapper

    def install(self) -> None:
        app = self.app
        for name, label in (
            ("build_initial_source", "source_object"),
            ("emit_initial_rays", "ray_generation"),
            ("build_initial_scene", "scene_build"),
            ("write_plotly_trajectories", "plotly_html"),
            ("_add_mirror_angle_controls", "html_controls"),
            ("detector_energy_summary", "result_summaries"),
        ):
            setattr(app, name, self.timed(label, getattr(app, name)))

        original_write_csv = app.write_csv

        def write_csv_wrapper(path: Path, rows: Any) -> bool:
            label = "csv_segments" if path.name == "segments.csv" else "csv_detector_hits"
            started = time.perf_counter()
            try:
                return original_write_csv(path, rows)
            finally:
                elapsed = time.perf_counter() - started
                self.add_timing(label, elapsed)
                self.add_timing("csv_write", elapsed)

        app.write_csv = write_csv_wrapper

        for name in (
            "make_circle_outline",
            "make_cylindrical_surface_overlays",
            "make_disk_overlays",
            "make_rectangle_outline",
            "make_rectangular_prism_overlays",
            "make_triangular_prism_overlays",
            "build_bundle_overlays",
        ):
            setattr(app, name, self.timed("overlay_geometry", getattr(app, name)))

        original_compile = rt.Scene.compile

        def compile_wrapper(
            scene: Any,
            backend: str = "numpy",
            *,
            use_broad_phase: bool = True,
        ) -> Any:
            started = time.perf_counter()
            try:
                return original_compile(
                    scene,
                    backend,
                    use_broad_phase=use_broad_phase,
                )
            finally:
                self.add_timing("geometry_compile", time.perf_counter() - started)

        rt.Scene.compile = compile_wrapper

        original_find = rt.CompiledScene.find_nearest

        def find_wrapper(scene: Any, rays: Any, **kwargs: Any) -> Any:
            self.active_rays.append(int(rays.n_rays))
            self.produced_rays.append(0)
            self.surface_counts.append(len(scene.surfaces))
            self.broad_group_counts.append(sum(not isinstance(entry, int) for entry in scene._search_plan))
            started = time.perf_counter()
            try:
                return original_find(scene, rays, **kwargs)
            finally:
                self.add_timing("nearest_intersections", time.perf_counter() - started)

        rt.CompiledScene.find_nearest = find_wrapper

        for surface_class in (rt.PlaneSurface, rt.SphericalCapSurface, rt.CylinderSurface):
            original_intersect = surface_class.intersect

            def make_intersect_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
                def intersect_wrapper(surface: Any, rays: Any) -> Any:
                    self.intersection_calls += 1
                    self.intersection_ray_tests += rays.n_rays
                    if surface.name.startswith("BUNDLE_"):
                        self.bundle_intersection_ray_tests += rays.n_rays
                    return function(surface, rays)

                return intersect_wrapper

            surface_class.intersect = make_intersect_wrapper(original_intersect)

        original_interact = rt.Surface.interact

        def interact_wrapper(surface: Any, rays: Any, hit: Any, tracer: Any) -> Any:
            started = time.perf_counter()
            try:
                children, records = original_interact(surface, rays, hit, tracer)
                if self.produced_rays:
                    self.produced_rays[-1] += sum(child.n_rays for child in children)
                return children, records
            finally:
                self.add_timing("surface_interactions", time.perf_counter() - started)

        rt.Surface.interact = interact_wrapper

        original_trace = rt.RayTracer.trace

        def trace_wrapper(tracer: Any, rays: Any) -> Any:
            started = time.perf_counter()
            try:
                self.result = original_trace(tracer, rays)
                return self.result
            finally:
                self.add_timing("trace_total", time.perf_counter() - started)

        rt.RayTracer.trace = trace_wrapper

        original_segment_block = rt.RayTracer._make_segment_block

        def segment_block_wrapper(tracer: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original_segment_block(tracer, *args, **kwargs)
            finally:
                self.add_timing("segment_collection", time.perf_counter() - started)

        rt.RayTracer._make_segment_block = segment_block_wrapper

        original_clip = rt.RayTracer._nearest_bundle_cylinder_clip
        rt.RayTracer._nearest_bundle_cylinder_clip = self.timed(
            "bundle_boundary_checks", original_clip
        )

        original_summary = rt.TraceResult.detector_power_summary
        rt.TraceResult.detector_power_summary = self.timed(
            "result_summaries", original_summary
        )

        original_subset = rt.RayBundle.subset

        def subset_wrapper(bundle: Any, mask: Any) -> Any:
            self.subset_calls += 1
            self.subset_input_rays += bundle.n_rays
            started = time.perf_counter()
            try:
                out = original_subset(bundle, mask)
                self.subset_output_rays += out.n_rays
                return out
            finally:
                self.add_timing("raybundle_subset", time.perf_counter() - started)

        rt.RayBundle.subset = subset_wrapper

        original_concat = rt.RayBundle.concat.__func__

        def concat_wrapper(cls: Any, chunks: Any, backend: str | None = None) -> Any:
            self.concat_calls += 1
            self.concat_input_chunks += len(chunks)
            started = time.perf_counter()
            try:
                return original_concat(cls, chunks, backend)
            finally:
                self.add_timing("raybundle_concat", time.perf_counter() - started)

        rt.RayBundle.concat = classmethod(concat_wrapper)

        original_to_numpy = rt.to_numpy

        def to_numpy_wrapper(value: Any) -> np.ndarray:
            self.to_numpy_calls += 1
            self.to_numpy_input_bytes += int(getattr(value, "nbytes", 0))
            return original_to_numpy(value)

        rt.to_numpy = to_numpy_wrapper
        app.to_numpy = to_numpy_wrapper

    def report(self, import_seconds: float, total_seconds: float, mode: str) -> dict[str, Any]:
        result = self.result
        segment_rows = 0
        detector_rows = 0
        if result is not None:
            segment_rows = sum(len(block["ray_id"]) for block in result.segments)
            detector_rows = sum(len(block["ray_id"]) for block in result.detector_hits)

        exhaustive_checks = sum(
            active * surfaces
            for active, surfaces in zip(self.active_rays, self.surface_counts)
        )
        peak = peak_working_set_bytes()
        return {
            "mode": mode,
            "parameters": {
                "backend": "numpy",
                "initial_rays": int(self.app.INITIAL_RAY_COUNT),
                "max_interactions": int(self.app.MAX_SECONDARY_RAY_GENERATIONS),
                "max_time_s": float(self.app.INTEGRATION_TIME_S),
                "beam_radial_positions": int(self.app.BEAM_RADIAL_POSITIONS),
                "cutoff_ratio": float(self.app.BEAM_CUTOFF_RATIO),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpus": os.cpu_count(),
                "numpy": np.__version__,
            },
            "import_seconds": import_seconds,
            "total_seconds": total_seconds,
            "peak_working_set_bytes": peak,
            "timings": self.timings,
            "trace_metrics": {
                "compiled_surfaces": self.surface_counts[0] if self.surface_counts else 0,
                "iterations": len(self.active_rays),
                "active_rays_by_iteration": self.active_rays,
                "produced_rays_by_iteration": self.produced_rays,
                "ray_surface_intersection_tests": self.intersection_ray_tests,
                "exhaustive_ray_surface_intersection_tests": exhaustive_checks,
                "bundle_ray_surface_intersection_tests": self.bundle_intersection_ray_tests,
                "broad_phase_bound_ray_tests": sum(
                    active * groups
                    for active, groups in zip(self.active_rays, self.broad_group_counts)
                ),
                "final_rays": result.final_rays.n_rays if result is not None else 0,
                "segment_blocks": len(result.segments) if result is not None else 0,
                "segment_rows": segment_rows,
                "detector_blocks": len(result.detector_hits) if result is not None else 0,
                "detector_rows": detector_rows,
                "subset_calls": self.subset_calls,
                "subset_input_rays": self.subset_input_rays,
                "subset_output_rays": self.subset_output_rays,
                "concat_calls": self.concat_calls,
                "concat_input_chunks": self.concat_input_chunks,
                "to_numpy_calls": self.to_numpy_calls,
                "to_numpy_input_bytes": self.to_numpy_input_bytes,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("full", "interactive", "trace"), default="full"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results.json")
    args = parser.parse_args()

    app, import_seconds = load_application()
    instrumentation = Instrumentation(app)
    instrumentation.install()

    output_dir = ROOT / f"benchmark_output_{args.mode}"
    app_args = [str(APP_PATH), "--backend", "numpy", "--outdir", str(output_dir)]
    if args.mode == "interactive":
        app_args.extend(("--no-open-plot", "--skip-csv"))
    elif args.mode == "trace":
        app_args.extend(("--no-plot", "--skip-csv"))
    else:
        app_args.append("--no-open-plot")

    profiler = cProfile.Profile() if args.profile else None
    old_argv = sys.argv
    started = time.perf_counter()
    try:
        sys.argv = app_args
        if profiler is not None:
            profiler.enable()
        app.main()
    finally:
        if profiler is not None:
            profiler.disable()
        total_seconds = time.perf_counter() - started
        sys.argv = old_argv

    report = instrumentation.report(import_seconds, total_seconds, args.mode)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if profiler is not None:
        profile_path = args.output.with_suffix(".prof")
        text_path = args.output.with_suffix(".profile.txt")
        profiler.dump_stats(profile_path)
        with text_path.open("w", encoding="utf-8") as stream:
            stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
            stats.print_stats(80)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
