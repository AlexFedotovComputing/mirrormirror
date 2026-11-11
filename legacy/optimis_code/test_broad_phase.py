"""Differential regression tests for exhaustive and broad-phase ray tracing."""

from __future__ import annotations

import unittest

import numpy as np

from benchmark_performance import load_application
from raytrace import GaussianBeamSource, RayBundle, RayTracer


RAY_FIELDS = (
    "position",
    "direction",
    "power",
    "wavelength_m",
    "refractive_index",
    "polarization",
    "intensity",
    "time_s",
    "ray_id",
    "parent_id",
    "depth",
    "bundle_reflected",
)


def make_tracer(app, scene, *, use_broad_phase: bool) -> RayTracer:
    return RayTracer(
        scene=scene,
        backend="numpy",
        max_interactions=app.MAX_SECONDARY_RAY_GENERATIONS,
        max_time_s=app.INTEGRATION_TIME_S,
        bundle_clip_inner_radius_m=app.BUNDLE_RAY_INNER_CYLINDER_RADIUS_M,
        bundle_clip_outer_radius_m=app.BUNDLE_RAY_OUTER_CYLINDER_RADIUS_M,
        bundle_clip_z_min_m=app.BUNDLE_RAY_Z_MIN_M,
        bundle_clip_z_max_m=app.BUNDLE_RAY_Z_MAX_M,
        use_broad_phase=use_broad_phase,
    )


class BroadPhaseRegressionTests(unittest.TestCase):
    def assert_array_identical(self, left, right, message: str) -> None:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        self.assertEqual(left_array.shape, right_array.shape, message)
        if left_array.dtype.kind in "OUS" or right_array.dtype.kind in "OUS":
            np.testing.assert_array_equal(left_array, right_array, err_msg=message)
        else:
            # The broad phase only selects candidate rows. The retained narrow
            # phase performs the same operations, so bitwise equality is expected.
            np.testing.assert_array_equal(left_array, right_array, err_msg=message)

    def assert_ray_bundle_identical(self, left, right, prefix: str) -> None:
        self.assertEqual(left.n_rays, right.n_rays, prefix)
        for field in RAY_FIELDS:
            self.assert_array_identical(
                getattr(left, field),
                getattr(right, field),
                f"{prefix}.{field}",
            )

    def assert_block_lists_identical(self, left, right, prefix: str) -> None:
        self.assertEqual(len(left), len(right), f"{prefix}.block_count")
        for block_index, (left_block, right_block) in enumerate(zip(left, right)):
            self.assertEqual(
                tuple(left_block.keys()),
                tuple(right_block.keys()),
                f"{prefix}[{block_index}].keys",
            )
            for key in left_block:
                self.assert_array_identical(
                    left_block[key],
                    right_block[key],
                    f"{prefix}[{block_index}].{key}",
                )

    def assert_results_identical(self, exhaustive, optimized, prefix: str) -> None:
        self.assertEqual(exhaustive.surface_names, optimized.surface_names, prefix)
        self.assert_ray_bundle_identical(
            exhaustive.final_rays,
            optimized.final_rays,
            f"{prefix}.final_rays",
        )
        self.assert_block_lists_identical(
            exhaustive.segments,
            optimized.segments,
            f"{prefix}.segments",
        )
        self.assert_block_lists_identical(
            exhaustive.detector_hits,
            optimized.detector_hits,
            f"{prefix}.detector_hits",
        )
        self.assertEqual(
            exhaustive.detector_power_summary(),
            optimized.detector_power_summary(),
            f"{prefix}.detector_power_summary",
        )

    def trace_pair(self, app, rays):
        scene = app.build_initial_scene()
        exhaustive = make_tracer(app, scene, use_broad_phase=False).trace(rays)
        optimized = make_tracer(app, scene, use_broad_phase=True).trace(rays)
        return exhaustive, optimized

    def test_find_nearest_random_and_boundary_rays(self) -> None:
        app, _ = load_application()
        compiled = app.build_initial_scene().compile("numpy", use_broad_phase=True)
        rng = np.random.default_rng(20260924)
        positions = rng.uniform((-1.2, -1.2, -0.1), (1.2, 1.2, 0.2), size=(10_000, 3))
        directions = rng.normal(size=(10_000, 3))

        # Exact AABB-boundary and axis-parallel cases for every bundle group.
        groups = [entry for entry in compiled._search_plan if not isinstance(entry, int)]
        boundary_positions = []
        boundary_directions = []
        for group in groups:
            minimum = np.asarray(group.minimum)
            maximum = np.asarray(group.maximum)
            center = 0.5 * (minimum + maximum)
            boundary_positions.extend((minimum, maximum, center, (minimum[0], center[1], center[2])))
            boundary_directions.extend(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
        positions = np.concatenate((positions, np.asarray(boundary_positions)), axis=0)
        directions = np.concatenate((directions, np.asarray(boundary_directions)), axis=0)
        count = len(positions)
        rays = RayBundle(
            position=positions,
            direction=directions,
            power=np.ones(count),
            wavelength_m=np.full(count, 532e-9),
            refractive_index=np.ones(count),
        )

        for skip_repeated in (False, True):
            if skip_repeated:
                rays.bundle_reflected[::2] = True
            exhaustive = compiled.find_nearest_exhaustive(
                rays,
                skip_repeated_bundle_reflections=skip_repeated,
            )
            optimized = compiled.find_nearest_broad_phase(
                rays,
                skip_repeated_bundle_reflections=skip_repeated,
            )
            for field in ("surface_index", "distance", "points", "normals", "local_u", "local_v"):
                self.assert_array_identical(
                    getattr(exhaustive, field),
                    getattr(optimized, field),
                    f"nearest.{field}.skip={skip_repeated}",
                )

    def test_full_20000_ray_trace_is_bitwise_identical(self) -> None:
        app, _ = load_application()
        rays = app.emit_initial_rays(app.build_initial_source("numpy"), target_count=20_000)
        exhaustive, optimized = self.trace_pair(app, rays)
        self.assert_results_identical(exhaustive, optimized, "full_20000")

    def test_each_controlled_mirror_angle(self) -> None:
        # Full-size correctness is covered above. These eight scenarios exercise
        # changed paths through every controlled mirror at lower diagnostic ray
        # count so the regression suite remains practical.
        for mirror_index in range(8):
            with self.subTest(mirror_index=mirror_index):
                app, _ = load_application()
                angles = app._empty_controlled_mirror_angles()
                mirror = app.CONTROLLED_MIRRORS[mirror_index][1]
                angles[mirror.name][("x", "y", "z")[mirror_index % 3]] = 0.005
                app._apply_controlled_mirror_angles(angles)
                rays = app.emit_initial_rays(app.build_initial_source("numpy"), target_count=2_000)
                exhaustive, optimized = self.trace_pair(app, rays)
                self.assert_results_identical(
                    exhaustive,
                    optimized,
                    f"controlled_mirror_{mirror_index}",
                )

    def test_full_20000_all_controlled_mirrors_changed(self) -> None:
        app, _ = load_application()
        angles = app._empty_controlled_mirror_angles()
        axes = ("x", "y", "z")
        for mirror_index, (_, mirror) in enumerate(app.CONTROLLED_MIRRORS):
            angles[mirror.name][axes[mirror_index % 3]] = 0.005
        app._apply_controlled_mirror_angles(angles)
        rays = app.emit_initial_rays(app.build_initial_source("numpy"), target_count=20_000)
        exhaustive, optimized = self.trace_pair(app, rays)
        self.assert_results_identical(
            exhaustive,
            optimized,
            "full_20000_all_controlled_mirrors_changed",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
