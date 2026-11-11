"""Physics and candidate-filter regressions for the preserved optical engine."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

from mirrorlab.engine import (
    CylinderSurface, Detector, GaussianBeamSource, InteractionMode, PlaneMirror,
    PlaneSurface, RayTracer, Scene, SemiTransparentMirror, SphericalCapSurface,
    SurfaceOptics,
)


def source(axis=(0., 0., 1.)):
    return GaussianBeamSource((0., 0., 0.), axis, .01, 1e6, 266e-9, radial_positions=2)


def assert_same_trace(test, left, right):
    test.assertEqual(len(left.segments), len(right.segments))
    test.assertEqual(len(left.detector_hits), len(right.detector_hits))
    for first, second in zip(left.segments, right.segments):
        for key in first:
            if key == "surface":
                np.testing.assert_array_equal(first[key], second[key])
            else:
                np.testing.assert_allclose(first[key], second[key], rtol=1e-11, atol=1e-11)
    test.assertEqual(left.detector_power_summary().keys(), right.detector_power_summary().keys())
    for key, value in left.detector_power_summary().items():
        test.assertAlmostEqual(value, right.detector_power_summary()[key], places=9)


class EngineTests(unittest.TestCase):
    def test_split_power_and_reflection_match_legacy_engine(self):
        path = Path(__file__).resolve().parents[1] / "legacy" / "raytrace.py"
        spec = importlib.util.spec_from_file_location("legacy_physics_baseline", path)
        legacy = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = legacy
        spec.loader.exec_module(legacy)

        def run(module):
            scene = module.Scene().add(
                module.SemiTransparentMirror("split", (0, 0, 1), (0, 0, 1),
                                             shape="disk", radius=.1, reflectance=.3, transmittance=.7),
                module.Detector("transmitted", (0, 0, 2), (0, 0, 1), shape="disk", radius=.1),
                module.Detector("reflected", (0, 0, -1), (0, 0, 1), shape="disk", radius=.1),
            )
            return module.RayTracer(scene, max_interactions=8).trace(source().emit())

        import mirrorlab.engine as current
        expected, actual = run(legacy), run(current)
        assert_same_trace(self, expected, actual)
        total = sum(actual.detector_power_summary().values())
        self.assertAlmostEqual(total, float(source().emit().power.sum()), places=9)
        self.assertAlmostEqual(actual.detector_power_summary()["reflected"] / total, .3)

    def test_candidate_bounds_preserve_off_axis_hits_and_misses(self):
        optics = SurfaceOptics(InteractionMode.ABSORB)
        surfaces = [
            PlaneSurface("cluster:disk", optics, (0, 0, 1), (0, 0, 1), shape="disk", radius=.2),
            PlaneSurface("cluster:rectangle", optics, (2, 0, 1), (0, 0, 1), width=.3, height=.5),
            PlaneSurface("cluster:triangle", optics, (-2, 0, 1), (0, 0, 1), shape="triangle",
                         triangle_vertices=[[-.2, -.2], [.2, -.2], [0, .2]]),
            CylinderSurface("cylinder", optics, (0, 2, 1), (0, 0, 1), .2, 1),
            SphericalCapSurface("sphere", optics, (0, -2, 1), (0, 0, 1), .3, .2),
            PlaneSurface("infinite", optics, (0, 0, 5), (0, 0, 1), shape="infinite"),
        ]
        from mirrorlab.engine import CompiledScene
        compiled = CompiledScene(surfaces)
        rays = source().emit()
        rng = np.random.default_rng(981)
        for _ in range(20):
            rays.position = rng.uniform(-3, 3, rays.position.shape)
            rays.direction = rng.normal(size=rays.direction.shape)
            rays.direction /= np.linalg.norm(rays.direction, axis=1)[:, None]
            full = compiled.find_nearest(rays, collision_mode="exhaustive")
            filtered = compiled.find_nearest(rays, collision_mode="expected")
            np.testing.assert_array_equal(full.surface_index, filtered.surface_index)
            np.testing.assert_allclose(full.distance, filtered.distance)
        self.assertLess(compiled.intersection_tests, compiled.possible_tests)

    def test_bounds_include_tangencies_and_parallel_rays(self):
        from mirrorlab.configurations import load_configuration
        from mirrorlab.engine import RayBundle
        compiled = load_configuration().scene.compile()
        positions, directions = [], []
        for group in compiled._search_plan:
            if isinstance(group, int):
                continue
            minimum, maximum = np.asarray(group.minimum), np.asarray(group.maximum)
            center = (minimum + maximum) / 2
            positions.extend((minimum, maximum, center, [minimum[0], center[1], center[2]]))
            directions.extend(((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 1, 0)))
        count = len(positions)
        rays = RayBundle(np.asarray(positions), np.asarray(directions), np.ones(count),
                         np.full(count, 266e-9), np.ones(count))
        first = compiled.find_nearest(rays, collision_mode="exhaustive")
        second = compiled.find_nearest(rays, collision_mode="expected")
        for name in ("surface_index", "distance", "points", "normals", "local_u", "local_v"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    def test_exhaustive_ignores_old_bundle_route_shortcut(self):
        scene = Scene().add(PlaneMirror("BUNDLE_1_1 Mirror 1", (0, 0, 1), (0, 0, 1), shape="disk", radius=.1))
        rays = source().emit().with_bundle_reflected()
        hit = scene.compile().find_nearest(rays, collision_mode="exhaustive", skip_repeated_bundle_reflections=True)
        self.assertTrue(np.all(hit.surface_index == 0))

    def test_interaction_cap_stops_at_last_intersection(self):
        scene = Scene().add(
            PlaneMirror("turn", (0, 0, 1), (0, 0, 1), shape="disk", radius=.1),
            Detector("next", (0, 0, -1), (0, 0, 1), shape="disk", radius=.1),
        )
        result = RayTracer(scene, max_interactions=1, max_time_s=1e-6).trace(source().emit())
        self.assertTrue(result.diagnostics["terminated_by_interaction_limit"])
        self.assertEqual(result.diagnostics["unfinished_count"], 19)
        self.assertEqual(len(result.segments), 1)
        np.testing.assert_allclose(result.final_rays.position[:, 2], 1., atol=2e-7)
        self.assertEqual(result.detector_hits, [])

    def test_invalid_search_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            RayTracer(Scene(), collision_mode="typo")

    def test_only_reflected_bundle_branches_are_clipped(self):
        for mode in ("exhaustive", "expected"):
            for coated_side in (True, False):
                with self.subTest(mode=mode, coated_side=coated_side):
                    scene = Scene().add(
                        SemiTransparentMirror("BUNDLE_1_1 Mirror 1", (0, 0, 1), (0, 0, 1),
                                              shape="disk", radius=.1, reflectance=.3, transmittance=.7,
                                              reflect_from_minus_side=coated_side),
                        Detector("transmitted", (0, 0, 2), (0, 0, 1), shape="disk", radius=.1),
                        Detector("reflected", (0, 0, -1), (0, 0, 1), shape="disk", radius=.1),
                    )
                    rays = source().emit()
                    result = RayTracer(scene, max_interactions=4, collision_mode=mode,
                                       bundle_clip_z_max_m=1.5).trace(rays)
                    power = result.detector_power_summary()
                    total = float(rays.power.sum())
                    self.assertAlmostEqual(sum(power.values()), total)
                    self.assertAlmostEqual(power["transmitted"] / total, .7 if coated_side else 1.)
                    self.assertAlmostEqual(power.get("reflected", 0.) / total, .3 if coated_side else 0.)
                    for block in result.segments:
                        transmitted = block["surface"] == "transmitted"
                        reflected = block["surface"] == "reflected"
                        self.assertFalse(block["bundle_reflected"][transmitted].any())
                        self.assertTrue(block["bundle_reflected"][reflected].all())


if __name__ == "__main__":
    unittest.main()
