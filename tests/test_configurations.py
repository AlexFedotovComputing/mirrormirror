import dataclasses
import json
import unittest

import numpy as np

from mirrorlab.configurations import (
    apply_fitted_rays, configuration_ray_references, load_configuration,
)
from mirrorlab.engine import RayTracer


class ConfigurationTests(unittest.TestCase):
    def test_all_four_snapshots_and_collision_modes(self):
        for direction in ("west_to_east", "east_to_west"):
            for kind in ("ideal", "real"):
                with self.subTest(direction=direction, kind=kind):
                    config = load_configuration(direction, kind)
                    self.assertEqual(len(configuration_ray_references(config)), 112)
                    self.assertEqual(len(config.scene.build_surfaces()), 200)
                    rays = config.emit(217)
                    results = []
                    for mode in ("exhaustive", "expected"):
                        tracer = RayTracer(config.scene, collision_mode=mode,
                                           max_interactions=20, max_time_s=config.integration_time_s,
                                           **config.tracer_options)
                        results.append(tracer.trace(rays))
                    full, selected = results
                    self.assertEqual(len(full.segments), len(selected.segments))
                    for first, second in zip(full.segments, selected.segments):
                        np.testing.assert_array_equal(first["surface"], second["surface"])
                        for field in first:
                            np.testing.assert_array_equal(first[field], second[field])
                    self.assertEqual(len(full.detector_hits), len(selected.detector_hits))
                    for first, second in zip(full.detector_hits, selected.detector_hits):
                        for field in first:
                            np.testing.assert_array_equal(first[field], second[field])
                    for field in ("position", "direction", "power", "polarization", "ray_id", "parent_id", "depth", "time_s"):
                        np.testing.assert_array_equal(getattr(full.final_rays, field), getattr(selected.final_rays, field))
                    self.assertLess(selected.diagnostics["intersection_tests"], full.diagnostics["intersection_tests"])

    def test_real_inherits_all_common_optics_from_ideal(self):
        def snapshot(value):
            return json.dumps(dataclasses.asdict(value), default=lambda x: x.tolist(), sort_keys=True)

        for direction in ("west_to_east", "east_to_west"):
            ideal, real = [load_configuration(direction, kind) for kind in ("ideal", "real")]
            with self.subTest(direction=direction):
                common = lambda c: [snapshot(e) for e in c.scene.elements if not e.name.startswith("BUNDLE_")]
                self.assertEqual(common(ideal), common(real))
                self.assertEqual(snapshot(ideal.source), snapshot(real.source))
                self.assertEqual(ideal.tracer_options, real.tracer_options)
                self.assertEqual(ideal.integration_time_s, real.integration_time_s)
                self.assertEqual(ideal.max_interactions, real.max_interactions)
                # Configuration instances are independently mutable for future alignment work.
                real.scene.elements[0].surfaces[0].center[0] += 1
                self.assertNotEqual(common(ideal), common(real))

    def test_world_frame_and_ring_controls_opposite_half(self):
        for direction, source_sign, injection in (("west_to_east", -1, "EAST"), ("east_to_west", 1, "WEST")):
            config = load_configuration(direction)
            self.assertGreater(source_sign * config.source.waist_position[2], 0)
            self.assertEqual(config.metadata["injection_side"], injection)
            for reference in configuration_ray_references(config).values():
                self.assertLess(source_sign * reference["point_m"][2], 0)

    def test_ideal_micro_mirrors_turn_rays_into_both_halves(self):
        for direction in ("west_to_east", "east_to_west"):
            config = load_configuration(direction)
            surfaces = {s.name: s for s in config.scene.build_surfaces()}
            for mode in ("exhaustive", "expected"):
                with self.subTest(direction=direction, mode=mode):
                    result = RayTracer(config.scene, collision_mode=mode, max_interactions=48,
                                       max_time_s=config.integration_time_s,
                                       **config.tracer_options).trace(config.emit(721))
                    first_hits = {}
                    for block in result.segments:
                        for i, name in enumerate(block["surface"]):
                            if not name.startswith("BUNDLE_") or block["bundle_reflected"][i]:
                                continue
                            incoming = np.array([block[a + "1"][i] - block[a + "0"][i] for a in "xyz"])
                            incoming /= np.linalg.norm(incoming)
                            normal = surfaces[name].normal
                            first_hits[block["ray_id"][i]] = (name, incoming - 2 * np.dot(incoming, normal) * normal)
                    self.assertEqual(len({name for name, _ in first_hits.values()}), 112)
                    children = 0
                    for block in result.segments:
                        for i, parent in enumerate(block["parent_id"]):
                            if parent not in first_hits:
                                continue
                            start = np.array([block[a + "0"][i] for a in "xyz"])
                            outgoing = np.array([block[a + "1"][i] for a in "xyz"]) - start
                            outgoing /= np.linalg.norm(outgoing)
                            np.testing.assert_allclose(outgoing, first_hits[parent][1], atol=1e-10)
                            self.assertLess(np.dot(start[:2], outgoing[:2]), 0)
                            self.assertTrue(block["bundle_reflected"][i])
                            children += 1
                    self.assertEqual(children, len(first_hits))
                    self.assertEqual(result.diagnostics["unfinished_count"], 0)

    def test_sampling_is_reproducible_and_preserves_grid_power(self):
        config = load_configuration()
        for count in (1, 19, 100, 217):
            first, second = config.emit(count), config.emit(count)
            self.assertEqual(first.n_rays, count)
            np.testing.assert_array_equal(first.position, second.position)
            self.assertTrue(np.isfinite(first.power).all())
            self.assertGreater(float(first.power.sum()), 0)
        # 100 is sampled from a 127-point grid; sampling changes no total power.
        self.assertAlmostEqual(float(config.emit(100).power.sum()), float(config.emit(127).power.sum()), places=7)
        for invalid in (0, 50001, 1.5, True):
            with self.assertRaises(ValueError):
                config.emit(invalid)

    def test_fitted_axes_update_normals_and_bound_translation(self):
        config = load_configuration(kind="real")
        reference = configuration_ray_references(config)["1_1_1"]
        old_center = np.asarray(reference["point_m"])
        target = {"ray_id": "1_1_1", "point_m": (old_center + [.01, 0, 0]).tolist(), "direction": [1., 0., 0.]}
        report = apply_fitted_rays(config, [target])
        self.assertEqual(report["updated_count"], 1)
        self.assertLessEqual(report["mirrors"][0]["applied_shift_m"], .002)
        updated = configuration_ray_references(config)["1_1_1"]
        np.testing.assert_allclose(updated["direction"], target["direction"], atol=1e-12)
        fresh = configuration_ray_references(load_configuration(kind="real"))["1_1_1"]
        np.testing.assert_allclose(fresh["point_m"], old_center)

    def test_unregistered_unknown_ray_cannot_mutate_geometry(self):
        config = load_configuration(kind="real")
        with self.assertRaises(ValueError):
            apply_fitted_rays(config, [{"ray_id": "99_1_1", "point_m": [0, 0, 0], "direction": [1, 0, 0]}])


if __name__ == "__main__":
    unittest.main()
