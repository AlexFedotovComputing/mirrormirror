import json
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from mirrorlab.engine import RayBundle, RayTracer, Scene
from mirrorlab.engine.bodies import MirrorBody
from mirrorlab.configurations import load_configuration, _bundle_surfaces, StaticElement
from mirrorlab.calibration.physical import (
    body_pair_clearance, incoming_at_mirrors, install_bodies, reflecting_normal,
    relax_body_rolls, relax_line_centers, incoming_at_points,
    rod_containment, rotation_from_normal, validate_bodies, validate_reflections, waist_side,
)


def rays(points, directions):
    return RayBundle(np.array(points, float), np.array(directions, float),
                     np.ones(len(points)), np.full(len(points), 266e-9), np.ones(len(points)))


class PhysicalBodyTests(unittest.TestCase):
    def setUp(self):
        self.body = MirrorBody([0, 0, 0], np.eye(3), "1_1_1")

    def test_exact_support_and_cut_ellipse(self):
        for v in ([0, 0, 1], [0, 0, -1], [1, 2, -3], [3, -2, 1], [0, 0, 0]):
            v = np.asarray(v)
            expected = (.0005 * np.hypot(v[0], v[1]) + .0095 * v[2] if v[2] >= 0
                        else .0005 * np.hypot(v[0] + v[2], v[1]))
            self.assertAlmostEqual(self.body.support(v), expected, places=14)
            self.assertTrue(self.body.contains(self.body.support_point(v), 1e-12))
        cut = self.body.build_surfaces()[0]
        hit = cut.intersect(rays([[.00049, 0, -.01], [0, .0006, -.01]], [[0, 0, 1]] * 2))
        self.assertTrue(hit.valid[0])  # ellipse's long semi-axis = sqrt(2)*r
        self.assertFalse(hit.valid[1])  # a circular sqrt(2)*r aperture would accept this
        self.assertAlmostEqual(np.dot(self.body.axis, self.body.normal), -np.sqrt(.5))
        self.assertAlmostEqual(self.body.axial_height_m + self.body.radius_m, .01)

    def test_so3_roll_and_serialized_axis_normal_consistency(self):
        normal = reflecting_normal([.12, -.04, 1.], [-.7, .5, .12])
        rotation = rotation_from_normal(normal, [0, 0, -1.])
        body = MirrorBody([0, 0, 0], rotation)
        np.testing.assert_allclose(body.normal, normal, atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        np.testing.assert_allclose(MirrorBody.from_dict(body.to_dict()).rotation, rotation)
        invalid = body.to_dict(); invalid["normal"] = [1, 0, 0]
        with self.assertRaises(ValueError):
            MirrorBody.from_dict(invalid)
        with self.assertRaises(ValueError):
            MirrorBody([0, 0, 0], np.diag([1, 1, -1]))
        with self.assertRaises(ValueError):
            reflecting_normal([0, 0, 1], [0, 0, 1])

    def test_only_cut_reflects_and_side_back_absorb(self):
        surfaces = self.body.build_surfaces()
        scene = Scene([StaticElement("body", surfaces)])
        source = rays([[0, 0, -.002], [.002, 0, .005], [0, 0, .012]],
                      [[0, 0, 1], [-1, 0, 0], [0, 0, -1]])
        trace = RayTracer(scene, max_interactions=3, max_time_s=1e-9).trace(source)
        first = trace.segments[0]
        self.assertEqual(list(first["surface"]), ["BUNDLE_1_1 Mirror 1", "BUNDLE_1_1 Mirror 1:side", "BUNDLE_1_1 Mirror 1:back"])
        children = [int(p) for b in trace.segments[1:] for p in b["parent_id"]]
        self.assertIn(0, children)
        self.assertNotIn(1, children)
        self.assertNotIn(2, children)
        block = next(b for b in trace.segments[1:] if 0 in b["parent_id"])
        i = list(block["parent_id"]).index(0)
        outgoing = np.array([block[a + "1"][i] - block[a + "0"][i] for a in "xyz"])
        outgoing /= np.linalg.norm(outgoing)
        np.testing.assert_allclose(outgoing, [1, 0, 0], atol=1e-12)

    def test_separation_touching_overlap_and_physical_gap_are_distinct(self):
        expected = ((.002, "separated"), (.001, "touching_within_tolerance"), (.0008, "intersecting"))
        for x, status in expected:
            report = body_pair_clearance(self.body, MirrorBody([x, 0, 0], np.eye(3), "b"))
            self.assertEqual(report["status"], status)
            if status == "intersecting":
                self.assertGreater(report["common_ball_radius_m"], 0)
        touching = MirrorBody([.001, 0, 0], np.eye(3), "b")
        self.assertFalse(body_pair_clearance(self.body, touching, required_gap_m=1e-5)["physical_gap_satisfied"])
        near = MirrorBody([.001000001, 0, 0], np.eye(3), "b")
        self.assertEqual(body_pair_clearance(self.body, near)["status"], "touching_within_tolerance")

    def test_repeated_reflection_flag_never_makes_physical_bodies_transparent(self):
        blocker = MirrorBody([.003, 0, -.004], np.eye(3), "1_2_1")
        scene = Scene([StaticElement("body", self.body.build_surfaces()),
                       StaticElement("blocker", blocker.build_surfaces())])
        trace = RayTracer(scene, max_interactions=4, max_time_s=1e-9, collision_mode="expected",
                          skip_repeated_bundle_reflections=True).trace(rays([[0, 0, -.002]], [[0, 0, 1]]))
        self.assertEqual(trace.segments[0]["surface"][0], "BUNDLE_1_1 Mirror 1")
        self.assertEqual(trace.segments[1]["surface"][0], "BUNDLE_1_2 Mirror 1:side")
        self.assertEqual(len(trace.segments), 2)

    def test_long_cylinders_can_collide_far_from_cut_centres(self):
        other = MirrorBody([-.005, 0, .005], Rotation.from_euler("y", 90, degrees=True).as_matrix(), "1_2_1")
        self.assertGreater(np.linalg.norm(other.center_m - self.body.center_m), .005)
        self.assertEqual(body_pair_clearance(self.body, other)["status"], "intersecting")
        report = validate_bodies([self.body, other], check_length_alternative=False)
        self.assertEqual(report["status"], "infeasible")
        self.assertEqual(report["conflicts"][0]["pair"], ["1_1_1", "1_2_1"])

    def test_capsule_overlap_does_not_reject_separated_skew_bodies(self):
        other = MirrorBody([-.0008, 0, .0095], np.eye(3), "b")
        # Capsule axis endpoints are only .8 mm apart, less than 2*r.
        self.assertLess(np.linalg.norm(self.body.axis * .0095 - other.center_m), .001)
        report = body_pair_clearance(self.body, other)
        self.assertEqual(report["status"], "separated")
        self.assertGreater(report["gap_lower_bound_m"], .0002)
        u = np.array(report["separating_direction"])
        self.assertGreaterEqual(-other.support(-u) - self.body.support(u), report["gap_lower_bound_m"] - 1e-12)

    def test_each_body_gets_its_own_nearest_gap_not_global_minimum(self):
        bodies = [self.body, MirrorBody([.001, 0, 0], np.eye(3), "1_1_2"),
                  MirrorBody([.004, 0, 0], np.eye(3), "1_2_1")]
        report = validate_bodies(bodies, check_length_alternative=False)
        self.assertEqual(report["minimum_gap_m"], 0.)
        self.assertAlmostEqual(report["per_body_minimum_gap_m"]["1_2_1"], .002, places=12)
        self.assertEqual(report["per_body_nearest_pair"]["1_2_1"], ["1_1_2", "1_2_1"])
        self.assertLessEqual(report["per_body_clearance_interval_m"]["1_2_1"][0], .002)

    def test_roll_relaxation_preserves_observations_and_requires_certified_clearance(self):
        crossing = MirrorBody([-.005, 0, .005], Rotation.from_euler("y", 90, degrees=True).as_matrix(), "1_1_2")
        originals = [self.body, crossing]
        relaxed, report = relax_body_rolls(originals, required_gap_m=2e-5, starts=2)
        self.assertEqual(report["status"], "body_clearance_verified")
        self.assertGreater(body_pair_clearance(*relaxed)["gap_lower_bound_m"], 2e-5)
        for before, after in zip(originals, relaxed):
            np.testing.assert_array_equal(before.center_m, after.center_m)
            np.testing.assert_allclose(before.normal, after.normal, atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(after.rotation), 1.)
        # Identical cut centres and normals make these opaque volumes overlap
        # for any roll; an optimizer termination must not claim feasibility.
        duplicate = MirrorBody([0, 0, 0], np.eye(3), "1_1_2")
        _, failed = relax_body_rolls([self.body, duplicate], max_iterations=8, starts=1)
        self.assertEqual(failed["status"], "no_global_feasible_solution")
        self.assertTrue(failed["final_geometry"]["conflicts"])

    def test_entire_body_rod_containment_and_waist(self):
        region = {"origin_m": [0, 0, 0], "axis": [0, 0, 1], "radius_m": .005,
                  "axial_min_m": -.001, "axial_max_m": .005}
        self.assertEqual(rod_containment(self.body, region)["status"], "outside")
        self.assertEqual(rod_containment(self.body, None)["status"], "not_evaluated")
        self.assertEqual(waist_side([0, 0, 0], [1, 0, 0])["status"], "not_evaluated")
        self.assertEqual(waist_side([0, 0, 0], [1, 0, 0], [-1, 0, 0])["status"], "wrong_side")

    def test_mesh_and_intersections_share_same_solid(self):
        rotation = Rotation.from_rotvec([.3, -.2, .7]).as_matrix()
        body = MirrorBody([.2, 1., -.5], rotation)
        for p in body.mesh(48)["vertices_m"]:
            self.assertTrue(body.contains(p, 1e-12))
        cut_indices = set(i for triangle in body.mesh(48)["cut_faces"] for i in triangle)
        for i in cut_indices:
            self.assertTrue(body.on_cut(body.mesh(48)["vertices_m"][i]))


class IncidentFieldTests(unittest.TestCase):
    def test_along_line_relaxation_respects_source_and_preserves_measured_line(self):
        config = load_configuration("east_to_west")
        key, surface = next(iter(_bundle_surfaces(config)))
        din = np.array(incoming_at_mirrors(config)[key]["direction"])
        dout = din - 2 * np.dot(din, surface.normal) * surface.normal
        body = MirrorBody(surface.center + .015 * dout, rotation_from_normal(surface.normal), key)
        self.assertEqual(incoming_at_points(config, {key: body.center_m})[key]["status"], "outside_illumination")
        relaxed, report = relax_line_centers(config, [body], {key: dout},
                                             {"axial_center_sigma_m": .01, "axial_center_limit_sigma": 3., "physical_starts": 2})
        shift = relaxed[0].center_m - body.center_m
        self.assertEqual(report["status"], "body_clearance_verified")
        self.assertGreater(np.linalg.norm(shift), .005)
        self.assertLessEqual(np.linalg.norm(shift), .03)
        np.testing.assert_allclose(np.cross(shift, dout), [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(relaxed[0].normal, body.normal, atol=1e-12)
        self.assertEqual(incoming_at_points(config, {key: relaxed[0].center_m})[key]["status"], "available")
        self.assertEqual(validate_reflections(config, relaxed)["validated_count"], 1)

    def test_full_optics_field_detectably_differs_from_constant_z(self):
        config = load_configuration("east_to_west")
        field = incoming_at_mirrors(config)
        self.assertEqual(len(field), 112)
        for value in field.values():
            self.assertEqual(value["status"], "available")
            self.assertLess(value["full_trace_miss_m"], 1e-9)
            self.assertGreater(np.linalg.norm(np.array(value["direction"]) - [0, 0, -1]), 1e-5)
            self.assertIn("MP1", value["interaction_chain"])
            self.assertTrue(any(name.startswith("MC") for name in value["interaction_chain"]))
        self.assertIn("PR3:side_1", field["4_1_1"]["interaction_chain"])

    def test_historical_26_coating_errors_repaired_with_real_daughters(self):
        config = load_configuration("west_to_east", "real")
        self.assertEqual(len(config.metadata["historical_coating_repairs"]), 26)
        bodies = [MirrorBody(s.center, rotation_from_normal(s.normal), key) for key, s in _bundle_surfaces(config)]
        report = validate_reflections(config, bodies, physical_bodies=False)
        self.assertEqual(len(report["mirrors"]), 112)
        for mirror in report["mirrors"].values():
            if "actual_reflected_child" not in mirror:
                self.assertEqual(mirror["reason"], "Another physical surface is hit before the requested cut")
                continue
            self.assertTrue(mirror["actual_reflected_child"])
            self.assertLess(mirror["incident_dot_normal"], 0)
            self.assertGreater(mirror["outgoing_dot_normal"], 0)
            self.assertTrue(mirror["hit_on_ellipse"])
        # Historical planes can occlude outgoing rays; do not hide this by
        # skipping later bundle intersections to get an artificial pass.
        self.assertTrue(any(v.get("outgoing_blockers") for v in report["mirrors"].values()))
        field = incoming_at_mirrors(config)
        for key, surface in _bundle_surfaces(config):
            d = np.array(field[key]["direction"])
            trace = RayTracer(Scene([StaticElement("test", [surface])]), max_interactions=2,
                              max_time_s=1e-9).trace(rays([surface.center - 1e-4*d], [d]))
            self.assertEqual(len(trace.segments), 2, key)
            self.assertEqual(trace.segments[1]["parent_id"][0], 0, key)

    def test_targeted_physical_cut_and_opaque_body_use_current_configuration(self):
        config = load_configuration("east_to_west")
        key, surface = next(iter(_bundle_surfaces(config)))
        body = MirrorBody(surface.center + .0002 * surface.normal, rotation_from_normal(surface.normal), key)
        old_count = len(config.scene.build_surfaces())
        old_center = surface.center.copy()
        report = validate_reflections(config, [body])["mirrors"][key]
        self.assertEqual(report["status"], "validated")
        self.assertTrue(report["actual_reflected_child"])
        np.testing.assert_allclose(report["hit_m"], body.center_m, atol=1e-9)
        self.assertEqual(len(config.scene.build_surfaces()), old_count)
        np.testing.assert_array_equal(surface.center, old_center)


if __name__ == "__main__":
    unittest.main()
