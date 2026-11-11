import unittest

import numpy as np

from mirrorlab.configurations import configuration_ray_references, load_configuration
from mirrorlab.coordinates import intersect_ray_group, survey_frames
from mirrorlab.engine import RayTracer
from mirrorlab.plotting import PlotlyRenderer


class CoordinateTests(unittest.TestCase):
    def test_recover_intersection_and_reject_parallel_rays(self):
        point = np.array([.3, -1.2, .7])
        directions = np.array([[1, 1, .01], [-1, 1, .02], [0, 1, -.03]])
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        origins = point + directions * np.array([.5, 1, 2])[:, None]
        actual, rms = intersect_ray_group(origins, directions)
        np.testing.assert_allclose(actual, point, atol=1e-12)
        self.assertLess(rms, 1e-12)
        with self.assertRaises(ValueError):
            intersect_ray_group(origins, np.tile([0, 1, 0], (3, 1)))

    def test_both_volumes_share_survey_axis_and_opposite_rings(self):
        frames = survey_frames()
        for direction, injection_sign in (("west_to_east", -1), ("east_to_west", 1)):
            config = load_configuration(direction)
            frame = frames[direction]
            with self.subTest(direction=direction):
                np.testing.assert_allclose(frame.rotation.T @ frame.rotation, np.eye(3), atol=1e-14)
                self.assertEqual(frame.report["status"], "estimated")
                self.assertEqual(frame.report["survey_window_to_legacy_window"], {"1": 4, "2": 3, "3": 2, "4": 1})
                self.assertEqual(len(frame.report["groups"]), 16)
                source = frame.points(config.source.waist_position)
                self.assertEqual(frame.report["survey_z_signs"],
                                 dict(laser=-injection_sign, ring=-injection_sign, injection=injection_sign))
                self.assertLess(injection_sign * source[2], 0)
                self.assertAlmostEqual(frame.report["ring_center_m"][2], -injection_sign * 1.7905)
                for reference in configuration_ray_references(config).values():
                    self.assertGreater(injection_sign * frame.points(reference["point_m"])[2], 0)
                limits = [config.tracer_options[f"bundle_clip_z_{end}_m"] for end in ("min", "max")]
                extent = frame.points([[0, 0, z] for z in limits])
                np.testing.assert_allclose(sorted(extent[:, 2]), [-1.665, 1.665], atol=1e-12)
                np.testing.assert_allclose(extent.mean(0), frame.report["gas_axis_point_m"], atol=1e-12)
                # Window numbering is reconciled explicitly, with finite residuals
                # retained rather than claiming an exact fit to measured optics.
                errors = [g["residual_mm"] for g in frame.report["groups"]]
                self.assertTrue(np.isfinite(errors).all())
                self.assertAlmostEqual(frame.report["max_mm"], max(errors))
                self.assertGreater(frame.report["rms_mm"], 2.)
        np.testing.assert_array_equal(frames["west_to_east"].report["gas_axis_point_m"],
                                      frames["east_to_west"].report["gas_axis_point_m"])

    def test_every_plotted_model_point_transforms_but_survey_points_stay_original(self):
        for direction, frame in survey_frames().items():
            config = load_configuration(direction)
            result = RayTracer(config.scene, max_interactions=4,
                               max_time_s=config.integration_time_s, **config.tracer_options).trace(config.emit(19))
            renderer = PlotlyRenderer()
            native, _ = renderer.render(config, result)
            survey, _ = renderer.render(config, result, frame)
            with self.subTest(direction=direction):
                for original, moved in zip(native["data"][:-1], survey["data"][:-2]):
                    first = np.array([original[a] for a in "xyz"], float).T
                    second = np.array([moved[a] for a in "xyz"], float).T
                    mask = np.isfinite(first).all(axis=1)
                    np.testing.assert_allclose(second[mask], frame.points(first[mask]), atol=1e-12)
                    np.testing.assert_array_equal(np.isnan(first), np.isnan(second))
                measured = survey["data"][-2]
                points = np.array([p for row in frame.measurements for p in row.screens
                                   if all(v is not None for v in p)])
                np.testing.assert_array_equal(np.array([measured[a] for a in "xyz"]).T, points)
                self.assertEqual(measured["name"], "Геодезические точки")
                self.assertEqual(survey["data"][-1]["z"], [0] * 4)


if __name__ == "__main__":
    unittest.main()
