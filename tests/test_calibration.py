import json
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from mirrorlab.calibration import analyze_measurements, fit_line, fit_rigid_to_lines, read_lavsan, read_survey
from mirrorlab.calibration.registration import fit_rigid_points

ROOT = Path(__file__).resolve().parents[1]


class GeometryFitTests(unittest.TestCase):
    def test_line_orientation_and_orthogonal_residual(self):
        fit = fit_line([[1, 2, 3], [1, 2, 7]])
        np.testing.assert_allclose(fit.point, [1, 2, 3])
        np.testing.assert_allclose(fit.direction, [0, 0, 1])
        self.assertAlmostEqual(fit.rms, 0)
        noisy = fit_line([[0, 0.001, 0], [1, -.001, 0], [2, .001, 0]])
        self.assertLess(noisy.rms, .001)
        with self.assertRaises(ValueError):
            fit_line([[1, 1, 1], [1, 1, 1]])

    def test_recovers_known_rigid_transform_from_lines(self):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(30, 3))
        rotation = Rotation.from_euler("xyz", [.2, -.13, .6]).as_matrix()
        translation = np.array([1.2, -.3, 2.7])
        moved = points @ rotation.T + translation
        directions = rng.normal(size=(30, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        origins = moved + directions * rng.uniform(-1, 1, size=(30, 1))
        result = fit_rigid_to_lines(points, origins, directions, starts=4)
        self.assertTrue(result["success"])
        np.testing.assert_allclose(result["rotation"], rotation, atol=1e-7)
        np.testing.assert_allclose(result["translation_m"], translation, atol=1e-7)
        self.assertLess(result["rms_mm"], 1e-5)
        self.assertAlmostEqual(np.linalg.det(result["rotation"]), 1)

    def test_underdetermined_registration_rejected(self):
        points = np.arange(18).reshape(6, 3)
        with self.assertRaises(ValueError):
            fit_rigid_to_lines(points, points, np.tile([0, 0, 1], (6, 1)))
        with self.assertRaises(ValueError):
            fit_rigid_points(points, points)


class SurveyTests(unittest.TestCase):
    def test_saved_equations_reproduction_and_missing_data(self):
        report = analyze_measurements()
        self.assertEqual(report["counts"], dict(complete=85, single_screen=17, missing=10, total=112))
        audit = report["saved_equations_comparison"]
        self.assertEqual(audit["matched"], 111)
        self.assertEqual([row["id"] for row in audit["differences"]], ["4_1_3"])
        first = report["rays"][0]
        np.testing.assert_allclose(first["coefficients_mm"], [433.70, -473.87, -365.07, 17.52, -491.44, 15.31])
        self.assertIsNone(next(ray for ray in report["rays"] if ray["id"] == "1_1_5")["direction"])
        json.dumps(report, allow_nan=False)

    def test_numeric_cells_with_date_formats_are_not_dates(self):
        records, warnings = read_survey(ROOT / "Геодезические измерения из+к-.xlsx")
        by_id = {record.id: record for record in records}
        self.assertAlmostEqual(by_id["1_4_1"].screens[1][0], .00212, places=10)
        self.assertAlmostEqual(by_id["3_3_2"].screens[1][0], .00719, places=10)
        self.assertEqual(len(warnings), 2)
        self.assertEqual(len(records), 112)

    def test_known_scene_registration_is_recovered_from_tables(self):
        registered = analyze_measurements()["scene_registration"]
        self.assertTrue(registered["transform"]["accepted"])
        self.assertLess(registered["transform"]["max_mm"], .001)
        self.assertFalse(registered["accepted"])
        self.assertEqual(registered["rays"], [])
        self.assertEqual(len(registered["candidate_rays"]), 85)
        self.assertGreater(registered["compatibility"]["max_line_distance_mm"], 60)
        np.testing.assert_allclose(registered["transform"]["translation_m"], [.00159, .00879, 3.44722], atol=1e-7)

    def test_lavsan_keeps_inconsistent_window_data_and_flags_it(self):
        points = read_lavsan(ROOT / "лавсан_из+к-.txt")
        self.assertAlmostEqual(points[3, 3, 1][2], 1.83535)
        report = analyze_measurements("east_to_west", fit_lavsan=True)
        self.assertEqual(report["counts"]["complete"], 83)
        lavsan = report["lavsan"]
        self.assertEqual(lavsan["status"], "partial")
        third = next(rod for rod in lavsan["rods"] if rod["rod"] == 3)
        self.assertFalse(third["accepted"])
        self.assertGreater(third["rms_mm"], 100)
        self.assertTrue(all(not ray["id"].startswith("3_") for ray in lavsan["refined_rays"]))
        self.assertIsNone(report["scene_registration"])
        json.dumps(report, allow_nan=False)

    def test_direction_validation_prevents_cross_half_lavsan(self):
        with self.assertRaises(ValueError):
            analyze_measurements("west_to_east", fit_lavsan=True)
        with self.assertRaises(ValueError):
            analyze_measurements("east")


if __name__ == "__main__":
    unittest.main()
