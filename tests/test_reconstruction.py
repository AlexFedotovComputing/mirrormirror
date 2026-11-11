"""Independent failure cases for empirical reconstruction and provenance."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from mirrorlab.calibration.empirical import (angular_distance, assign_fan, gauge_template,
                                              learn_template, rigid_rotation, unit)
from mirrorlab.calibration.observations import (exclude_sources, observation_registry,
                                                read_lasertest_template)
from mirrorlab.calibration.reconstruction import (_options, fit_assembly, group_observations,
                                                  masked_line_residual)
from mirrorlab.calibration.validation import screen_errors, validate_withheld

ROOT = Path(__file__).resolve().parents[1]


def synthetic_groups(count=8, seed=12):
    rng = np.random.default_rng(seed)
    angles = np.array([-.58, -.39, -.19, .005, .21, .40, .61])
    model = np.c_[np.cos(angles), np.sin(angles), np.zeros(7)]
    groups = {}
    for b in range(count):
        keep = np.delete(np.arange(7), b % 7)
        rot = Rotation.from_euler("xyz", [rng.normal(0, .03), rng.normal(0, .03), rng.uniform(-3, 3)]).as_matrix()
        dirs = unit(model[keep] @ rot.T + rng.normal(0, .0003, (6, 3)))
        groups[f"{b//4+1}_{b%4+1}"] = dict(directions=dirs.tolist())
    return model, groups


def synthetic_registry(count=4):
    angles = np.linspace(-.6, .6, 7)
    dirs = np.c_[np.cos(angles), np.sin(angles), np.zeros(7)]
    records = []
    for b in range(count):
        center = np.array([0., .03*b, .1*b])
        rotation = Rotation.from_euler("yz", [.02, .12*b]).as_matrix()
        for j in range(7):
            direction = rotation @ dirs[j]
            identifier = f"synthetic:{b+1}:1:{j+1}"
            records.append(dict(id=identifier, independent_source_id=identifier,
                                old_id=f"{b+1}_1_{j+1}", rod=b+1, window=1, ray=j+1,
                                screens_m=[(center + .5*direction).tolist(), (center+direction).tolist()],
                                complete_points=2))
    return dict(records=records)


class ObservationLedgerTests(unittest.TestCase):
    def test_original_counts_masks_and_provenance(self):
        for direction, expected in [("west_to_east", (85, 17, 10)), ("east_to_west", (83, 18, 11))]:
            ledger = observation_registry(ROOT, direction)
            self.assertEqual(112, ledger["counts"]["total"])
            self.assertEqual(expected, tuple(ledger["counts"][name] for name in ("complete", "single_screen", "missing")))
            for row in ledger["records"]:
                self.assertEqual(64, len(row["source_sha256"]))
                self.assertTrue(row["sheet"])
                self.assertEqual([[v is None for v in p] for p in row["screens_m"]], row["missing_mask"])
                self.assertIsNone(row["metrology_sigma_m"])

    def test_conflict_and_date_cells_are_preserved(self):
        west = observation_registry(ROOT, "west_to_east")
        conflict = next(x for x in west["conflicts"] if x["old_id"] == "4_1_3")
        row = next(x for x in west["records"] if x["old_id"] == "4_1_3")
        self.assertEqual("primary_only", conflict["decision"])
        self.assertIsNone(row["screens_m"][1][0])
        self.assertIsNone(row["screens_m"][1][1])
        east = observation_registry(ROOT, "east_to_west")
        cells = {entry["cell"]: row for row in east["records"] for entry in row["transformations"] if isinstance(entry, dict)}
        self.assertAlmostEqual(.00212, cells["F24"]["screens_m"][1][0])
        self.assertAlmostEqual(.00719, cells["F80"]["screens_m"][1][0])
        rod = next(x for x in east["lavsan"]["rods"] if x["rod"] == 3)
        self.assertIn([3, 4], rod["repeated_levels"])
        self.assertEqual("excluded_inconsistent_levels", rod["status"])

    def test_only_known_lasertest_zero_placeholders_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_coords"
            path.write_text("Окно 1\n7\t0\t0\t0\nОкно 2\n7\t0\t0\t0\n1\t0\t2\t3\n", encoding="utf-8")
            result = read_lasertest_template(path)
            self.assertEqual([0., 0., 0.], result[0]["coordinates_m"])
            self.assertEqual([None]*3, result[1]["coordinates_m"])
            self.assertEqual(0., result[2]["coordinates_m"][0])

    def test_hidden_source_filter_removes_derived_copies(self):
        records = [dict(id="primary", independent_source_id="a"),
                   dict(id="copy", independent_source_id="b", derived_from=["a"]),
                   dict(id="other", independent_source_id="c")]
        self.assertEqual(["other"], [r["id"] for r in exclude_sources(dict(records=records), {"a"})])


class EmpiricalTemplateTests(unittest.TestCase):
    def test_partial_training_and_geometric_permutation_invariance(self):
        truth, groups = synthetic_groups()
        first = learn_template(groups, iterations=4, ambiguity_rad=.003)
        shuffled = copy.deepcopy(groups)
        for group in shuffled.values():
            group["directions"] = list(reversed(group["directions"]))
        second = learn_template(shuffled, iterations=4, ambiguity_rad=.003)
        self.assertEqual(0, first["complete_assemblies"])
        np.testing.assert_allclose(first["directions_local"], second["directions_local"], atol=1e-9)
        self.assertLess(np.max(angular_distance(first["directions_local"], gauge_template(truth))), .006)
        self.assertTrue(all(s["independent_assemblies"] >= 4 for s in first["statistics"]))

    def test_symmetric_missing_end_has_separate_assignment_modes(self):
        angle = np.linspace(-.6, .6, 7)
        template = np.c_[np.cos(angle), np.sin(angle), np.zeros(7)]
        fit = assign_fan(template, template[1:5], ambiguity_rad=.001)
        self.assertTrue(fit["ambiguous"])
        self.assertGreaterEqual(fit["alternative_count"], 3)
        self.assertFalse(fit["cost_is_probability"])

    def test_predictive_scatter_does_not_shrink_with_sample_size(self):
        _, groups = synthetic_groups()
        small = learn_template(groups, iterations=3, ambiguity_rad=.003)
        enlarged = {f"{rep*2+int(k.split('_')[0])}_{k.split('_')[1]}": copy.deepcopy(g)
                    for rep in range(4) for k, g in groups.items()}
        large = learn_template(enlarged, iterations=3, ambiguity_rad=.003)
        for a, b in zip(small["statistics"], large["statistics"]):
            self.assertGreater(b["observed_predictive_sigma_rad"], .85 * a["observed_predictive_sigma_rad"])
            self.assertLess(b["mean_sigma_rad"], a["mean_sigma_rad"])

    def test_shared_translation_and_data_rank(self):
        ledger = synthetic_registry()
        groups = group_observations(ledger["records"])
        template = learn_template({k:g for k,g in groups.items() if k != "1_1"}, iterations=2, ambiguity_rad=.003)
        target = copy.deepcopy(groups["1_1"])
        for row in target["records"]:
            row["screens_m"] = (np.asarray(row["screens_m"]) + [0., .004, -.003]).tolist()
        target = group_observations(target["records"])["1_1"]
        result = fit_assembly(template, target, np.zeros((7,3)), dict(rotation=np.eye(3),translation_m=[0,0,0]), _options(None))
        self.assertGreater(result["rigid_translation_m"][1], .0035)
        self.assertLess(np.max(np.linalg.norm(result["template_deformations_m"], axis=1)), .0003)
        self.assertLess(result["data_jacobian_rank"], result["full_parameter_count"])

    def test_partial_components_not_filled_and_real_xy_section(self):
        residual = masked_line_residual(np.array([0.,0.,0.]), unit([1.,1.,1.]), [[1.,None,1.]])
        np.testing.assert_allclose(residual, 0., atol=1e-12)
        self.assertEqual(2, len(residual))
        report = screen_errors(np.array([0.,0.,0.]), unit([1.,0.,1.]), [[2.,0.,2.]])
        self.assertAlmostEqual(0., report[0]["distance_m"])
        self.assertEqual([2.,0.,2.], report[0]["predicted_point_m"])

    def test_shared_rotation_is_not_stored_as_individual_deformation(self):
        rng = np.random.default_rng(93)
        angles = np.array([-.60, -.39, -.18, .02, .21, .42, .64])
        template_d = gauge_template(np.c_[np.cos(angles), np.sin(angles), np.zeros(7)])
        deformed = unit(template_d + rng.normal(0., .004, (7, 3)))
        # Define the synthetic local deformation by removing its best common
        # rotation independently (Wahba), then add a known large assembly pose.
        deformed = deformed @ rigid_rotation(template_d, deformed)
        true_rotation = Rotation.from_rotvec([.18, -.11, .24]).as_matrix()
        directions = deformed @ true_rotation.T
        records = []
        for j, direction in enumerate(directions):
            distance = .25 + .12*j
            records.append(dict(id=f"rotation:{j}", old_id=f"1_1_{j+1}", rod=1, window=1, ray=j+1,
                                complete_points=2, screens_m=[(distance*direction).tolist(), ((distance+1)*direction).tolist()]))
        model = dict(directions_local=template_d.tolist(), statistics=[dict(observed_predictive_sigma_rad=s)
                     for s in [.004, .006, .005, .008, .004, .007, .005]])
        result = fit_assembly(model, group_observations(records)["1_1"], np.zeros((7, 3)),
                              dict(rotation=np.eye(3), translation_m=[0., 0., 0.]),
                              _options(dict(measurement_sigma_m=1e-5, individual_center_sigma_m=1e-5)))
        common_error = Rotation.from_matrix(result["rotation"] @ true_rotation.T).magnitude()
        recovered_local = result["directions"] @ result["rotation"]
        self.assertLess(common_error, 2e-4)
        self.assertLess(np.max(angular_distance(recovered_local, deformed)), 2e-4)
        self.assertGreater(np.linalg.norm(result["template_deformations_rad"]), .001)
        self.assertEqual(11, result["angular_gauge"]["individual_parameters"])
        self.assertEqual(38, result["full_parameter_count"])
        np.testing.assert_allclose(result["angular_gauge"]["common_mode_residual_rad"], 0., atol=1e-12)

    def test_withheld_predictions_cannot_read_historical_geometry(self):
        class ForbiddenGeometry:
            def __getattribute__(self, name):
                raise AssertionError("Independent holdout accessed derived historical geometry")
        ledger = synthetic_registry(3)
        report = validate_withheld(ForbiddenGeometry(), ledger, _options(dict(template_iterations=1, max_nfev=20)))
        self.assertEqual(3, report["predicted"])
        for fold in report["folds"]:
            self.assertNotIn(fold["hidden_observation_id"], fold["target_visible_observation_ids"])
            self.assertFalse(set(fold["training_observation_ids"]) & set(fold["excluded_independent_sources"]))
            self.assertEqual("not_required_raw_survey_frame", fold["coordinate_registration"])
            self.assertEqual("not_evaluated_without_independent_body_layout", fold["physical_body_validation"])


if __name__ == "__main__":
    unittest.main()
