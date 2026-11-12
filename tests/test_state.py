import copy
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from mirrorlab.legacy_import import import_legacy
from mirrorlab.service import _historical_document, dispatch
from mirrorlab.state import (configuration_from_dataset, digest, load_document,
                             save_document, validate_document)

ROOT = Path(__file__).resolve().parents[1]


class PhysicalStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = _historical_document(dict(direction="west_to_east", kind="real"))

    def test_round_trip_retains_physical_state_and_traces_without_repository(self):
        doc = copy.deepcopy(self.document)
        mirror = doc["datasets"]["west_to_east"]["mirrors"][0]
        mirror["center_m"][0] += .0001
        mirror["uncertainty"]["test_interval_m"] = [.001, .005]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "physical.json"
            save_document(doc, path)
            restored = load_document(path)
        self.assertEqual(restored, doc)
        payload = dict(direction="west_to_east", kind="real", ray_count=19, max_interactions=48)
        with patch("mirrorlab.service.load_configuration", side_effect=AssertionError("disk geometry leaked")), \
             patch("mirrorlab.service.survey_frames", side_effect=AssertionError("coordinates re-fitted")):
            first = dispatch("trace", {**payload, "active_configuration": doc})
            second = dispatch("trace", {**payload, "active_configuration": restored})
        for key in ("segment_count", "detector_hits", "detector_power_w"):
            self.assertEqual(first["summary"][key], second["summary"][key])
        self.assertEqual(first["figure"], second["figure"])
        config = configuration_from_dataset(restored["datasets"]["west_to_east"])
        self.assertEqual(config.tracer_options["surface_epsilon"], 1e-7)
        self.assertEqual(config.tracer_options["min_power"], 1e-18)
        self.assertTrue(any(np.allclose(s.center, mirror["center_m"]) for s in config.scene.build_surfaces()
                            if s.name == "BUNDLE_1_1 Mirror 1"))

    def test_invalid_candidates_do_not_mutate_accepted_document(self):
        original = copy.deepcopy(self.document)
        cases = []
        for field, value in (("schema_version", 99), ("units", "mm"), ("status", "success")):
            candidate = copy.deepcopy(original)
            candidate[field] = value
            cases.append(candidate)
        for change in ("nan", "reflection", "negative_radius", "duplicate", "wrong_axis", "dangling", "optics_hash", "path_shape", "covariance"):
            candidate = copy.deepcopy(original)
            dataset = candidate["datasets"]["west_to_east"]
            mirror = dataset["mirrors"][0]
            if change == "nan":
                mirror["center_m"][0] = float("nan")
            elif change == "reflection":
                mirror["rotation"][0] = [-x for x in mirror["rotation"][0]]
            elif change == "negative_radius":
                mirror["radius_m"] = -1
            elif change == "duplicate":
                dataset["mirrors"][1] = copy.deepcopy(mirror)
            elif change == "wrong_axis":
                mirror["axis"] = [1., 0., 0.]
            elif change == "dangling":
                mirror["observation_ids"] = ["absent"]
            elif change == "optics_hash":
                dataset["optics_snapshot"]["source"]["parameters"]["waist_radius"] *= 2
            elif change == "path_shape":
                mirror["trajectory"] = dict(point_m=[0., 0., 0.], direction=[0., 0., 1.], points_m=[[0., 0.]])
            elif change == "covariance":
                mirror["uncertainty"]["center_covariance_m2"] = [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            cases.append(candidate)
        for candidate in cases:
            with self.subTest(candidate=cases.index(candidate) if candidate.get("schema_version") == 99 else "invalid"):
                with self.assertRaises(ValueError):
                    validate_document(candidate)
                self.assertEqual(original, self.document)

    def test_atomic_replace_failure_preserves_previous_file(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "state.json"
            target.write_text("previous file", encoding="utf-8")
            with patch("mirrorlab.state.os.replace", side_effect=OSError("simulated interruption")):
                with self.assertRaises(OSError):
                    save_document(self.document, target)
            self.assertEqual(target.read_text(encoding="utf-8"), "previous file")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_optical_hash_survives_electron_json_number_round_trip(self):
        node = ROOT / ".tools" / "node" / "node.exe"
        if not node.exists():
            self.skipTest("Local Node runtime unavailable; covered by desktop integration test")
        encoded = json.dumps(self.document)
        result = subprocess.run([str(node), "-e",
            "let b='';process.stdin.on('data',c=>b+=c);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(b))));"],
            input=encoded, text=True, encoding="utf-8", capture_output=True, check=True)
        transit = json.loads(result.stdout)
        validate_document(transit)
        self.assertEqual(digest(transit), digest(self.document))

    def test_coordinate_reflection_is_distinct_from_body_reflection(self):
        candidate = copy.deepcopy(self.document)
        frame = candidate["datasets"]["west_to_east"]["display_frame"]
        frame.update(rotation=[[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], handedness_change=True)
        validate_document(candidate)
        frame["handedness_change"] = False
        with self.assertRaises(ValueError):
            validate_document(candidate)

    def test_native_parser_rejects_duplicate_keys_and_nonfinite(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "state.json"
            for value in ('{"format":1,"format":2}', '{"value":NaN}'):
                target.write_text(value, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_document(target)

    def test_legacy_json_deltas_require_exact_base(self):
        base_dir = ROOT / "data" / "static"
        path = base_dir / "west_to_east_real.json"
        doc = import_legacy(path)
        self.assertIsNotNone(doc["datasets"]["west_to_east"]["optics_snapshot"])
        self.assertEqual(doc["status"], "legacy_unvalidated")
        with TemporaryDirectory() as directory:
            delta = Path(directory) / path.name
            delta.write_bytes(path.read_bytes())
            doc = import_legacy(delta)
            self.assertIsNone(doc["datasets"]["west_to_east"]["optics_snapshot"])
            bad_base = json.loads((base_dir / "west_to_east_ideal.json").read_text(encoding="utf-8"))
            bad_base["source"]["parameters"]["waist_radius"] = .5
            (Path(directory) / "west_to_east_ideal.json").write_text(json.dumps(bad_base), encoding="utf-8")
            doc = import_legacy(delta)
            self.assertIsNone(doc["datasets"]["west_to_east"]["optics_snapshot"])

    def test_legacy_tables_archive_and_code_is_never_executed(self):
        source = ROOT / "Восстановленные_микросборки.txt"
        with self.assertRaises(ValueError):
            import_legacy(source)
        doc = import_legacy(source, "west_to_east", "legacy")
        self.assertEqual(len(doc["datasets"]["west_to_east"]["mirrors"]), 112)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "Optimis_code-main.zip"
            sentinel = Path(directory) / "executed.txt"
            with ZipFile(path, "w") as archive:
                archive.writestr("root/data.txt", source.read_bytes())
                archive.writestr("root/malicious.py", f"open({str(sentinel)!r}, 'w').write('executed')")
            restored = import_legacy(path, "west_to_east", "legacy")
            self.assertFalse(sentinel.exists())
            for a, b in zip(doc["datasets"]["west_to_east"]["mirrors"], restored["datasets"]["west_to_east"]["mirrors"]):
                self.assertEqual(a["center_m"], b["center_m"])
                self.assertEqual(a["rotation"], b["rotation"])
            with ZipFile(path, "w") as archive:
                archive.writestr("../escape.txt", "data")
            with self.assertRaises(ValueError):
                import_legacy(path, "west_to_east", "legacy")

    def test_windows_base_paths_rejected_before_read(self):
        original = json.loads((ROOT / "data/static/west_to_east_real.json").read_text(encoding="utf-8"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "delta.json"
            for name in ("..\\outside.json", "C:\\outside.json", "../outside.json", "/outside.json"):
                original["base_configuration"] = name
                path.write_text(json.dumps(original), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "sibling"):
                    import_legacy(path)

    def test_legacy_coating_convention_converted_with_traced_field(self):
        from mirrorlab.calibration.physical import incoming_at_mirrors
        doc = import_legacy(ROOT / "data/static/west_to_east_real.json")
        dataset = doc["datasets"]["west_to_east"]
        field = incoming_at_mirrors(configuration_from_dataset(dataset))
        for m in dataset["mirrors"]:
            self.assertLess(np.dot(m["normal"], field[m["mirror_id"]]["direction"]), 0)
            self.assertEqual(m["physical_normal_conversion"], "outward_against_traced_incident_branch")

    def test_invalid_surface_type_or_material_rejected_at_load(self):
        for change in ("type", "index", "mode"):
            doc = copy.deepcopy(self.document)
            d = doc["datasets"]["west_to_east"]
            item = d["optics_snapshot"]["elements"][0]["surfaces"][0]
            if change == "type":
                item.clear()
                item.update(type="Material", parameters=dict(name="fake surface", refractive_index_model=1.))
            elif change == "index":
                item["parameters"]["optics"]["parameters"]["n_minus"] = -1
            else:
                item["parameters"]["optics"]["parameters"]["mode"] = "execute"
            d["optics_sha256"] = digest(d["optics_snapshot"])
            with self.assertRaises(ValueError):
                validate_document(doc)

    def test_historical_ideal_optimization_rejected_before_reconstruction(self):
        with patch("mirrorlab.calibration.reconstruction.reconstruct") as reconstruct:
            with self.assertRaisesRegex(ValueError, "все лучи заданы"):
                dispatch("optimize", dict(direction="both", kind="ideal"))
            reconstruct.assert_not_called()

    def test_loading_old_document_adds_geometric_report_without_changing_source(self):
        doc = copy.deepcopy(self.document)
        dataset = doc['datasets']['west_to_east']
        q = np.asarray(dataset['display_frame']['rotation'])
        t = np.asarray(dataset['display_frame']['translation_m'])
        dataset['observations'] = [dict(id='experiment:unrelated_number',
            screens_m=[(q @ np.array([x, 0., 0.]) + t).tolist() for x in (1., 2.)])]
        dataset['mirrors'][0]['center_m'] = [0., 0., 0.]
        dataset['mirrors'][0]['outgoing_direction'] = [1., 0., 0.]
        dataset['diagnostics']['reflections'] = dict(mirrors={'4_4_7': dict(
            reflected_trajectories=[dict(segments=[dict(start_m=[0., 0., 0.], end_m=[3., 0., 0.])])])})
        before = copy.deepcopy(doc)
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'old.json'
            save_document(doc, source)
            source_bytes = source.read_bytes()
            result = dispatch('load_configuration', dict(path=str(source)))
            matching = result['reports'][0]['diagnostics']['trajectory_matching']
            geometric = result['reports'][0]['diagnostics']['geometric_fit']
            self.assertEqual(1, matching['matched_count'])
            self.assertEqual(1, geometric['matched_count'])
            self.assertEqual(before, doc)
            self.assertEqual(source_bytes, source.read_bytes())
            self.assertEqual(before['status'], result['status'])
            saved = Path(directory) / 'with-report.json'
            dispatch('save_configuration', dict(document=result['document'], path=str(saved)))
            self.assertEqual(matching, load_document(saved)['datasets']['west_to_east']['diagnostics']['trajectory_matching'])
            self.assertEqual(geometric, load_document(saved)['datasets']['west_to_east']['diagnostics']['geometric_fit'])
            traced = dispatch('trace', dict(direction='west_to_east', kind='real', ray_count=1,
                                            max_interactions=1, active_configuration=result['document']))
            self.assertEqual(geometric, traced['comparison_reports'][0]['diagnostics']['geometric_fit'])
            self.assertEqual(matching, traced['comparison_reports'][0]['diagnostics']['trajectory_matching'])

    def test_active_configuration_optimization_ignores_historical_selection(self):
        from mirrorlab.legacy_import import historical_dataset
        with patch("mirrorlab.calibration.reconstruction.reconstruct",
                   side_effect=lambda config, *args, **kwargs: historical_dataset(config)) as reconstruct:
            result = dispatch("optimize", dict(direction="west_to_east", kind="ideal",
                                               active_configuration=self.document))
        reconstruct.assert_called_once()
        self.assertIn("west_to_east", result["document"]["datasets"])

    def test_failure_of_one_side_retains_other_side_diagnostics(self):
        from mirrorlab.legacy_import import historical_dataset
        def candidate(config, *args, **kwargs):
            if config.direction == "east_to_west":
                raise ValueError("contradictory input")
            dataset = historical_dataset(config)
            dataset["status"] = "partial"
            return dataset
        with patch("mirrorlab.calibration.reconstruction.reconstruct", side_effect=candidate):
            result = dispatch("optimize", dict(direction="both", kind="real"))
        datasets = result["document"]["datasets"]
        self.assertEqual(datasets["west_to_east"]["status"], "partial")
        self.assertEqual(datasets["east_to_west"]["status"], "infeasible")
        self.assertEqual(len(datasets["east_to_west"]["mirrors"]), 112)
        self.assertFalse(result["can_activate"])

    def test_no_silent_direction_fallback(self):
        with self.assertRaises(ValueError):
            dispatch("trace", dict(direction="both", active_configuration=self.document))


if __name__ == "__main__":
    unittest.main()
