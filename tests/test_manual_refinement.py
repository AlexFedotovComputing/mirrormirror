import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from mirrorlab.manual_schema import BUNDLE, import_manual
from mirrorlab.legacy_import import import_legacy
from mirrorlab.state import new_document, validate_document, save_document, load_document
from mirrorlab.calibration.manual_refinement import masked_point_residual, refine_manual


class ManualImportTests(unittest.TestCase):
    def test_coordinates_and_independent_identity(self):
        west = import_manual(BUNDLE / "west_manual_corrected_survey.json")["datasets"]["east_to_west"]
        q, t = np.array(west["display_frame"]["rotation"]), np.array(west["display_frame"]["translation_m"])
        self.assertEqual(len(west["mirrors"]), 112)
        self.assertEqual(len({m["canonical_id"] for m in west["mirrors"]}), 112)
        for m in west["mirrors"]:
            np.testing.assert_allclose(q @ m["center_m"] + t, m["manual_reference_survey"]["survey_center_m"], atol=1e-14)
        self.assertIsNone(west["mirrors"][0]["canonical_slot"])
        self.assertNotEqual(west["mirrors"][1]["source_manual_id"], west["mirrors"][1]["survey_observation_id"].split(":")[-1])
        east = import_manual(BUNDLE / "east_recovered_centers.json")["datasets"]["west_to_east"]
        gap = next(m for m in east["mirrors"] if m["source_manual_id"] == "3_2_7")
        self.assertEqual(gap["canonical_slot"], 5)
        self.assertFalse(gap["raw_manual_direction_used_as_prior"])
        single = next(r for r in east["observations"]["records"] if r["old_id"] == "4_1_3")
        self.assertEqual(single["complete_points"], 1)

    def test_text_and_duplicate_zip_preserve_lineage(self):
        for path in (BUNDLE / "upload").glob("*.txt"):
            dataset = next(iter(import_manual(path)["datasets"].values()))
            source = dataset["manual_references"]["original_text"]
            self.assertEqual(source["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.zip"
            with ZipFile(path, "w") as archive:
                for name in ("west_manual_corrected_survey.json", "east_recovered_centers.json"):
                    archive.write(BUNDLE / name, name)
                    archive.write(BUNDLE / name, "analysis/" + name)
            document = import_legacy(path)
            self.assertEqual(len(document["datasets"]), 2)
            self.assertEqual(len(document["datasets"]["west_to_east"]["observations"]["records"]), 112)

    def test_reject_double_transform(self):
        with self.assertRaises(ValueError):
            import_manual(BUNDLE / "east_recovered_centers.json", coordinate_frame="legacy")

    def test_masked_residual_is_projection_not_zero_fill(self):
        np.testing.assert_allclose(masked_point_residual([0,0,0], [1,0,0], [5,None,2]), [0,2])
        self.assertEqual(len(masked_point_residual([0,0,0], [1,0,0], [None,None,None])), 0)

    def test_full_refinement_preserves_raw_and_rejects_collisions(self):
        original = import_manual(BUNDLE / "east_recovered_centers.json")["datasets"]["west_to_east"]
        old = copy.deepcopy(original)
        messages = []
        result = refine_manual(original, {"max_nfev": 10}, messages.append)
        self.assertEqual(original, old)
        self.assertEqual(original["observations"], result["observations"])
        self.assertEqual(original["manual_references"], result["manual_references"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["physical_validation"]["validated_count"], 112)
        self.assertFalse(result["mechanics"]["conflicts"])
        self.assertAlmostEqual(result["manual_refinement"]["before"]["complete_point_rms_mm"], 13.04764567271065, places=6)
        self.assertTrue(messages[-1]["fraction"] == 1)
        for m in result["mirrors"]:
            self.assertLess(m["refinement"]["proposed_fit"]["primary_jacobian_rank"], 6)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "native.json"
            doc = validate_document(new_document({"west_to_east": result}))
            save_document(doc, target)
            self.assertEqual(load_document(target), doc)


if __name__ == "__main__":
    unittest.main()
