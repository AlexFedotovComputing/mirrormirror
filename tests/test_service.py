import json
from pathlib import Path
import subprocess
import sys
import unittest

from mirrorlab.service import TraceRequest, dispatch
from unittest.mock import patch
from tempfile import TemporaryDirectory


class ServiceTests(unittest.TestCase):
    def test_parameter_validation(self):
        for payload in ({"ray_count": True}, {"ray_count": 0}, {"ray_count": 50001},
                        {"max_interactions": 101}, {"collision_mode": "fast"},
                        {"direction": "east"}, {"kind": "dynamic"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                TraceRequest(**payload)
        with self.assertRaises(ValueError):
            dispatch("trace", [])

    def test_trace_json_and_limit_warning(self):
        result = dispatch("trace", {"direction": "west_to_east", "ray_count": 19, "max_interactions": 1})
        json.dumps(result, allow_nan=False)
        self.assertGreater(result["summary"]["unfinished_count"], 0)
        self.assertTrue(any("Лимит" in message for message in result["warnings"]))
        self.assertEqual(result["metadata"]["source_side"], "WEST")
        self.assertGreater(len(result["figure"]["data"]), 10)
        # Validate the complete wire figure against Plotly's schema.
        import plotly.graph_objects as go
        go.Figure(result["figure"])

    def test_default_build_contains_both_sides_and_preserves_each_calculation(self):
        result = dispatch("trace", {"ray_count": 19})
        self.assertEqual(result["request"]["direction"], "both")
        self.assertEqual({s["metadata"]["injection_side"] for s in result["sides"]}, {"WEST", "EAST"})
        self.assertEqual(result["summary"]["ray_count"], 38)
        self.assertEqual(result["metadata"]["coordinate_frame"], "survey")
        for side in result["sides"]:
            single = dispatch("trace", {"direction": side["direction"], "ray_count": 19})
            for key in ("segment_count", "detector_hits", "detector_power_w", "unfinished_count"):
                self.assertEqual(side["summary"][key], single["summary"][key])
            name = side["metadata"]["injection_side"]
            combined = [t for t in result["figure"]["data"] if t.get("legendgroup") in (name, f"{name}:survey")]
            separate = [t for t in single["figure"]["data"] if not t.get("name", "").startswith("Граница")]
            self.assertEqual(len(combined), len(separate))
            for a, b in zip(combined, separate):
                for axis in "xyz":
                    self.assertEqual(a[axis], b[axis])
        self.assertTrue(any(t.get("name") == "Лазер EAST → ввод WEST" for t in result["figure"]["data"]))
        self.assertTrue(any(t.get("name") == "Лазер WEST → ввод EAST" for t in result["figure"]["data"]))
        import plotly.graph_objects as go
        go.Figure(result["figure"])

    def test_real_build_automatically_uses_available_screens_and_reports_partial_use(self):
        result = dispatch("trace", {"kind": "real", "ray_count": 19})
        reports = {r["direction"]: r for r in result["measurements"]["reports"]}
        west = reports["east_to_west"]["additional_screens"]
        self.assertEqual(west["status"], "used_partial")
        self.assertEqual(west["accepted_rods"], [1, 2, 4])
        self.assertEqual(west["rejected_rods"], [3])
        self.assertEqual(west["refined_ray_count"], 75)
        self.assertFalse(west["applied_to_mirror_geometry"])
        self.assertEqual(reports["west_to_east"]["additional_screens"]["status"], "not_available")
        json.dumps(result, allow_nan=False)

    def test_missing_additional_file_is_not_reported_as_used(self):
        with TemporaryDirectory() as directory, patch("mirrorlab.service.ROOT", Path(directory)):
            result = dispatch("calibrate", {"direction": "east_to_west"})
        self.assertEqual(result["additional_screens"]["status"], "not_available")
        self.assertEqual(result["lavsan"]["status"], "not_requested")

    def test_worker_recovers_after_bad_request(self):
        messages = "not JSON\n" + json.dumps({"id": 2, "method": "catalog", "payload": {}}) + "\n"
        process = subprocess.run([sys.executable, "-u", "-m", "mirrorlab.worker"],
                                 cwd=Path(__file__).resolve().parents[1], input=messages,
                                 text=True, encoding="utf-8", capture_output=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        first, second = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertIn("error", first)
        self.assertEqual(second["id"], 2)
        self.assertEqual(second["result"]["modules"]["static"], "ready")


if __name__ == "__main__":
    unittest.main()
