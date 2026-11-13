"""Differential checks for exact field reuse and stable trace indexes."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mirrorlab.calibration import physical
from mirrorlab.configurations import StaticElement
from mirrorlab.engine import GaussianBeamSource, Scene
from mirrorlab.engine.bodies import MirrorBody


class PhysicalCacheUnitTests(unittest.TestCase):
    def setUp(self):
        physical._FIELD_CACHE.clear()
        physical._INCOMING_CACHE.clear()

    def tearDown(self):
        self.setUp()

    def test_indexes_reproduce_scan_order_and_missing_keys(self):
        rows = {7: dict(id=7, parent=2, root=0, chain=["MP", "MC"]),
                3: dict(id=3, parent=2, root=0, chain=["MP", "MC"]),
                5: dict(id=5, parent=1, root=1, chain=["MP"])}
        branches, parents = physical._record_indexes(rows)
        for row in rows.values():
            self.assertEqual(branches[(row["root"], tuple(row["chain"]))],
                [r for r in rows.values() if r["root"] == row["root"] and r["chain"] == row["chain"]])
            self.assertEqual(parents[row["parent"]], [r for r in rows.values() if r["parent"] == row["parent"]])
        self.assertEqual([r["id"] for r in parents[2]], [7, 3])
        self.assertEqual(branches.get((99, ()), []), [])

    def test_cached_values_are_isolated_and_keys_include_exact_points_ids_and_order(self):
        config = SimpleNamespace(source=SimpleNamespace(waist_radius=1., cutoff_ratio=1.))
        with patch.object(physical, "_field_models", return_value=({}, "fixed-common-optics")):
            points = {"1_1_1": np.zeros(3), "1_1_2": np.ones(3)}
            expected = physical.incoming_at_points(config, points)
            first = physical.incoming_at_points(config, points)
            first["1_1_1"]["alternatives"].append({"status": "forged"})
            expected["1_1_2"]["status"] = "forged"
            actual = physical.incoming_at_points(config, points)
            self.assertEqual(actual["1_1_1"]["alternatives"], [])
            self.assertEqual(actual["1_1_2"]["status"], "unavailable")
            self.assertEqual(len(physical._INCOMING_CACHE), 1)
            points["1_1_1"][0] = np.nextafter(0., 1.)
            physical.incoming_at_points(config, points)
            physical.incoming_at_points(config, dict(reversed(list(points.items()))))
            physical.incoming_at_points(config, {"1_1_3": points["1_1_1"], "1_1_2": points["1_1_2"]})
            self.assertEqual(len(physical._INCOMING_CACHE), 4)
            for i in range(20):
                physical.incoming_at_points(config, {"1_1_1": [0., 0., float(i)]})
            self.assertEqual(len(physical._INCOMING_CACHE), 8)

    def test_field_hash_invalidates_optics_source_trace_time_and_effective_depth(self):
        config = SimpleNamespace(scene=Scene([]), source=GaussianBeamSource([0, 0, 0], [0, 0, 1], .01, 1., 266e-9),
            integration_time_s=1e-7, max_interactions=12, tracer_options={}, emit=lambda n: None)
        points = {"1_1_1": [0., 0., 1.]}
        with patch.object(physical, "_trace", return_value=SimpleNamespace(segments=[])) as trace:
            original = physical.incoming_at_points(config, points)["1_1_1"]["field_hash"]
            config.max_interactions = 24  # Still 48 effective steps.
            self.assertEqual(physical.incoming_at_points(config, points)["1_1_1"]["field_hash"], original)
            self.assertEqual(trace.call_count, 1)
            for mutate in [lambda c: setattr(c, "max_interactions", 49),
                           lambda c: setattr(c, "integration_time_s", 2e-7),
                           lambda c: setattr(c.source, "waist_radius", .02),
                           lambda c: c.tracer_options.update(surface_epsilon=2e-7),
                           lambda c: c.scene.elements.append(StaticElement("MC_changed", []))]:
                candidate = copy.deepcopy(config)
                mutate(candidate)
                self.assertNotEqual(physical.incoming_at_points(candidate, points)["1_1_1"]["field_hash"], original)
            self.assertEqual(trace.call_count, 6)
            # Bodies do not affect the common-optics field; they are tested by
            # the uncached full physical scene in validate_reflections.
            config.scene.elements.append(StaticElement("BUNDLE_changed", []))
            self.assertEqual(physical.incoming_at_points(config, points)["1_1_1"]["field_hash"], original)
            self.assertEqual(trace.call_count, 6)


class _ScanIndex:
    """Reference implementation of the original full scans."""
    def __init__(self, rows, branch):
        self.rows, self.branch = rows, branch

    def get(self, key, default=None):
        if self.branch:
            return [r for r in self.rows.values() if r["root"] == key[0] and tuple(r["chain"]) == key[1]]
        return [r for r in self.rows.values() if r["parent"] == key]


class PhysicalCacheRealDataTests(unittest.TestCase):
    def test_both_sides_match_uncached_scans_and_always_trace_full_bodies(self):
        from mirrorlab.manual_schema import import_manual
        from mirrorlab.state import configuration_from_dataset
        bundle = Path(__file__).resolve().parents[1] / "data" / "manual_refinement"
        for name in ("west_manual_corrected_survey.json", "east_recovered_centers.json"):
            with self.subTest(name=name):
                dataset = next(iter(import_manual(bundle / name)["datasets"].values()))
                config = configuration_from_dataset(dataset)
                bodies = [MirrorBody.from_dict(m) for m in dataset["mirrors"]]
                physical._INCOMING_CACHE.clear()
                with patch.object(physical, "_record_indexes", side_effect=lambda rows: (_ScanIndex(rows, True), _ScanIndex(rows, False))):
                    reference = physical.validate_reflections(config, bodies)
                physical._INCOMING_CACHE.clear()
                with patch.object(physical, "_trace", wraps=physical._trace) as trace:
                    fresh = physical.validate_reflections(config, bodies)
                    calls_after_fresh = trace.call_count
                    cached = physical.validate_reflections(config, bodies)
                    # The reused incident field saves work, but the body scene
                    # is still traced again, even with identical geometry.
                    self.assertEqual(trace.call_count, calls_after_fresh + 1)
                    self.assertEqual(fresh, reference)
                    self.assertEqual(cached, reference)


if __name__ == "__main__":
    unittest.main()
