from pathlib import Path
import unittest

from mirrorlab.calibration.empirical import assign_fan, learn_template
from mirrorlab.calibration.observations import observation_registry
from mirrorlab.calibration.reconstruction import group_observations


class OriginalNumberingControl(unittest.TestCase):
    def test_blank_seventh_row_is_an_internal_fan_omission(self):
        root = Path(__file__).resolve().parents[1]
        registry = observation_registry(root, "west_to_east")
        groups = group_observations(registry["records"])
        template = learn_template(groups, iterations=3, ambiguity_rad=.006)
        target = groups["3_2"]
        self.assertEqual(len(target["directions"]), 6)
        fit = assign_fan(template["directions_local"], target["directions"], ambiguity_rad=.006)
        missing = set(range(7)) - set(fit["slots"])
        self.assertEqual(len(missing), 1)
        self.assertFalse(missing & {0, 6}, "An empty row must not force an edge slot")
        self.assertEqual(template["complete_assemblies"], 0)
        self.assertTrue(all(s["independent_assemblies"] > 1 for s in template["statistics"]))


if __name__ == "__main__":
    unittest.main()
