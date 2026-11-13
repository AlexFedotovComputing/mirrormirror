"""Differential oracle frozen before batch assignment optimization."""
from itertools import combinations
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from mirrorlab.calibration import empirical
from mirrorlab.calibration.empirical import unit, local_fan, rigid_rotation, angular_distance

def scalar_assign_fan(template, directions, *, ambiguity_rad=.002, max_alternatives=6, allow_outlier=True):
    """Enumerate monotone subsets and at most one explicitly excluded outlier."""
    template, directions = unit(template), unit(directions)
    if not 3 <= len(directions) <= 7:
        raise ValueError("A fan needs 3 to 7 complete rays")
    local, order, basis = local_fan(directions)
    candidates = []
    masks = [np.arange(len(local))]
    if allow_outlier and len(local) > 4:
        masks += [np.delete(np.arange(len(local)), k) for k in range(len(local))]
    for keep in masks:
        for slots in combinations(range(7), len(keep)):
            slots = np.asarray(slots, int)
            rotation = rigid_rotation(template[slots], local[keep])
            errors = angular_distance(template[slots] @ rotation.T, local[keep])
            # Removing a point requires a 5-sigma gross error, not an automatic match.
            cost = float(np.sum(np.minimum(errors**2, (10 * ambiguity_rad)**2))
                         + (len(local) - len(keep)) * (5 * ambiguity_rad)**2)
            global_rotation = basis @ rotation
            candidates.append(dict(slots=slots.tolist(), observation_indices=order[keep].tolist(),
                                   outliers=order[np.setdiff1d(np.arange(len(local)), keep)].tolist(),
                                   cost=cost, rotation=global_rotation.tolist(), angular_residual_rad=errors.tolist()))
    candidates.sort(key=lambda c: (c["cost"], c["slots"]))
    best = candidates[0]
    tolerance = len(directions) * ambiguity_rad**2
    alternatives = [c for c in candidates if c["cost"] <= best["cost"] + tolerance]
    return {**best, "alternatives": alternatives[:max_alternatives], "alternative_count": len(alternatives),
            "ambiguous": len(alternatives) > 1, "cost_is_probability": False,
            "unmatched_cost_rad2": (5 * ambiguity_rad)**2}


class BatchAssignmentTests(unittest.TestCase):
    def assert_oracle(self, template, directions, **options):
        self.assertEqual(empirical.assign_fan(template, directions, **options),
                         scalar_assign_fan(template, directions, **options))

    def test_synthetic_all_cardinalities_and_outlier_modes(self):
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(7451)
        theta = np.linspace(-.28, .31, 7)
        template = unit(np.c_[np.cos(theta), np.sin(theta), rng.normal(0, .002, 7)])
        for count in range(3, 8):
            for trial in range(4):
                selection = sorted(rng.choice(7, count, replace=False))
                rotation = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
                points = unit(template[selection] @ rotation.T + rng.normal(0, .001, (count, 3)))
                if trial == 3:
                    points[0] = unit(points[0] + [.3, -.2, .1])
                points = points[rng.permutation(count)]
                for allow_outlier in (False, True):
                    for maximum in (0, 1, 6, 999):
                        self.assert_oracle(template, points, allow_outlier=allow_outlier,
                                           max_alternatives=maximum)

    def test_exact_near_ties_and_rank_deficiency(self):
        theta = np.linspace(-.3, .3, 7)
        planar = np.c_[np.cos(theta), np.sin(theta), np.zeros(7)]
        for perturbation in (0., 1e-15, 1e-10):
            template = planar.copy()
            template[3, 2] += perturbation
            for count in range(3, 8):
                for ambiguity in (0., .002, 1.):
                    self.assert_oracle(template, template[:count], ambiguity_rad=ambiguity,
                                       max_alternatives=999)
        repeated = np.tile([1., 0., 0.], (7, 1))
        for count in range(3, 8):
            self.assert_oracle(repeated, repeated[:count])
            self.assert_oracle(planar, repeated[:count])
        for scale in (1e-2, 1e-4, 1e-6):
            near_rank_one = np.c_[np.cos(theta * scale), np.sin(theta * scale), np.zeros(7)]
            for count in range(3, 8):
                self.assert_oracle(near_rank_one, near_rank_one[:count], ambiguity_rad=scale * .002)

    def test_cache_is_immutable_and_not_corrupted_between_calls(self):
        for slots, keeps, omitted in empirical._fan_hypotheses(6, True):
            self.assertFalse(slots.flags.writeable)
            self.assertFalse(keeps.flags.writeable)
            self.assertFalse(omitted.flags.writeable)
            with self.assertRaises(ValueError):
                slots[0, 0] = 2

    def test_alternative_threshold_rounding_boundary(self):
        theta = np.array([-.31, -.20, -.11, .003, .10, .23, .34])
        template = unit(np.c_[np.cos(theta), np.sin(theta), np.linspace(-.002, .003, 7)])
        directions = template[[0, 1, 3, 4, 6]]
        full = scalar_assign_fan(template, directions, ambiguity_rad=1.,
                                 allow_outlier=False, max_alternatives=999)
        for candidate in full['alternatives'][1:]:
            boundary = np.sqrt((candidate['cost'] - full['cost']) / len(directions))
            for ambiguity in (np.nextafter(boundary, 0.), boundary, np.nextafter(boundary, np.inf)):
                self.assert_oracle(template, directions, ambiguity_rad=ambiguity,
                                   allow_outlier=False, max_alternatives=999)

    def test_proper_rotation_reflection_degeneracy(self):
        # Full-rank cross matrix with negative determinant and repeated
        # smallest singular values: rank-one checks alone cannot detect this.
        template = unit(np.r_[np.eye(3), [[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1]]])
        for perturbation in (0., 1e-14, 1e-7):
            reflected = template[:3] @ np.diag([1., 1., -1.])
            reflected[0, 1] += perturbation
            for ambiguity in (0., .002, 1.):
                self.assert_oracle(template, reflected, ambiguity_rad=ambiguity,
                                   max_alternatives=999, allow_outlier=False)

    def test_input_failure_contract(self):
        template = np.tile([1., 0., 0.], (7, 1))
        for count in (0, 1, 2, 8):
            with self.assertRaisesRegex(ValueError, '3 to 7'):
                empirical.assign_fan(template, np.tile([1., 0., 0.], (count, 1)))


class RealTemplateDifferentialTests(unittest.TestCase):
    def test_both_real_templates_and_every_real_assembly_are_exact(self):
        from mirrorlab.calibration.observations import observation_registry
        from mirrorlab.calibration.reconstruction import group_observations
        root = Path(__file__).resolve().parents[1]
        for direction in ('east_to_west', 'west_to_east'):
            groups = group_observations(observation_registry(root, direction)['records'])
            with patch.object(empirical, 'assign_fan', scalar_assign_fan):
                expected = empirical.learn_template(groups, iterations=5, ambiguity_rad=.006)
            actual = empirical.learn_template(groups, iterations=5, ambiguity_rad=.006)
            self.assertEqual(actual, expected)
            template = actual['directions_local']
            for group in groups.values():
                self.assertEqual(empirical.assign_fan(template, group['directions'], ambiguity_rad=.006),
                                 scalar_assign_fan(template, group['directions'], ambiguity_rad=.006))


if __name__ == '__main__':
    unittest.main()

