import copy
import unittest

import numpy as np

from mirrorlab.calibration.trajectory_matching import (
    best_fit_trajectories, compare_dataset_geometric_fit, compare_dataset_trajectories,
    match_trajectories, traced_output_trajectories,
)


def observation(identifier, y=0.):
    return dict(id=identifier, screens_m=[[1., y, 0.], [2., y, 0.]])


def trajectory(identifier, y=0., length=3.):
    return dict(id=identifier, point_m=[0., y, 0.], direction=[1., 0., 0.], length_m=length)


def segment(start, end):
    return dict(start_m=start, end_m=end)


class TrajectoryMatchingTests(unittest.TestCase):
    def test_renumbering_reordering_and_reversed_screen_order_do_not_change_geometry(self):
        observations = [observation('1_1_1', 0.), observation('4_3_7', .1)]
        candidates = [trajectory('4_3_7', 0.), trajectory('1_1_1', .1)]
        result = match_trajectories(observations, candidates)
        self.assertEqual(2, result['matched_count'])
        self.assertEqual({('1_1_1', '4_3_7'), ('4_3_7', '1_1_1')},
                         {(p['observation_id'], p['trajectory_id']) for p in result['matches']})
        shuffled = copy.deepcopy(list(reversed(observations)))
        for i, row in enumerate(shuffled):
            row['id'] = f'renamed:{i}'
            row['screens_m'].reverse()
        other = match_trajectories(shuffled, list(reversed(candidates)))
        self.assertEqual(2, other['matched_count'])
        self.assertEqual([0., 0.], [p['maximum_distance_m'] for p in other['matches']])

    def test_one_path_cannot_explain_two_measurements(self):
        result = match_trajectories([observation('a'), observation('b')], [trajectory('c')])
        self.assertEqual(1, result['matched_count'])
        self.assertEqual(1, result['unmatched_observation_count'])
        self.assertEqual('one_to_one_conflict', result['unmatched_observations'][0]['reason'])

    def test_global_assignment_maximizes_matches_not_greedy_nearest(self):
        result = match_trajectories([observation('a', 0.), observation('b', -.0009)],
                                    [trajectory('c', -.0001), trajectory('d', .0009)], distance_tolerance_m=.001)
        self.assertEqual(2, result['matched_count'])
        self.assertEqual({('a', 'd'), ('b', 'c')},
                         {(p['observation_id'], p['trajectory_id']) for p in result['matches']})

    def test_parallel_offset_and_crossing_wrong_direction_are_not_matches(self):
        crossing = dict(id='crossing', point_m=[1.5, -1., 0.], direction=[0., 1., 0.], length_m=2.)
        result = match_trajectories([observation('a')], [trajectory('offset', .02), crossing])
        self.assertEqual(0, result['matched_count'])
        self.assertEqual(2, result['unmatched_trajectory_count'])

    def test_does_not_extrapolate_past_blocker_or_behind_origin(self):
        for candidate in [trajectory('blocked', length=1.5),
                          dict(id='backward', point_m=[3., 0., 0.], direction=[1., 0., 0.])]:
            self.assertEqual(0, match_trajectories([observation('a')], [candidate])['matched_count'])
        self.assertEqual(1, match_trajectories([observation('a')], [trajectory('passed', length=2.5)])['matched_count'])

    def test_angle_gate_applies_even_when_both_points_are_close(self):
        row = dict(id='short', screens_m=[[1., 0., 0.], [1.0001, .0001, 0.]])
        result = match_trajectories([row], [trajectory('a')], angle_tolerance_rad=.006)
        self.assertEqual(0, result['matched_count'])
        nearest = result['unmatched_observations'][0]['nearest']
        self.assertLess(nearest['maximum_distance_m'], result['distance_tolerance_m'])
        self.assertGreater(nearest['angular_error_rad'], result['angle_tolerance_rad'])

    def test_default_angle_is_descriptive_even_for_large_angular_error(self):
        row = dict(id='short', screens_m=[[1., 0., 0.], [1.0001, .0001, 0.]])
        result = match_trajectories([row], [trajectory('a')])
        self.assertEqual(1, result['matched_count'])
        self.assertIsNone(result['angle_tolerance_rad'])
        self.assertGreater(result['matches'][0]['angular_error_rad'], .006)

    def test_best_fit_keeps_outliers_and_reports_statistics_over_all_assignments(self):
        rows = [observation('first', 0.), observation('second', .1), observation('outlier', .3)]
        candidates = [trajectory('renumbered-b', .1005), trajectory('renumbered-a', .002), trajectory('renumbered-c', .31)]
        result = best_fit_trajectories(rows, candidates)
        self.assertEqual(3, result['assigned_count'])
        self.assertEqual(2, result['matched_count'])
        self.assertEqual(1, result['counts_within_1mm'])
        self.assertEqual(2, result['counts_within_3mm'])
        self.assertAlmostEqual(2., result['position_error_mm']['median'])
        self.assertAlmostEqual(10., result['position_error_mm']['maximum'])
        self.assertAlmostEqual(9.2, result['position_error_mm']['p95'])
        self.assertEqual('outlier', result['outliers'][0]['observation_id'])
        self.assertEqual(0, result['unmatched_observation_count'])
        reversed_result = best_fit_trajectories(list(reversed(rows)), list(reversed(candidates)))
        self.assertEqual(result['position_error_mm'], reversed_result['position_error_mm'])
        self.assertEqual({(p['observation_id'], p['trajectory_id']) for p in result['matches']},
                         {(p['observation_id'], p['trajectory_id']) for p in reversed_result['matches']})

    def test_best_fit_cannot_reuse_a_ray_and_missing_data_has_no_false_zero_error(self):
        result = best_fit_trajectories([observation('a'), observation('b')], [trajectory('one')])
        self.assertEqual(1, result['assigned_count'])
        self.assertEqual(1, result['unmatched_observation_count'])
        empty = best_fit_trajectories([observation('a')], [])
        self.assertIsNone(empty['position_error_mm']['median'])
        self.assertEqual(0, empty['assigned_count'])

    def test_old_tolerances_do_not_reintroduce_angular_gate_in_dataset_reports(self):
        row = dict(id='short', screens_m=[[1., 0., 0.], [1.0001, .002, 0.]])
        dataset = dict(direction='west_to_east', status='partial', coordinate_frame='survey',
            observations=[row], options=dict(trajectory_distance_tolerance_m=.001, trajectory_angle_tolerance_rad=.006),
            mirrors=[dict(mirror_id='1_1_1', center_m=[0., 0., 0.], outgoing_direction=[1., 0., 0.])],
            diagnostics=dict(reflections=dict(mirrors={'1_1_1': dict(reflected_trajectories=[dict(
                segments=[segment([0., 0., 0.], [3., 0., 0.])])])})))
        for compare in (compare_dataset_geometric_fit, compare_dataset_trajectories):
            result = compare(dataset)
            self.assertEqual(1, result['matched_count'])
            self.assertEqual(.003, result['distance_tolerance_m'])
            self.assertIsNone(result['angle_tolerance_rad'])

    def test_partial_and_coincident_points_not_promoted_to_known_trajectories(self):
        rows = [dict(id='partial', screens_m=[[1., 0., 0.], [2., None, 0.]]),
                dict(id='coincident', screens_m=[[1., 0., 0.], [1., 0., 0.]])]
        result = match_trajectories(rows, [trajectory('a')])
        self.assertEqual(0, result['observation_count'])
        self.assertEqual(2, len(result['excluded_observations']))
        self.assertEqual(1, result['unmatched_trajectory_count'])

    def test_empty_sets_and_bad_tolerances(self):
        result = match_trajectories([observation('a')], [])
        self.assertEqual(1, result['unmatched_observation_count'])
        self.assertIsNone(result['unmatched_observations'][0]['nearest'])
        self.assertEqual(0, match_trajectories([], [])['matched_count'])
        for tolerance in [0., -1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                match_trajectories([], [], distance_tolerance_m=tolerance)

    def test_trace_prefix_continues_through_screen_but_stops_at_turn(self):
        prefix = [segment([0., 0., 0.], [.5, 0., 0.]),
                  segment([.5000001, 0., 0.], [3., 0., 0.])]
        reflections = dict(mirrors={'arbitrary': dict(status='infeasible', reflected_trajectories=[
            dict(segments=prefix + [segment([3., 0., 0.], [3., 2., 0.])]),
            dict(segments=prefix + [segment([3., 0., 0.], [3., -2., 0.])])])})
        # Coordinate basis changes, including reflection, must apply to both
        # position and direction. Trace verdict labels do not control matching.
        frame = dict(rotation=np.diag([-1., 1., 1.]).tolist(), translation_m=[5., 2., 0.])
        paths = traced_output_trajectories(reflections, frame)
        self.assertEqual(1, len(paths))
        self.assertEqual(3., paths[0]['length_m'])
        row = dict(id='different_assembly', screens_m=[[4., 2., 0.], [3., 2., 0.]])
        report = compare_dataset_trajectories(dict(observations=[row], display_frame=frame,
                     diagnostics=dict(reflections=reflections)))
        self.assertEqual(1, report['matched_count'])

    def test_missing_frame_is_not_silently_treated_as_identity(self):
        report = compare_dataset_trajectories(dict(observations=[observation('a')],
                     diagnostics=dict(reflections={'mirrors': {}}), coordinate_frame='legacy'))
        self.assertEqual('not_evaluated', report['status'])


if __name__ == '__main__':
    unittest.main()
