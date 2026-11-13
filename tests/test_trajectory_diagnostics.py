import copy
import json
import unittest

import numpy as np

from mirrorlab.calibration.trajectory_matching import dataset_comparison_report


def dataset(end=1.5, surface='BUNDLE_1_1 Mirror 2:side'):
    return dict(direction='west_to_east', status='partial', coordinate_frame='survey',
                observations=[dict(id='measured', screens_m=[[1., 0., 0.], [2., 0., 0.]])],
                mirrors=[dict(mirror_id='1_1_1', center_m=[0., 0., 0.], outgoing_direction=[1., 0., 0.],
                              radius_m=.0005)],
                diagnostics=dict(reflections=dict(mirrors={'1_1_1': dict(
                    status='infeasible', incoming=dict(status='available'), actual_reflected_child=True,
                    reflected_trajectories=[dict(segments=[dict(start_m=[0., 0., 0.],
                        end_m=[end, 0., 0.], surface=surface)])])})))


def diagnose(data):
    return dataset_comparison_report(data)['diagnostics']['trajectory_diagnostics']


class TrajectoryDiagnosticsTests(unittest.TestCase):
    def test_blocker_distinguishes_fitted_line_from_finite_path_without_mutation(self):
        data = dataset()
        original = copy.deepcopy(data)
        report = diagnose(data)
        self.assertEqual(data, original)
        json.dumps(report, allow_nan=False)
        row = report['observations'][0]
        self.assertTrue(row['fit_only'])
        self.assertEqual(['outgoing_mirror_interception'], row['reason_codes'])
        self.assertEqual('BUNDLE_1_1 Mirror 2:side', row['path']['terminal_surface'])
        self.assertEqual([0., .5], [p['beyond_end_m'] for p in row['path']['point_checks']])
        self.assertEqual(0., row['path']['extension_maximum_distance_m'])
        self.assertAlmostEqual(2**.5, row['fitted_mirror']['dimensions']['ellipse_major_axis_mm'])

    def test_blocker_beyond_measurements_does_not_explain_a_failure(self):
        data = dataset(end=2.5)
        data['diagnostics']['reflections']['mirrors']['1_1_1']['outgoing_blockers'] = ['BUNDLE_1_1 Mirror 2:side']
        data['diagnostics']['geometry'] = dict(conflicts=[dict(pair=['1_1_1', '1_1_2'], status='intersecting')])
        row = diagnose(data)['observations'][0]
        self.assertEqual(['matched'], row['reason_codes'])
        self.assertFalse(row['fit_only'])
        self.assertTrue(row['fitted_mirror']['body_conflicts'])

    def test_time_limit_and_unknown_endpoint_are_not_called_opaque_blockers(self):
        for surface, expected in [('<time_limit>', 'trace_time_limit'), (None, 'trace_ended_before_points'),
                                  ('SCREEN', 'trace_ended_before_points'), ('<bundle_system_boundary>', 'trace_boundary')]:
            with self.subTest(surface=surface):
                row = diagnose(dataset(surface=surface))['observations'][0]
                self.assertEqual([expected], row['reason_codes'])

    def test_turn_preserves_actual_stopping_surface_not_later_descendant(self):
        data = dataset(surface='TURN')
        data['diagnostics']['reflections']['mirrors']['1_1_1']['reflected_trajectories'][0]['segments'].append(
            dict(start_m=[1.5, 0., 0.], end_m=[1.5, 3., 0.], surface='BUNDLE_9_9 Mirror 9:side'))
        row = diagnose(data)['observations'][0]
        self.assertEqual(['path_turn_or_discontinuity'], row['reason_codes'])
        self.assertEqual('TURN', row['path']['terminal_surface'])
        self.assertEqual(1, len(row['path']['segments']))

    def test_incoming_blocker_in_legacy_report_requires_no_retrace(self):
        data = dataset()
        data['diagnostics']['reflections']['mirrors']['1_1_1'] = dict(status='infeasible',
            incoming=dict(status='available'), first_surface='BUNDLE_1_1 Mirror 2:back')
        row = diagnose(data)['observations'][0]
        self.assertEqual(['intercepted_before_mirror'], row['reason_codes'])
        self.assertIsNone(row['path'])
        self.assertIsNone(row['fitted_mirror']['incident_segment'])

    def test_no_optical_evidence_does_not_invent_obstruction(self):
        data = dataset()
        data['diagnostics']['reflections']['mirrors'] = {}
        row = diagnose(data)['observations'][0]
        self.assertEqual(['optical_evidence_missing'], row['reason_codes'])
        data['diagnostics'] = {}
        row = diagnose(data)['observations'][0]
        self.assertEqual('not_evaluated', row['status'])
        self.assertFalse(row['fit_only'])

    def test_incoming_and_coating_failures(self):
        for check, expected in [
            (dict(first_surface='<time_limit>'), 'incoming_trace_time_limit'),
            (dict(first_surface='<bundle_system_boundary>'), 'incoming_trace_boundary'),
            (dict(incoming=dict(status='outside_illumination')), 'outside_illumination'),
            (dict(reason='Incoming branch not present in physical scene'), 'incoming_branch_missing'),
            (dict(incident_dot_normal=1., actual_reflected_child=False), 'wrong_reflecting_side'),
            (dict(hit_on_ellipse=False, actual_reflected_child=False), 'outside_reflecting_ellipse'),
        ]:
            data = dataset()
            data['diagnostics']['reflections']['mirrors']['1_1_1'] = check
            self.assertIn(expected, diagnose(data)['observations'][0]['reason_codes'])

    def test_geometric_error_is_not_mislabelled_as_truncation(self):
        data = dataset()
        data['observations'][0]['screens_m'] = [[1., .1, 0.], [2., .1, 0.]]
        row = diagnose(data)['observations'][0]
        self.assertEqual(['geometric_mismatch', 'physical_path_mismatch'], row['reason_codes'])
        self.assertFalse(row['fit_only'])

    def test_assignment_conflict_identifies_measurement_using_path(self):
        data = dataset(end=3.)
        data['observations'].append(dict(id='duplicate', screens_m=[[1., 0., 0.], [2., 0., 0.]]))
        rows = diagnose(data)['observations']
        failed = next(r for r in rows if r['status'] == 'unmatched')
        passed = next(r for r in rows if r['status'] == 'matched')
        self.assertIn('one_to_one_conflict', failed['reason_codes'])
        self.assertEqual(passed['observation_id'], failed['physical']['assigned_observation_id'])

    def test_unrelated_nearest_path_is_not_blame_for_fit_mirror(self):
        data = dataset()
        reports = data['diagnostics']['reflections']['mirrors']
        reports['2_2_2'] = reports.pop('1_1_1')
        reports['1_1_1'] = dict(incoming=dict(status='outside_illumination'))
        row = diagnose(data)['observations'][0]
        self.assertEqual('2_2_2', row['physical']['mirror_id'])
        self.assertEqual(['outside_illumination'], row['reason_codes'])
        self.assertIsNone(row['path'])

    def test_partial_observations_and_transformed_endpoints(self):
        data = dataset()
        data['display_frame'] = dict(rotation=np.diag([-1., 1., 1.]).tolist(), translation_m=[5., 2., 0.])
        data['observations'] = [dict(id='a', screens_m=[[4., 2., 0.], [3., 2., 0.]]),
                                dict(id='partial', screens_m=[[1., None, 0.], [2., 0., 0.]])]
        data['diagnostics']['reflections']['mirrors']['1_1_1']['incident_segment'] = dict(
            start_m=[0., -1., 0.], end_m=[0., 0., 0.], surface='cut')
        rows = diagnose(data)['observations']
        self.assertEqual([3.5, 2., 0.], rows[0]['path']['end_m'])
        self.assertEqual([5., 2., 0.], rows[0]['fitted_mirror']['incident_segment']['end_m'])
        self.assertEqual('excluded', rows[1]['status'])


if __name__ == '__main__':
    unittest.main()
