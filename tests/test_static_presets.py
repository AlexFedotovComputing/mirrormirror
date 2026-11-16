import unittest
from unittest.mock import patch

import numpy as np

from mirrorlab.configurations import load_configuration
from mirrorlab.service import dispatch, static_manual_document
from mirrorlab.state import configuration_from_dataset, digest, validate_document


class StaticPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manual = static_manual_document('both')

    def test_manual_preset_contains_supplied_mirrors_and_matching_optics(self):
        validate_document(self.manual)
        for direction, dataset in self.manual['datasets'].items():
            self.assertEqual(len(dataset['mirrors']), 112)
            self.assertNotIn('manual_rays', dataset)
            raw = dataset['manual_references']['payload']
            if direction == 'east_to_west':
                optics = raw['recommended_optimization_candidate']['calibrated_optics_snapshot']
                bodies = {m['source_manual_id']: m['recommended_optimization_seed']['body_native']
                          for m in raw['mirrors']}
            else:
                optics = raw['optics_snapshot']
                bodies = {m['source_id']: m['cylinder'] for m in raw['mirrors']}
            self.assertEqual(dataset['ring_alignment_previous_optics'], optics)
            self.assertEqual(dataset['optics_sha256'], digest(dataset['optics_snapshot']))
            self.assertTrue(dataset['optics_snapshot']['metadata']['ring_alignment'])
            for mirror in dataset['mirrors']:
                supplied = bodies[mirror['source_manual_id']]
                np.testing.assert_allclose(mirror['center_m'], supplied['center_m'], rtol=0, atol=1e-12)
                np.testing.assert_allclose(mirror['rotation'], supplied['rotation'], rtol=0, atol=1e-12)

    def test_real_trace_uses_manual_preset_and_requested_ray_count(self):
        with patch('mirrorlab.state.configuration_from_dataset', wraps=configuration_from_dataset) as build, \
                patch('mirrorlab.calibration.manual_rays.add_manual_rays', side_effect=AssertionError('aperture audit')):
            result = dispatch('trace_static', dict(direction='west_to_east', kind='real',
                ray_count=37, active_configuration={'invalid': 'must not be used'}))
        used = build.call_args.args[0]
        expected = self.manual['datasets']['west_to_east']
        self.assertEqual(used['mirrors'], expected['mirrors'])
        self.assertEqual(used['optics_snapshot'], expected['optics_snapshot'])
        self.assertEqual(result['summary']['ray_count'], 37)
        self.assertGreater(result['summary']['segment_count'], 0)
        self.assertEqual(result['metadata']['static_configuration'], 'manual')

    def test_ideal_trace_ignores_previously_loaded_manual_geometry(self):
        with patch('mirrorlab.service.load_configuration', wraps=load_configuration) as load, \
                patch('mirrorlab.state.configuration_from_dataset', side_effect=AssertionError('stale candidate')):
            result = dispatch('trace_static', dict(direction='west_to_east', kind='ideal',
                ray_count=7, active_configuration=self.manual, candidate_configuration=self.manual))
        load.assert_called_once_with('west_to_east', 'ideal')
        self.assertEqual(result['summary']['ray_count'], 7)
        self.assertEqual(result['metadata']['static_configuration'], 'ideal')


if __name__ == '__main__':
    unittest.main()
