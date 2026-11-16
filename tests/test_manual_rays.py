import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from mirrorlab.calibration.manual_rays import current_analysis, one_to_one_aperture, manual_geometry_on_base
from mirrorlab.service import dispatch, document_result
from mirrorlab.state import load_document, save_document


class ManualRaysTests(unittest.TestCase):
    def test_manual_overlay_preserves_base_optics_and_survey_pose(self):
        from mirrorlab.service import _historical_document
        from mirrorlab.manual_schema import import_manual, BUNDLE
        from mirrorlab.state import validate_document, new_document, configuration_from_dataset
        from mirrorlab.engine.bodies import MirrorBody
        base = _historical_document({'direction': 'east_to_west', 'kind': 'real'})['datasets']['east_to_west']
        manual = import_manual(BUNDLE / 'west_manual_corrected_survey.json')['datasets']['east_to_west']
        original = copy.deepcopy(base)
        # Exercise a distinct frame, including a handedness change.
        base['display_frame']['rotation'] = np.eye(3).tolist()
        base['display_frame']['translation_m'] = [1., 2., 3.]
        merged = manual_geometry_on_base(base, manual)
        self.assertEqual(merged['optics_snapshot'], base['optics_snapshot'])
        self.assertEqual(merged['display_frame'], base['display_frame'])
        q = np.asarray(manual['display_frame']['rotation'])
        t = np.asarray(manual['display_frame']['translation_m'])
        raw = manual['manual_references']['payload']['mirrors'][0]['recommended_optimization_seed']['body_native']
        expected = q @ np.asarray(raw['center_m']) + t
        actual = np.asarray(merged['mirrors'][0]['center_m']) + [1., 2., 3.]
        np.testing.assert_allclose(actual, expected)
        body = MirrorBody.from_dict(merged['mirrors'][0])
        np.testing.assert_allclose(body.normal, q @ MirrorBody.from_dict(raw).normal)
        validate_document(new_document({'east_to_west': merged}))
        config = configuration_from_dataset(merged)
        self.assertEqual(config.reconstruction['mirrors'], merged['mirrors'])
        self.assertEqual(original['mirrors'], base['mirrors'])

    @classmethod
    def setUpClass(cls):
        # This suite reproduces the archived WEST optical audit explicitly.
        policy = patch('mirrorlab.service.ENABLED_DIRECTIONS', ('east_to_west', 'west_to_east'))
        policy.start()
        cls.addClassCleanup(policy.stop)
        cls.result = dispatch('add_manual_rays', {'direction': 'east_to_west', 'kind': 'real'})

    def test_full_scene_reproduces_audit_with_unique_physical_mirrors(self):
        d = self.result['document']['datasets']['east_to_west']
        a = d['manual_rays']
        self.assertEqual((a['baseline_count'], a['matched_count'], a['observation_count']), (54, 80, 83))
        self.assertEqual(len({m['mirror_id'] for m in a['matches']}), 80)
        self.assertEqual(len(a['full_trace']['paths']), 90)
        self.assertIsNone(a['power_fraction'])
        self.assertFalse(self.result['can_activate'])
        self.assertTrue(all(p['length_m'] > 0 for p in a['full_trace']['paths']))
        json.dumps(self.result, allow_nan=False)

    def test_persistence_and_repeated_action_do_not_duplicate_paths_or_change_base(self):
        original = copy.deepcopy(self.result['document'])
        with TemporaryDirectory() as directory:
            p = Path(directory) / 'manual.json'
            save_document(original, p)
            loaded = dispatch('load_configuration', {'path': str(p)})
            self.assertEqual(loaded['document']['datasets']['east_to_west']['manual_rays'],
                             original['datasets']['east_to_west']['manual_rays'])
            again = dispatch('add_manual_rays', {'direction': 'east_to_west',
                                                'candidate_configuration': load_document(p)})
        self.assertEqual(original, self.result['document'])
        d = again['document']['datasets']['east_to_west']
        self.assertEqual(d['manual_rays'], original['datasets']['east_to_west']['manual_rays'])
        traces = [t for t in again['figure']['data'] if t.get('meta', {}).get('kind') == 'manual_aperture_ray']
        self.assertEqual(len(traces), 90)
        self.assertEqual({t['meta']['direction'] for t in traces}, {'east_to_west'})

    def test_changed_geometry_invalidates_saved_paths(self):
        doc = copy.deepcopy(self.result['document'])
        d = doc['datasets']['east_to_west']
        d['mirrors'][0]['center_m'][0] += .0001
        self.assertIsNone(current_analysis(d))
        result = document_result(doc)
        self.assertNotIn('manual_rays', result['document']['datasets']['east_to_west'])
        self.assertFalse(any(t.get('meta', {}).get('kind') == 'manual_aperture_ray' for t in result['figure']['data']))

    def test_many_samples_from_one_mirror_do_not_inflate_channel_count(self):
        comparison = dict(trajectory_diagnostics=dict(observations=[dict(status='unmatched',
            observation_id=str(i), screens_m=[[1, 0, 0], [2, 0, 0]]) for i in range(2)]),
            trajectory_matching=dict(trajectories=[]))
        paths = [dict(id=str(i), mirror_id='same', point_m=[0, 0, 0], direction=[1, 0, 0], length_m=3.) for i in range(2)]
        self.assertEqual(one_to_one_aperture(comparison, paths)['0.003']['count'], 1)
        self.assertEqual(one_to_one_aperture(comparison, [])['0.003']['count'], 0)


if __name__ == '__main__':
    unittest.main()
