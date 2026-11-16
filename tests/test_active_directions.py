"""The temporary WEST lock must also cover old documents and non-UI calls."""
import copy
import unittest
from unittest.mock import patch

from mirrorlab.configurations import load_configuration
from mirrorlab.legacy_import import historical_dataset
from mirrorlab.service import (
    TraceRequest, catalog, dispatch, document_result, refine_manual_configuration,
    selected_directions,
)
from mirrorlab.state import new_document


class ActiveDirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = new_document({
            d: historical_dataset(load_configuration(d))
            for d in ('west_to_east', 'east_to_west')
        })

    def test_catalog_and_default_offer_only_east(self):
        result = catalog()
        self.assertEqual([d['id'] for d in result['directions']], ['west_to_east'])
        self.assertEqual(result['defaults']['direction'], 'west_to_east')
        self.assertEqual(selected_directions('both'), ('west_to_east',))
        self.assertEqual([d['direction'] for d in result['additional_screens']], ['west_to_east'])

    def test_explicit_west_is_rejected_before_any_operation(self):
        with self.assertRaisesRegex(ValueError, 'временно отключён'):
            TraceRequest(direction='east_to_west')
        for method in ('trace', 'trace_static', 'calibrate', 'optimize', 'refine_manual',
                       'add_manual_rays', 'import_legacy', 'import_manual', 'save_configuration'):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'временно отключён'):
                dispatch(method, {'direction': 'east_to_west'})

    def test_old_both_document_preserves_west_without_rendering_or_analyzing_it(self):
        before = copy.deepcopy(self.document)
        from mirrorlab.calibration.trajectory_matching import dataset_comparison_report
        with patch('mirrorlab.calibration.trajectory_matching.dataset_comparison_report',
                   wraps=dataset_comparison_report) as compare:
            result = document_result(self.document)
        self.assertEqual(self.document, before)
        self.assertEqual(result['document']['datasets']['east_to_west'], before['datasets']['east_to_west'])
        self.assertEqual(result['enabled_directions'], ['west_to_east'])
        self.assertEqual(result['scene_direction'], 'west_to_east')
        self.assertEqual([d['direction'] for d in result['reports']], ['west_to_east'])
        self.assertEqual([call.args[0]['direction'] for call in compare.call_args_list], ['west_to_east'])
        self.assertTrue(all(t.get('meta', {}).get('direction') != 'east_to_west'
                            for t in result['figure']['data']))

    def test_west_only_document_is_unavailable(self):
        document = new_document({'east_to_west': self.document['datasets']['east_to_west']})
        with self.assertRaisesRegex(ValueError, 'временно отключён'):
            document_result(document)

    def test_both_trace_contains_only_east(self):
        result = dispatch('trace', {'direction': 'both', 'ray_count': 7, 'cpu_workers': 1,
                                    'max_interactions': 1, 'active_configuration': self.document})
        self.assertEqual([s['direction'] for s in result['sides']], ['west_to_east'])
        self.assertEqual(result['summary']['ray_count'], 7)
        self.assertEqual([s['direction'] for s in result['comparison_reports']], ['west_to_east'])

    def test_manual_refinement_of_both_never_passes_west_to_optimizer(self):
        # Restrict this check to service routing; the manual optimizer has its
        # own physical validation suite and must not be invoked for WEST.
        document = copy.deepcopy(self.document)
        for dataset in document['datasets'].values():
            dataset['manual_references'] = {'test': True}
        with patch('mirrorlab.state.validate_document'), \
                patch('mirrorlab.calibration.manual_refinement.refine_manual', side_effect=lambda d, **k: d) as refine, \
                patch('mirrorlab.service.document_result', side_effect=lambda d: d):
            result = refine_manual_configuration({'direction': 'both', 'document': document})
        self.assertEqual([call.args[0]['direction'] for call in refine.call_args_list], ['west_to_east'])
        self.assertEqual(result['datasets']['east_to_west'], document['datasets']['east_to_west'])


if __name__ == '__main__':
    unittest.main()
