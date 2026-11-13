import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import plotly.graph_objects as go

from mirrorlab.manual_schema import import_manual
from mirrorlab.service import dispatch, document_result
from mirrorlab.state import load_document, save_document, validate_document

ROOT = Path(__file__).resolve().parents[1]


class ManualServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = import_manual(ROOT / 'data/manual_refinement/east_recovered_centers.json')

    def test_saved_preview_is_self_contained_and_never_activates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'east.json'
            save_document(self.document, path)
            with patch('mirrorlab.configurations.load_configuration', side_effect=AssertionError('repository geometry')):
                result = dispatch('load_configuration', {'path': str(path)})
                go.Figure(result['figure'])
            self.assertEqual(self.document, load_document(path))
        self.assertFalse(result['can_activate'])
        kinds = {t.get('meta', {}).get('kind') for t in result['figure']['data']}
        self.assertTrue({'manual_reference', 'manual_primary'} <= kinds)

    def test_raw_tampering_is_detected(self):
        for field in ('raw_utf8', 'payload'):
            doc = copy.deepcopy(self.document)
            ref = doc['datasets']['west_to_east']['manual_references']
            if field == 'raw_utf8':
                ref[field] += ' '
            else:
                ref[field]['units'] = 'mm'
            with self.assertRaisesRegex(ValueError, 'Manual source'):
                validate_document(doc)

    def test_optimize_routes_candidate_without_mutating_accepted_input(self):
        original = copy.deepcopy(self.document)
        with patch('mirrorlab.calibration.manual_refinement.refine_manual', side_effect=lambda d, **kw: d) as refine:
            result = dispatch('optimize', {'candidate_configuration': self.document,
                'kind': 'ideal', 'direction': 'both', 'configuration_version': 17})
        self.assertEqual(refine.call_count, 1)
        self.assertEqual(result['document']['configuration_version'], 17)
        self.assertEqual(original, self.document)
        self.assertFalse(result['can_activate'])


if __name__ == '__main__':
    unittest.main()
