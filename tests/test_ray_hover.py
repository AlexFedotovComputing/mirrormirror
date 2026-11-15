"""Ray labels and survey XYZ must survive grouping and display subsampling."""
import json
import unittest
from types import SimpleNamespace

import numpy as np

from mirrorlab.visualization import grouped_segments


def segment_block(ids, starts, ends):
    starts, ends = np.asarray(starts), np.asarray(ends)
    return dict(ray_id=ids, intensity=np.ones(len(ids)),
                **{axis + '0': starts[:, i] for i, axis in enumerate('xyz')},
                **{axis + '1': ends[:, i] for i, axis in enumerate('xyz')})


class RayHoverTests(unittest.TestCase):
    def test_survey_coordinates_and_direction_keep_individual_ray_identity(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        translation = np.array([10., 20., 30.])
        frame = SimpleNamespace(points=lambda points: np.asarray(points) @ rotation.T + translation)
        result = SimpleNamespace(segments=[segment_block(
            [21, 22], [[1., 2., 3.], [4., 5., 6.]], [[1., 3., 3.], [4., 5., 8.]])])
        trace, = grouped_segments(result, {21: '2_3_4', 22: '2_3_4'}, frame)
        self.assertEqual(trace['meta']['coordinate_frame'], 'survey')
        self.assertEqual(trace['meta']['bundle'], 3)
        self.assertEqual(len(trace['customdata']), len(trace['x']))
        np.testing.assert_allclose(trace['customdata'][0][:3], [8., 21., 33.])
        np.testing.assert_allclose(trace['customdata'][1][:3], [7., 21., 33.])
        np.testing.assert_allclose(trace['customdata'][0][8:], [-1., 0., 0.])
        np.testing.assert_allclose(trace['customdata'][3][8:], [0., 0., 1.])
        self.assertEqual(trace['customdata'][0][3:8], [21, '2_3_4', 2, 3, 4])
        self.assertEqual(trace['customdata'][3][3:8], [22, '2_3_4', 2, 3, 4])
        self.assertIsNone(trace['customdata'][2])
        self.assertIn('Z=%{customdata[2]', trace['hovertemplate'])
        json.dumps(trace, allow_nan=False)

    def test_subsampling_keeps_metadata_aligned_with_segments(self):
        starts = np.column_stack((np.arange(205), np.zeros(205), np.ones(205)))
        result = SimpleNamespace(segments=[segment_block(
            list(range(205)), starts, starts + [0, 0, 1])])
        trace, = grouped_segments(result, {i: '1_4_7' for i in range(205)}, None)
        self.assertEqual(trace['meta']['segment_count'], 205)
        self.assertLessEqual(trace['meta']['displayed_segments'], 100)
        for index in range(0, len(trace['x']), 3):
            row = trace['customdata'][index]
            self.assertEqual(row[0], row[3])
            self.assertEqual(row[0], trace['x'][index])
            self.assertEqual(row[2], trace['z'][index])
            self.assertEqual(row[4], '1_4_7')

    def test_common_and_zero_length_segments_have_finite_metadata(self):
        result = SimpleNamespace(segments=[segment_block([1], [[0., 0., 0.]], [[0., 0., 0.]])])
        trace, = grouped_segments(result, {}, None)
        self.assertEqual(trace['customdata'][0][3:8], [1, None, None, None, None])
        self.assertEqual(trace['customdata'][0][8:], [0., 0., 0.])
        self.assertEqual(trace['meta']['coordinate_frame'], 'legacy')
        json.dumps(trace, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
