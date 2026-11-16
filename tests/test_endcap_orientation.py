"""Drawing registration and physical coordinates survive endcap projection."""
import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from mirrorlab.endcap import drawing_xy, endcap_projection, experimental_points, mechanical_basis, roc_geometry


def landmarks():
    # A slanted periscope and asymmetric branch positions expose a wrong
    # rotation or a missed reflection.
    points = {
        'MP1': [-.35, -.8, 0.], 'MP2': [-.075, -.323686, 0.],
        'MS1': [-.93, .93, 0.], 'MS2': [-.93, -.93, 0.],
        'MS3': [.93, -.93, 0.], 'MS4': [.93, .93, 0.],
        'ML3': [-.215, .222, 0.], 'MR2': [.217, .222, 0.],
    }
    return {name: SimpleNamespace(name=name, center=np.asarray(point))
            for name, point in points.items()}


def frame_for(angle=0., parity=1, sign=1):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -parity*s, 0.], [s, parity*c, 0.], [0., 0., sign]])
    offset = np.array([.02, -.04, 0.])
    return SimpleNamespace(
        points=lambda values: np.asarray(values) @ rotation.T + offset,
        report={'survey_z_signs': {'injection': sign}})


class EndcapOrientationTests(unittest.TestCase):
    def test_measured_points_keep_survey_coordinates_missing_screens_and_model_links(self):
        frame = frame_for(.7, -1, -1)
        records = [dict(id='obs4', old_id='2_2_4', rod=2, window=2,
                        screens_m=[[.1, .2, -.6], [.3, .4, -.7]]),
                   dict(id='obs5', old_id='2_2_5', rod=2, window=2,
                        screens_m=[[None, .2, -.6], [.5, .6, -.7]])]
        config = SimpleNamespace(reconstruction=dict(observations={'records': records},
            mirrors=[dict(canonical_id='2_3_4', observation_ids=['obs4'])]))
        before = copy.deepcopy(config.reconstruction)
        axis = [.02, -.04, 0.]
        basis = mechanical_basis(landmarks(), frame)
        result = experimental_points(config, frame, axis, basis)
        self.assertEqual([len(t['x']) for t in result], [2, 1])
        self.assertEqual(result[0]['meta']['mirror_ids'], ['2_3_4'])
        self.assertEqual(result[1]['meta']['bundle'], 3)
        self.assertEqual(result[1]['meta']['mirror_ids'], [])
        self.assertEqual(result[1]['customdata'][0], [.5, .6, -.7, '2 2 5', 2])
        expected = drawing_xy(records[0]['screens_m'], axis, basis)*1000
        np.testing.assert_allclose(np.c_[result[0]['x'], result[0]['y']], expected)
        self.assertEqual(config.reconstruction, before)

    def test_camera_is_outside_each_half_and_numbering_cannot_flip_the_view(self):
        surfaces = landmarks()
        source_points = np.array([surfaces[name].center for name in surfaces])
        base_frame = frame_for()
        baseline = drawing_xy(base_frame.points(source_points), base_frame.points([0., 0., 0.]),
                              mechanical_basis(surfaces, base_frame))
        for angle in (0., .7, -2.1):
            for parity in (-1, 1):
                for sign in (-1, 1):
                    with self.subTest(angle=angle, parity=parity, sign=sign):
                        frame = frame_for(angle, parity, sign)
                        basis = mechanical_basis(surfaces, frame)
                        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-14)
                        self.assertAlmostEqual(np.linalg.det(basis), sign)
                        positions = drawing_xy(frame.points(source_points), frame.points([0., 0., 0.]), basis)
                        np.testing.assert_allclose(positions, baseline*[parity*sign, 1], atol=1e-14)
                        named = dict(zip(surfaces, positions))
                        pipe = named['MP1'] - named['MP2']
                        self.assertAlmostEqual(pipe[0], 0.)
                        self.assertLess(pipe[1], 0.)
                        swapped = {**surfaces, 'MS1': surfaces['MS4'], 'MS4': surfaces['MS1']}
                        np.testing.assert_array_equal(mechanical_basis(swapped, frame), basis)

    def test_degenerate_periscope_has_no_invented_orientation(self):
        surfaces = landmarks()
        surfaces['MP1'].center = surfaces['MP2'].center + [0., 0., 1.]
        with self.assertRaisesRegex(ValueError, 'MP2'):
            mechanical_basis(surfaces, frame_for())

    def test_clipping_preserves_survey_z_and_identity_on_each_half(self):
        surfaces = landmarks()
        trace = dict(
            name='Бандл 2_3 · зеркало 4', line={'color': '#123456'},
            meta={'mirror_id': '2_3_4', 'rod': 2, 'bundle': 3, 'mirror': 4},
            x=[0., 6., None], y=[1., 4., None], z=[-3., 3., None],
            customdata=[[0., 1., -3., 41, '2_3_4', 2, 3, 4, 2/3, 1/3, 2/3],
                        [6., 4., 3., 41, '2_3_4', 2, 3, 4, 2/3, 1/3, 2/3], None],
            hovertemplate='Z=%{customdata[2]}')
        original = copy.deepcopy(trace)
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                frame = frame_for(.8, -1, sign)
                config = SimpleNamespace(
                    scene=SimpleNamespace(build_surfaces=lambda: list(surfaces.values())),
                    tracer_options={'bundle_clip_z_min_m': -2., 'bundle_clip_z_max_m': 2.},
                    metadata={'ring_side': 'EAST' if sign == 1 else 'WEST',
                              'injection_side': 'WEST' if sign == 1 else 'EAST'})
                result = endcap_projection(config, frame, [trace])
                projected, = result['rays']
                expected = ([[3., 2.5, 0.], [5., 3.5, 2.]] if sign == 1
                            else [[1., 1.5, -2.], [3., 2.5, 0.]])
                np.testing.assert_allclose([row[:3] for row in projected['customdata'][:2]], expected)
                for row in projected['customdata'][:2]:
                    self.assertEqual(row[3:8], [41, '2_3_4', 2, 3, 4])
                    np.testing.assert_allclose(row[8:], [2/3, 1/3, 2/3])
                self.assertIsNone(projected['customdata'][2])
                self.assertEqual(projected['meta']['mirror_id'], '2_3_4')
                uv = drawing_xy(expected, result['axis_m'], result['basis_survey'])*1000
                np.testing.assert_allclose(projected['x'][:2], uv[:, 0])
                np.testing.assert_allclose(projected['y'][:2], uv[:, 1])
                self.assertEqual(result['plane_z_m'], sign*2.)
                self.assertEqual(result['view_from_side'], config.metadata['injection_side'])
                self.assertEqual(result['viewing_direction_survey'], [0., 0., -sign])
                self.assertGreater(sign*result['camera_position_m'][2], sign*result['plane_z_m'])
                right, up = np.asarray(result['basis_survey']).T
                np.testing.assert_allclose(np.cross(np.r_[right, 0.], np.r_[up, 0.]), [0., 0., sign], atol=1e-14)
                json.dumps(result, allow_nan=False)
        self.assertEqual(trace, original)

    def test_roc_top_and_bottom_are_centered_on_screen_vertical(self):
        outlines = roc_geometry(1)['outlines']
        self.assertEqual(len(outlines), 12)
        for index, sign in ((3, 1), (9, -1)):
            outline = outlines[index]
            self.assertAlmostEqual(np.mean(outline['x'][:4]), 0.)
            self.assertGreater(sign*np.mean(outline['y'][:4]), 0.)
            self.assertLess(min(outline['x']), 0.)
            self.assertGreater(max(outline['x']), 0.)


if __name__ == '__main__':
    unittest.main()
