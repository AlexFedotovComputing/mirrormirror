"""Drawing registration and physical coordinates survive endcap projection."""
import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from mirrorlab.endcap import drawing_xy, endcap_projection, mechanical_basis, roc_geometry


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
    def test_rotations_and_reflections_preserve_the_drawing(self):
        surfaces = landmarks()
        source_points = np.array([surfaces[name].center for name in surfaces])
        baseline = None
        for angle in (0., .7, -2.1):
            for parity in (-1, 1):
                with self.subTest(angle=angle, parity=parity):
                    frame = frame_for(angle, parity)
                    basis = mechanical_basis(surfaces, frame)
                    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-14)
                    positions = drawing_xy(frame.points(source_points), frame.points([0., 0., 0.]), basis)
                    if baseline is None:
                        baseline = positions
                    np.testing.assert_allclose(positions, baseline, atol=1e-14)
                    named = dict(zip(surfaces, positions))
                    pipe = named['MP1'] - named['MP2']
                    self.assertAlmostEqual(pipe[0], 0.)
                    self.assertLess(pipe[1], 0.)
                    self.assertLess(named['MS1'][0], 0.)
                    self.assertGreater(named['MS4'][0], 0.)

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
                    metadata={'ring_side': 'EAST'})
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
