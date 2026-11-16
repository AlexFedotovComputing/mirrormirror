import json
import unittest
from types import SimpleNamespace

import numpy as np

from mirrorlab.visualization import spot_metrics, field_data, unwrap, ray_origins
from mirrorlab.service import dispatch
from mirrorlab.endcap import clip_half_segment, outer_xy, roc_geometry


class VisualizationTests(unittest.TestCase):
    def test_roc_frames_keep_tdr_dimensions_on_both_outer_views(self):
        west, east = roc_geometry(1)['outlines'], roc_geometry(-1)['outlines']
        self.assertEqual(len(west), 12)
        for w, e in zip(west, east):
            np.testing.assert_allclose(w['x'], -np.asarray(e['x']))
            np.testing.assert_allclose(w['y'], e['y'])
            p = np.column_stack((w['x'],w['y']))
            self.assertAlmostEqual(np.linalg.norm(p[1]-p[2]),642)
            self.assertAlmostEqual(np.linalg.norm(p[3]-p[0]),213)
            self.assertAlmostEqual(np.linalg.norm((p[1]+p[2]-p[0]-p[3])/2),800)

    def test_outer_views_have_opposite_horizontal_axes_and_same_up(self):
        points = np.array([[.8, .9, 1.], [-.4, .2, -1.]])
        axis = [.1, -.1, 0.]
        west = outer_xy(points, 1, axis)
        east = outer_xy(points, -1, axis)
        np.testing.assert_allclose(west[:, 0], -east[:, 0])
        np.testing.assert_allclose(west[:, 1], east[:, 1])
        np.testing.assert_allclose(west[0], [.7, 1.])

    def test_projected_paths_clip_at_cathode_and_endcap_without_extension(self):
        for sign in (-1, 1):
            a, b = clip_half_segment([0, 0, -2*sign], [4, 0, 2*sign], sign, 1.5)
            np.testing.assert_allclose(a, [2, 0, 0])
            np.testing.assert_allclose(b, [3.5, 0, 1.5*sign])
            self.assertIsNone(clip_half_segment([0, 0, -2*sign], [1, 0, -sign], sign, 1.5))
            self.assertIsNone(clip_half_segment([0, 0, 2*sign], [1, 0, 2*sign], sign, 1.5))
            a, b = clip_half_segment([0, 0, sign], [1, 0, sign], sign, 1.5)
            np.testing.assert_allclose([a, b], [[0, 0, sign], [1, 0, sign]])

    def test_ellipse_metrics_and_degenerate_samples(self):
        angles = np.linspace(0, 2*np.pi, 400, endpoint=False)
        uv = np.column_stack((.002*np.cos(angles), .001*np.sin(angles)))
        points = np.column_stack((uv, np.ones(len(uv))))
        metrics = spot_metrics(uv, points, np.ones(400), np.full(400, .01))
        self.assertAlmostEqual(metrics['diameter_m'], .003)
        self.assertAlmostEqual(metrics['ellipticity'], .5)
        self.assertAlmostEqual(metrics['mean_irradiance_w_m2'], 4/(np.pi*.002*.001))
        np.testing.assert_allclose(metrics['center_m'], [0, 0, 1], atol=1e-15)
        for n in (0, 1):
            m = spot_metrics(uv[:n], points[:n], np.ones(n), np.ones(n))
            self.assertIsNone(m['diameter_m'])
            json.dumps(m, allow_nan=False)

    def test_field_keeps_empty_space_and_constant_intensity(self):
        field = field_data([[0,0], [.001, .001]], np.array([42.,42.]),
                           bounds=[[-.01,-.01],[.01,.01]])
        self.assertTrue(any(v is None for row in field['z'] for v in row))
        np.testing.assert_allclose([v for row in field['z'] for v in row if v is not None], 42.)
        json.dumps(field, allow_nan=False)

    def test_cylinder_seam_does_not_inflate_spot(self):
        surface = SimpleNamespace(center=np.zeros(3), axis=[0,0,1], radius=1.)
        angles = np.array([np.pi-.001, -np.pi+.001])
        p = np.column_stack((np.cos(angles),np.sin(angles),np.zeros(2)))
        uv = unwrap(p, surface)
        self.assertAlmostEqual(np.ptp(uv[:,0]), .002)

    def test_ancestry_labels_reflected_branch_only(self):
        def block(ids, parents, surfaces, flags):
            return dict(ray_id=ids, parent_id=parents, surface=surfaces, bundle_reflected=flags)
        result = SimpleNamespace(segments=[
            block([1],[-1],['BUNDLE_2_3 Mirror 4'],[False]),
            block([2,3],[1,1],['screen_b_2_3','Screen 1'],[True,False]),
            block([4],[2],['<boundary>'],[True])])
        self.assertEqual(ray_origins(result), {1:None, 2:'2_3_4', 3:None, 4:'2_3_4'})

    def test_presets_report_all_mirrors_and_conserve_camera_power(self):
        for kind in ('ideal', 'real'):
            result = dispatch('trace_static', dict(direction='both',kind=kind,ray_count=217,cpu_workers=1))
            json.dumps(result, allow_nan=False)
            self.assertEqual(len(result['visualization']['sides']),1)
            self.assertEqual(result['visualization']['sides'][0]['side'], 'EAST')
            for side, original in zip(result['visualization']['sides'],result['sides']):
                self.assertEqual(len(side['gas_rows']),112)
                self.assertEqual(len(side['cylinders']),16)
                self.assertEqual(len(side['cameras']),4)
                self.assertEqual(side['source']['ray_count'],217)
                endcap = side['endcap']
                self.assertTrue(endcap['available'])
                self.assertEqual(endcap['sign'], 1 if side['side'] == 'WEST' else -1)
                self.assertAlmostEqual(abs(endcap['plane_z_m']), 1.665)
                self.assertEqual(len(endcap['outlines']), 12)
                self.assertTrue(endcap['rays'])
                for camera in side['cameras']:
                    expected=original['summary']['detector_power_w'].get(camera['name'],0)
                    self.assertAlmostEqual(camera['metrics']['power_w'],expected)
                self.assertTrue(any(t['meta']['mirror_id'] for t in side['scene_rays']))
                self.assertTrue(any(row['ray_count'] for row in side['gas_rows']))


if __name__ == '__main__':
    unittest.main()
