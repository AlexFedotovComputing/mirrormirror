import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from mirrorlab.configurations import StaticElement
from mirrorlab.engine import RayBundle, RayTracer, Scene
from mirrorlab.engine.bodies import MirrorBody


class BodyBroadPhaseTests(unittest.TestCase):
    def test_exact_body_bounds_preserve_all_surface_hits_and_daughters(self):
        bodies = [MirrorBody([x, 0, 0], Rotation.from_rotvec([.2, -.4, .1]).as_matrix(),
                             mirror_id=f"{i+1}_1_1") for i, x in enumerate((-.01, 0., .01))]
        scene = Scene([StaticElement(b.mirror_id, b.build_surfaces()) for b in bodies])
        xy = np.meshgrid(np.linspace(-.012, .012, 81), np.linspace(-.001, .001, 9))
        p = np.c_[xy[0].ravel(), xy[1].ravel(), np.full(xy[0].size, -.005)]
        rays = RayBundle(p, np.tile([0., 0., 1.], (len(p), 1)), np.ones(len(p)),
                         np.full(len(p), 266e-9), np.ones(len(p)))
        fast = RayTracer(scene, max_interactions=5, max_time_s=1e-9, collision_mode="expected",
                         skip_repeated_bundle_reflections=True)
        slow = RayTracer(scene, max_interactions=5, max_time_s=1e-9)
        a, b = fast.trace(rays), slow.trace(rays)
        self.assertEqual(len(a.segments), len(b.segments))
        for first, second in zip(a.segments, b.segments):
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
        self.assertLess(fast.compiled_scene.intersection_tests, slow.compiled_scene.intersection_tests)
        self.assertTrue(any(np.any(s["bundle_reflected"]) for s in a.segments))


if __name__ == "__main__":
    unittest.main()
