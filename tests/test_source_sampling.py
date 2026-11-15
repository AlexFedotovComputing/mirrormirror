"""Area conservation and spatial coverage of source-ray quadrature."""
import unittest

import numpy as np

from mirrorlab.engine.source import (
    GaussianBeamSource, _concentric_disk, _stratified_square_samples,
    stratified_disk_samples,
)


class SourceSamplingTests(unittest.TestCase):
    def test_every_equal_area_cell_has_exactly_one_point(self):
        for count in (1, 2, 3, 19, 100, 217, 1000):
            with self.subTest(count=count):
                points = _stratified_square_samples(count)
                rows = max(1, round(np.sqrt(count)))
                edges = np.arange(rows + 1) * count // rows
                occupied = []
                for start, end in zip(edges[:-1], edges[1:]):
                    width, height = 1 / (end-start), (end-start) / count
                    self.assertAlmostEqual(width * height, 1 / count)
                    row = points[(points[:, 1] >= start/count) & (points[:, 1] < end/count)]
                    occupied.extend(np.floor(row[:, 0] / width).astype(int) + start)
                self.assertEqual(sorted(occupied), list(range(count)))

    def test_disk_mapping_preserves_area_and_handles_origin(self):
        np.testing.assert_array_equal(_concentric_disk(np.array([[.5, .5]])), [[0., 0.]])
        # Numerical Jacobian in every wedge, away from derivative boundaries.
        points = np.array([[.8,.6], [.6,.8], [.2,.6], [.4,.8],
                           [.2,.4], [.4,.2], [.8,.4], [.6,.2]])
        step = 1e-6
        dx = (_concentric_disk(points+[step,0])-_concentric_disk(points-[step,0]))/(2*step)
        dy = (_concentric_disk(points+[0,step])-_concentric_disk(points-[0,step]))/(2*step)
        jacobian = dx[:,0]*dy[:,1]-dx[:,1]*dy[:,0]
        np.testing.assert_allclose(jacobian, np.pi, rtol=1e-8)

    def test_disk_coverage_is_uniform_in_area_and_azimuth(self):
        points = stratified_disk_samples(10000)
        radius_squared = np.sum(points**2, axis=1)
        self.assertTrue(np.all(radius_squared < 1))
        radial_counts = np.histogram(radius_squared, bins=np.linspace(0, 1, 21))[0]
        self.assertLess(np.max(np.abs(radial_counts-500)), 60)
        angles = np.arctan2(points[:,1], points[:,0])
        angular_counts = np.histogram(angles, bins=np.linspace(-np.pi, np.pi, 17))[0]
        self.assertLess(np.max(np.abs(angular_counts-625)), 60)
        np.testing.assert_allclose(np.mean(points, axis=0), [0,0], atol=.005)
        np.testing.assert_allclose(np.mean(points**2, axis=0), [.25,.25], atol=.005)

    def test_sampling_reproducible_and_not_affected_by_global_rng(self):
        first = stratified_disk_samples(217)
        np.random.seed(987)
        np.testing.assert_array_equal(first, stratified_disk_samples(217))
        self.assertFalse(np.array_equal(first, stratified_disk_samples(217, seed=1)))
        self.assertEqual(len(np.unique(first, axis=0)), 217)

    def test_gaussian_weights_and_transverse_plane_for_tilted_source(self):
        source = GaussianBeamSource((2.,3.,4.), (1.,2.,3.), .01, 1e6, 266e-9)
        rays = source.emit(1000)
        relative = rays.position - source.waist_position
        axis = np.array(source.axis) / np.linalg.norm(source.axis)
        np.testing.assert_allclose(relative @ axis, 0, atol=1e-14)
        expected = source.peak_intensity*np.exp(-2*np.sum(relative**2,axis=1)/source.waist_radius**2)
        np.testing.assert_allclose(rays.intensity, expected)
        np.testing.assert_allclose(rays.power/rays.intensity, np.repeat(rays.power[0]/rays.intensity[0], 1000))
        power = source.peak_intensity*np.pi*source.waist_radius**2/2*(-np.expm1(-2))
        self.assertAlmostEqual(float(rays.power.sum()), power, places=10)


if __name__ == '__main__':
    unittest.main()
