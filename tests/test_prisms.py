"""Prism closure, cap intersections and the geometry sent to Plotly."""
from collections import Counter
import unittest

import numpy as np

from mirrorlab.configurations import load_configuration
from mirrorlab.engine import RayBundle, Scene, TriangularPrism
from mirrorlab.plotting import _surface_traces


def prism(normal=(0., 0., 1.)):
    return TriangularPrism("prism", (1., 2., 3.), normal, (0., 1., 0.),
                           [(-.026, -.013), (.026, -.013), (0., .026)],
                           .05, 1.5, reflectance=0., transmittance=1.)


class PrismTests(unittest.TestCase):
    def assert_closed(self, surfaces):
        edges = Counter()
        vertices = set()
        for surface in surfaces:
            outline, mesh = _surface_traces(surface)
            points = np.column_stack([outline[a] for a in "xyz"])
            np.testing.assert_allclose(points, np.column_stack([mesh[a] for a in "xyz"]))
            face = [tuple(point) for point in np.round(points, 9)]
            vertices.update(face)
            for start, end in zip(face, face[1:]):
                edges[tuple(sorted((start, end)))] += 1
        self.assertEqual(len(surfaces), 5)
        self.assertEqual(len(vertices), 6)
        self.assertEqual(len(edges), 9)
        self.assertEqual(set(edges.values()), {2}, "Each prism edge must join two faces")

    def test_prism_builder_joins_all_five_faces(self):
        for normal in ((0., 0., 1.), (1., -2., 3.), (0., 0., -1.)):
            with self.subTest(normal=normal):
                self.assert_closed(prism(normal).build_surfaces())

    def test_cap_intersections_match_from_both_ends(self):
        for normal in ((0., 0., 1.), (1., -2., 3.), (0., 0., -1.)):
            element = prism(normal)
            bottom, top, *_ = element.build_surfaces()
            # Points near each vertex exercise areas lost by mirroring a cap;
            # the last point is outside, in the formerly spurious cap area.
            local = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]]) @ np.array(element.vertices_2d)
            local = np.vstack((local, [0., -.024]))
            footprint = local[:, :1] * bottom._u + local[:, 1:] * bottom._v
            for sign, entry, exit_cap in ((1, bottom, top), (-1, top, bottom)):
                direction = sign * bottom.normal
                rays = RayBundle(entry.center + footprint - .01 * direction,
                                 np.tile(direction, (4, 1)), np.ones(4), np.full(4, 266e-9), np.ones(4))
                with self.subTest(normal=normal, sign=sign):
                    for cap in (entry, exit_cap):
                        hit = cap.intersect(rays)
                        np.testing.assert_array_equal(hit.valid, [True, True, True, False])
                    # Inside the solid, both search modes must find the exit cap.
                    rays.position += .02 * direction
                    compiled = Scene().add(element).compile()
                    for mode in ("exhaustive", "expected"):
                        hit = compiled.find_nearest(rays, collision_mode=mode)
                        names = [compiled.surfaces[i].name for i in hit.surface_index[:3]]
                        self.assertEqual(names, [exit_cap.name] * 3)

    def test_all_static_prisms_are_closed_in_plotly(self):
        for direction in ("west_to_east", "east_to_west"):
            for kind in ("ideal", "real"):
                config = load_configuration(direction, kind)
                prisms = [e for e in config.scene.elements
                          if any(getattr(s, "shape", None) == "triangle" for s in e.surfaces)]
                self.assertEqual(len(prisms), 5)
                for element in prisms:
                    with self.subTest(direction=direction, kind=kind, prism=element.name):
                        self.assert_closed(element.surfaces)


if __name__ == "__main__":
    unittest.main()
