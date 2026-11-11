"""One physical model of a 45-degree cut cylindrical mirror, in metres.

Local solid: x*x+y*y <= r*r, x <= z <= h.  ``center_m`` always
means the centre of the reflecting ellipse, never the centre of mass.
The default 10 mm length is the full axial extent (h+r); this is an
explicit working convention, not a dimension verified against a drawing.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import numpy as np

from .backend import get_backend, to_numpy
from .materials import InteractionMode, SurfaceOptics
from .rays import Intersection
from .surfaces import Surface

LOCAL_NORMAL = np.array([1., 0., -1.]) / math.sqrt(2.)


@dataclass
class MirrorBody:
    center_m: object
    rotation: object
    mirror_id: str = ""
    radius_m: float = .0005
    axial_height_m: float = .0095
    length_definition: str = "full_extent"

    def __post_init__(self):
        self.center_m = np.asarray(self.center_m, dtype=float)
        self.rotation = np.asarray(self.rotation, dtype=float)
        if self.center_m.shape != (3,) or self.rotation.shape != (3, 3):
            raise ValueError("Mirror centre must be (3,), rotation must be (3,3)")
        if not np.isfinite(self.center_m).all() or not np.isfinite(self.rotation).all():
            raise ValueError("Mirror pose contains nonfinite values")
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-8, rtol=0) or not np.isclose(np.linalg.det(self.rotation), 1., atol=1e-8, rtol=0):
            raise ValueError("Physical mirror rotation must belong to SO(3)")
        if not math.isfinite(self.radius_m) or not math.isfinite(self.axial_height_m) or self.radius_m <= 0 or self.axial_height_m < self.radius_m:
            raise ValueError("Mirror dimensions require finite h >= r > 0")
        if self.length_definition not in {"full_extent", "cut_center_to_back"}:
            raise ValueError("Unknown mirror length definition")

    @property
    def axis(self):
        return self.rotation[:, 2]

    @property
    def normal(self):
        return self.rotation @ LOCAL_NORMAL

    @property
    def center(self):
        return self.center_m

    def to_dict(self):
        return {"mirror_id": self.mirror_id, "center_m": self.center_m.tolist(),
                "center_definition": "reflecting_cut_center", "rotation": self.rotation.tolist(),
                "axis": self.axis.tolist(), "normal": self.normal.tolist(),
                "radius_m": float(self.radius_m), "axial_height_m": float(self.axial_height_m),
                "length_definition": self.length_definition,
                "nonreflecting_surface_model": "opaque_absorbing_assumption"}

    @classmethod
    def from_dict(cls, value):
        body = cls(value["center_m"], value["rotation"],
                   value.get("mirror_id", value.get("canonical_id", value.get("old_id", ""))),
                   value.get("radius_m", .0005), value.get("axial_height_m", .0095),
                   value.get("length_definition", "full_extent"))
        for field in ("axis", "normal"):
            if field in value and not np.allclose(value[field], getattr(body, field), atol=1e-8, rtol=0):
                raise ValueError(f"Stored {field} is inconsistent with body rotation")
        if value.get("center_definition", "reflecting_cut_center") != "reflecting_cut_center":
            raise ValueError("Unsupported definition of mirror centre")
        return body

    def local(self, points):
        return (np.asarray(points) - self.center_m) @ self.rotation

    def contains(self, point, tolerance_m=0.):
        p = self.local(point)
        return bool(np.hypot(p[0], p[1]) <= self.radius_m + tolerance_m and
                    p[2] >= p[0] - math.sqrt(2) * tolerance_m and
                    p[2] <= self.axial_height_m + tolerance_m)

    def on_cut(self, point, tolerance_m=1e-9):
        p = self.local(point)
        return bool(abs(p[2] - p[0]) <= math.sqrt(2) * tolerance_m and
                    np.hypot(p[0], p[1]) <= self.radius_m + tolerance_m)

    def support_point(self, direction):
        v = self.rotation.T @ np.asarray(direction, dtype=float)
        xy = v[:2].copy()
        if v[2] < 0:
            xy[0] += v[2]
        length = np.linalg.norm(xy)
        xy = self.radius_m * xy / length if length > 1e-30 else np.zeros(2)
        z = self.axial_height_m if v[2] >= 0 else xy[0]
        return self.center_m + self.rotation @ np.r_[xy, z]

    def support(self, direction):
        v = np.asarray(direction, dtype=float)
        return float(np.dot(v, self.support_point(v)))

    def bounds(self):
        axes = np.eye(3)
        return (np.array([-self.support(-v) for v in axes]),
                np.array([self.support(v) for v in axes]))

    def mesh(self, segments=32):
        """Display-only polygon mesh; tracing/collisions remain analytic."""
        segments = max(8, int(segments))
        theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        x, y = self.radius_m * np.cos(theta), self.radius_m * np.sin(theta)
        cut = np.column_stack([x, y, x])
        back = np.column_stack([x, y, np.full(segments, self.axial_height_m)])
        vertices = np.vstack([cut, back, [0, 0, 0], [0, 0, self.axial_height_m]])
        faces, cut_faces = [], []
        for i in range(segments):
            j = (i + 1) % segments
            faces.extend([[i, j, segments + i], [j, segments + j, segments + i],
                          [2 * segments + 1, segments + i, segments + j]])
            cut_faces.append([2 * segments, j, i])
        return {"vertices_m": (vertices @ self.rotation.T + self.center_m).tolist(),
                "body_faces": faces, "cut_faces": cut_faces,
                "display_radial_error_m": self.radius_m * (1 - math.cos(math.pi / segments))}

    def build_surfaces(self, name=None):
        if name is None:
            parts = self.mirror_id.split("_")
            name = f"BUNDLE_{parts[0]}_{parts[1]} Mirror {parts[2]}" if len(parts) == 3 else f"BUNDLE_{self.mirror_id}"
        cut = SurfaceOptics(mode=InteractionMode.MIRROR, reflectance=1.,
                            release_reflected=True, release_transmitted=False,
                            reflect_from_plus_side=True, reflect_from_minus_side=False)
        absorb = SurfaceOptics(mode=InteractionMode.ABSORB, release_transmitted=False,
                               reflect_from_minus_side=False, reflect_from_plus_side=False)
        return [CutCylinderSurface(name, cut, self, "cut"),
                CutCylinderSurface(name + ":side", copy.deepcopy(absorb), self, "side"),
                CutCylinderSurface(name + ":back", copy.deepcopy(absorb), self, "back")]


@dataclass
class CutCylinderSurface(Surface):
    body: MirrorBody
    part: str

    @property
    def center(self):
        return self.body.center_m

    @property
    def normal(self):
        return self.body.normal

    def to_backend(self, backend="numpy"):
        # Geometry parameters are small host arrays; intersect converts them
        # to the ray backend without moving ray bundles to the CPU.
        get_backend(backend)
        return copy.deepcopy(self)

    def intersect(self, rays):
        xp = rays.xp
        R = xp.asarray(self.body.rotation)
        p = (rays.position - xp.asarray(self.body.center_m)) @ R
        d = rays.direction @ R
        r, h = self.body.radius_m, self.body.axial_height_m
        if self.part == "cut":
            denominator = d[:, 2] - d[:, 0]
            good = xp.abs(denominator) > 1e-14
            t = (p[:, 0] - p[:, 2]) / xp.where(good, denominator, 1.)
            q = p + t[:, None] * d
            good &= (t > 1e-9) & (q[:, 0] ** 2 + q[:, 1] ** 2 <= r * r + 1e-20)
            normals = xp.broadcast_to(xp.asarray(LOCAL_NORMAL), q.shape)
        elif self.part == "back":
            good = xp.abs(d[:, 2]) > 1e-14
            t = (h - p[:, 2]) / xp.where(good, d[:, 2], 1.)
            q = p + t[:, None] * d
            good &= (t > 1e-9) & (q[:, 0] ** 2 + q[:, 1] ** 2 <= r * r + 1e-20)
            normals = xp.broadcast_to(xp.asarray([0., 0., 1.]), q.shape)
        elif self.part == "side":
            A = xp.sum(d[:, :2] ** 2, axis=1)
            B = xp.sum(p[:, :2] * d[:, :2], axis=1)
            C = xp.sum(p[:, :2] ** 2, axis=1) - r * r
            discriminant = B * B - A * C
            valid_quad = (A > 1e-24) & (discriminant >= 0)
            root = xp.sqrt(xp.maximum(discriminant, 0))
            t = xp.full(rays.n_rays, xp.inf)
            # Test BOTH roots against the skew cap. The nearest root can lie
            # below the cut while the farther root is on the actual solid.
            for candidate in ((-B - root) / xp.where(valid_quad, A, 1.),
                              (-B + root) / xp.where(valid_quad, A, 1.)):
                q = p + candidate[:, None] * d
                valid = valid_quad & (candidate > 1e-9) & (q[:, 2] >= q[:, 0] - 1e-12) & (q[:, 2] <= h + 1e-12)
                t = xp.minimum(t, xp.where(valid, candidate, xp.inf))
            good = xp.isfinite(t)
            q = p + xp.where(good, t, 0.)[:, None] * d
            normals = xp.column_stack([q[:, 0] / r, q[:, 1] / r, xp.zeros(rays.n_rays)])
        else:
            raise ValueError(f"Unknown cylindrical mirror boundary: {self.part}")
        distance = xp.where(good, t, xp.inf)
        points = rays.position + xp.where(good, t, 0.)[:, None] * rays.direction
        return Intersection(valid=good, distance=distance, points=points,
                            normals=normals @ R.T, local_u=q[:, 0], local_v=q[:, 1])
