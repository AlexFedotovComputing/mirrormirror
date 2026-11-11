from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


from .backend import get_array_module, get_backend, to_numpy, asarray1, dot, norm, normalize, cross, orthonormal_basis, safe_where, project_to_transverse, count_nonzero


from .materials import InteractionMode, SurfaceOptics


from .rays import RayBundle, Intersection, NearestHit


from .physics import fresnel_amplitudes


@dataclass
class Surface:
    name: str
    optics: SurfaceOptics

    def to_backend(self, backend: str = "numpy") -> "Surface":
        return copy.deepcopy(self)

    def intersect(self, rays: RayBundle) -> Intersection:
        raise NotImplementedError

    # ------------------------------ optics handling ---------------------------

    def interact(self, rays: RayBundle, hit: NearestHit, tracer: "RayTracer") -> Tuple[List[RayBundle], List[Dict[str, Any]]]:
        mode = self.optics.mode
        if mode == InteractionMode.ABSORB:
            records = self._make_hit_records(rays, hit)
            return [], records
        if mode == InteractionMode.TRANSPARENT:
            return self._transparent(rays, hit, tracer), self._make_hit_records(rays, hit)
        if mode == InteractionMode.MIRROR:
            children = self._mirror(rays, hit, tracer)
            return children, self._make_hit_records(rays, hit)
        if mode in {InteractionMode.REFRACT, InteractionMode.BEAMSPLITTER}:
            children = self._refract_like(rays, hit, tracer)
            return children, self._make_hit_records(rays, hit)
        raise ValueError(f"Unsupported interaction mode: {mode}")

    def _make_hit_records(self, rays: RayBundle, hit: NearestHit) -> List[Dict[str, Any]]:
        if not self.optics.detector:
            return []
        return [
            {
                "surface": self.name,
                "ray_id": to_numpy(rays.ray_id),
                "parent_id": to_numpy(rays.parent_id),
                "depth": to_numpy(rays.depth),
                "position": to_numpy(hit.points),
                "direction": to_numpy(rays.direction),
                "power": to_numpy(rays.power),
                "intensity": to_numpy(rays.intensity),
                "local_u": to_numpy(hit.local_u),
                "local_v": to_numpy(hit.local_v),
            }
        ]

    def _with_new_state(
        self,
        parent: RayBundle,
        new_pos: Any,
        new_dir: Any,
        new_power: Any,
        new_intensity: Any,
        new_n: Any,
        new_pol: Any,
        new_time_s: Any,
    ) -> RayBundle:
        xp = parent.xp
        return RayBundle(
            position=new_pos,
            direction=new_dir,
            power=new_power,
            wavelength_m=parent.wavelength_m,
            refractive_index=new_n,
            polarization=new_pol,
            intensity=new_intensity,
            time_s=new_time_s,
            ray_id=xp.zeros(parent.n_rays, dtype=np.int64),
            parent_id=parent.ray_id.copy(),
            depth=parent.depth + 1,
            bundle_reflected=parent.bundle_reflected,
        )

    def _mirror(self, rays: RayBundle, hit: NearestHit, tracer: "RayTracer") -> List[RayBundle]:
        xp = rays.xp
        n_geo = hit.normals
        d = rays.direction
        cos_proj = dot(d, n_geo)
        going_plus = cos_proj > 0.0
        reflect_allowed = xp.where(going_plus, self.optics.reflect_from_minus_side, self.optics.reflect_from_plus_side)
        d_ref = normalize(d - 2.0 * cos_proj[:, None] * n_geo)
        new_pos = hit.points + tracer.surface_epsilon * d_ref
        refl = 1.0 if self.optics.reflectance is None else float(self.optics.reflectance)
        new_power = rays.power * refl
        new_intensity = rays.intensity * refl
        new_pol = project_to_transverse(rays.polarization, d_ref)
        child = self._with_new_state(
            rays,
            new_pos,
            d_ref,
            new_power,
            new_intensity,
            rays.refractive_index,
            new_pol,
            rays.time_s,
        )
        keep = (child.power > tracer.min_power) & reflect_allowed
        if self.name.startswith("BUNDLE_"):
            child = child.with_bundle_reflected()
        return [child.subset(keep)] if count_nonzero(keep) > 0 else []

    def _transparent(self, rays: RayBundle, hit: NearestHit, tracer: "RayTracer") -> List[RayBundle]:
        xp = rays.xp
        tcoef = 1.0 if self.optics.transmittance is None else float(self.optics.transmittance)
        new_pos = hit.points + tracer.surface_epsilon * rays.direction
        child = self._with_new_state(
            rays,
            new_pos,
            rays.direction,
            rays.power * tcoef,
            rays.intensity * tcoef,
            rays.refractive_index,
            rays.polarization,
            rays.time_s,
        )
        keep = child.power > tracer.min_power
        return [child.subset(keep)] if count_nonzero(keep) > 0 else []

    def _refract_like(self, rays: RayBundle, hit: NearestHit, tracer: "RayTracer") -> List[RayBundle]:
        xp = rays.xp
        d_in = rays.direction
        n_geo = hit.normals
        going_plus = dot(d_in, n_geo) > 0.0
        reflect_allowed = xp.where(going_plus, self.optics.reflect_from_minus_side, self.optics.reflect_from_plus_side)

        n_minus = self.optics.n_minus_values(rays.wavelength_m)
        n_plus = self.optics.n_plus_values(rays.wavelength_m)
        n1 = xp.where(going_plus, n_minus, n_plus)
        n2 = xp.where(going_plus, n_plus, n_minus)
        n_face = xp.where(going_plus[:, None], -n_geo, n_geo)  # opposes the incident direction

        cos_i = -dot(d_in, n_face)
        eta = n1 / n2
        sin2_t = eta * eta * xp.maximum(0.0, 1.0 - cos_i * cos_i)
        tir = sin2_t > 1.0
        cos_t = xp.sqrt(xp.maximum(0.0, 1.0 - sin2_t))

        d_ref = normalize(d_in + 2.0 * cos_i[:, None] * n_face)
        d_trn = normalize(eta[:, None] * d_in + (eta * cos_i - cos_t)[:, None] * n_face)

        s_hat, p_in, p_ref, p_trn = self._polarization_frames(d_in, d_ref, d_trn, n_face, rays.polarization)
        es = dot(rays.polarization, s_hat)
        ep = dot(rays.polarization, p_in)
        w_sum = es * es + ep * ep
        es2 = xp.where(w_sum > 1e-15, (es * es) / w_sum, 0.5)
        ep2 = xp.where(w_sum > 1e-15, (ep * ep) / w_sum, 0.5)

        if self.optics.use_fresnel:
            rs, rp, ts, tp = fresnel_amplitudes(cos_i, cos_t, n1, n2, tir)
            Rs = rs * rs
            Rp = rp * rp
            Ts = ((n2 * cos_t) / xp.maximum(n1 * cos_i, 1e-15)) * ts * ts
            Tp = ((n2 * cos_t) / xp.maximum(n1 * cos_i, 1e-15)) * tp * tp
            R = es2 * Rs + ep2 * Rp
            T = es2 * Ts + ep2 * Tp
            R = xp.where(tir, 1.0, R)
            T = xp.where(tir, 0.0, T)
        else:
            R = xp.full(rays.n_rays, 0.0 if self.optics.reflectance is None else float(self.optics.reflectance), dtype=float)
            T = xp.full(rays.n_rays, 1.0 if self.optics.transmittance is None else float(self.optics.transmittance), dtype=float)
            T = xp.where(tir, 0.0, T)
            if self.optics.allow_total_internal_reflection:
                R = xp.where(tir, 1.0, R)

        # A coating's sidedness does not suppress total internal reflection
        # at the dielectric boundary (including uncoated prism faces).
        R = xp.where(tir & self.optics.allow_total_internal_reflection,
                     1.0, xp.where(reflect_allowed, R, 0.0))
        T = xp.where(reflect_allowed, T, xp.where(tir, 0.0, 1.0))

        out: List[RayBundle] = []

        if self.optics.release_reflected or count_nonzero(tir) > 0:
            refl_mask = (R > tracer.min_power) & (self.optics.release_reflected | (tir & self.optics.allow_total_internal_reflection))
            if count_nonzero(refl_mask) > 0:
                pol_ref = normalize(es[:, None] * s_hat + ep[:, None] * p_ref)
                child_ref = self._with_new_state(
                    rays,
                    hit.points + tracer.surface_epsilon * d_ref,
                    d_ref,
                    rays.power * R,
                    rays.intensity * R,
                    n1,
                    pol_ref,
                    rays.time_s,
                )
                if self.name.startswith("BUNDLE_"):
                    child_ref = child_ref.with_bundle_reflected()
                out.append(child_ref.subset(refl_mask))

        trn_mask = self.optics.release_transmitted & (~tir) & (T > tracer.min_power)
        if count_nonzero(trn_mask) > 0:
            pol_trn = normalize(es[:, None] * s_hat + ep[:, None] * p_trn)
            child_trn = self._with_new_state(
                rays,
                hit.points + tracer.surface_epsilon * d_trn,
                d_trn,
                rays.power * T,
                rays.intensity * T,
                n2,
                pol_trn,
                rays.time_s,
            )
            out.append(child_trn.subset(trn_mask))

        return out

    def _polarization_frames(self, d_in: Any, d_ref: Any, d_trn: Any, n_face: Any, pol: Any):
        xp = get_array_module(d_in, d_ref, d_trn, n_face, pol)
        s = cross(d_in, n_face)
        s_norm = norm(s)
        fallback = project_to_transverse(pol, d_in)
        use_fb = s_norm < 1e-12
        s = safe_where(use_fb[:, None], fallback, s)
        s = normalize(s)
        p_in = normalize(cross(s, d_in))
        p_ref = normalize(cross(s, d_ref))
        p_trn = normalize(cross(s, d_trn))
        return s, p_in, p_ref, p_trn

@dataclass
class PlaneSurface(Surface):
    center: Any
    normal: Any
    shape: str = "rectangle"  # rectangle | disk | infinite
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    in_plane_reference: Optional[Any] = None
    triangle_vertices: Optional[Sequence[Sequence[float]]] = None
    _u: Any = field(init=False, repr=False, default=None)
    _v: Any = field(init=False, repr=False, default=None)
    _w: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        xp = get_array_module(self.center, self.normal)
        self.center = asarray1(self.center, xp=xp)
        self.normal = normalize(asarray1(self.normal, xp=xp))
        u, v, w = orthonormal_basis(self.normal, self.in_plane_reference, xp=xp)
        self._u = u
        self._v = v
        self._w = w
        shape = self.shape.lower()
        if shape not in {"rectangle", "disk", "infinite", "triangle"}:
            raise ValueError(f"Unsupported plane shape: {self.shape}")
        self.shape = shape
        if self.shape == "rectangle" and (self.width is None or self.height is None):
            raise ValueError("Rectangle plane requires width and height.")
        if self.shape == "disk" and self.radius is None:
            raise ValueError("Disk plane requires radius.")
        if self.shape == "triangle" and self.triangle_vertices is None:
            raise ValueError("Triangle plane requires triangle_vertices.")

    def to_backend(self, backend: str = "numpy") -> "PlaneSurface":
        xp = get_backend(backend)
        new = copy.deepcopy(self)
        new.center = asarray1(to_numpy(self.center), xp=xp)
        new.normal = asarray1(to_numpy(self.normal), xp=xp)
        new._u = asarray1(to_numpy(self._u), xp=xp)
        new._v = asarray1(to_numpy(self._v), xp=xp)
        new._w = asarray1(to_numpy(self._w), xp=xp)
        new.triangle_vertices = self.triangle_vertices
        return new

    def intersect(self, rays: RayBundle) -> Intersection:
        xp = rays.xp
        center = xp.asarray(self.center)
        normal = xp.asarray(self.normal)
        u = xp.asarray(self._u)
        v = xp.asarray(self._v)
        denom = dot(rays.direction, normal)
        valid = xp.abs(denom) > 1e-12
        t = dot(center - rays.position, normal) / xp.where(valid, denom, 1.0)
        valid = valid & (t > 1e-9)

        points = rays.position + t[:, None] * rays.direction
        rel = points - center
        local_u = dot(rel, u)
        local_v = dot(rel, v)

        if self.shape == "rectangle":
            valid = valid & (xp.abs(local_u) <= 0.5 * self.width) & (xp.abs(local_v) <= 0.5 * self.height)
        elif self.shape == "disk":
            valid = valid & ((local_u * local_u + local_v * local_v) <= self.radius * self.radius)
        elif self.shape == "triangle":
            u1, v1 = self.triangle_vertices[0]
            u2, v2 = self.triangle_vertices[1]
            u3, v3 = self.triangle_vertices[2]
            d1 = (local_u - u1) * (v2 - v1) - (u2 - u1) * (local_v - v1)
            d2 = (local_u - u2) * (v3 - v2) - (u3 - u2) * (local_v - v2)
            d3 = (local_u - u3) * (v1 - v3) - (u1 - u3) * (local_v - v3)
            in_tri = ((d1 >= -1e-12) & (d2 >= -1e-12) & (d3 >= -1e-12)) | ((d1 <= 1e-12) & (d2 <= 1e-12) & (d3 <= 1e-12))
            valid = valid & in_tri

        t = xp.where(valid, t, xp.inf)
        normals = xp.broadcast_to(normal, points.shape)
        return Intersection(valid=valid, distance=t, points=points, normals=normals, local_u=local_u, local_v=local_v)

@dataclass
class SphericalCapSurface(Surface):
    vertex: Any
    axis: Any
    curvature_radius: float
    aperture_radius: float
    _center: Any = field(init=False, repr=False, default=None)
    _axis: Any = field(init=False, repr=False, default=None)
    _sign: float = field(init=False, repr=False, default=1.0)

    def __post_init__(self) -> None:
        if abs(self.curvature_radius) < 1e-12:
            raise ValueError("curvature_radius must be non-zero for a spherical surface")
        xp = get_array_module(self.vertex, self.axis)
        self.vertex = asarray1(self.vertex, xp=xp)
        self.axis = normalize(asarray1(self.axis, xp=xp))
        self._axis = self.axis
        self._center = self.vertex + self.curvature_radius * self.axis
        self._sign = 1.0 if self.curvature_radius > 0.0 else -1.0

    def to_backend(self, backend: str = "numpy") -> "SphericalCapSurface":
        xp = get_backend(backend)
        new = copy.deepcopy(self)
        new.vertex = asarray1(to_numpy(self.vertex), xp=xp)
        new.axis = asarray1(to_numpy(self.axis), xp=xp)
        new._axis = asarray1(to_numpy(self._axis), xp=xp)
        new._center = asarray1(to_numpy(self._center), xp=xp)
        return new

    def intersect(self, rays: RayBundle) -> Intersection:
        xp = rays.xp
        center = xp.asarray(self._center)
        axis = xp.asarray(self._axis)
        radius = abs(self.curvature_radius)

        oc = rays.position - center
        b = dot(rays.direction, oc)
        c = dot(oc, oc) - radius * radius
        disc = b * b - c
        valid = disc >= 0.0
        sqrt_disc = xp.sqrt(xp.maximum(disc, 0.0))
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc

        inf = xp.full_like(t1, xp.inf)
        t1 = xp.where(t1 > 1e-9, t1, inf)
        t2 = xp.where(t2 > 1e-9, t2, inf)
        t = xp.minimum(t1, t2)
        valid = valid & xp.isfinite(t)

        # Misses carry infinite distance, but their temporary geometry must stay
        # finite to avoid NaNs/warnings before the validity mask is applied.
        points = rays.position + xp.where(valid, t, 0.)[:, None] * rays.direction
        rel_vertex = points - self.vertex
        axial = dot(rel_vertex, axis)
        radial_vec = rel_vertex - axial[:, None] * axis
        radial = norm(radial_vec)
        valid = valid & (radial <= self.aperture_radius)

        sphere_normal = normalize(points - center)
        normals = -self._sign * sphere_normal  # makes the normal at the vertex equal to +axis
        t = xp.where(valid, t, xp.inf)
        return Intersection(valid=valid, distance=t, points=points, normals=normals, local_u=radial, local_v=axial)

@dataclass
class CylinderSurface(Surface):
    center: Any
    axis: Any
    radius: float
    length: float

    def __post_init__(self) -> None:
        xp = get_array_module(self.center, self.axis)
        self.center = asarray1(self.center, xp=xp)
        self.axis = normalize(asarray1(self.axis, xp=xp))

    def to_backend(self, backend: str = "numpy") -> "CylinderSurface":
        xp = get_backend(backend)
        new = copy.deepcopy(self)
        new.center = asarray1(to_numpy(self.center), xp=xp)
        new.axis = asarray1(to_numpy(self.axis), xp=xp)
        return new

    def intersect(self, rays: RayBundle) -> Intersection:
        xp = rays.xp
        a = xp.asarray(self.axis)
        c0 = xp.asarray(self.center)

        dp = rays.position - c0
        d_par = dot(rays.direction, a)
        p_par = dot(dp, a)
        d_perp = rays.direction - d_par[:, None] * a
        p_perp = dp - p_par[:, None] * a

        A = dot(d_perp, d_perp)
        B = 2.0 * dot(d_perp, p_perp)
        C = dot(p_perp, p_perp) - self.radius * self.radius
        disc = B * B - 4.0 * A * C
        valid = (disc >= 0.0) & (A > 1e-14)
        sqrt_disc = xp.sqrt(xp.maximum(disc, 0.0))
        t1 = (-B - sqrt_disc) / xp.where(A > 1e-14, 2.0 * A, 1.0)
        t2 = (-B + sqrt_disc) / xp.where(A > 1e-14, 2.0 * A, 1.0)

        inf = xp.full_like(t1, xp.inf)
        t1 = xp.where(t1 > 1e-9, t1, inf)
        t2 = xp.where(t2 > 1e-9, t2, inf)
        t = xp.minimum(t1, t2)
        valid = valid & xp.isfinite(t)

        t_safe = xp.where(valid, t, 0.0)
        points = rays.position + t_safe[:, None] * rays.direction
        axial = dot(points - c0, a)
        valid = valid & (xp.abs(axial) <= 0.5 * self.length)
        radial_vec = points - c0 - axial[:, None] * a
        normals = normalize(radial_vec)
        t = xp.where(valid, t, xp.inf)
        return Intersection(valid=valid, distance=t, points=points, normals=normals, local_u=axial, local_v=norm(radial_vec))
