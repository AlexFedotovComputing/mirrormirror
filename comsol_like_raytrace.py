from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Any
Number = Union[int, float]
IndexModel = Union[Number, Callable[[float], float], "Material"]
LIGHT_SPEED_M_S = 299_792_458.0


# -----------------------------------------------------------------------------
# Backend helpers
# -----------------------------------------------------------------------------


def get_array_module(*xs: Any):
    """Return numpy or cupy depending on the first array-like argument."""
    for x in xs:
        if x is None:
            continue
        module = type(x).__module__.split(".")[0]
        if module == "cupy":
            import cupy as cp

            return cp
    return np



def get_backend(name: str = "numpy"):
    name = name.lower()
    if name in {"numpy", "np", "cpu"}:
        return np
    if name in {"cupy", "cp", "cuda", "gpu"}:
        import cupy as cp

        return cp
    raise ValueError(f"Unsupported backend: {name}")



def to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return x
    xp = get_array_module(x)
    if xp is np:
        return np.asarray(x)
    return xp.asnumpy(x)



def scalar_to_python(x: Any) -> Union[int, float]:
    arr = to_numpy(x)
    if np.ndim(arr) == 0:
        return arr.item()
    raise TypeError("Expected a scalar-like object.")


# -----------------------------------------------------------------------------
# Small vector helpers
# -----------------------------------------------------------------------------


def asarray1(x: Sequence[float], xp=np) -> Any:
    arr = xp.asarray(x, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {arr.shape}")
    return arr



def dot(a: Any, b: Any) -> Any:
    xp = get_array_module(a, b)
    return xp.sum(a * b, axis=-1)



def norm(a: Any, axis: int = -1, keepdims: bool = False, eps: float = 0.0) -> Any:
    xp = get_array_module(a)
    out = xp.sqrt(xp.sum(a * a, axis=axis, keepdims=keepdims))
    if eps > 0:
        out = xp.maximum(out, eps)
    return out



def normalize(a: Any, eps: float = 1e-15) -> Any:
    xp = get_array_module(a)
    return a / norm(a, keepdims=True, eps=eps)



def cross(a: Any, b: Any) -> Any:
    xp = get_array_module(a, b)
    return xp.cross(a, b)



def orthonormal_basis(axis: Sequence[float], reference: Optional[Sequence[float]] = None, xp=np):
    w = normalize(asarray1(axis, xp=xp))
    if reference is None:
        if abs(float(to_numpy(w)[0])) < 0.9:
            reference = (1.0, 0.0, 0.0)
        else:
            reference = (0.0, 1.0, 0.0)
    ref = asarray1(reference, xp=xp)
    ref = ref - dot(ref, w) * w
    ref_norm = float(to_numpy(norm(ref)))
    if ref_norm < 1e-12:
        fallback = (0.0, 0.0, 1.0) if abs(float(to_numpy(w)[2])) < 0.9 else (0.0, 1.0, 0.0)
        ref = asarray1(fallback, xp=xp)
        ref = ref - dot(ref, w) * w
    u = normalize(ref)
    v = normalize(cross(w, u))
    u = normalize(cross(v, w))
    return u, v, w



def safe_where(condition: Any, x: Any, y: Any) -> Any:
    xp = get_array_module(condition, x, y)
    return xp.where(condition, x, y)



def project_to_transverse(vec: Any, direction: Any, fallback: Optional[Any] = None) -> Any:
    xp = get_array_module(vec, direction, fallback)
    out = vec - dot(vec, direction)[..., None] * direction
    mask = norm(out) < 1e-12
    if int(np.count_nonzero(to_numpy(mask))) > 0:
        if fallback is None:
            fb_u, _, _ = orthonormal_basis(to_numpy(direction[0]) if direction.ndim == 2 else to_numpy(direction), xp=xp)
            fallback = fb_u
        fallback_arr = fallback if getattr(fallback, "ndim", 1) == out.ndim else xp.broadcast_to(fallback, out.shape)
        out = safe_where(mask[..., None], fallback_arr, out)
    return normalize(out)



def count_nonzero(mask: Any) -> int:
    xp = get_array_module(mask)
    return int(scalar_to_python(xp.count_nonzero(mask)))


# -----------------------------------------------------------------------------
# Material and optics definitions
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    name: str
    refractive_index_model: IndexModel = 1.0

    def n(self, wavelength_m: float) -> float:
        model = self.refractive_index_model
        if isinstance(model, Material):
            return model.n(wavelength_m)
        if callable(model):
            return float(model(wavelength_m))
        return float(model)


AIR = Material("air", 1.0)


class InteractionMode(str, Enum):
    REFRACT = "refract"
    MIRROR = "mirror"
    BEAMSPLITTER = "beamsplitter"
    TRANSPARENT = "transparent"
    ABSORB = "absorb"


@dataclass
class SurfaceOptics:
    mode: InteractionMode
    label: str = ""
    n_minus: IndexModel = AIR
    n_plus: IndexModel = AIR
    reflectance: Optional[float] = None
    transmittance: Optional[float] = None
    release_reflected: bool = False
    release_transmitted: bool = True
    use_fresnel: bool = True
    allow_total_internal_reflection: bool = True
    detector: bool = False

    def _evaluate_model(self, model: IndexModel, wavelengths: Any) -> Any:
        xp = get_array_module(wavelengths)
        if isinstance(model, Material):
            model = model.refractive_index_model
        if callable(model):
            wl_host = to_numpy(wavelengths).reshape(-1)
            vals = np.asarray([float(model(float(w))) for w in wl_host], dtype=float).reshape(to_numpy(wavelengths).shape)
            return xp.asarray(vals)
        return xp.full(wavelengths.shape, float(model), dtype=float)

    def n_minus_values(self, wavelengths: Any) -> Any:
        return self._evaluate_model(self.n_minus, wavelengths)

    def n_plus_values(self, wavelengths: Any) -> Any:
        return self._evaluate_model(self.n_plus, wavelengths)


# -----------------------------------------------------------------------------
# Ray bundle
# -----------------------------------------------------------------------------


@dataclass
class RayBundle:
    position: Any
    direction: Any
    power: Any
    wavelength_m: Any
    refractive_index: Any
    polarization: Optional[Any] = None
    intensity: Optional[Any] = None
    time_s: Optional[Any] = None
    ray_id: Optional[Any] = None
    parent_id: Optional[Any] = None
    depth: Optional[Any] = None

    def __post_init__(self) -> None:
        xp = get_array_module(self.position, self.direction, self.power, self.wavelength_m, self.refractive_index)
        self.position = xp.asarray(self.position, dtype=float)
        self.direction = normalize(xp.asarray(self.direction, dtype=float))
        self.power = xp.asarray(self.power, dtype=float).reshape(-1)
        self.wavelength_m = xp.asarray(self.wavelength_m, dtype=float).reshape(-1)
        self.refractive_index = xp.asarray(self.refractive_index, dtype=float).reshape(-1)

        n = self.position.shape[0]
        if self.position.shape != (n, 3):
            raise ValueError(f"position must have shape (N, 3), got {self.position.shape}")
        if self.direction.shape != (n, 3):
            raise ValueError(f"direction must have shape (N, 3), got {self.direction.shape}")
        for name, arr in {
            "power": self.power,
            "wavelength_m": self.wavelength_m,
            "refractive_index": self.refractive_index,
        }.items():
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape (N,), got {arr.shape}")

        if self.polarization is None:
            u0, _, _ = orthonormal_basis(to_numpy(self.direction[0]), xp=xp)
            self.polarization = xp.broadcast_to(u0, (n, 3)).copy()
        else:
            self.polarization = project_to_transverse(xp.asarray(self.polarization, dtype=float), self.direction)

        if self.intensity is None:
            self.intensity = self.power.copy()
        else:
            self.intensity = xp.asarray(self.intensity, dtype=float).reshape(-1)

        if self.time_s is None:
            self.time_s = xp.zeros(n, dtype=float)
        else:
            self.time_s = xp.asarray(self.time_s, dtype=float).reshape(-1)
        if self.time_s.shape != (n,):
            raise ValueError(f"time_s must have shape (N,), got {self.time_s.shape}")

        if self.ray_id is None:
            self.ray_id = xp.arange(n, dtype=np.int64)
        else:
            self.ray_id = xp.asarray(self.ray_id, dtype=np.int64).reshape(-1)
        if self.parent_id is None:
            self.parent_id = xp.full(n, -1, dtype=np.int64)
        else:
            self.parent_id = xp.asarray(self.parent_id, dtype=np.int64).reshape(-1)
        if self.depth is None:
            self.depth = xp.zeros(n, dtype=np.int64)
        else:
            self.depth = xp.asarray(self.depth, dtype=np.int64).reshape(-1)

    @property
    def xp(self):
        return get_array_module(self.position)

    @property
    def n_rays(self) -> int:
        return int(self.position.shape[0])

    def subset(self, mask: Any) -> "RayBundle":
        return RayBundle(
            position=self.position[mask],
            direction=self.direction[mask],
            power=self.power[mask],
            wavelength_m=self.wavelength_m[mask],
            refractive_index=self.refractive_index[mask],
            polarization=self.polarization[mask],
            intensity=self.intensity[mask],
            time_s=self.time_s[mask],
            ray_id=self.ray_id[mask],
            parent_id=self.parent_id[mask],
            depth=self.depth[mask],
        )

    def replace_ids(self, ray_id: Any, parent_id: Any) -> "RayBundle":
        return RayBundle(
            position=self.position,
            direction=self.direction,
            power=self.power,
            wavelength_m=self.wavelength_m,
            refractive_index=self.refractive_index,
            polarization=self.polarization,
            intensity=self.intensity,
            time_s=self.time_s,
            ray_id=ray_id,
            parent_id=parent_id,
            depth=self.depth,
        )

    def propagate(self, new_position: Any, delta_time_s: Any) -> "RayBundle":
        return RayBundle(
            position=new_position,
            direction=self.direction,
            power=self.power,
            wavelength_m=self.wavelength_m,
            refractive_index=self.refractive_index,
            polarization=self.polarization,
            intensity=self.intensity,
            time_s=self.time_s + delta_time_s,
            ray_id=self.ray_id,
            parent_id=self.parent_id,
            depth=self.depth,
        )

    @classmethod
    def concat(cls, chunks: Sequence["RayBundle"], backend: Optional[str] = None) -> "RayBundle":
        chunks = [c for c in chunks if c is not None and c.n_rays > 0]
        if not chunks:
            xp = get_backend(backend or "numpy")
            empty_vec = xp.zeros((0, 3), dtype=float)
            empty = xp.zeros((0,), dtype=float)
            empty_i = xp.zeros((0,), dtype=np.int64)
            return RayBundle(
                position=empty_vec,
                direction=empty_vec,
                power=empty,
                wavelength_m=empty,
                refractive_index=empty,
                polarization=empty_vec,
                intensity=empty,
                time_s=empty,
                ray_id=empty_i,
                parent_id=empty_i,
                depth=empty_i,
            )
        xp = chunks[0].xp
        return RayBundle(
            position=xp.concatenate([c.position for c in chunks], axis=0),
            direction=xp.concatenate([c.direction for c in chunks], axis=0),
            power=xp.concatenate([c.power for c in chunks], axis=0),
            wavelength_m=xp.concatenate([c.wavelength_m for c in chunks], axis=0),
            refractive_index=xp.concatenate([c.refractive_index for c in chunks], axis=0),
            polarization=xp.concatenate([c.polarization for c in chunks], axis=0),
            intensity=xp.concatenate([c.intensity for c in chunks], axis=0),
            time_s=xp.concatenate([c.time_s for c in chunks], axis=0),
            ray_id=xp.concatenate([c.ray_id for c in chunks], axis=0),
            parent_id=xp.concatenate([c.parent_id for c in chunks], axis=0),
            depth=xp.concatenate([c.depth for c in chunks], axis=0),
        )


# -----------------------------------------------------------------------------
# Intersections
# -----------------------------------------------------------------------------


@dataclass
class Intersection:
    valid: Any
    distance: Any
    points: Any
    normals: Any
    local_u: Optional[Any] = None
    local_v: Optional[Any] = None


@dataclass
class NearestHit:
    surface_index: Any
    distance: Any
    points: Any
    normals: Any
    local_u: Any
    local_v: Any

    def subset(self, mask: Any) -> "NearestHit":
        return NearestHit(
            surface_index=self.surface_index[mask],
            distance=self.distance[mask],
            points=self.points[mask],
            normals=self.normals[mask],
            local_u=self.local_u[mask],
            local_v=self.local_v[mask],
        )


# -----------------------------------------------------------------------------
# Surface primitives
# -----------------------------------------------------------------------------


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
        )

    def _mirror(self, rays: RayBundle, hit: NearestHit, tracer: "RayTracer") -> List[RayBundle]:
        xp = rays.xp
        n_geo = hit.normals
        d = rays.direction
        cos_proj = dot(d, n_geo)
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
        keep = child.power > tracer.min_power
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
        if shape not in {"rectangle", "disk", "infinite"}:
            raise ValueError(f"Unsupported plane shape: {self.shape}")
        self.shape = shape
        if self.shape == "rectangle" and (self.width is None or self.height is None):
            raise ValueError("Rectangle plane requires width and height.")
        if self.shape == "disk" and self.radius is None:
            raise ValueError("Disk plane requires radius.")

    def to_backend(self, backend: str = "numpy") -> "PlaneSurface":
        xp = get_backend(backend)
        new = copy.deepcopy(self)
        new.center = asarray1(to_numpy(self.center), xp=xp)
        new.normal = asarray1(to_numpy(self.normal), xp=xp)
        new._u = asarray1(to_numpy(self._u), xp=xp)
        new._v = asarray1(to_numpy(self._v), xp=xp)
        new._w = asarray1(to_numpy(self._w), xp=xp)
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

        points = rays.position + t[:, None] * rays.direction
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

        points = rays.position + t[:, None] * rays.direction
        axial = dot(points - c0, a)
        valid = valid & (xp.abs(axial) <= 0.5 * self.length)
        radial_vec = points - c0 - axial[:, None] * a
        normals = normalize(radial_vec)
        t = xp.where(valid, t, xp.inf)
        return Intersection(valid=valid, distance=t, points=points, normals=normals, local_u=axial, local_v=norm(radial_vec))


# -----------------------------------------------------------------------------
# Composite optical elements
# -----------------------------------------------------------------------------


class OpticalElement:
    def build_surfaces(self) -> List[Surface]:
        raise NotImplementedError


@dataclass
class PlaneMirror(OpticalElement):
    name: str
    center: Sequence[float]
    normal: Sequence[float]
    shape: str = "rectangle"
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    in_plane_reference: Optional[Sequence[float]] = None
    reflectance: float = 1.0

    def build_surfaces(self) -> List[Surface]:
        optics = SurfaceOptics(
            mode=InteractionMode.MIRROR,
            label=self.name,
            reflectance=self.reflectance,
            transmittance=0.0,
            release_reflected=True,
            release_transmitted=False,
            use_fresnel=False,
        )
        return [
            PlaneSurface(
                name=self.name,
                optics=optics,
                center=self.center,
                normal=self.normal,
                shape=self.shape,
                width=self.width,
                height=self.height,
                radius=self.radius,
                in_plane_reference=self.in_plane_reference,
            )
        ]


@dataclass
class BeamSplitter(OpticalElement):
    name: str
    center: Sequence[float]
    normal: Sequence[float]
    n_minus: IndexModel = AIR
    n_plus: IndexModel = AIR
    reflectance: float = 0.5
    transmittance: float = 0.5
    shape: str = "rectangle"
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    in_plane_reference: Optional[Sequence[float]] = None

    def build_surfaces(self) -> List[Surface]:
        optics = SurfaceOptics(
            mode=InteractionMode.BEAMSPLITTER,
            label=self.name,
            n_minus=self.n_minus,
            n_plus=self.n_plus,
            reflectance=self.reflectance,
            transmittance=self.transmittance,
            release_reflected=True,
            release_transmitted=True,
            use_fresnel=False,
        )
        return [
            PlaneSurface(
                name=self.name,
                optics=optics,
                center=self.center,
                normal=self.normal,
                shape=self.shape,
                width=self.width,
                height=self.height,
                radius=self.radius,
                in_plane_reference=self.in_plane_reference,
            )
        ]


@dataclass
class BeamDump(OpticalElement):
    name: str
    center: Sequence[float]
    normal: Sequence[float]
    shape: str = "rectangle"
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    in_plane_reference: Optional[Sequence[float]] = None
    detector: bool = False

    def build_surfaces(self) -> List[Surface]:
        optics = SurfaceOptics(
            mode=InteractionMode.ABSORB,
            label=self.name,
            detector=self.detector,
            release_reflected=False,
            release_transmitted=False,
        )
        return [
            PlaneSurface(
                name=self.name,
                optics=optics,
                center=self.center,
                normal=self.normal,
                shape=self.shape,
                width=self.width,
                height=self.height,
                radius=self.radius,
                in_plane_reference=self.in_plane_reference,
            )
        ]


@dataclass
class Detector(BeamDump):
    detector: bool = True


@dataclass
class Window(OpticalElement):
    name: str
    center: Sequence[float]
    normal: Sequence[float]
    thickness: float
    n_glass: IndexModel
    n_outside: IndexModel = AIR
    shape: str = "disk"
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    in_plane_reference: Optional[Sequence[float]] = None
    use_fresnel: bool = True

    def build_surfaces(self) -> List[Surface]:
        center = np.asarray(self.center, dtype=float)
        normal = normalize(np.asarray(self.normal, dtype=float))
        front = center - 0.5 * self.thickness * normal
        back = center + 0.5 * self.thickness * normal

        optics_front = SurfaceOptics(
            mode=InteractionMode.REFRACT,
            label=f"{self.name}:front",
            n_minus=self.n_outside,
            n_plus=self.n_glass,
            release_reflected=False,
            release_transmitted=True,
            use_fresnel=self.use_fresnel,
        )
        optics_back = SurfaceOptics(
            mode=InteractionMode.REFRACT,
            label=f"{self.name}:back",
            n_minus=self.n_glass,
            n_plus=self.n_outside,
            release_reflected=False,
            release_transmitted=True,
            use_fresnel=self.use_fresnel,
        )
        return [
            PlaneSurface(
                name=f"{self.name}:front",
                optics=optics_front,
                center=front,
                normal=normal,
                shape=self.shape,
                width=self.width,
                height=self.height,
                radius=self.radius,
                in_plane_reference=self.in_plane_reference,
            ),
            PlaneSurface(
                name=f"{self.name}:back",
                optics=optics_back,
                center=back,
                normal=normal,
                shape=self.shape,
                width=self.width,
                height=self.height,
                radius=self.radius,
                in_plane_reference=self.in_plane_reference,
            ),
        ]


@dataclass
class SphericalLens(OpticalElement):
    """
    A simple thick lens assembled from two spherical caps.

    Sign convention:
    - front vertex = center - 0.5 * thickness * axis
    - back vertex  = center + 0.5 * thickness * axis
    - the geometric normal at both vertices points along +axis
    - therefore a typical biconvex lens is: radius_front > 0, radius_back < 0
    """

    name: str
    center: Sequence[float]
    axis: Sequence[float]
    thickness: float
    aperture_radius: float
    radius_front: float
    radius_back: float
    n_lens: IndexModel
    n_outside: IndexModel = AIR
    use_fresnel: bool = True

    def build_surfaces(self) -> List[Surface]:
        c = np.asarray(self.center, dtype=float)
        a = normalize(np.asarray(self.axis, dtype=float))
        v_front = c - 0.5 * self.thickness * a
        v_back = c + 0.5 * self.thickness * a
        front = SphericalCapSurface(
            name=f"{self.name}:front",
            optics=SurfaceOptics(
                mode=InteractionMode.REFRACT,
                label=f"{self.name}:front",
                n_minus=self.n_outside,
                n_plus=self.n_lens,
                release_reflected=False,
                release_transmitted=True,
                use_fresnel=self.use_fresnel,
            ),
            vertex=v_front,
            axis=a,
            curvature_radius=self.radius_front,
            aperture_radius=self.aperture_radius,
        )
        back = SphericalCapSurface(
            name=f"{self.name}:back",
            optics=SurfaceOptics(
                mode=InteractionMode.REFRACT,
                label=f"{self.name}:back",
                n_minus=self.n_lens,
                n_plus=self.n_outside,
                release_reflected=False,
                release_transmitted=True,
                use_fresnel=self.use_fresnel,
            ),
            vertex=v_back,
            axis=a,
            curvature_radius=self.radius_back,
            aperture_radius=self.aperture_radius,
        )
        return [front, back]


# -----------------------------------------------------------------------------
# Scene and compile step
# -----------------------------------------------------------------------------


@dataclass
class CompiledScene:
    surfaces: List[Surface]

    def find_nearest(self, rays: RayBundle) -> NearestHit:
        xp = rays.xp
        n = rays.n_rays
        best_t = xp.full(n, xp.inf, dtype=float)
        best_idx = xp.full(n, -1, dtype=np.int64)
        best_points = xp.zeros((n, 3), dtype=float)
        best_normals = xp.zeros((n, 3), dtype=float)
        best_u = xp.zeros(n, dtype=float)
        best_v = xp.zeros(n, dtype=float)

        for i, surface in enumerate(self.surfaces):
            hit = surface.intersect(rays)
            better = hit.valid & (hit.distance < best_t)
            if count_nonzero(better) == 0:
                continue
            best_t = xp.where(better, hit.distance, best_t)
            best_idx = xp.where(better, i, best_idx)
            best_points = xp.where(better[:, None], hit.points, best_points)
            best_normals = xp.where(better[:, None], hit.normals, best_normals)
            if hit.local_u is not None:
                best_u = xp.where(better, hit.local_u, best_u)
            if hit.local_v is not None:
                best_v = xp.where(better, hit.local_v, best_v)

        return NearestHit(
            surface_index=best_idx,
            distance=best_t,
            points=best_points,
            normals=best_normals,
            local_u=best_u,
            local_v=best_v,
        )


@dataclass
class Scene:
    elements: List[OpticalElement] = field(default_factory=list)

    def add(self, *elements: OpticalElement) -> "Scene":
        self.elements.extend(elements)
        return self

    def build_surfaces(self) -> List[Surface]:
        out: List[Surface] = []
        for element in self.elements:
            out.extend(element.build_surfaces())
        return out

    def compile(self, backend: str = "numpy") -> CompiledScene:
        return CompiledScene([surface.to_backend(backend) for surface in self.build_surfaces()])


# -----------------------------------------------------------------------------
# Gaussian source
# -----------------------------------------------------------------------------


@dataclass
class GaussianBeamSource:
    waist_position: Sequence[float]
    axis: Sequence[float]
    waist_radius: float
    peak_intensity: float
    wavelength_m: float
    polarization_reference: Sequence[float] = (0.0, 1.0, 0.0)
    radial_positions: int = 15
    cutoff_ratio: float = 1.0
    n_medium: float = 1.0
    backend: str = "numpy"

    def emit(self) -> RayBundle:
        xp = get_backend(self.backend)
        waist = asarray1(self.waist_position, xp=xp)
        u, v, w = orthonormal_basis(self.axis, self.polarization_reference, xp=xp)
        pol = project_to_transverse(xp.broadcast_to(asarray1(self.polarization_reference, xp=xp), (1, 3)).copy(), xp.broadcast_to(w, (1, 3)))[0]

        n_r = int(self.radial_positions)
        Rc = self.cutoff_ratio * self.waist_radius

        points: List[Any] = []
        dirs: List[Any] = []
        intensities: List[float] = []
        powers: List[float] = []
        pols: List[Any] = []

        radii = [Rc * k / n_r for k in range(n_r + 1)]
        boundaries = [0.0]
        for k in range(1, n_r + 1):
            boundaries.append(0.5 * (radii[k - 1] + radii[k]))
        boundaries.append(Rc)

        for k, r in enumerate(radii):
            if k == 0:
                m = 1
                thetas = np.array([0.0], dtype=float)
            else:
                m = 6 * k
                thetas = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False, dtype=float)

            r_in = boundaries[k]
            r_out = boundaries[k + 1]
            annulus_area = math.pi * (r_out * r_out - r_in * r_in)
            cell_area = annulus_area / m if m > 0 else 0.0

            for theta in thetas:
                cth = float(math.cos(theta))
                sth = float(math.sin(theta))
                p = waist + r * cth * u + r * sth * v
                I = self.peak_intensity * math.exp(-2.0 * (r / self.waist_radius) ** 2)
                P = I * cell_area
                points.append(p)
                dirs.append(w)
                intensities.append(I)
                powers.append(P)
                pols.append(pol)

        position = xp.stack(points, axis=0)
        direction = xp.stack(dirs, axis=0)
        polarization = xp.stack(pols, axis=0)
        n = position.shape[0]
        return RayBundle(
            position=position,
            direction=direction,
            power=xp.asarray(powers, dtype=float),
            wavelength_m=xp.full(n, self.wavelength_m, dtype=float),
            refractive_index=xp.full(n, self.n_medium, dtype=float),
            polarization=polarization,
            intensity=xp.asarray(intensities, dtype=float),
            time_s=xp.zeros(n, dtype=float),
            ray_id=xp.arange(n, dtype=np.int64),
            parent_id=xp.full(n, -1, dtype=np.int64),
            depth=xp.zeros(n, dtype=np.int64),
        )


# -----------------------------------------------------------------------------
# Tracer and result
# -----------------------------------------------------------------------------


@dataclass
class TraceResult:
    final_rays: RayBundle
    segments: List[Dict[str, Any]]
    detector_hits: List[Dict[str, Any]]
    surface_names: List[str]

    def detector_power_summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for item in self.detector_hits:
            name = str(item["surface"])
            power = float(np.sum(item["power"]))
            out[name] = out.get(name, 0.0) + power
        return out


@dataclass
class RayTracer:
    scene: Scene
    backend: str = "numpy"
    max_interactions: int = 16
    max_time_s: Optional[float] = None
    min_power: float = 1e-18
    surface_epsilon: float = 1e-7
    record_segments: bool = True
    max_active_rays: Optional[int] = None
    compiled_scene: CompiledScene = field(init=False)

    def __post_init__(self) -> None:
        self.compiled_scene = self.scene.compile(self.backend)

    def _distance_to_time(self, distance_m: Any, refractive_index: Any) -> Any:
        xp = get_array_module(distance_m, refractive_index)
        return xp.asarray(distance_m, dtype=float) * xp.asarray(refractive_index, dtype=float) / LIGHT_SPEED_M_S

    def _time_to_distance(self, delta_time_s: Any, refractive_index: Any) -> Any:
        xp = get_array_module(delta_time_s, refractive_index)
        return xp.asarray(delta_time_s, dtype=float) * LIGHT_SPEED_M_S / xp.asarray(refractive_index, dtype=float)

    def _make_segment_block(self, rays: RayBundle, end_points: Any, surface_names: Any, end_time_s: Any) -> Dict[str, Any]:
        return {
            "ray_id": to_numpy(rays.ray_id),
            "parent_id": to_numpy(rays.parent_id),
            "depth": to_numpy(rays.depth),
            "surface": to_numpy(surface_names),
            "x0": to_numpy(rays.position[:, 0]),
            "y0": to_numpy(rays.position[:, 1]),
            "z0": to_numpy(rays.position[:, 2]),
            "x1": to_numpy(end_points[:, 0]),
            "y1": to_numpy(end_points[:, 1]),
            "z1": to_numpy(end_points[:, 2]),
            "power": to_numpy(rays.power),
            "intensity": to_numpy(rays.intensity),
            "t0_s": to_numpy(rays.time_s),
            "t1_s": to_numpy(end_time_s),
        }

    def _propagate_to_time_limit(self, rays: RayBundle, delta_time_s: Any) -> RayBundle:
        end_distance = self._time_to_distance(delta_time_s, rays.refractive_index)
        end_points = rays.position + end_distance[:, None] * rays.direction
        return rays.propagate(end_points, delta_time_s)

    def trace(self, rays: RayBundle) -> TraceResult:
        xp = get_backend(self.backend)
        active = rays
        finished: List[RayBundle] = []
        segments: List[Dict[str, Any]] = []
        detector_hits: List[Dict[str, Any]] = []

        next_id = int(np.max(to_numpy(active.ray_id))) + 1 if active.n_rays > 0 else 0

        for depth in range(self.max_interactions):
            if active.n_rays == 0:
                break
            if self.max_active_rays is not None and active.n_rays > self.max_active_rays:
                order = np.argsort(-to_numpy(active.power))[: self.max_active_rays]
                active = active.subset(xp.asarray(order, dtype=np.int64))

            if self.max_time_s is not None:
                remaining_time = float(self.max_time_s) - active.time_s
                within_time_mask = remaining_time > 0.0
                if count_nonzero(~within_time_mask) > 0:
                    finished.append(active.subset(~within_time_mask))
                if count_nonzero(within_time_mask) == 0:
                    active = RayBundle.concat([], backend=self.backend)
                    break
                active = active.subset(within_time_mask)
                remaining_time = float(self.max_time_s) - active.time_s

            nearest = self.compiled_scene.find_nearest(active)
            hit_mask = nearest.surface_index >= 0
            hit_time = self._distance_to_time(nearest.distance, active.refractive_index)
            if self.max_time_s is not None:
                hit_mask = hit_mask & (hit_time <= remaining_time + 1e-15)
            miss_mask = ~hit_mask

            if count_nonzero(miss_mask) > 0:
                missed = active.subset(miss_mask)
                if self.max_time_s is not None:
                    miss_remaining_time = float(self.max_time_s) - missed.time_s
                    miss_propagated = self._propagate_to_time_limit(missed, miss_remaining_time)
                    if self.record_segments:
                        surface_names = np.full(miss_propagated.n_rays, "<time_limit>", dtype=object)
                        segments.append(
                            self._make_segment_block(
                                missed,
                                miss_propagated.position,
                                surface_names,
                                miss_propagated.time_s,
                            )
                        )
                    finished.append(miss_propagated)
                else:
                    finished.append(missed)

            if count_nonzero(hit_mask) == 0:
                active = RayBundle.concat([], backend=self.backend)
                break

            if self.record_segments:
                idx_host = to_numpy(nearest.surface_index[hit_mask]).astype(int)
                hit_rays = active.subset(hit_mask)
                hit_end_time = active.time_s[hit_mask] + hit_time[hit_mask]
                surface_names = np.asarray([self.compiled_scene.surfaces[i].name for i in idx_host], dtype=object)
                segments.append(
                    self._make_segment_block(
                        hit_rays,
                        nearest.points[hit_mask],
                        surface_names,
                        hit_end_time,
                    )
                )

            children: List[RayBundle] = []
            for i, surface in enumerate(self.compiled_scene.surfaces):
                surf_mask = hit_mask & (nearest.surface_index == i)
                if count_nonzero(surf_mask) == 0:
                    continue
                rays_sub = active.subset(surf_mask).propagate(nearest.points[surf_mask], hit_time[surf_mask])
                hit_sub = nearest.subset(surf_mask)
                new_children, new_hits = surface.interact(rays_sub, hit_sub, self)

                for record in new_hits:
                    detector_hits.append(record)

                for child in new_children:
                    if child.n_rays == 0:
                        continue
                    new_ids = xp.arange(next_id, next_id + child.n_rays, dtype=np.int64)
                    next_id += child.n_rays
                    child = child.replace_ids(ray_id=new_ids, parent_id=child.parent_id)
                    keep = child.power > self.min_power
                    if count_nonzero(keep) > 0:
                        children.append(child.subset(keep))

            active = RayBundle.concat(children, backend=self.backend)

        if active.n_rays > 0:
            if self.max_time_s is not None:
                remaining_time = float(self.max_time_s) - active.time_s
                positive_time_mask = remaining_time > 0.0
                if count_nonzero(positive_time_mask) > 0:
                    active_to_limit = active.subset(positive_time_mask)
                    propagated = self._propagate_to_time_limit(active_to_limit, remaining_time[positive_time_mask])
                    if self.record_segments:
                        surface_names = np.full(propagated.n_rays, "<time_limit>", dtype=object)
                        segments.append(
                            self._make_segment_block(
                                active_to_limit,
                                propagated.position,
                                surface_names,
                                propagated.time_s,
                            )
                        )
                    finished.append(propagated)
                if count_nonzero(~positive_time_mask) > 0:
                    finished.append(active.subset(~positive_time_mask))
            else:
                finished.append(active)

        final_rays = RayBundle.concat(finished, backend=self.backend)
        return TraceResult(
            final_rays=final_rays,
            segments=segments,
            detector_hits=detector_hits,
            surface_names=[s.name for s in self.compiled_scene.surfaces],
        )


# -----------------------------------------------------------------------------
# Fresnel
# -----------------------------------------------------------------------------



def fresnel_amplitudes(cos_i: Any, cos_t: Any, n1: Any, n2: Any, tir: Any):
    xp = get_array_module(cos_i, cos_t, n1, n2, tir)
    denom_s = n1 * cos_i + n2 * cos_t
    denom_p = n2 * cos_i + n1 * cos_t
    rs = (n1 * cos_i - n2 * cos_t) / xp.maximum(denom_s, 1e-15)
    rp = (n2 * cos_i - n1 * cos_t) / xp.maximum(denom_p, 1e-15)
    ts = (2.0 * n1 * cos_i) / xp.maximum(denom_s, 1e-15)
    tp = (2.0 * n1 * cos_i) / xp.maximum(denom_p, 1e-15)
    rs = xp.where(tir, 1.0, rs)
    rp = xp.where(tir, 1.0, rp)
    ts = xp.where(tir, 0.0, ts)
    tp = xp.where(tir, 0.0, tp)
    return rs, rp, ts, tp


# -----------------------------------------------------------------------------
# Convenience builders inspired by the COMSOL report
# -----------------------------------------------------------------------------



def build_demo_scene(n_glass: float = 1.50) -> Scene:
    """
    Small demonstration scene with a 50/50 beamsplitter, one mirror and a detector.
    The geometry is *not* the imported SAT geometry; it is a clean primitive-based
    optical bench showing the intended architecture.
    """

    scene = Scene()
    scene.add(
        BeamSplitter(
            name="BS_50_50",
            center=(-0.34448, -0.80321, 3.05),
            normal=normalize(np.asarray((0.0, 1.0, 1.0), dtype=float)),
            shape="rectangle",
            width=0.05,
            height=0.05,
            reflectance=0.5,
            transmittance=0.5,
            n_minus=AIR,
            n_plus=AIR,
            in_plane_reference=(1.0, 0.0, 0.0),
        ),
        PlaneMirror(
            name="Mirror_arm",
            center=(-0.34448, -0.76, 2.96),
            normal=(0.0, -1.0, 0.0),
            shape="rectangle",
            width=0.08,
            height=0.05,
            in_plane_reference=(1.0, 0.0, 0.0),
        ),
        Window(
            name="FS_window",
            center=(-0.34448, -0.80321, 2.86),
            normal=(0.0, 0.0, -1.0),
            thickness=0.008,
            n_glass=n_glass,
            n_outside=AIR,
            shape="disk",
            radius=0.02,
            use_fresnel=True,
        ),
        Detector(
            name="Detector_main",
            center=(-0.34448, -0.80321, 2.60),
            normal=(0.0, 0.0, 1.0),
            shape="rectangle",
            width=0.08,
            height=0.08,
            in_plane_reference=(1.0, 0.0, 0.0),
        ),
        BeamDump(
            name="Dump_side",
            center=(-0.34448, -0.65, 2.95),
            normal=(0.0, -1.0, 0.0),
            shape="rectangle",
            width=0.10,
            height=0.10,
            in_plane_reference=(1.0, 0.0, 0.0),
        ),
    )
    return scene


__all__ = [
    "AIR",
    "Material",
    "InteractionMode",
    "SurfaceOptics",
    "RayBundle",
    "PlaneSurface",
    "SphericalCapSurface",
    "CylinderSurface",
    "OpticalElement",
    "PlaneMirror",
    "BeamSplitter",
    "BeamDump",
    "Detector",
    "Window",
    "SphericalLens",
    "Scene",
    "CompiledScene",
    "GaussianBeamSource",
    "RayTracer",
    "TraceResult",
    "build_demo_scene",
    "get_backend",
    "get_array_module",
    "to_numpy",
]
