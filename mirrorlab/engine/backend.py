from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np


ArrayLike = Any
Number = Union[int, float]
IndexModel = Union[Number, Callable[[float], float], "Material"]
LIGHT_SPEED_M_S = 299_792_458.0


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
