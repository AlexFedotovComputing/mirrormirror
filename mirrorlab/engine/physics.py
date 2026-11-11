from __future__ import annotations

from typing import Any



from .backend import get_array_module


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
