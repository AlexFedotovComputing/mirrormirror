from __future__ import annotations

import raytrace as _impl

# Backward-compatible shim. New code should import from `comsol_raytrace`.
__all__ = list(getattr(_impl, "__all__", ()))
globals().update({name: getattr(_impl, name) for name in __all__})
