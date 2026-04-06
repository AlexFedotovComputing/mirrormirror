from __future__ import annotations

import vizual as _impl

# Backward-compatible shim. New code should import from `comsol_vizual`.
__all__ = list(getattr(_impl, "__all__", ()))
if not __all__:
    __all__ = [name for name in dir(_impl) if not name.startswith("_")]
globals().update({name: getattr(_impl, name) for name in __all__})
