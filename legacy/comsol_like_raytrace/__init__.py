from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_IMPL_MODULE_NAME = f"{__name__}._impl"
_IMPL_PATH = Path(__file__).resolve().parent.parent / "comsol_like_raytrace.py"

_impl = sys.modules.get(_IMPL_MODULE_NAME)
if _impl is None:
    _spec = spec_from_file_location(_IMPL_MODULE_NAME, _IMPL_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Could not load ray-tracing implementation from {_IMPL_PATH}")
    _impl = module_from_spec(_spec)
    sys.modules[_IMPL_MODULE_NAME] = _impl
    _spec.loader.exec_module(_impl)

__all__ = list(getattr(_impl, "__all__", ()))
globals().update({name: getattr(_impl, name) for name in __all__})
