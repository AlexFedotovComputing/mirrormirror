from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np


from .backend import get_array_module, to_numpy, IndexModel


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
    reflect_from_minus_side: bool = True
    reflect_from_plus_side: bool = True
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
