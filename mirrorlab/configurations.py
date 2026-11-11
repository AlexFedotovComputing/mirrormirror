"""Static optics data, direction conventions and deterministic source sampling."""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .engine import (
    CylinderSurface, GaussianBeamSource, Material, OpticalElement, PlaneSurface,
    Scene, SphericalCapSurface, SurfaceOptics, to_numpy,
)

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "static"
TYPES = {cls.__name__: cls for cls in (
    CylinderSurface, GaussianBeamSource, Material, PlaneSurface,
    SphericalCapSurface, SurfaceOptics,
)}
DIRECTIONS = {
    "west_to_east": {"source_side": "WEST", "source_sign": "+", "injection_side": "EAST", "injection_sign": "−"},
    "east_to_west": {"source_side": "EAST", "source_sign": "−", "injection_side": "WEST", "injection_sign": "+"},
}


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if "type" in value and "parameters" in value:
            cls = TYPES[value["type"]]
            return cls(**{name: _decode(item) for name, item in value["parameters"].items()})
        return {name: _decode(item) for name, item in value.items()}
    return value


@dataclass
class StaticElement(OpticalElement):
    """A preserved optical element with its explicitly compiled boundary geometry."""
    name: str
    surfaces: list

    def build_surfaces(self):
        return self.surfaces


@dataclass
class StaticConfiguration:
    direction: str
    kind: str
    scene: Scene
    source: GaussianBeamSource
    integration_time_s: float
    max_interactions: int
    tracer_options: dict
    metadata: dict
    overlays: list = field(default_factory=list)

    def emit(self, ray_count: int = 217):
        """Sample a hexapolar Gaussian grid reproducibly, retaining its total power."""
        if isinstance(ray_count, bool) or not isinstance(ray_count, int) or not 1 <= ray_count <= 50000:
            raise ValueError("ray_count must be an integer from 1 to 50000")
        source = copy.deepcopy(self.source)
        source.radial_positions = max(1, math.ceil((math.sqrt(12 * ray_count - 3) - 3) / 6))
        rays = source.emit()
        if rays.n_rays != ray_count:
            total_power = float(np.sum(to_numpy(rays.power)))
            indices = np.floor(np.linspace(0, rays.n_rays, ray_count, endpoint=False)).astype(int)
            rays = rays.subset(rays.xp.asarray(indices))
            rays.power *= total_power / float(np.sum(to_numpy(rays.power)))
        return rays


def load_configuration(direction: str = "west_to_east", kind: str = "ideal", backend: str = "numpy") -> StaticConfiguration:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction: {direction}")
    if kind not in {"ideal", "real"}:
        raise ValueError("kind must be 'ideal' or 'real'")
    if backend not in {"numpy", "cupy"}:
        raise ValueError("backend must be 'numpy' or 'cupy'")
    path = DATA_DIRECTORY / f"{direction}_{kind}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if kind == "real":
        base_name = f"{direction}_ideal.json"
        if data.get("base_configuration") != base_name:
            raise ValueError(f"Real configuration must inherit common optics from {base_name}")
        base = json.loads((DATA_DIRECTORY / base_name).read_text(encoding="utf-8"))
        overrides = {item["name"]: item for item in data["elements"]}
        expected = {item["name"] for item in base["elements"] if item["name"].startswith("BUNDLE_")}
        if set(overrides) != expected or len(overrides) != len(data["elements"]):
            raise ValueError("Real configuration must override exactly the ideal microassemblies")
        data = {**base, **data,
                "elements": [overrides.get(item["name"], item) for item in base["elements"]],
                "provenance": {**data["provenance"], "common_optics_file": base_name,
                               "common_optics_provenance": base["provenance"]}}
    source = _decode(data["source"])
    source.backend = backend
    elements = [StaticElement(item["name"], [_decode(s) for s in item["surfaces"]]) for item in data["elements"]]
    return StaticConfiguration(
        direction=direction, kind=kind, scene=Scene(elements), source=source,
        integration_time_s=data["integration_time_s"], max_interactions=data["max_interactions"],
        tracer_options=data["tracer_options"],
        metadata={**DIRECTIONS[direction], "coordinate_frame": "legacy",
                  "units": "m", "provenance": data["provenance"]},
    )


def _bundle_surfaces(config):
    for surface in config.scene.build_surfaces():
        match = re.fullmatch(r"BUNDLE_(\d+)_(\d+) Mirror (\d+)", surface.name)
        if match:
            yield "_".join(match.groups()), surface


def configuration_ray_references(config: StaticConfiguration) -> dict:
    """Expected outgoing axes in the application's world frame (metres)."""
    incoming = np.array((0., 0., 1. if config.direction == "west_to_east" else -1.))
    references = {}
    for ray_id, surface in _bundle_surfaces(config):
        normal = to_numpy(surface.normal)
        outgoing = incoming - 2 * np.dot(incoming, normal) * normal
        references[ray_id] = {"point_m": to_numpy(surface.center).tolist(), "direction": outgoing.tolist()}
    return references


def apply_fitted_rays(config: StaticConfiguration, rays: list, max_shift_m: float = 0.002) -> dict:
    """Apply already registered outgoing axes; raw survey coordinates are invalid.

    Rotation follows specular reflection. Translation toward each fitted axis is
    capped and reported, so uncertain registration cannot silently move hardware
    over macroscopic distances. Callers own frame registration and acceptance.
    """
    if not math.isfinite(max_shift_m) or max_shift_m < 0:
        raise ValueError("max_shift_m must be finite and nonnegative")
    surfaces = dict(_bundle_surfaces(config))
    incoming = np.array((0., 0., 1. if config.direction == "west_to_east" else -1.))
    updates = []
    seen = set()
    for ray in rays:
        key = ray["ray_id"]
        if key in seen:
            raise ValueError(f"Duplicate fitted ray: {key}")
        seen.add(key)
        if key not in surfaces:
            raise ValueError(f"Unknown fitted ray: {key}")
        surface = surfaces[key]
        point = np.asarray(ray["point_m"], dtype=float)
        outgoing = np.asarray(ray["direction"], dtype=float)
        if point.shape != (3,) or outgoing.shape != (3,) or not np.isfinite([point, outgoing]).all():
            raise ValueError(f"Invalid fitted ray coordinates: {key}")
        length = np.linalg.norm(outgoing)
        if length < 1e-12:
            raise ValueError(f"Zero direction: {key}")
        outgoing /= length
        old = np.asarray(surface.center)
        nearest = point + np.dot(old - point, outgoing) * outgoing
        requested_shift = nearest - old
        requested_distance = float(np.linalg.norm(requested_shift))
        normal = incoming - outgoing
        if np.linalg.norm(normal) < 1e-12:
            raise ValueError(f"Fitted ray does not define a reflective normal: {key}")
        normal /= np.linalg.norm(normal)
        if np.dot(normal, surface.normal) < 0:
            normal *= -1
        scale = min(1., max_shift_m / max(requested_distance, 1e-15))
        if abs(normal[2]) > 1e-12:
            chief_offset = np.r_[-requested_shift[:2], np.dot(normal[:2], requested_shift[:2]) / normal[2]]
            scale = min(scale, 0.5 * surface.radius / max(float(np.linalg.norm(chief_offset)), 1e-15))
        new_center = old + scale * requested_shift
        reference = np.asarray(surface._u) - np.dot(surface._u, normal) * normal
        if np.linalg.norm(reference) < 1e-12:
            reference = np.eye(3)[np.argmin(np.abs(normal))]
        # Validate everything before mutating; each reconstructed plane derives
        # its local basis again from its new normal and in-plane reference.
        updates.append((surface, new_center, normal, reference, key, requested_distance, scale))
    report = []
    for surface, center, normal, reference, key, requested, scale in updates:
        surface.center, surface.normal, surface.in_plane_reference = center, normal, reference
        surface.__post_init__()
        report.append({"ray_id": key, "requested_shift_m": requested, "applied_shift_m": requested * scale,
                       "remaining_line_distance_m": requested * (1 - scale)})
    config.metadata["fitted_mirrors"] = len(report)
    return {"updated_count": len(report), "mirrors": report}
