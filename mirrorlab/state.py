"""Portable physical configurations. Loading is validation, never recalibration.

All coordinates are metres. A body's rotation is proper; a documented change
of coordinate basis may reverse handedness. The two are validated separately.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

import numpy as np

from .configurations import StaticConfiguration, StaticElement, TYPES, _decode
from .directions import ROUTES
from .engine import Scene

FORMAT = "mirrorlab.physical-configuration"
SCHEMA_VERSION = 1
STATUSES = {"validated", "partial", "infeasible", "legacy_unvalidated"}
MAX_BYTES = 100_000_000
# Versioned format defaults, deliberately independent of future RayTracer
# constructor defaults. Early v1 documents may omit these ordinary settings.
SCHEMA_TRACE_DEFAULTS = dict(min_power=1e-18, surface_epsilon=1e-7,
    record_segments=True, max_active_rays=None, bundle_clip_inner_radius_m=None,
    bundle_clip_outer_radius_m=None, bundle_clip_z_min_m=None, bundle_clip_z_max_m=None,
    bundle_clip_epsilon_m=1e-9, skip_repeated_bundle_reflections=False)


def encode(value):
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {"type": type(value).__name__, "parameters": {
            f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init}}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1:
        # File timestamps/opaque identifiers must not lose bits in Electron.
        return str(value)
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if callable(value):
        raise ValueError("Callable optical models cannot be saved as portable data")
    return value


def digest(value):
    # JSON's number type has no int/float distinction. Electron serializes 1.0
    # as 1 and -0.0 as 0; hashes must survive that lossless numeric round trip.
    def canonical(item):
        if isinstance(item, float) and math.isfinite(item) and item.is_integer():
            return int(item)
        if isinstance(item, list):
            return [canonical(x) for x in item]
        if isinstance(item, dict):
            return {k: canonical(v) for k, v in item.items()}
        return item
    return hashlib.sha256(json.dumps(canonical(encode(value)), ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def snapshot(config):
    """Common optics are copied, not referenced by a local repository path."""
    return dict(source=encode(config.source),
                elements=[dict(name=e.name, surfaces=[encode(s) for s in e.build_surfaces()])
                          for e in config.scene.elements if not e.name.startswith("BUNDLE_")],
                integration_time_s=config.integration_time_s,
                max_interactions=config.max_interactions,
                tracer_options={**SCHEMA_TRACE_DEFAULTS, **copy.deepcopy(config.tracer_options)},
                metadata=copy.deepcopy(config.metadata))


def _finite_tree(value, path="document", depth=0):
    if depth > 80:
        raise ValueError("Configuration nesting is too deep")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite number: {path}")
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        raise ValueError(f"Large integer must be represented as a string: {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}", depth + 1)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _finite_tree(item, f"{path}[{i}]", depth + 1)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError(f"Non-JSON value: {path}")


def vector(value, size=3, label="vector"):
    a = np.asarray(value, float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"Invalid {label}: expected {size} finite values")
    return a


def rotation(value, proper=True):
    a = np.asarray(value, float)
    if a.shape != (3, 3) or not np.isfinite(a).all() or not np.allclose(a.T @ a, np.eye(3), atol=1e-8):
        raise ValueError("Rotation must be an orthogonal 3 by 3 matrix")
    det = np.linalg.det(a)
    if (proper and not np.isclose(det, 1, atol=1e-8)) or not np.isclose(abs(det), 1, atol=1e-8):
        raise ValueError("Physical body rotation must have det(R)=+1")
    return a


def _validate_encoded(value):
    if isinstance(value, list):
        for item in value:
            _validate_encoded(item)
    elif isinstance(value, dict):
        if "type" in value or "parameters" in value:
            if set(value) != {"type", "parameters"} or value["type"] not in TYPES:
                raise ValueError("Unsupported optical object type")
            cls = TYPES[value["type"]]
            allowed = {f.name for f in dataclasses.fields(cls) if f.init}
            if not isinstance(value["parameters"], dict) or set(value["parameters"]) - allowed:
                raise ValueError("Unsupported optical object parameters")
        for item in value.values():
            _validate_encoded(item)


def _validate_snapshot(data):
    if not isinstance(data, dict):
        raise ValueError("Missing embedded optics")
    required = {"source", "elements", "integration_time_s", "max_interactions", "tracer_options", "metadata"}
    if not required <= data.keys():
        raise ValueError("Incomplete optics snapshot")
    _validate_encoded(data["source"])
    if data["source"].get("type") != "GaussianBeamSource":
        raise ValueError("Unsupported source")
    source = _decode(data["source"])
    vector(source.waist_position)
    if np.linalg.norm(vector(source.axis)) < 1e-12 or source.waist_radius <= 0 or source.wavelength_m <= 0:
        raise ValueError("Invalid light source dimensions/direction")
    if source.n_medium <= 0 or source.peak_intensity < 0:
        raise ValueError("Invalid light source medium/power")
    if type(data["max_interactions"]) is not int or not 1 <= data["max_interactions"] <= 10000:
        raise ValueError("Invalid interaction limit")
    if not isinstance(data["integration_time_s"], (float, int)) or data["integration_time_s"] <= 0:
        raise ValueError("Invalid integration time")
    from .engine import RayTracer
    allowed_options = {f.name for f in dataclasses.fields(RayTracer) if f.init} - {
        "scene", "backend", "max_interactions", "max_time_s", "collision_mode"}
    if not isinstance(data["tracer_options"], dict) or set(data["tracer_options"]) - allowed_options:
        raise ValueError("Unsupported tracer settings")
    names, surfaces = set(), set()
    for e in data["elements"]:
        if not isinstance(e.get("name"), str) or e["name"] in names or e["name"].startswith("BUNDLE_"):
            raise ValueError("Duplicate or invalid common optical element ID")
        names.add(e["name"])
        for item in e["surfaces"]:
            _validate_encoded(item)
            if item["type"] not in {"PlaneSurface", "CylinderSurface", "SphericalCapSurface"}:
                raise ValueError("Common optical boundary must be a supported Surface")
            for name in ("axis", "normal"):
                if name in item["parameters"] and not np.isclose(np.linalg.norm(vector(item["parameters"][name])), 1., atol=1e-8):
                    raise ValueError("Serialized surface directions must already be normalized")
            s = _decode(item)
            from .engine.materials import InteractionMode, Material
            if s.optics.mode not in {mode.value for mode in InteractionMode}:
                raise ValueError("Unsupported optical interaction mode")
            for name in ("reflectance", "transmittance"):
                fraction = getattr(s.optics, name)
                if fraction is not None and (isinstance(fraction, bool) or not 0 <= fraction <= 1):
                    raise ValueError("Optical power fractions must be between zero and one")
            for name in ("n_minus", "n_plus"):
                medium = getattr(s.optics, name)
                value = medium.refractive_index_model if isinstance(medium, Material) else medium
                if isinstance(value, bool) or not isinstance(value, (float, int)) or value <= 0:
                    raise ValueError("Embedded refractive index must be a positive numeric constant")
            if getattr(s, "shape", None) == "triangle":
                vertices = np.asarray(s.triangle_vertices, float)
                if vertices.shape != (3, 2) or abs(np.linalg.det(np.c_[vertices, np.ones(3)])) < 1e-18:
                    raise ValueError("Invalid optical triangle")
            if s.name in surfaces:
                raise ValueError("Duplicate surface name")
            surfaces.add(s.name)
            for field in ("center", "vertex"):
                if hasattr(s, field):
                    vector(getattr(s, field), label=field)
            for field in ("normal", "axis"):
                if hasattr(s, field) and not np.isclose(np.linalg.norm(vector(getattr(s, field))), 1, atol=1e-8):
                    raise ValueError("Invalid optical surface direction")
            for field in ("radius", "length", "width", "height"):
                v = getattr(s, field, None)
                if v is not None and v <= 0:
                    raise ValueError("Optical dimensions must be positive")


def validate_document(document):
    """Validate the complete candidate without mutating it or an active scene."""
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise ValueError("Неизвестный формат; для архивных данных выберите «Загрузить легаси».")
    if type(document.get("schema_version")) is not int or document["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Неподдерживаемая schema_version: {document.get('schema_version')}")
    _finite_tree(document)
    if document.get("units") != "m" or document.get("status") not in STATUSES:
        raise ValueError("Invalid document units/status")
    if not isinstance(document.get("configuration_id"), str) or not document["configuration_id"]:
        raise ValueError("Missing configuration ID")
    datasets = document.get("datasets")
    if not isinstance(datasets, dict) or not datasets or set(datasets) - set(ROUTES):
        raise ValueError("Directions must be explicitly separated")
    from .engine.bodies import MirrorBody
    for direction, dataset in datasets.items():
        if dataset.get("direction") != direction or dataset.get("status") not in STATUSES:
            raise ValueError("Dataset direction/status mismatch")
        if dataset.get("physical_sides") != encode(ROUTES[direction].metadata()):
            raise ValueError("Physical side metadata mismatch")
        optics = dataset.get("optics_snapshot")
        if optics is not None:
            if digest(optics) != dataset.get("optics_sha256"):
                raise ValueError("Embedded optics hash mismatch")
            _validate_snapshot(optics)
        elif dataset["status"] == "validated":
            raise ValueError("Validated configuration cannot lack common optics")
        frame = dataset.get("display_frame")
        if frame is not None:
            r = rotation(frame.get("rotation"), proper=False)
            vector(frame.get("translation_m"))
            if np.linalg.det(r) < 0 and not frame.get("handedness_change"):
                raise ValueError("Coordinate handedness change must be documented")
        observations = dataset.get("observations", [])
        if isinstance(observations, dict):
            observations = observations.get("records", observations.get("observations", []))
        obs_ids = [o.get("observation_id", o.get("id")) for o in observations]
        if len(set(obs_ids)) != len(obs_ids) or any(x is None for x in obs_ids):
            raise ValueError("Duplicate or missing observation IDs")
        for observation in observations:
            screens = observation.get("screens_m")
            if screens is not None:
                if len(screens) != 2 or any(not isinstance(p, list) or len(p) != 3 for p in screens):
                    raise ValueError("Survey observations require two masked XYZ vectors")
                for p in screens:
                    if any(v is not None and (isinstance(v, bool) or not isinstance(v, (float, int))) for v in p):
                        raise ValueError("Survey components must be finite numbers or null")
                mask = observation.get("missing_mask")
                if mask is not None and mask != [[v is None for v in p] for p in screens]:
                    raise ValueError("Survey missing-component mask disagrees with raw observations")
                if observation.get("units", "m") != "m":
                    raise ValueError("Normalized observations must be in metres")
        mirrors = dataset.get("mirrors", [])
        if len(mirrors) != 112 and not dataset.get("import_incomplete"):
            raise ValueError("Each processed direction requires all 112 mirrors")
        ids, old_ids = set(), set()
        for m in mirrors:
            cid, oid = m.get("canonical_id"), m.get("old_id")
            if not isinstance(cid, str) or not re.fullmatch(r"[1-4]_[1-4]_[1-7]", cid) or cid in ids or not re.fullmatch(r"[1-4]_[1-4]_[1-7]", str(oid)) or oid in old_ids:
                raise ValueError("Duplicate or invalid mirror IDs")
            ids.add(cid)
            old_ids.add(oid)
            if m.get("mirror_id", cid) != cid:
                raise ValueError("Physical surface ID must equal the canonical mirror ID")
            if m.get("status") not in {"data_supported", "data_and_mechanics", "prior_dominated", "ambiguous", "infeasible", "legacy_unvalidated", "failed"}:
                raise ValueError("Unknown mirror evidence status")
            if m.get("observation_status") not in {"measured", "partial", "predicted", "ambiguous", "failed"}:
                raise ValueError("Unknown trajectory observation status")
            vector(m.get("center_m"), label="cut centre")
            r = rotation(m.get("rotation"))
            for name, local in (("axis", [0, 0, 1]), ("normal", np.array([1, 0, -1]) / np.sqrt(2))):
                if name in m and not np.allclose(vector(m[name]), r @ local, atol=1e-8):
                    raise ValueError(f"Body {name} is inconsistent with its rotation")
            if m.get("length_definition", "full_extent") not in {"full_extent", "cut_center_to_back"}:
                raise ValueError("Unknown cylinder length convention")
            radius, height = m.get("radius_m", .0005), m.get("axial_height_m", .0095)
            if radius <= 0 or height < radius:
                raise ValueError("Invalid skew-cylinder dimensions")
            MirrorBody.from_dict(m)
            trajectory = m.get("trajectory")
            if trajectory and trajectory.get("direction") is not None:
                if not np.isclose(np.linalg.norm(vector(trajectory["direction"])), 1, atol=1e-7):
                    raise ValueError("Trajectory direction must be unit length")
                vector(trajectory.get("point_m", m["center_m"]))
            if trajectory:
                for path in [trajectory, *trajectory.get("paths", [])]:
                    for point in path.get("points_m", []):
                        vector(point, label="optical path point")
                    for segment in path.get("segments", []):
                        vector(segment.get("start_m"), label="segment start")
                        vector(segment.get("end_m"), label="segment end")
                        if segment.get("direction") is not None and not np.isclose(np.linalg.norm(vector(segment["direction"])), 1., atol=1e-7):
                            raise ValueError("Optical segment direction must be unit length")
            for name, covariance in m.get("uncertainty", {}).items():
                if "covariance" in name and covariance is not None:
                    matrix = np.asarray(covariance, float)
                    if matrix.shape != (3, 3) or not np.allclose(matrix, matrix.T, atol=1e-10) or np.linalg.eigvalsh(matrix).min() < -1e-10:
                        raise ValueError("Mirror covariance must be a 3 by 3 PSD matrix")
            for observation_id in m.get("observation_ids", []):
                if observation_id not in obs_ids:
                    raise ValueError("Dangling mirror observation reference")
        assignments = dataset.get("assignments", [])
        if isinstance(assignments, list):
            for a in assignments:
                if a.get("canonical_id") is not None and a["canonical_id"] not in ids:
                    raise ValueError("Dangling canonical assignment")
                for oid in a.get("observation_ids", []):
                    if oid not in obs_ids:
                        raise ValueError("Dangling assignment observation")
                for hypothesis in a.get("alternatives", []):
                    destinations = []
                    sources = []
                    for link in hypothesis.get("mapping", []):
                        if link.get("canonical_id") not in ids or link.get("observation_id") not in obs_ids:
                            raise ValueError("Dangling alternative assignment")
                        destinations.append(link["canonical_id"])
                        sources.append(link["observation_id"])
                    if len(set(destinations)) != len(destinations) or len(set(sources)) != len(sources):
                        raise ValueError("Each assignment hypothesis must be one-to-one")
        template = dataset.get("template")
        if template:
            directions = np.asarray(template.get("directions_local"), float)
            if directions.shape != (7, 3) or not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-7):
                raise ValueError("Template requires seven unit directions")
            for statistics in template.get("statistics", []):
                covariance = statistics.get("covariance_direction")
                if covariance is not None:
                    covariance = np.asarray(covariance, float)
                    if covariance.shape != (3, 3) or not np.allclose(covariance, covariance.T, atol=1e-10) or np.linalg.eigvalsh(covariance).min() < -1e-10:
                        raise ValueError("Template covariance must be symmetric positive semidefinite")
    if document["status"] == "validated" and any(d["status"] != "validated" for d in datasets.values()):
        raise ValueError("Overall status cannot promote an unvalidated side")
    return document


def new_document(datasets, version=0):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    statuses = {x["status"] for x in datasets.values()}
    status = "validated" if statuses == {"validated"} else (
        "infeasible" if statuses == {"infeasible"} else "legacy_unvalidated" if statuses == {"legacy_unvalidated"} else "partial")
    source_root = Path(__file__).resolve().parent
    source_hashes = {str(p.relative_to(source_root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source_root.rglob("*.py"))}
    return dict(format=FORMAT, schema_version=SCHEMA_VERSION, application_commit=commit,
                implementation_sha256=digest(source_hashes), implementation_files_sha256=source_hashes,
                created_at=datetime.now(timezone.utc).isoformat(), configuration_id=str(uuid4()),
                configuration_version=version, units="m", status=status, datasets=encode(datasets))


def dataset_from_config(config, result, display_frame=None):
    result = copy.deepcopy(result)
    result["direction"] = config.direction
    result["physical_sides"] = ROUTES[config.direction].metadata()
    result["optics_snapshot"] = snapshot(config)
    result["optics_sha256"] = digest(result["optics_snapshot"])
    if display_frame is not None:
        result["display_frame"] = encode(display_frame)
        result["display_frame"]["handedness_change"] = bool(np.linalg.det(display_frame["rotation"]) < 0)
    return result


def configuration_from_dataset(dataset):
    data = dataset.get("optics_snapshot")
    if data is None:
        raise ValueError("В импорте отсутствует общая оптика; трассировка недоступна.")
    config = StaticConfiguration(direction=dataset["direction"], kind="reconstruction",
        scene=Scene([StaticElement(e["name"], [_decode(s) for s in e["surfaces"]]) for e in data["elements"]]),
        source=_decode(data["source"]), integration_time_s=data["integration_time_s"],
        max_interactions=data["max_interactions"], tracer_options={**SCHEMA_TRACE_DEFAULTS, **copy.deepcopy(data["tracer_options"])},
        metadata={**copy.deepcopy(data["metadata"]), "configuration_status": dataset["status"],
                  "model_kind": dataset.get("model_kind", "reconstruction")})
    from .calibration.physical import install_bodies
    from .engine.bodies import MirrorBody
    bodies = [MirrorBody.from_dict(m) for m in dataset["mirrors"]]
    install_bodies(config, bodies)
    config.reconstruction = copy.deepcopy(dataset)
    return config


def save_document(document, path):
    validate_document(document)
    encoded = (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ValueError("Configuration exceeds 100 MB")
    target = Path(path)
    if target.suffix.lower() != ".json":
        raise ValueError("Configuration filename must end in .json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=f".{target.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return dict(saved=True, path=str(target), sha256=hashlib.sha256(encoded).hexdigest())


def read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Configuration exceeds 100 MB")
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"Non-finite JSON value: {v}")))


def load_document(path):
    return validate_document(read_json(path))
