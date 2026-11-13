"""Restricted adapters for the data formats shipped with MirrorLab.

No archive extraction, Python execution, AST evaluation, or filename-based
physical-side inference is performed. Missing optics remains missing.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

from .configurations import StaticConfiguration, StaticElement, _decode, _bundle_surfaces
from .directions import ROUTES
from .engine import Scene
from .state import MAX_BYTES, dataset_from_config, new_document, read_json, validate_document

# Reviewed export pair hashes from the requested base e57e447. A newer local
# base cannot silently replace one of these files during an old delta import.
BASELINE_HASHES = {
    "east_to_west_ideal.json": "2744aa001c880e8b3f72e60d65b6585293f06b759dad166ef3f94067b49ff4c0",
    "east_to_west_real.json": "30e41d5a3e0ea2d171b61ba380cbe19ca1ad3e50bebc3c164d55ffded6c92fa4",
    "west_to_east_ideal.json": "200b00cb6a79d6a2441e7d7e0c31fc4d3de9892e053f33e12556e6b3c38c6107",
    "west_to_east_real.json": "b3b6f9e2a12e645ca5e90b8f6eb0c1b60875f8028139495b92ee3a22318c3bef",
}


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _safe_base_name(name):
    return (isinstance(name, str) and bool(name) and name not in {".", ".."}
            and not any(c in name for c in ("/", "\\", ":"))
            and PurePosixPath(name).name == name)


def _loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite legacy JSON")))


def _mirror(oid, center, normal, provenance):
    from .calibration.physical import rotation_from_normal
    from .engine.bodies import MirrorBody
    body = MirrorBody(center_m=center, rotation=rotation_from_normal(normal), mirror_id=oid)
    return {**body.to_dict(), "old_id": oid, "canonical_id": oid,
            "status": "prior_dominated", "observation_status": "predicted", "trajectory": None,
            "uncertainty": {"body_roll": "not_observed", "centre": "legacy_unvalidated"},
            "provenance": provenance,
            "assumptions": ["Cylinder roll chosen by deterministic normal-to-SO(3) prior",
                            "1 mm diameter; 10 mm total extent; opaque absorbing side/back assumed"]}


def historical_dataset(config, provenance=None):
    from .calibration.physical import incoming_at_mirrors, rotation_from_normal
    from .engine.bodies import MirrorBody
    field = incoming_at_mirrors(config)
    mirrors = []
    for oid, surface in _bundle_surfaces(config):
        normal = np.asarray(surface.normal).copy()
        din = field.get(oid, {}).get("direction")
        original = dict(normal=normal.tolist(), reflect_from_minus_side=surface.optics.reflect_from_minus_side,
                        reflect_from_plus_side=surface.optics.reflect_from_plus_side)
        if din is not None and np.dot(normal, din) > 0:
            normal *= -1
        mirror = _mirror(oid, np.asarray(surface.center).tolist(), normal.tolist(),
                         provenance or config.metadata.get("provenance", {}))
        if din is not None:
            mirror.update(MirrorBody(mirror["center_m"], rotation_from_normal(normal, axis_hint=din), mirror_id=oid).to_dict())
        mirror["imported_surface_convention"] = original
        mirror["physical_normal_conversion"] = "outward_against_traced_incident_branch" if din is not None else "not_evaluated"
        mirrors.append(mirror)
    return dataset_from_config(config, dict(direction=config.direction, status="legacy_unvalidated",
        model_kind="historical_" + config.kind, mirrors=mirrors, observations=[], assignments=[], template=None,
        mechanics={"rod_containment": "not_evaluated", "waist_side": "not_evaluated"},
        diagnostics={"limitations": ["Legacy plane normals do not measure cylinder roll or body pose",
                                      "Physical body geometry introduced by explicit working assumptions"]}))


def _static_json(raw, members, origin, expected_direction=None):
    data = _loads(raw)
    direction = data.get("direction")
    if direction not in ROUTES:
        raise ValueError("Legacy JSON must specify its physical direction")
    if expected_direction not in (None, "both", direction):
        raise ValueError("Выбранная сторона не совпадает с метаданными JSON.")
    provenance = dict(file=origin, sha256=_hash(raw), source_kind="derived_legacy",
                      original=data.get("provenance", {}))
    missing = []
    if data.get("base_configuration"):
        base_name = data["base_configuration"]
        if not _safe_base_name(base_name):
            raise ValueError("Base configuration must be a sibling filename")
        base_raw = members.get(base_name)
        expected_hash = data.get("base_sha256", data.get("base_configuration_sha256"))
        if _hash(raw) == BASELINE_HASHES.get(f"{direction}_real.json"):
            expected_hash = BASELINE_HASHES[f"{direction}_ideal.json"]
        if base_raw is None or expected_hash is None or _hash(base_raw) != expected_hash:
            missing.append("Common optics: declared base is missing or its identity/hash cannot be established")
        else:
            base = _loads(base_raw)
            if base.get("direction") != direction or base.get("kind") != "ideal" or base.get("base_configuration"):
                raise ValueError("Incompatible legacy base")
            overrides = {e["name"]: e for e in data["elements"]}
            expected = {e["name"] for e in base["elements"] if e["name"].startswith("BUNDLE_")}
            if set(overrides) != expected or len(overrides) != len(data["elements"]):
                raise ValueError("Legacy real delta must override exactly its base bundles")
            provenance["base_sha256"] = _hash(base_raw)
            data = {**base, "kind": "real", "elements": [overrides.get(e["name"], e) for e in base["elements"]]}
    from .state import _validate_encoded
    _validate_encoded(data.get("source", {}))
    for element in data.get("elements", []):
        for surface in element["surfaces"]:
            _validate_encoded(surface)
    elements = [StaticElement(e["name"], [_decode(s) for s in e["surfaces"]]) for e in data["elements"]]
    if not missing:
        config = StaticConfiguration(direction=direction, kind=data.get("kind", "ideal"), scene=Scene(elements),
            source=_decode(data["source"]), integration_time_s=data["integration_time_s"],
            max_interactions=data["max_interactions"], tracer_options=data["tracer_options"],
            metadata={**ROUTES[direction].metadata(), "coordinate_frame": "legacy", "units": "m", "provenance": provenance})
        dataset = historical_dataset(config, provenance)
    else:
        mirrors = []
        for element in elements:
            for s in element.surfaces:
                match = re.fullmatch(r"BUNDLE_([1-4])_([1-4]) Mirror ([1-7])", s.name)
                if match:
                    mirrors.append(_mirror("_".join(match.groups()), s.center, s.normal, provenance))
        dataset = dict(direction=direction, physical_sides=ROUTES[direction].metadata(), status="legacy_unvalidated",
                       mirrors=mirrors, import_incomplete=True, optics_snapshot=None, observations=[], assignments=[],
                       template=None, diagnostics={"limitations": missing}, model_kind="historical_delta")
    return dataset


def _table(raw, direction, coordinate_frame, origin):
    if direction not in ROUTES or coordinate_frame not in {"legacy", "survey"}:
        raise ValueError("Для TXT выберите одну физическую сторону и систему координат (архивная или геодезическая). Имя файла этого не определяет.")
    text = raw.decode("utf-8-sig")
    mirrors, observations = [], []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"\s*([1-4])[._]([1-4])[._]([1-7])\s+", line)
        if not match:
            continue
        values = re.findall(r"\(([^()]+)\)", line[match.end():])
        points = []
        for value in values:
            parts = value.split(",")
            if len(parts) != 3:
                raise ValueError(f"{origin}:{line_number}: expected 3-vector")
            points.append(np.array([float(x) for x in parts]))
        if len(points) not in {2, 4, 6}:
            raise ValueError(f"{origin}:{line_number}: supported table has 4 corners, optional centre/normal, or centre/normal")
        if len(points) >= 4:
            center = np.mean(points[:4], axis=0)
            normal = np.cross(points[1] - points[0], points[3] - points[0])
            if len(points) == 6:
                if not np.allclose(center, points[4], atol=1e-6):
                    raise ValueError("Declared cut centre disagrees with corner centroid")
                normal = points[5]
        else:
            center, normal = points
        if np.linalg.norm(normal) < 1e-12:
            raise ValueError("Degenerate legacy cut normal")
        normal /= np.linalg.norm(normal)
        provenance = dict(file=origin, sha256=_hash(raw), line=line_number, units="m",
                          coordinate_frame=coordinate_frame, source_kind="derived_legacy", original_line=line)
        oid = "_".join(match.groups())
        mirrors.append(_mirror(oid, center, normal, provenance))
    if not mirrors:
        # Known tabular line-equation format: rod, window_ray, x0,y0,z0,dx,dy,dz.
        rod = None
        for line_number, line in enumerate(text.splitlines(), 1):
            cells = line.split("\t")
            if len(cells) < 8 or not re.fullmatch(r"[1-4]_[1-7]", cells[1].strip()):
                continue
            if cells[0].strip():
                rod = int(cells[0])
            if rod not in range(1, 5):
                raise ValueError("Invalid rod in legacy equations")
            from .calibration.io import number
            vals = [number(v) for v in cells[2:8]]
            oid = f"{rod}_{cells[1].strip()}"
            observations.append(dict(id=oid, observation_id=oid, original_id=oid, source_kind="derived_equation",
                values=vals, missing_mask=[v is None for v in vals], source_units="mm", coordinate_frame=coordinate_frame,
                file=origin, sha256=_hash(raw), line=line_number, used_for_training=False))
        if not observations:
            raise ValueError("Unsupported TXT: expected supplied four-corner/centre-normal table or ray equations")
    return dict(direction=direction, physical_sides=ROUTES[direction].metadata(), status="legacy_unvalidated",
        model_kind="historical_table" if mirrors else "derived_observations_only", mirrors=mirrors,
        import_incomplete=len(mirrors) != 112, observations=observations, assignments=[], template=None,
        optics_snapshot=None, coordinate_frame=coordinate_frame,
        diagnostics={"limitations": ["Common optics, mechanical registration, and cylinder roll are not established",
                                    "Imported plane corners are metadata; the physical cut is an ellipse",
                                    "Ray equations alone are not a physical configuration"]})


def import_legacy(path, direction=None, coordinate_frame=None, kind="real"):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Legacy file exceeds 100 MB")
    from .manual_schema import is_manual_data, import_manual, _known_text
    if path.suffix.lower() == ".json" and is_manual_data(_loads(path.read_bytes())):
        return import_manual(path, direction, coordinate_frame)
    if path.suffix.lower() == ".txt":
        try:
            manual = _known_text(path.read_bytes(), path.name)
        except ValueError:
            manual = None
        if manual is not None:
            return import_manual(path, direction, coordinate_frame)
    members = {}
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            infos = archive.infolist()
            if sum(i.file_size for i in infos) > MAX_BYTES:
                raise ValueError("Legacy archive expands beyond 100 MB")
            for info in infos:
                name = PurePosixPath(info.filename.replace("\\", "/"))
                if name.is_absolute() or ".." in name.parts or ":" in str(name):
                    raise ValueError("Unsafe archive path")
                if name.suffix.lower() not in {".json", ".txt"} or info.is_dir():
                    continue
                raw = archive.read(info)
                if name.name in members and members[name.name] != raw:
                    raise ValueError("Archive contains ambiguous duplicate data filenames")
                members[name.name] = raw
        candidates = []
        if any(is_manual_data(_loads(raw)) for name, raw in members.items() if name.endswith(".json")):
            return import_manual(path, direction, coordinate_frame)
        for name, raw in members.items():
            if name.endswith(".json"):
                try:
                    data = _loads(raw)
                    if data.get("direction") in ROUTES and data.get("elements") and data.get("kind") == kind:
                        if direction in (None, "both", data["direction"]):
                            candidates.append((name, raw))
                except (ValueError, AttributeError):
                    continue
        if candidates:
            datasets = {}
            for name, raw in candidates:
                dataset = _static_json(raw, members, f"{path.name}!{name}", direction)
                if dataset["direction"] in datasets:
                    raise ValueError("Archive contains several configurations for one direction")
                datasets[dataset["direction"]] = dataset
            return validate_document(new_document(datasets))
        tables = [(name, raw) for name, raw in members.items() if re.search(
            rb"(?m)^\s*[1-4][._][1-4][._][1-7]\s+\(", raw)]
        # Identical duplicated archive data count as a single source.
        unique = {raw: name for name, raw in tables}
        if len(unique) != 1:
            raise ValueError("Архив должен содержать однозначную таблицу микросборок или статический JSON; Python не выполняется.")
        raw, name = next(iter(unique.items()))
        dataset = _table(raw, direction, coordinate_frame, f"{path.name}!{name}")
    elif path.suffix.lower() == ".json":
        raw = path.read_bytes()
        data = _loads(raw)
        base_name = data.get("base_configuration")
        if base_name and not _safe_base_name(base_name):
            raise ValueError("Base configuration must be a sibling filename")
        if base_name:
            base_path = path.parent / base_name
            if base_path.is_file() and base_path.stat().st_size <= MAX_BYTES:
                members[base_name] = base_path.read_bytes()
        dataset = _static_json(raw, members, path.name, direction)
    elif path.suffix.lower() == ".txt":
        dataset = _table(path.read_bytes(), direction, coordinate_frame, path.name)
    else:
        raise ValueError("Поддерживаются статические JSON, таблицы TXT и ZIP с данными. Python-файлы не выполняются.")
    return validate_document(new_document({dataset["direction"]: dataset}))
