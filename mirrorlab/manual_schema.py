"""Data-only adapters for the reviewed manual reconstruction bundle.

Archive programs are never imported. Manual reconstructions remain dependent
references, not additional survey measurements or proof of physical validity.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

from .state import MAX_BYTES, digest, new_document, snapshot, validate_document
from .directions import ROUTES
from .engine.bodies import MirrorBody
from .calibration.physical import rotation_from_normal

BUNDLE = Path(__file__).resolve().parents[1] / "data" / "manual_refinement"
FORMATS = {"mirrorlab.manual-reconstruction-input", "analysis.manual-reconstruction-seeds"}


def is_manual_data(data):
    return isinstance(data, dict) and data.get("format") in FORMATS


def _loads(raw):
    from .legacy_import import _loads as strict_loads
    return strict_loads(raw)


def _primary(direction, payload):
    if direction == "west_to_east":
        return copy.deepcopy(payload["primary_measurement_ledger"])
    from .calibration.observations import observation_registry
    # The delivered workbook is part of this versioned input, not a mutable
    # current-working-directory file. Its normalized ledger is embedded below.
    return observation_registry(BUNDLE / "primary_measurements", direction)


def dataset_from_manual(data, raw, origin):
    if not is_manual_data(data) or data.get("schema_version") != 1:
        raise ValueError("Unsupported manual reconstruction schema")
    direction = data.get("route")
    west = direction == "east_to_west"
    if direction not in ROUTES or data.get("units") != "m":
        raise ValueError("Manual data must declare a physical route and metre units")
    if data.get("canonical_coordinate_frame", data.get("coordinate_frame")) != "survey":
        raise ValueError("Manual data must explicitly use survey coordinates")
    if len(data.get("mirrors", [])) != 112:
        raise ValueError("Manual data requires all 112 unique source records")
    if west:
        policy = data["coordinate_policy"]
        frame = dict(rotation=policy["current_native_to_survey_rotation"],
                     translation_m=policy["current_native_to_survey_translation_m"])
        from .configurations import load_configuration
        optics = snapshot(load_configuration(direction, "ideal"))
        optics_origin = "current_versioned_ideal_snapshot; explicit import baseline"
    else:
        frame = copy.deepcopy(data["native_to_survey"])
        optics = copy.deepcopy(data["optics_snapshot"])
        if digest(optics) != data["optics_sha256"]:
            raise ValueError("Manual optics snapshot hash mismatch")
        optics_origin = "embedded_manual_snapshot"
    q, t = np.asarray(frame["rotation"]), np.asarray(frame["translation_m"])
    if q.shape != (3, 3) or not np.allclose(q.T @ q, np.eye(3), atol=1e-8):
        raise ValueError("Invalid manual coordinate transform")
    frame.update(coordinate_frame="survey", calculation_frame="legacy",
                 handedness_change=bool(np.linalg.det(q) < 0))
    ledger = _primary(direction, data)
    observations = {r["id"]: r for r in ledger["records"]}
    mirrors, assignments = [], []
    for record in data["mirrors"]:
        source_id = record["source_manual_id"] if west else record["source_id"]
        native_id = record["native_stable_seed_id"] if west else record["native_id"]
        center_survey = np.asarray(record["survey_center_m"] if west else record["center_survey_m"], float)
        center = q.T @ (center_survey - t)
        if west:
            normal = q.T @ np.asarray(record["survey_original_plane_normal"])
            body = MirrorBody(center, rotation_from_normal(normal), mirror_id=native_id)
            matched = record.get("matched_primary_survey_id")
            observation_id = f"{direction}:survey:{matched}" if matched else None
            slot = None  # The native seed index is explicitly not a physical number.
            outgoing = record["survey_target_outgoing"]
        else:
            body = MirrorBody.from_dict(record["cylinder"])
            if not np.allclose(body.center_m, center, atol=1e-10, rtol=0):
                raise ValueError("EAST native/survey centres disagree")
            observation_id = record.get("primary_observation_id")
            slot = record["canonical_slot"]
            outgoing = record["outgoing_survey"]
        if observation_id is not None and observation_id not in observations:
            raise ValueError("Unknown primary observation in manual assignment")
        observation = observations.get(observation_id, {})
        count = observation.get("complete_points", 0)
        m = {**body.to_dict(), "old_id": source_id, "canonical_id": native_id,
             "source_manual_id": source_id, "survey_observation_id": observation_id,
             "physical_native_id": native_id, "canonical_slot": slot,
             "physical_number_verified": False, "status": "prior_dominated",
             "observation_status": "measured" if count == 2 else "partial" if count else "predicted",
             "observation_ids": [observation_id] if observation_id else [],
             "manual_reference_survey": copy.deepcopy(record),
             "target_outgoing_native": (q.T @ np.asarray(outgoing)).tolist(),
             "trajectory": None,
             "uncertainty": {"body_roll": "not_observed", "longitudinal_center": "prior_dominated",
                             "metrology": "unknown"},
             "mapping_origin": "bundle_hypothesis_not_independent_measurement"}
        if not west and source_id == "3_2_7":
            m.update(mapping_origin="bundle_confirmed_internal_gap_hypothesis", raw_manual_direction_used_as_prior=False)
            if slot != 5:
                raise ValueError("Bundled EAST 3.2 missing ray must occupy internal slot 5")
        mirrors.append(m)
        assignments.append(dict(canonical_id=native_id, source_manual_id=source_id,
                                observation_ids=m["observation_ids"], canonical_slot=slot,
                                mapping_origin=m["mapping_origin"]))
    used_observations = [m["survey_observation_id"] for m in mirrors if m["survey_observation_id"] is not None]
    if len(used_observations) != len(set(used_observations)):
        raise ValueError("A primary observation cannot be assigned to multiple manual mirrors")
    payload_hash = hashlib.sha256(raw).hexdigest()
    return dict(direction=direction, physical_sides=ROUTES[direction].metadata(), status="legacy_unvalidated",
        model_kind="manual_reconstruction", mirrors=mirrors, observations=ledger, assignments=assignments,
        template=None, display_frame=frame, optics_snapshot=optics, optics_sha256=digest(optics),
        original_optics_snapshot=copy.deepcopy(optics), optics_origin=optics_origin,
        manual_references=dict(schema_version=1, source_file=origin, sha256=payload_hash,
            lineage_id=f"manual:{payload_hash}", source_kind="dependent_manual_reconstruction",
            units="m", coordinate_frame="survey", raw_utf8=raw.decode("utf-8"),
            payload=copy.deepcopy(data), used_as_primary_measurements=False),
        mechanics=dict(rod_containment="not_evaluated", holder_containment="not_evaluated", waist_side="not_evaluated"),
        diagnostics=dict(limitations=["Imported manual geometry is not a validated physical configuration",
            "Mechanical holder/rod/waist dimensions and metrology are unknown",
            "Manual references do not contribute additional independent measurements"],
            supplied_baseline=copy.deepcopy(data.get("validation", data.get("summary", {})))))


def _known_text(raw, origin):
    """Exact source identity selects a reviewed correction, never the filename.

Unknown edited tables must not silently inherit historical registration or
the internal missing-ray hypothesis from a different measurement epoch.
"""
    sha = hashlib.sha256(raw).hexdigest()
    for filename in ("west_manual_corrected_survey.json", "east_recovered_centers.json"):
        candidate_raw = (BUNDLE / filename).read_bytes()
        data = _loads(candidate_raw)
        if sha == data.get("source_sha256", data.get("source_manual_sha256")):
            result = dataset_from_manual(data, candidate_raw, filename)
            result["manual_references"]["original_text"] = dict(source_file=origin, sha256=sha,
                raw_utf8=raw.decode("utf-8"), source_kind="manual_reference",
                derived_adapter="hash_matched_reviewed_bundle_correction")
            return result
    raise ValueError("Неизвестная версия ручного TXT: для неё нужны явная привязка и JSON с метаданными. Поддерживаются оба исходных TXT поставленного пакета.")


def import_manual(path, direction=None, coordinate_frame=None):
    if coordinate_frame not in (None, "survey"):
        raise ValueError("Manual bundle coordinates are survey; no second legacy transform is permitted")
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Manual input exceeds 100 MB")
    datasets = {}
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            infos = archive.infolist()
            if sum(i.file_size for i in infos) > MAX_BYTES:
                raise ValueError("Manual archive expands beyond 100 MB")
            for info in infos:
                name = PurePosixPath(info.filename.replace("\\", "/"))
                if name.is_absolute() or ".." in name.parts or ":" in str(name):
                    raise ValueError("Unsafe manual archive path")
                if info.is_dir() or name.suffix.lower() != ".json":
                    continue
                raw = archive.read(info)
                data = _loads(raw)
                if not is_manual_data(data):
                    continue
                route = data.get("route")
                if direction not in (None, "both", route):
                    continue
                if route in datasets:
                    if datasets[route]["manual_references"]["sha256"] != hashlib.sha256(raw).hexdigest():
                        # Analysis directory has earlier iterations; only canonical
                        # top-level deliverables are import inputs.
                        if len(name.parts) > 1:
                            continue
                        raise ValueError("Conflicting manual inputs for one side")
                    continue
                if len(name.parts) > 1:
                    continue
                datasets[route] = dataset_from_manual(data, raw, f"{path.name}!{name}")
    else:
        raw = path.read_bytes()
        if path.suffix.lower() == ".json":
            result = dataset_from_manual(_loads(raw), raw, path.name)
        elif path.suffix.lower() == ".txt":
            result = _known_text(raw, path.name)
        else:
            raise ValueError("Manual import supports JSON, TXT and data ZIP")
        if direction not in (None, "both", result["direction"]):
            raise ValueError("Selected side conflicts with manual metadata")
        datasets[result["direction"]] = result
    if not datasets:
        raise ValueError("No recognized manual reconstruction inputs")
    return validate_document(new_document(datasets))
