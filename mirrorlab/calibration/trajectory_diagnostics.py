"""Evidence for individual comparisons, without changing fits or verdicts.

A fitted assignment is a diagnostic hypothesis, not a verified mirror identity.
Only a finite traced segment can explain a measurement. Its extension is used
solely to distinguish a truncated path from a geometric miss.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from .trajectory_matching import _complete_observations, _dataset_context, _pair_metrics


def _optical_causes(check):
    if not check:
        return ["optical_evidence_missing"]
    incoming = check.get("incoming", {})
    status = incoming.get("status")
    if status and status != "available":
        return [{"outside_illumination": "outside_illumination", "ambiguous": "incoming_ambiguous",
                 "branch_not_reached": "incoming_branch_missing",
                 "verification_failed": "incoming_verification_failed"}.get(status, "incoming_unavailable")]
    if check.get("first_surface") == "<time_limit>":
        return ["incoming_trace_time_limit"]
    if (check.get("first_surface") or "").startswith("<bundle_"):
        return ["incoming_trace_boundary"]
    if check.get("first_surface"):
        return ["intercepted_before_mirror"]
    if check.get("reason") == "Incoming branch not present in physical scene":
        return ["incoming_branch_missing"]
    codes = []
    if check.get("incident_dot_normal", -1.) >= -1e-8:
        codes.append("wrong_reflecting_side")
    if check.get("hit_on_ellipse") is False:
        codes.append("outside_reflecting_ellipse")
    if check.get("actual_reflected_child") is False:
        codes.append("no_reflected_child")
    return codes


def _path_evidence(row, points, path, tolerance):
    metric = _pair_metrics(row, points, path)
    extension = _pair_metrics(row, points, {**path, "length_m": np.inf})
    origin, direction = np.asarray(path["point_m"]), np.asarray(path["direction"])
    direction = direction / np.linalg.norm(direction)
    along = (points - origin) @ direction
    length = path["length_m"]
    checks = [dict(point_index=i + 1, point_m=point.tolist(), distance_m=metric["point_distances_m"][i],
                   extension_distance_m=extension["point_distances_m"][i],
                   along_path_m=float(along[i]), beyond_end_m=float(max(0., along[i] - length)),
                   behind_origin_m=float(max(0., -along[i]))) for i, point in enumerate(points)]
    code = "physical_path_mismatch"
    if metric["maximum_distance_m"] <= tolerance:
        code = "path_within_tolerance"
    elif extension["maximum_distance_m"] <= tolerance and np.any(along > length):
        surface = path.get("terminal_surface") or ""
        if surface.startswith("BUNDLE_"):
            code = "outgoing_mirror_interception"
        elif surface == "<time_limit>":
            code = "trace_time_limit"
        elif surface.startswith("<bundle_"):
            code = "trace_boundary"
        elif path.get("stop_reason") == "direction_change_or_discontinuity":
            code = "path_turn_or_discontinuity"
        else:
            code = "trace_ended_before_points"
    elif np.any(along < -tolerance):
        code = "points_behind_reflection"
    return dict(**metric, mirror_id=path.get("mirror_id"), length_m=length,
                end_m=path.get("end_m"), terminal_surface=path.get("terminal_surface"),
                stop_reason=path.get("stop_reason"), reason_code=code,
                extension_maximum_distance_m=extension["maximum_distance_m"],
                point_checks=checks, segments=path.get("segments", []))


def _mirror_evidence(mirror_id, mirrors, reflections, frame, geometry):
    if mirror_id is None:
        return None
    mirror = mirrors.get(mirror_id, {})
    check = reflections.get(mirror_id, mirror.get("optical_validation", {}))
    body = mirror.get("body", mirror)
    radius = body.get("radius_m")
    dimensions = (dict(cylinder_diameter_mm=2 * radius * 1000,
                       ellipse_minor_axis_mm=2 * radius * 1000,
                       ellipse_major_axis_mm=2 * math.sqrt(2) * radius * 1000)
                  if radius is not None else None)
    incident = check.get("incident_segment")
    if incident:
        q, t = np.asarray(frame["rotation"]), np.asarray(frame["translation_m"])
        incident = dict(surface=incident.get("surface"),
                        start_m=(np.asarray(incident["start_m"]) @ q.T + t).tolist(),
                        end_m=(np.asarray(incident["end_m"]) @ q.T + t).tolist())
    conflicts = [p for p in geometry.get("conflicts", []) if mirror_id in p.get("pair", [])]
    return dict(mirror_id=mirror_id, status=check.get("status", "not_evaluated"),
                incoming_status=check.get("incoming", {}).get("status"),
                optical_reason_codes=_optical_causes(check), first_surface=check.get("first_surface"),
                target_surface=check.get("target_surface"), incident_segment=incident,
                outgoing_blockers=check.get("outgoing_blockers", []),
                violations=mirror.get("violations", []), body_conflicts=conflicts,
                dimensions=dimensions, clearance_m=mirror.get("clearance_m"),
                axial_shift_m=mirror.get("template_deviation", {}).get("axial_physical_relaxation_m"))


def diagnose_trajectories(dataset, fit, physical):
    observations, frame, reason = _dataset_context(dataset)
    if reason:
        return dict(status="not_evaluated", reason=reason, observations=[])
    complete, excluded = _complete_observations(observations)
    fit_pairs = {p["observation_id"]: p for p in fit.get("matches", [])}
    fit_paths = {p["id"]: p for p in fit.get("trajectories", [])}
    paths = {p["id"]: p for p in physical.get("trajectories", [])}
    matches = {p["observation_id"]: p for p in physical.get("matches", [])}
    misses = {p["observation_id"]: p for p in physical.get("unmatched_observations", [])}
    assigned_to = {p["trajectory_id"]: p["observation_id"] for p in physical.get("matches", [])}
    mirrors = {m.get("mirror_id", m.get("canonical_id")): m for m in dataset.get("mirrors", [])}
    diagnostics = dataset.get("diagnostics", {})
    reflections = diagnostics.get("reflections", {}).get("mirrors", {})
    tolerance = physical.get("distance_tolerance_m", fit.get("distance_tolerance_m", .003))
    rows = []
    for row, points in complete:
        identifier = row["id"]
        fitted = fit_pairs.get(identifier)
        fit_mirror = fit_paths.get(fitted["trajectory_id"], {}).get("mirror_id") if fitted else None
        matched = matches.get(identifier)
        miss = misses.get(identifier, {})
        nearest = matched or miss.get("nearest")
        physical_mirror = paths.get(nearest["trajectory_id"], {}).get("mirror_id") if nearest else None
        evidence = _mirror_evidence(fit_mirror, mirrors, reflections, frame, diagnostics.get("geometry", {}))
        # Preserve the difference between the fit-associated mirror and a
        # possibly unrelated globally nearest path. Never blame the latter's
        # obstruction for failure of the fitted mirror.
        associated = [p for p in paths.values() if fit_mirror is not None and p.get("mirror_id") == fit_mirror]
        reference = paths[matched["trajectory_id"]] if matched else (
            min(associated, key=lambda p: _pair_metrics(row, points, p)["maximum_distance_m"]) if associated else None)
        path_check = _path_evidence(row, points, reference, tolerance) if reference else None
        reasons = []
        if physical.get("status") != "evaluated":
            status = "not_evaluated"
            reasons.append("physical_comparison_unavailable")
        elif matched:
            status = "matched"
            reasons.append("matched")
        else:
            status = "unmatched"
            if miss.get("reason") == "one_to_one_conflict":
                reasons.append("one_to_one_conflict")
            if fit.get("status") != "evaluated":
                reasons.append("geometric_fit_unavailable")
            elif fitted is None:
                reasons.append("no_fitted_assignment")
            elif not fitted.get("within_tolerance"):
                reasons.append("geometric_mismatch")
            if evidence:
                reasons.extend(evidence["optical_reason_codes"])
                if path_check and path_check["reason_code"] != "path_within_tolerance":
                    reasons.append(path_check["reason_code"])
                elif not path_check and not evidence["optical_reason_codes"]:
                    reasons.append("reflected_path_missing")
            if not reasons:
                reasons.append("physical_path_mismatch")
        rows.append(dict(observation_id=identifier, status=status, reason_codes=list(dict.fromkeys(reasons)),
                         fit_only=bool(status == "unmatched" and fitted and fitted.get("within_tolerance")),
                         fitted={**fitted, "mirror_id": fit_mirror} if fitted else None,
                         physical={**nearest, "mirror_id": physical_mirror,
                                   "matched": bool(matched),
                                   "assigned_observation_id": assigned_to.get(nearest["trajectory_id"])} if nearest else None,
                         fitted_mirror=evidence, path=path_check,
                         path_basis="matched_physical_path" if matched else "fitted_mirror_path",
                         screens_m=points.tolist()))
    rows.extend(dict(observation_id=e["observation_id"], status="excluded", fit_only=False,
                     reason_codes=[e["reason"]], fitted=None, physical=None, fitted_mirror=None, path=None)
                for e in excluded)
    return dict(status="evaluated", coordinate_frame="survey", distance_tolerance_m=tolerance,
                fit_only_count=sum(r["fit_only"] for r in rows),
                reason_counts=dict(Counter(code for r in rows for code in r["reason_codes"])),
                observations=rows, attribution="fitted_assignment_is_a_hypothesis_not_verified_identity",
                mechanics_are_separate=True)
