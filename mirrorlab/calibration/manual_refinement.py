"""Conservative, bounded refinement of imported manual reference poses.

Primary masked points alone form the data likelihood. The bundled optical
candidate is an explicit conditional start; every returned body is retraced.
Unknown mechanical/metrology inputs always prevent automatic acceptance.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..engine.bodies import MirrorBody
from ..state import configuration_from_dataset, digest
from .physical import (incoming_at_points, reflecting_normal, rotation_from_normal,
                       validate_bodies, validate_reflections, unit)


def masked_point_residual(center, direction, point):
    """Distance after minimizing the unknown position along a masked line."""
    mask = np.array([v is not None for v in point])
    if not mask.any():
        return np.empty(0)
    delta = np.asarray([v for v in point if v is not None], float) - np.asarray(center)[mask]
    vector = np.asarray(direction)[mask]
    square = float(vector @ vector)
    return delta - vector * (float(delta @ vector) / square) if square > 1e-20 else delta


def _points(mirror, ledger):
    record = ledger.get(mirror.get("survey_observation_id"), {})
    return [p for p in record.get("screens_m", []) if any(v is not None for v in p)]


def _metrics(mirrors, centers, directions, q, t, ledger):
    by_mirror, all_errors, complete_errors, single_errors, groups = {}, [], [], [], {}
    for m, c, d in zip(mirrors, centers, directions):
        errors = [float(np.linalg.norm(masked_point_residual(q @ c + t, q @ d, p)))
                  for p in _points(m, ledger)]
        all_errors.extend(errors)
        record = ledger.get(m.get("survey_observation_id"), {})
        full = [error for p, error in zip(_points(m, ledger), errors) if all(v is not None for v in p)]
        if record.get("complete_points") == 2:
            complete_errors.extend(full)
        else:
            single_errors.extend(full)
        group = "_".join(m["source_manual_id"].split("_")[:2])
        g = groups.setdefault(group, dict(errors=[], complete_line_errors=[]))
        g["errors"].extend(errors)
        if record.get("complete_points") == 2:
            g["complete_line_errors"].extend(full)
        by_mirror[m["canonical_id"]] = dict(point_residuals_mm=(np.array(errors) * 1000).tolist(),
            rms_mm=float(np.sqrt(np.mean(np.square(errors))) * 1000) if errors else None,
            observation_id=m.get("survey_observation_id"))
    def summary(errors):
        return dict(count=len(errors), rms_mm=float(np.sqrt(np.mean(np.square(errors))) * 1000) if errors else None,
                    max_mm=float(max(errors) * 1000) if errors else None)
    return dict(point_count=len(all_errors), rms_mm=float(np.sqrt(np.mean(np.square(all_errors))) * 1000)
                if all_errors else None, complete_point_count=len(complete_errors),
                complete_point_rms_mm=float(np.sqrt(np.mean(np.square(complete_errors))) * 1000)
                if complete_errors else None, complete_point_definition="points_from_two_complete_screen_records",
                single_points=summary(single_errors),
                assemblies={group: dict(all_points=summary(g["errors"]), complete_lines=summary(g["complete_line_errors"]))
                            for group, g in groups.items()}, mirrors=by_mirror)


def refine_manual(dataset, options=None, progress=None):
    options = dict(options or {})
    if not dataset.get("manual_references"):
        raise ValueError("refine_manual requires an imported manual dataset")
    settings = dict(point_sigma_m=.001, center_sigma_m=.0005, center_bound_m=.0015,
                    direction_sigma_rad=.003, direction_bound_rad=.015,
                    max_nfev=45, required_gap_m=0., numerical_tolerance_m=1e-8,
                    use_supplied_optical_seed=True)
    unknown = set(options) - set(settings)
    if unknown:
        raise ValueError("Unsupported manual refinement options: " + ", ".join(sorted(unknown)))
    settings.update(options)
    for key, value in settings.items():
        if key == "use_supplied_optical_seed":
            if type(value) is not bool:
                raise ValueError("Optical seed selection must be boolean")
        elif not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or (value == 0 and key != "required_gap_m"):
            raise ValueError("Refinement scales must be finite positive scenarios")
    if type(settings["max_nfev"]) is not int or settings["max_nfev"] > 500:
        raise ValueError("max_nfev must be an integer between 1 and 500")
    def notify(stage, fraction):
        if progress:
            progress(dict(stage=stage, fraction=fraction))
    notify("manual_inputs", .02)
    result = copy.deepcopy(dataset)
    raw = result["manual_references"]["payload"]
    west = dataset["direction"] == "east_to_west"
    mirrors = result["mirrors"]
    q = np.asarray(dataset["display_frame"]["rotation"])
    t = np.asarray(dataset["display_frame"]["translation_m"])
    ledger = {r["id"]: r for r in result["observations"]["records"]}
    initial_centers = [np.asarray(m["center_m"]) for m in mirrors]
    initial_directions = [unit(m["target_outgoing_native"]) for m in mirrors]
    before = _metrics(mirrors, initial_centers, initial_directions, q, t, ledger)
    optical_changes = []
    if west and settings["use_supplied_optical_seed"]:
        candidate = raw["recommended_optimization_candidate"]
        optics = copy.deepcopy(candidate["calibrated_optics_snapshot"])
        if digest(optics) != candidate["calibrated_optics_sha256"]:
            raise ValueError("Supplied calibrated optics hash mismatch")
        result["optics_snapshot"] = optics
        result["optics_sha256"] = digest(optics)
        optical_changes = copy.deepcopy(candidate["MC_steering"])
        for m, source in zip(mirrors, raw["mirrors"]):
            m.update(copy.deepcopy(source["recommended_optimization_seed"]["body_native"]))
    config = configuration_from_dataset(result)
    centers = np.array([m["center_m"] for m in mirrors])
    directions = np.array(initial_directions)
    field = incoming_at_points(config, {m["canonical_id"]: c for m, c in zip(mirrors, centers)})
    # Reconstruct normals from the actual incident branch. Historic normals
    # are preserved in raw references and never treated as measured tilt.
    seed_bodies = []
    for m, c, d in zip(mirrors, centers, directions):
        incoming = field[m["canonical_id"]].get("direction")
        if incoming is None:
            seed_bodies.append(MirrorBody.from_dict(m))
        elif west and not settings["use_supplied_optical_seed"]:
            seed_bodies.append(MirrorBody(c, rotation_from_normal(reflecting_normal(incoming, d), incoming), mirror_id=m["canonical_id"]))
        else:
            seed_bodies.append(MirrorBody.from_dict(m))
    notify("manual_primary_fit", .15)
    fitted_centers, fitted_directions, fit_reports = [], [], {}
    for index, (m, c0, d0) in enumerate(zip(mirrors, centers, directions)):
        points = _points(m, ledger)
        # Dimensionless parameters permit meaningful bounds across m and rad.
        center_scale = settings["center_bound_m"] / math.sqrt(3)
        angle_scale = settings["direction_bound_rad"] / math.sqrt(3)
        def unpack(x):
            return c0 + x[:3] * center_scale, Rotation.from_rotvec(x[3:] * angle_scale).apply(d0)
        def data_residual(x):
            c, d = unpack(x)
            return np.concatenate([masked_point_residual(q @ c + t, q @ d, p) /
                settings["point_sigma_m"] for p in points]) if points else np.empty(0)
        def residual(x):
            return np.r_[data_residual(x), x[:3] * center_scale / settings["center_sigma_m"],
                         x[3:] * angle_scale / settings["direction_sigma_rad"]]
        fit = least_squares(residual, np.zeros(6), bounds=(-1., 1.), loss="soft_l1",
                            max_nfev=settings["max_nfev"])
        c, d = unpack(fit.x)
        fitted_centers.append(c); fitted_directions.append(d)
        # Rank of primary data alone, not the regularized optimizer Jacobian.
        h = 1e-5
        jac = np.column_stack([(data_residual(fit.x + np.eye(6)[j] * h) -
                                data_residual(fit.x - np.eye(6)[j] * h)) / (2*h) for j in range(6)])
        singular = np.linalg.svd(jac, compute_uv=False) if points else np.empty(0)
        rank = int(np.sum(singular > max(float(singular[0]) * 1e-7, 1e-8))) if len(singular) else 0
        fit_reports[m["canonical_id"]] = dict(optimizer_success=bool(fit.success), primary_jacobian_rank=rank,
            primary_jacobian_singular_values=singular.tolist(), parameter_count=6,
            center_shift_m=float(np.linalg.norm(c-c0)), direction_shift_rad=float(np.arccos(np.clip(d@d0,-1,1))),
            active_parameter_bounds=np.flatnonzero(np.abs(fit.x) > .999).tolist(),
            unobserved=["longitudinal_center", "body_roll"], uncertainty_kind="conditional_regularized_scenario")
        if index % 14 == 0:
            notify("manual_primary_fit", .15 + .35 * index / len(mirrors))
    notify("manual_physical_validation", .52)
    fitted_field = incoming_at_points(config, {m["canonical_id"]: c for m, c in zip(mirrors, fitted_centers)})
    fitted_bodies = []
    for m, old, c, d in zip(mirrors, seed_bodies, fitted_centers, fitted_directions):
        incoming = fitted_field[m["canonical_id"]].get("direction")
        if incoming is None:
            fitted_bodies.append(MirrorBody(c, old.rotation, mirror_id=old.mirror_id))
            continue
        normal = reflecting_normal(incoming, d)
        # Minimal rotation transports existing body roll without inventing an
        # observed roll constraint or breaking proper SO(3) body geometry.
        cross = np.cross(old.normal, normal)
        size = np.linalg.norm(cross)
        angle = np.arctan2(size, old.normal @ normal)
        rotate = Rotation.from_rotvec(cross / size * angle).as_matrix() if size > 1e-14 else np.eye(3)
        fitted_bodies.append(MirrorBody(c, rotate @ old.rotation, mirror_id=old.mirror_id))
    checks = []
    for label, bodies, dirs in (("conditional_start", seed_bodies, directions),
                                ("primary_fit", fitted_bodies, fitted_directions)):
        mechanical = validate_bodies(bodies, required_gap_m=settings["required_gap_m"],
            numerical_tolerance_m=settings["numerical_tolerance_m"], check_length_alternative=False)
        optical = validate_reflections(config, bodies)
        metrics = _metrics(mirrors, [b.center_m for b in bodies], dirs, q, t, ledger)
        checks.append(dict(label=label, bodies=bodies, directions=dirs, mechanics=mechanical,
                           optics=optical, metrics=metrics))
        notify("manual_physical_validation", .65 if label == "conditional_start" else .9)
    # Never trade a newly introduced collision/occlusion for a smaller data
    # penalty. A conditional seed with conflicts remains diagnostic as well.
    seed, fitted = checks
    admissible = [seed]
    if (len(fitted["mechanics"]["conflicts"]) <= len(seed["mechanics"]["conflicts"]) and
            fitted["optics"]["validated_count"] >= seed["optics"]["validated_count"]):
        admissible.append(fitted)
    chosen = min(admissible, key=lambda c: (len(c["mechanics"]["conflicts"]),
        112-c["optics"]["validated_count"], c["metrics"]["rms_mm"] if c["metrics"]["rms_mm"] is not None else math.inf))
    for m, body, outgoing in zip(mirrors, chosen["bodies"], chosen["directions"]):
        m.update(body.to_dict())
        physical = chosen["optics"]["mirrors"][body.mirror_id]
        m["physics"] = physical
        m["target_outgoing_native"] = np.asarray(outgoing).tolist()
        m["trajectory"] = ({**physical["reflected_child"], "paths": physical.get("reflected_trajectories", [])}
                           if physical.get("actual_reflected_child") else None)
        m["status"] = "infeasible" if physical["status"] != "validated" else "prior_dominated"
        m["refinement"] = dict(before=before["mirrors"][body.mirror_id],
            after=chosen["metrics"]["mirrors"][body.mirror_id],
            proposed_fit=fit_reports[body.mirror_id], selected_candidate=chosen["label"],
            total_center_shift_m=0.)
    for i, m in enumerate(mirrors):
        m["refinement"]["total_center_shift_m"] = float(np.linalg.norm(np.asarray(m["center_m"]) - initial_centers[i]))
    result["status"] = "infeasible" if chosen["mechanics"]["conflicts"] or chosen["optics"]["validated_count"] < 112 else "partial"
    result["mechanics"] = chosen["mechanics"]
    result["physical_validation"] = chosen["optics"]
    result["manual_refinement"] = dict(version=1, settings=settings, scales_are_measured_tolerances=False,
        before=before, after=chosen["metrics"], selected_candidate=chosen["label"],
        candidate_comparison=[dict(label=c["label"], metrics=c["metrics"],
            body_conflict_count=len(c["mechanics"]["conflicts"]), clear_optical_paths=c["optics"]["validated_count"])
            for c in checks], optical_changes=optical_changes,
        method="bounded_primary_masked_point_fit_with_independent_full_scene_candidate_selection",
        acceptance="diagnostic_only_unknown_holder_rod_waist_and_metrology",
        independent_validation=dict(status="not_evaluated", reason="Conditional manual priors are not independent holdout predictions"),
        limitations=["Individual bounded fits use the supplied assembly reference; no inferred manufacturing tolerances",
                     "Discrete assignments remain explicit bundle hypotheses; alternatives are not optimized",
                     "Common optics calibration is the supplied bounded hypothesis, not an independent measurement",
                     "No posterior confidence intervals without independent metrology"])
    result["diagnostics"]["limitations"] = result["manual_refinement"]["limitations"]
    result["diagnostics"]["manual_refinement"] = copy.deepcopy(result["manual_refinement"])
    notify("manual_complete", 1.)
    return result
