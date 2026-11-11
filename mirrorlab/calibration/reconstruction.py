"""Cross-fitted empirical reconstruction with explicit conditional mechanics.

No fitted observation replaces the ledger. Missing mechanical dimensions and
unreachable optical branches remain reportable failures, never successful fits.
"""
from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation

from .empirical import angular_distance, assign_fan, learn_template, unit
from .observations import observation_registry
from ..configurations import _bundle_surfaces
from ..coordinates import intersect_ray_group


DEFAULTS = dict(seed=0, template_iterations=5, assignment_resolution_rad=.006,
                measurement_sigma_m=.001, assembly_translation_sigma_m=.01,
                assembly_rotation_sigma_rad=.05, individual_center_sigma_m=.001,
                individual_angle_sigma_rad=.006, max_nfev=60,
                axial_center_sigma_m=.01, axial_center_limit_sigma=3.,
                physical_max_iterations=80, physical_starts=3,
                outlier_distance_m=.05, minimum_gap_m=0., numerical_epsilon_m=1e-9,
                validate_holdout=True, holdout_per_assembly=1, uncertainty_scenarios_m=[.0005, .001, .002])


def _notify(progress, stage, **values):
    if progress:
        progress(dict(stage=stage, **values))


def _options(values):
    result = {**DEFAULTS, **(values or {})}
    for name in ("assignment_resolution_rad", "measurement_sigma_m", "assembly_translation_sigma_m",
                 "assembly_rotation_sigma_rad", "individual_center_sigma_m", "individual_angle_sigma_rad",
                 "axial_center_sigma_m", "axial_center_limit_sigma",
                 "outlier_distance_m", "numerical_epsilon_m"):
        if not np.isfinite(result[name]) or result[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(result["minimum_gap_m"]) or result["minimum_gap_m"] < 0:
        raise ValueError("minimum_gap_m must be nonnegative and finite")
    if not 1 <= int(result["template_iterations"]) <= 50 or not 1 <= int(result["max_nfev"]) <= 1000:
        raise ValueError("Invalid reconstruction iteration limit")
    return result


def group_observations(records):
    groups = {}
    for row in records:
        groups.setdefault(f"{row['rod']}_{row['window']}", dict(records=[], directions=[], points=[], ids=[]))["records"].append(row)
    for key, group in groups.items():
        full = [r for r in group["records"] if r["complete_points"] == 2]
        if len(full) < 2:
            group.update(anchor=None, anchor_rms_m=None)
            continue
        points = np.asarray([r["screens_m"][0] for r in full])
        dirs = unit(np.asarray([r["screens_m"][1] for r in full]) - points)
        anchor, rms = intersect_ray_group(points, dirs)
        dirs[np.sum((points - anchor) * dirs, axis=1) < 0] *= -1
        group.update(anchor=anchor.tolist(), anchor_rms_m=float(rms), points=points.tolist(),
                     directions=dirs.tolist(), ids=[r["id"] for r in full])
    return groups


def geometric_registration(config, groups, *, exclude_groups=()):
    """Enumerate rod numbering, axial orders, and coordinate handedness.

    The gas-centre plane and parallel survey/scene rod axes are explicit external
    geometric assumptions. Neither a physical reflection nor a scale is fitted.
    """
    surfaces = dict(_bundle_surfaces(config))
    centres = {(r, w): np.mean([np.asarray(surfaces[f"{r}_{w}_{j}"].center) for j in range(1, 8)], axis=0)
               for r in range(1, 5) for w in range(1, 5)}
    keys = [k for k, g in groups.items() if g["anchor"] is not None and k not in exclude_groups]
    if len(keys) < 6:
        raise ValueError("Insufficient assemblies to register the survey coordinate frame")
    target = np.asarray([groups[k]["anchor"] for k in keys])
    parsed = [tuple(map(int, key.split("_"))) for key in keys]
    errors = np.asarray([groups[k]["anchor_rms_m"] for k in keys])
    weights = np.minimum(1., .002 / np.maximum(errors, 1e-12))
    midpoint = .5 * (config.tracer_options["bundle_clip_z_min_m"] + config.tracer_options["bundle_clip_z_max_m"])
    z_options = []
    for sign in (-1, 1):
        for order in permutations(range(1, 5)):
            z = np.array([sign * (centres[(1, order[w - 1])][2] - midpoint) for _, w in parsed])
            z_options.append((float(np.mean((z - target[:, 2])**2)), sign, order))
    z_options.sort()
    _, sign, order = z_options[0]
    candidates = []
    for rods in permutations(range(1, 5)):
        source = np.asarray([centres[(rods[r - 1], order[w - 1])] for r, w in parsed])
        for parity in (-1, 1):
            source_xy = source[:, :2].copy()
            source_xy[:, 1] *= parity
            x = source_xy - np.average(source_xy, axis=0, weights=weights)
            y = target[:, :2] - np.average(target[:, :2], axis=0, weights=weights)
            cross = float(np.sum(weights * (x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0])))
            dot = float(np.sum(weights[:, None] * x * y))
            angle = np.arctan2(cross, dot)

            def residual(v):
                c, s = np.cos(v[0]), np.sin(v[0])
                xy = np.array([[c, -parity * s], [s, parity * c]])
                return ((source[:, :2] @ xy.T + v[1:] - target[:, :2]) * np.sqrt(weights[:, None])).ravel()

            fit = least_squares(residual, [angle, 0., 0.], loss="soft_l1", f_scale=.005, max_nfev=30)
            c, s = np.cos(fit.x[0]), np.sin(fit.x[0])
            q = np.array([[c, -parity * s, 0.], [s, parity * c, 0.], [0., 0., sign]])
            t = np.r_[fit.x[1:], -sign * midpoint]
            distances = np.linalg.norm(source @ q.T + t - target, axis=1)
            cost = float(np.sum(weights * distances**2))
            candidates.append(dict(rotation=q.tolist(), translation_m=t.tolist(), rod_map={str(r): rods[r-1] for r in range(1, 5)},
                                   window_map={str(w): order[w-1] for w in range(1, 5)}, cost_m2=cost,
                                   rms_m=float(np.sqrt(np.mean(distances**2))), max_m=float(distances.max()),
                                   residuals_m=distances.tolist(), angle_rad=float(fit.x[0]), determinant=float(np.linalg.det(q))))
    candidates.sort(key=lambda a: (a["cost_m2"], abs(a["angle_rad"])))
    best_cost = candidates[0]["cost_m2"]
    close = [c for c in candidates if c["cost_m2"] <= best_cost + len(keys) * .001**2]
    # Fix numbering gauge by the smallest absolute in-plane coordinate rotation.
    # It is explicitly a convention, not evidence assigning physical rod labels.
    best = min(close, key=lambda c: (abs(c["angle_rad"]), c["determinant"]))
    return {**best, "coordinate_operation": "scene_to_survey", "status": "estimated",
            "method": "rod_permutation_axial_order_coordinate_parity_multistart",
            "alternatives": close, "alternative_count": len(close), "cost_is_probability": False,
            "excluded_groups": list(exclude_groups), "training_groups": keys,
            "source": "original_survey_lines_and_historical_mechanical_slots",
            "assumptions": ["Gas centre is survey z=0", "Rod axes are parallel to survey z",
                            "Historical assembly slots are a mechanical prior, not measurements",
                            "Rod-number gauge uses smallest absolute XY coordinate rotation"],
            "uncertainty": dict(registration_sigma_m=best["rms_m"], metrological_sigma_m=None,
                                 note="Fit discrepancy is not a certified survey uncertainty")}


def _scene_groups(groups, frame):
    q, t = np.asarray(frame["rotation"]), np.asarray(frame["translation_m"])
    result = {}
    for key, group in groups.items():
        result[key] = {**group, "directions": (np.asarray(group["directions"]) @ q).tolist() if group["directions"] else [],
                       "points": ((np.asarray(group["points"]) - t) @ q).tolist() if group["points"] else [],
                       "anchor": ((np.asarray(group["anchor"]) - t) @ q).tolist() if group["anchor"] is not None else None}
    return result


def masked_line_residual(point, direction, screens):
    """Residual in measured coordinates only, with forward propagation s>=0."""
    residuals = []
    for screen in screens:
        mask = np.array([x is not None for x in screen])
        if not np.any(mask):
            continue
        target = np.array([x for x in screen if x is not None])
        ds, ps = direction[mask], point[mask]
        length = float(ds @ ds)
        distance = max(0., float(ds @ (target - ps) / length)) if length > 1e-15 else 0.
        residuals.extend((ps + distance * ds - target).tolist())
    return np.asarray(residuals)


def _assign_with_partials(template, group, frame, options):
    fit = assign_fan(template, group["directions"], ambiguity_rad=options["assignment_resolution_rad"])
    by_id = {r["id"]: r for r in group["records"]}
    partials = [r for r in group["records"] if r["complete_points"] != 2
                and any(v is not None for p in r["screens_m"] for v in p)]
    q, t = np.asarray(frame["rotation"]), np.asarray(frame["translation_m"])
    candidates = []
    for alternative in fit["alternatives"]:
        assigned = {slot: by_id[group["ids"][index]] for slot, index in zip(alternative["slots"], alternative["observation_indices"])}
        free = [j for j in range(7) if j not in assigned]
        rotation = np.asarray(alternative["rotation"])
        directions = np.asarray(template) @ rotation.T @ q.T
        anchor = np.asarray(group["anchor"]) @ q.T + t
        partial_cost, rejected = 0., []
        if partials and free:
            costs = np.array([[np.sum(masked_line_residual(anchor, directions[j], row["screens_m"])**2)
                               for j in free] for row in partials])
            # Explicit unmatched columns prevent forcing suspicious observations.
            matrix = np.c_[costs, np.eye(len(partials)) * options["outlier_distance_m"]**2
                           + (1 - np.eye(len(partials))) * 1e6]
            left, right = linear_sum_assignment(matrix)
            for a, b in zip(left, right):
                if b < len(free) and costs[a, b] < options["outlier_distance_m"]**2:
                    assigned[free[b]] = partials[a]
                    partial_cost += float(costs[a, b])
                else:
                    rejected.append(partials[a]["id"])
        candidates.append({**alternative, "assigned": assigned, "partial_cost_m2": partial_cost,
                           "rejected_partial_ids": rejected,
                           "combined_cost": alternative["cost"] + partial_cost})
    candidates.sort(key=lambda x: x["combined_cost"])
    chosen = candidates[0]
    tolerance = options["assignment_resolution_rad"]**2 * len(group["directions"])
    alternatives = [c for c in candidates if c["combined_cost"] <= chosen["combined_cost"] + tolerance]
    return chosen, alternatives


def fit_assembly(template, group, centres, frame, options, *, allow_individual=True):
    """Shared rigid pose followed by limited individual centre/direction changes."""
    directions = np.asarray(template["directions_local"])
    chosen, alternatives = _assign_with_partials(directions, group, frame, options)
    rotation0 = np.asarray(chosen["rotation"])
    centres = np.asarray(centres)
    centre0 = centres.mean(axis=0)
    offsets = (centres - centre0) @ rotation0
    order = np.lexsort((offsets[:, 0], offsets[:, 1]))
    offsets = offsets[order]
    # Remove translation gauge of individual deformations in every evaluation.
    q, t = np.asarray(frame["rotation"]), np.asarray(frame["translation_m"])
    assigned = chosen["assigned"]
    sigma = options["measurement_sigma_m"]
    angle_sigma = np.array([max(options["individual_angle_sigma_rad"], s["observed_predictive_sigma_rad"] or 0.)
                            for s in template["statistics"]])
    tangent = []
    for direction in directions:
        a = unit(np.cross(direction, [0., 0., 1.]))
        tangent.append(np.array([a, unit(np.cross(direction, a))]))
    tangent = np.asarray(tangent)
    # A common infinitesimal rotation omega has tangent coordinates A @ omega.
    # Individual directions live in its orthogonal complement: they cannot
    # borrow a common rotation from the assembly pose merely to reduce a prior.
    rotation_modes = np.cross(directions[:, None, :], tangent).reshape(14, 3)
    _, mode_singular, mode_vt = np.linalg.svd(rotation_modes.T, full_matrices=True)
    mode_rank = int(np.sum(mode_singular > mode_singular[0] * 1e-10))
    angle_basis = mode_vt[mode_rank:].T
    individual_angle_parameters = angle_basis.shape[1]

    def unpack(parameters):
        rotation = Rotation.from_rotvec(parameters[3:6]).as_matrix() @ rotation0
        dc = parameters[6:27].reshape(7, 3) if len(parameters) > 6 else np.zeros((7, 3))
        dc = dc - dc.mean(axis=0)
        angles = (angle_basis @ parameters[27:]).reshape(7, 2) if len(parameters) > 6 else np.zeros((7, 2))
        local_d = unit(directions + np.einsum("ni,nij->nj", angles, tangent))
        return centre0 + parameters[:3] + (offsets + dc) @ rotation.T, local_d @ rotation.T, rotation, dc, angles

    def data_residual(parameters):
        c, d, _, _, _ = unpack(parameters)
        cs, ds = c @ q.T + t, d @ q.T
        return np.concatenate([masked_line_residual(cs[j], ds[j], row["screens_m"]) for j, row in sorted(assigned.items())]) / sigma

    def residual(parameters):
        chunks = [data_residual(parameters), parameters[:3] / options["assembly_translation_sigma_m"],
                  parameters[3:6] / options["assembly_rotation_sigma_rad"]]
        if len(parameters) > 6:
            chunks += [parameters[6:27] / options["individual_center_sigma_m"],
                       ((angle_basis @ parameters[27:]).reshape(7, 2) / angle_sigma[:, None]).ravel()]
        return np.concatenate(chunks)

    initial = np.zeros(6)
    rigid = least_squares(residual, initial, loss="soft_l1", f_scale=1., max_nfev=options["max_nfev"])
    fit = least_squares(residual, np.r_[rigid.x, np.zeros(21 + individual_angle_parameters)], loss="soft_l1", f_scale=1.,
                        max_nfev=options["max_nfev"]) if allow_individual else rigid
    centres_fit, outgoing, rotation, dc, angles = unpack(fit.x)
    # Rank is computed on dimensionlessly scaled coordinates/angles and on the
    # DATA Jacobian alone, never inferred from regularized optimizer success.
    scale = np.r_[np.full(3, options["assembly_translation_sigma_m"]),
                  np.full(3, options["assembly_rotation_sigma_rad"])]
    if len(fit.x) > 6:
        # These are numerical differentiation scales, not independent beta
        # priors. The actual anisotropic prior above remains alpha_j / sigma_j.
        beta_scale = np.sqrt(np.sum((angle_basis * np.repeat(angle_sigma, 2)[:, None])**2, axis=0))
        scale = np.r_[scale, np.full(21, options["individual_center_sigma_m"]), beta_scale]
    base = data_residual(fit.x)
    jac = np.empty((len(base), len(fit.x)))
    for k in range(len(fit.x)):
        trial = fit.x.copy()
        trial[k] += 1e-4 * scale[k]
        jac[:, k] = (data_residual(trial) - base) / 1e-4
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular > max(singular[0] * 1e-6, 1e-8))) if len(singular) else 0
    residuals = {j: float(np.sqrt(np.mean(masked_line_residual(centres_fit[j] @ q.T + t, outgoing[j] @ q.T,
                                                            row["screens_m"])**2))) for j, row in assigned.items()}
    return dict(centres=centres_fit, directions=outgoing, rotation=rotation, offsets=offsets, assigned=assigned,
                alternatives=alternatives, chosen=chosen, prior_order=order,
                rigid_residual_m=float(np.sqrt(np.mean(data_residual(rigid.x)**2))) * sigma,
                residual_m=float(np.sqrt(np.mean(base**2))) * sigma, residuals=residuals,
                template_deformations_m=dc, template_deformations_rad=angles,
                rigid_translation_m=fit.x[:3], rigid_rotation_rad=fit.x[3:6],
                angular_gauge=dict(method="local_tangent_nullspace_of_common_SO3_rotation",
                                   removed_common_rotation_dimensions=mode_rank,
                                   individual_parameters=individual_angle_parameters,
                                   common_mode_residual_rad=(rotation_modes.T @ angles.ravel()).tolist(),
                                   note="Exact gauge in the declared local tangent chart, not an Euler-angle average"),
                data_jacobian_rank=rank, full_parameter_count=len(fit.x),
                data_jacobian_singular_values=singular.tolist(),
                data_condition_number=float(singular[0] / singular[rank-1]) if rank else None,
                regularized_rank=int(np.linalg.matrix_rank(fit.jac * scale[None, :], tol=1e-8)),
                solver_success=bool(fit.success), solver_message=str(fit.message), evaluations=int(fit.nfev))


def _serial_assignment(fit, physical_group):
    alternatives = []
    for item in fit["alternatives"]:
        alternatives.append(dict(cost=item["combined_cost"], rotation=item["rotation"],
                                 mapping=[dict(canonical_id=f"{physical_group}_{j+1}", old_id=row["old_id"], observation_id=row["id"])
                                          for j, row in sorted(item["assigned"].items())],
                                 rejected_partial_ids=item["rejected_partial_ids"]))
    return dict(physical_group=physical_group, cost=fit["chosen"]["combined_cost"],
                alternatives=alternatives, ambiguous=len(alternatives) > 1, cost_is_probability=False,
                criterion="SO3 angular subset residual plus measured-component distances at both real screens",
                rejected_complete_indices=fit["chosen"]["outliers"],
                rejected_partial_ids=fit["chosen"]["rejected_partial_ids"])


def _source_conflict_scenarios(registry, scene_groups, frame, surfaces, loo_templates, mirrors, options):
    """Evaluate the documented TXT trust alternative without editing raw XLSX."""
    import copy
    result = []
    for conflict in registry.get("conflicts", []):
        old_id = conflict["old_id"]
        group_key = "_".join(old_id.split("_")[:2])
        if group_key not in loo_templates:
            continue
        rows = copy.deepcopy(scene_groups[group_key]["records"])
        row = next(r for r in rows if r["id"] == conflict["observation_id"])
        coefficients = np.asarray(conflict["coefficients_mm"]) / 1000
        point, direction = coefficients[:3], coefficients[3:]
        # These are synthetic points representing the trusted TXT line. They
        # are never entered in the original observation ledger as measurements.
        row["screens_m"] = [point.tolist(), (point + direction).tolist()]
        row["complete_points"] = 2
        alt_group = _scene_groups(group_observations(rows), frame)[group_key]
        r, w = group_key.split("_")
        physical = f"{frame['rod_map'][r]}_{frame['window_map'][w]}"
        centres = [surfaces[f"{physical}_{j}"].center for j in range(1, 8)]
        fit = fit_assembly(loo_templates[group_key], alt_group, centres, frame, options)
        slot = next((j for j, assigned in fit["assigned"].items() if assigned["id"] == row["id"]), None)
        primary = next((m for m in mirrors if row["id"] in m["observation_ids"]), None)
        report = dict(observation_id=row["id"], old_id=old_id,
                      primary_policy="Use only measured XLSX components",
                      alternative_policy="Trust complete TXT direction; synthetic samples constrain the line, not new measurements",
                      independent_evidence=False, raw_observations_modified=False,
                      alternative_assembly_residual_m=fit["residual_m"],
                      alternative_canonical_id=f"{physical}_{slot+1}" if slot is not None else None,
                      primary_canonical_id=primary["canonical_id"] if primary else None)
        if slot is not None:
            report["alternative_point_m"] = fit["centres"][slot].tolist()
            report["alternative_direction"] = fit["directions"][slot].tolist()
            original = next(r for r in registry["records"] if r["id"] == row["id"])
            q, t = np.asarray(frame["rotation"]), np.asarray(frame["translation_m"])
            components = masked_line_residual(fit["centres"][slot] @ q.T + t, fit["directions"][slot] @ q.T,
                                              original["screens_m"])
            report["alternative_original_component_rms_m"] = float(np.sqrt(np.mean(components**2)))
            if primary:
                report["angular_change_mrad"] = float(angular_distance(primary["outgoing_direction"], fit["directions"][slot]) * 1000)
                report["center_change_m"] = float(np.linalg.norm(np.asarray(primary["center_m"]) - fit["centres"][slot]))
        result.append(report)
    return result


def _section_intervals(center, direction, group, frame, sigma_angle, sigma_center):
    """Linearized conditional section covariance; grazing sections stay flagged."""
    q, t = np.asarray(frame["rotation"]), np.asarray(frame["translation_m"])
    c, d = np.asarray(center) @ q.T + t, np.asarray(direction) @ q.T
    sections = []
    for screen in range(2):
        z_values = [r["screens_m"][screen][2] for r in group["records"] if r["screens_m"][screen][2] is not None]
        if not z_values:
            continue
        z = float(np.median(z_values))
        if abs(d[2]) < max(1e-7, 3 * sigma_angle):
            sections.append(dict(z_m=z, status="grazing_or_angle_uncertainty_crosses_parallel", covariance_xy_m2=None,
                                 predictive_angle_sigma_rad=sigma_angle))
            continue
        distance = (z - c[2]) / d[2]
        projection = np.eye(3) - np.outer(d, [0., 0., 1.]) / d[2]
        direction_covariance = sigma_angle**2 / 2 * (np.eye(3) - np.outer(d, d))
        covariance = projection @ (sigma_center**2 * np.eye(3) + distance**2 * direction_covariance) @ projection.T
        sections.append(dict(z_m=z, point_m=(c + distance*d).tolist(), covariance_xy_m2=covariance[:2, :2].tolist(),
                             status="conditional_linearized", forward=bool(distance >= 0),
                             assumption="Fixed selected assignment and optical branch; separate modes are not averaged"))
    return sections


def reconstruct(config, root=None, options=None, progress=None):
    from .physical import (incoming_at_mirrors, incoming_at_points, rotation_from_normal,
                           validate_bodies, validate_reflections, relax_body_rolls, relax_line_centers)
    from ..engine.bodies import MirrorBody
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    opts = _options(options)
    _notify(progress, "проверка данных")
    embedded = getattr(config, "reconstruction", None)
    if embedded and embedded.get("observations"):
        import copy
        registry = copy.deepcopy(embedded.get("observation_registry", {}))
        registry["records"] = copy.deepcopy(embedded["observations"])
        registry.setdefault("sources", [{"sha256": "embedded", "type": "embedded_primary"}])
        registry.setdefault("conflicts", [])
        registry.setdefault("warnings", [])
    else:
        registry = observation_registry(root, config.direction)
    groups = group_observations(registry["records"])
    frame = geometric_registration(config, groups)
    scene_groups = _scene_groups(groups, frame)
    initial_field = incoming_at_mirrors(config)
    for key, group in scene_groups.items():
        r, w = key.split("_")
        slot = f"{frame['rod_map'][r]}_{frame['window_map'][w]}_1"
        group["incoming_direction"] = initial_field.get(slot, {}).get("direction")
    _notify(progress, "построение шаблона")
    template = learn_template(scene_groups, iterations=opts["template_iterations"], ambiguity_rad=opts["assignment_resolution_rad"])
    template["training_observation_ids"] = [identifier for group in scene_groups.values() for identifier in group["ids"]]
    template["source_sha256"] = registry["sources"][0]["sha256"]
    template["length_units"] = "m"
    _notify(progress, "сопоставление")
    surfaces = dict(_bundle_surfaces(config))
    mirrors, assignments, assembly_reports, loo_templates = [], [], [], {}
    geometry = {}
    for index, (key, group) in enumerate(sorted(scene_groups.items())):
        _notify(progress, "подгонка сборок", completed=index, total=len(scene_groups), assembly=key)
        r, w = map(int, key.split("_"))
        rod, window = frame["rod_map"][str(r)], frame["window_map"][str(w)]
        physical_group = f"{rod}_{window}"
        centres = np.array([surfaces[f"{physical_group}_{j}"].center for j in range(1, 8)])
        # The target assembly never trains its own deformation prior.
        external = {k: g for k, g in scene_groups.items() if k != key}
        external_template = learn_template(external, iterations=opts["template_iterations"], ambiguity_rad=opts["assignment_resolution_rad"])
        loo_templates[key] = external_template
        fit = fit_assembly(external_template, group, centres, frame, opts)
        assignment = _serial_assignment(fit, physical_group)
        assignment["observation_group"] = key
        assignments.append(assignment)
        _notify(progress, "индивидуальная подстройка", completed=index + 1, total=len(scene_groups), assembly=key)
        assembly_reports.append(dict(observation_group=key, physical_group=physical_group,
                                     rigid_residual_m=fit["rigid_residual_m"], residual_m=fit["residual_m"],
                                     translation_m=fit["rigid_translation_m"].tolist(), rotation_rad=fit["rigid_rotation_rad"].tolist(),
                                     rotation=fit["rotation"].tolist(), angular_gauge=fit["angular_gauge"],
                                     data_jacobian_rank=fit["data_jacobian_rank"], parameter_count=fit["full_parameter_count"],
                                     full_regularized_rank=fit["regularized_rank"],
                                     data_condition_number=fit["data_condition_number"],
                                     data_singular_values=fit["data_jacobian_singular_values"],
                                     solver_success=fit["solver_success"], solver_message=fit["solver_message"],
                                     template_training_groups=[k for k in scene_groups if k != key],
                                     center_layout_source="historical_mechanical_prior_not_empirical_measurement"))
        for j in range(7):
            old_id = f"{physical_group}_{int(fit['prior_order'][j]) + 1}"
            canonical_id = f"{physical_group}_{j+1}"
            row = fit["assigned"].get(j)
            obs_status = ("measured" if row and row["complete_points"] == 2 else "partial" if row else "predicted")
            ambiguous = len(fit["alternatives"]) > 1
            support = "ambiguous" if ambiguous else "data_and_mechanics" if row else "prior_dominated"
            stats = external_template["statistics"][j]
            sigma_angle = max(opts["individual_angle_sigma_rad"], stats["observed_predictive_sigma_rad"] or 0.)
            mirror = dict(canonical_id=canonical_id, mirror_id=canonical_id, old_id=old_id,
                          observation_ids=[row["id"]] if row else [], observation_old_ids=[row["old_id"]] if row else [],
                          observation_status=obs_status, status=support, center_m=fit["centres"][j].tolist(),
                          center_definition="reflecting_cut_center", outgoing_direction=fit["directions"][j].tolist(),
                          measurement_residual_m=fit["residuals"].get(j),
                          template_deviation=dict(center_m=fit["template_deformations_m"][j].tolist(),
                                                  angular_rad=float(np.linalg.norm(fit["template_deformations_rad"][j]))),
                          uncertainty=dict(center_status="prior_dominated", axial_roll_status="prior_dominated",
                                           center_sigma_m=opts["individual_center_sigma_m"],
                                           axial_center_sigma_m=opts["axial_center_sigma_m"],
                                           assembly_translation_sigma_m=opts["assembly_translation_sigma_m"],
                                           registration_sigma_m=frame["rms_m"],
                                           measured_coordinate_sigma_m=opts["measurement_sigma_m"],
                                           predictive_angle_sigma_rad=sigma_angle,
                                           mean_template_sigma_rad=stats["mean_sigma_rad"],
                                           interval_type="conditional_95_percent_normal_scenario_not_certified_coverage",
                                           measurement_error_known=False,
                                           unobservable_components=["center_along_output_line", "cylinder_roll_about_normal", "within_aperture_hit_offset"]),
                          violations=[], assumptions=["chief_ray_through_cut_centre working hypothesis; aperture offset unobserved",
                                                       "body roll selected by mechanical axial hint",
                                                       "metrology and mechanical scales are explicit sensitivity scenarios"])
            predictions = []
            for alternative in fit["alternatives"]:
                d = np.asarray(external_template["directions_local"])[j] @ np.asarray(alternative["rotation"]).T
                predictions.append(dict(direction=d.tolist(), assignment_cost=alternative["combined_cost"],
                                        observation_ids=[alternative["assigned"][j]["id"]] if j in alternative["assigned"] else []))
            mirror["assignment_predictions"] = predictions
            mirror["uncertainty"]["sections"] = _section_intervals(
                mirror["center_m"], mirror["outgoing_direction"], group, frame, sigma_angle,
                float(np.sqrt(opts["individual_center_sigma_m"]**2 + opts["assembly_translation_sigma_m"]**2 + frame["rms_m"]**2)))
            mirror["uncertainty"]["optics_sigma"] = None
            mirror["uncertainty"]["normal_sigma_basis"] = "Conditional on fixed traced optics; direction scatter propagated through reflection normal"
            mirror["uncertainty"]["axis_roll_sigma"] = None
            mirrors.append(mirror)
            geometry[canonical_id] = mirror["center_m"]
    _notify(progress, "проверка геометрии")
    field = incoming_at_points(config, geometry)
    bodies = []
    for mirror in mirrors:
        entry = field[mirror["canonical_id"]]
        mirror["incoming"] = entry
        incoming_direction = entry.get("direction") or initial_field.get(mirror["old_id"], {}).get("direction")
        if incoming_direction is None:
            raise ValueError(f"No traceable optical incoming branch for {mirror['canonical_id']}")
        din, dout = np.asarray(incoming_direction), np.asarray(mirror["outgoing_direction"])
        normal = unit(dout - din)
        rotation = rotation_from_normal(normal, axis_hint=din)
        body = MirrorBody(mirror["center_m"], rotation, mirror_id=mirror["canonical_id"])
        mirror.update(body.to_dict())
        sigma_angle = mirror["uncertainty"]["predictive_angle_sigma_rad"]
        direction_covariance = sigma_angle**2 / 2 * (np.eye(3) - np.outer(dout, dout))
        normal_jacobian = (np.eye(3) - np.outer(normal, normal)) / max(np.linalg.norm(dout-din), 1e-12)
        mirror["uncertainty"]["normal_covariance_conditional"] = (normal_jacobian @ direction_covariance @ normal_jacobian.T).tolist()
        mirror["uncertainty"]["axis_orientation"] = dict(status="prior_dominated", roll_angle_identified=False,
                                                       conditional_on_mechanical_clearance=True, covariance=None)
        mirror["trajectory"] = dict(point_m=mirror["center_m"], direction=dout.tolist(),
                                    origin_definition="chief_ray_intersection_hypothesis", segments=[],
                                    coordinate_frame="legacy", status="candidate_pending_full_trace")
        if entry.get("status") != "available":
            mirror["violations"].append(f"incoming_field:{entry.get('status')}")
        bodies.append(body)
    bodies, roll_report = relax_body_rolls(bodies, required_gap_m=opts["minimum_gap_m"],
                                          numerical_tolerance_m=opts["numerical_epsilon_m"],
                                          progress=lambda done, total: _notify(progress, "проверка геометрии", completed=done, total=total))
    initial_centers = {body.mirror_id: body.center_m.copy() for body in bodies}
    bodies, axial_report = relax_line_centers(
        config, bodies, {m["canonical_id"]: m["outgoing_direction"] for m in mirrors}, opts,
        progress=lambda done, total: _notify(progress, "проверка геометрии", completed=done, total=total))
    field = incoming_at_points(config, {b.mirror_id: b.center_m for b in bodies})
    for mirror, body in zip(mirrors, bodies):
        mirror.update(body.to_dict())
        mirror["incoming"] = field[body.mirror_id]
        mirror["violations"] = [v for v in mirror["violations"] if not v.startswith("incoming_field:")]
        if field[body.mirror_id].get("status") != "available":
            mirror["violations"].append(f"incoming_field:{field[body.mirror_id].get('status')}")
        mirror["trajectory"]["point_m"] = body.center_m.tolist()
        shift = body.center_m - initial_centers[body.mirror_id]
        mirror["template_deviation"]["axial_physical_relaxation_m"] = float(shift @ np.asarray(mirror["outgoing_direction"]))
        d = np.asarray(mirror["outgoing_direction"])
        mirror["uncertainty"]["center_covariance_conditional_m2"] = (
            opts["individual_center_sigma_m"]**2 * np.eye(3)
            + (opts["axial_center_sigma_m"]**2 - opts["individual_center_sigma_m"]**2) * np.outer(d, d)).tolist()
        mirror["assumptions"].append("Unobserved cylinder roll relaxed under exact body separation certificates")
        mirror["assumptions"].append("Centre may slide along its output line within explicit axial prior scenario; measured line is unchanged")
    mechanics = validate_bodies(bodies, required_gap_m=opts["minimum_gap_m"], numerical_tolerance_m=opts["numerical_epsilon_m"])
    mechanics["roll_relaxation"] = roll_report
    mechanics["axial_line_relaxation"] = axial_report
    reflections = validate_reflections(config, bodies)
    checks = reflections.get("mirrors", {})
    conflict_ids = {identifier for pair in mechanics.get("conflicts", []) for identifier in pair["pair"]}
    for mirror in mirrors:
        check = checks.get(mirror["canonical_id"], {})
        mirror["optical_validation"] = check
        if check.get("status") not in ("validated", "available", "passed"):
            mirror["violations"].append(f"reflection:{check.get('status', 'not_evaluated')}")
        else:
            mirror["trajectory"]["status"] = "traced_reflection"
        if check.get("reflected_trajectories"):
            mirror["trajectory"]["paths"] = check["reflected_trajectories"]
            mirror["trajectory"]["segments"] = check["reflected_trajectories"][0].get("segments", [])
            mirror["trajectory"]["points_m"] = check["reflected_trajectories"][0].get("points_m", [])
        if check.get("reflected_child"):
            mirror["trajectory"]["point_m"] = check["reflected_child"]["point_m"]
            mirror["trajectory"]["direction"] = check["reflected_child"]["direction"]
        if mirror["canonical_id"] in conflict_ids:
            mirror["violations"].append("body_clearance_not_certified")
        if mirror["violations"]:
            mirror["status"] = "infeasible"
        mirror["clearance_m"] = mechanics.get("per_body_minimum_gap_m", {}).get(mirror["canonical_id"])
        mirror["rod_containment"] = mechanics.get("rod_containment", {}).get(mirror["canonical_id"], {"status": "not_evaluated"})
        mirror["waist_side"] = "not_evaluated"
    _notify(progress, "оценка качества")
    from .validation import validate_withheld
    heldout = validate_withheld(config, registry, opts, progress=progress) if opts["validate_holdout"] else dict(status="not_requested")
    counts = dict(total=len(mirrors), measured=sum(m["observation_status"] == "measured" for m in mirrors),
                  partial=sum(m["observation_status"] == "partial" for m in mirrors),
                  predicted=sum(m["observation_status"] == "predicted" for m in mirrors),
                  ambiguous=sum(len(m["assignment_predictions"]) > 1 for m in mirrors),
                  failed=sum(m["status"] == "infeasible" for m in mirrors))
    source_scenarios = _source_conflict_scenarios(registry, scene_groups, frame, surfaces, loo_templates, mirrors, opts)
    template.pop("assignments", None)
    overall_status = "infeasible" if mechanics.get("status") == "infeasible" else "partial"
    return dict(direction=config.direction, status=overall_status, kind="empirical_reconstruction", units="m",
                coordinate_frame="legacy", display_frame=frame, mirrors=mirrors, template=template,
                observations=registry["records"], observation_registry=registry, assignments=assignments,
                options=opts, counts=counts,
                mechanics=dict(radius_m=.0005, axial_height_m=.0095, length_definition="full_extent",
                               rod_regions=None, waist=None, nonreflecting_surfaces="opaque_absorbing"),
                diagnostics=dict(assemblies=assembly_reports, geometry=mechanics, reflections=reflections,
                                 withheld=heldout, initial_field=initial_field,
                                 source_trust_scenarios=source_scenarios,
                                 unresolved=["Rod/holder boundaries unavailable", "Physical waist location unavailable",
                                             "Body centre offsets and roll are mechanically regularized, not measured",
                                             "Metric budgets are assumed scenarios; calibration has not certified them"],
                                 side_pooling="Separate templates; no evidence-based pooling claim",
                                 input_data_unchanged=True, optimizer_success_is_validation=False))
