"""Physical mirror geometry, deterministic incident field, and certificates.

No direction here is inferred from an archive filename or replaced by +/-Z.
The current common optics are planar: tracing three non-collinear launch
points gives an affine incident field. Every distinct field sample is checked
by a full traversal of the common optics at its solved launch. Exact repeated
requests may reuse that check; full scenes containing bodies are always traced.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import re

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from ..engine import RayBundle, RayTracer, Scene, PlaneSurface, to_numpy
from ..engine.bodies import MirrorBody, LOCAL_NORMAL


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    size = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(vector).all() or size < 1e-14:
        raise ValueError("Expected a finite nonzero 3-vector")
    return vector / size


def reflecting_normal(incoming, outgoing):
    incoming, outgoing = unit(incoming), unit(outgoing)
    if np.linalg.norm(outgoing - incoming) < 1e-8:
        raise ValueError("Degenerate grazing/forward reflection does not determine a normal")
    normal = unit(outgoing - incoming)
    if np.dot(incoming, normal) >= -1e-8:
        raise ValueError("Grazing incidence is not a stable physical mirror solution")
    return normal


def rotation_from_normal(normal, axis_hint=None):
    """Proper rotation with the requested cut normal and explicitly chosen roll.

The unobserved roll is chosen so that the cylinder axis is as close as
possible to ``axis_hint``. This is a mechanical prior, never an observation.
"""
    normal = unit(normal)
    hint = np.array([0., 0., 1.]) if axis_hint is None else unit(axis_hint)
    tangent = hint - np.dot(hint, normal) * normal
    if np.linalg.norm(tangent) < 1e-10:
        helper = np.eye(3)[np.argmin(np.abs(normal))]
        tangent = helper - np.dot(helper, normal) * normal
    tangent = unit(tangent)
    z_axis = (tangent - normal) / math.sqrt(2.)
    x_axis = (tangent + normal) / math.sqrt(2.)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _closest_simplex(points):
    """Closest point in a <=5-point convex hull, with barycentric witnesses."""
    points = np.asarray(points)
    best = None
    scale = max(float(np.max(np.linalg.norm(points, axis=1))), 1e-12)
    for count in range(1, min(len(points), 4) + 1):
        for subset in itertools.combinations(range(len(points)), count):
            matrix = points[list(subset)] / scale
            gram = matrix @ matrix.T
            kkt = np.block([[gram, np.ones((count, 1))], [np.ones((1, count)), np.zeros((1, 1))]])
            weights = np.linalg.lstsq(kkt, np.r_[np.zeros(count), 1.], rcond=1e-13)[0][:count]
            if np.min(weights) < -1e-10 or abs(weights.sum() - 1) > 1e-8:
                continue
            weights = np.maximum(weights, 0.)
            weights /= weights.sum()
            point = weights @ points[list(subset)]
            square = float(point @ point)
            if best is None or square < best[0]:
                best = square, list(subset), weights, point
    return best[1:]


def _intersection_ball(first, second, tolerance_m):
    """Convex SOCP-like interior test, independently checked primal certificate.

A positive feasible radius proves volume intersection. Failed convergence or
negative radius alone never proves separation.
"""
    origin = (first.center_m + second.center_m) / 2
    scale = 1000.  # solve in mm; avoid SLSQP finite differences at micrometres

    def margins(x):
        p = origin + x[:3] / scale
        values = []
        for body in (first, second):
            q = body.local(p) * scale
            values.extend([body.radius_m * scale - np.hypot(q[0], q[1]),
                           (q[2] - q[0]) / math.sqrt(2.), body.axial_height_m * scale - q[2]])
        return np.asarray(values)

    initial = np.zeros(4)
    initial[3] = float(margins(initial).min())
    result = minimize(lambda x: -x[3], initial, method="SLSQP",
                      constraints={"type": "ineq", "fun": lambda x: margins(x) - x[3]},
                      options={"maxiter": 120, "ftol": max(tolerance_m * scale * .01, 1e-11)})
    radius = float(margins(result.x).min() / scale)
    return radius, origin + result.x[:3] / scale, bool(result.success)


def body_pair_clearance(first, second, numerical_tolerance_m=1e-8, required_gap_m=0.):
    """Analytic-support GJK distance plus independent intersection certificate.

Separated results include a separating-plane lower bound; overlap results
include a common interior ball and a penetration lower bound, not a fictitious
unsigned negative distance. Touching is explicitly within numerical tolerance.
"""
    if not math.isfinite(numerical_tolerance_m) or not math.isfinite(required_gap_m) or numerical_tolerance_m <= 0 or required_gap_m < 0:
        raise ValueError("Numerical tolerance must be positive and physical gap nonnegative")
    direction = first.center_m - second.center_m
    if np.linalg.norm(direction) < 1e-15:
        direction = np.array([1., 0., 0.])
    witnesses = []
    closest = direction
    lower = -math.inf
    separation = None
    point_a = point_b = None
    for iteration in range(96):
        if np.linalg.norm(closest) <= numerical_tolerance_m * .05:
            break
        u = -unit(closest)
        a, b = first.support_point(u), second.support_point(-u)
        new = a - b
        gap = float(-np.dot(u, new))
        if gap > lower:
            lower, separation = gap, u
        if witnesses and np.linalg.norm(closest) - gap <= numerical_tolerance_m * .1:
            break
        if witnesses and min(np.linalg.norm(new - item[0]) for item in witnesses) < numerical_tolerance_m * .001:
            break
        witnesses.append((new, a, b))
        subset, weights, closest = _closest_simplex([item[0] for item in witnesses])
        witnesses = [witnesses[i] for i in subset]
        point_a = sum(w * item[1] for w, item in zip(weights, witnesses))
        point_b = sum(w * item[2] for w, item in zip(weights, witnesses))
    result = {"pair": [first.mirror_id, second.mirror_id],
              "numerical_tolerance_m": numerical_tolerance_m, "required_gap_m": required_gap_m,
              "iterations": iteration + 1, "algorithm": "analytic_support_GJK_plus_interior_certificate"}
    if separation is not None:
        # Recompute certificate from the exact support functions independently.
        lower = -second.support(-separation) - first.support(separation)
        result.update({"separating_direction": separation.tolist(), "gap_lower_bound_m": lower,
                       "witness_first_m": first.support_point(separation).tolist(),
                       "witness_second_m": second.support_point(-separation).tolist()})
    if lower > numerical_tolerance_m:
        result.update(status="separated", distance_m=float(np.linalg.norm(closest)),
                      physical_gap_satisfied=lower >= required_gap_m)
        if point_a is not None:
            result.update(closest_first_m=point_a.tolist(), closest_second_m=point_b.tolist())
        return result
    radius, point, converged = _intersection_ball(first, second, numerical_tolerance_m)
    if radius > numerical_tolerance_m:
        result.update(status="intersecting", distance_m=0., physical_gap_satisfied=False,
                      common_interior_point_m=point.tolist(), common_ball_radius_m=radius,
                      penetration_lower_bound_m=2 * radius)
    elif separation is not None and lower >= -numerical_tolerance_m and converged:
        result.update(status="touching_within_tolerance", distance_m=0.,
                      physical_gap_satisfied=required_gap_m == 0,
                      common_ball_radius_m=max(0., radius))
    else:
        result.update(status="indeterminate", distance_m=0., physical_gap_satisfied=False,
                      interior_solver_converged=converged, common_ball_radius_m=max(0., radius))
    return result


def rod_containment(body, region, numerical_tolerance_m=1e-8):
    if region is None:
        return {"status": "not_evaluated", "reason": "No independently established rod/holder geometry"}
    origin, axis = np.asarray(region["origin_m"]), unit(region["axis"])
    lower = -body.support(-axis) - np.dot(origin, axis)
    upper = body.support(axis) - np.dot(origin, axis)
    helper = np.eye(3)[np.argmin(abs(axis))]
    u = unit(np.cross(axis, helper)); v = np.cross(axis, u)
    count = 720
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    supports = [body.support(np.cos(a) * u + np.sin(a) * v) - np.dot(origin, np.cos(a) * u + np.sin(a) * v) for a in angles]
    # Circumscribed angular approximation: every vector is bracketed by two
    # sampled normals, so max sampled support / cos(pi/N) is an upper bound.
    radial_lower = max(supports)
    radial_upper = radial_lower / math.cos(math.pi / count)
    inside = (lower >= region["axial_min_m"] - numerical_tolerance_m and
              upper <= region["axial_max_m"] + numerical_tolerance_m and
              radial_upper <= region["radius_m"] + numerical_tolerance_m)
    outside = (lower < region["axial_min_m"] - numerical_tolerance_m or
               upper > region["axial_max_m"] + numerical_tolerance_m or
               radial_lower > region["radius_m"] + numerical_tolerance_m)
    return {"status": "inside" if inside else "outside" if outside else "indeterminate",
            "axial_extent_m": [lower, upper], "radial_upper_bound_m": radial_upper,
            "radial_approximation_error_m": radial_upper - radial_lower}


def waist_side(point, outgoing, waist_point=None, tolerance_m=1e-8):
    if waist_point is None:
        return {"status": "not_evaluated", "reason": "No independently established physical waist point"}
    distance = float(np.dot(np.asarray(waist_point) - point, unit(outgoing)))
    return {"status": "before_waist" if distance > tolerance_m else "wrong_side" if distance < -tolerance_m else "indeterminate",
            "forward_distance_m": distance}


def validate_bodies(bodies, required_gap_m=0., numerical_tolerance_m=1e-8, rod_regions=None,
                    check_length_alternative=True):
    bodies = [MirrorBody.from_dict(b) if isinstance(b, dict) else b for b in bodies]
    if len({b.mirror_id for b in bodies}) != len(bodies):
        raise ValueError("Duplicate body identifiers")
    bounds = [b.bounds() for b in bodies]
    candidates = []
    for i, first in enumerate(bodies):
        for j in range(i + 1, len(bodies)):
            alo, ahi = bounds[i]; blo, bhi = bounds[j]
            lower = float(np.linalg.norm(np.maximum(0., np.maximum(blo - ahi, alo - bhi))))
            candidates.append((lower, i, j))
    candidates.sort()
    results = []; minimum = math.inf; minimum_pair = None
    per_body = {b.mirror_id: math.inf for b in bodies}
    per_body_lower = {b.mirror_id: math.inf for b in bodies}
    per_body_pair = {b.mirror_id: None for b in bodies}
    for lower, i, j in candidates:
        # A pair can be excluded only when it cannot improve EITHER body's
        # nearest-neighbour gap. A global minimum (often zero) is insufficient
        # for honest per-mirror diagnostics.
        if lower > max(required_gap_m, per_body[bodies[i].mirror_id], per_body[bodies[j].mirror_id]) + numerical_tolerance_m:
            continue
        report = body_pair_clearance(bodies[i], bodies[j], numerical_tolerance_m, required_gap_m)
        results.append(report)
        gap = report.get("distance_m", 0.)
        if gap < minimum:
            minimum, minimum_pair = gap, report["pair"]
        for identifier in report["pair"]:
            if gap < per_body[identifier]:
                per_body[identifier], per_body_pair[identifier] = gap, report["pair"]
            lower_bound = max(0., report.get("gap_lower_bound_m", 0.)) if report["status"] == "separated" else 0.
            per_body_lower[identifier] = min(per_body_lower[identifier], lower_bound)
    bad = [r for r in results if not r["physical_gap_satisfied"]]
    containment = {b.mirror_id: rod_containment(b, (rod_regions or {}).get(b.mirror_id.split("_")[0]), numerical_tolerance_m) for b in bodies}
    status = "infeasible" if any(r["status"] != "indeterminate" for r in bad) or any(r["status"] == "outside" for r in containment.values()) else "partial"
    result = {"status": status, "pair_count": len(candidates), "exact_pair_count": len(results),
              "broad_phase_excluded_count": len(candidates) - len(results), "pairs": results,
              "minimum_gap_m": None if not math.isfinite(minimum) else minimum,
              "per_body_minimum_gap_m": {k: v if math.isfinite(v) else None for k, v in per_body.items()},
              "per_body_nearest_pair": per_body_pair,
              "per_body_clearance_interval_m": {k: [per_body_lower[k], v] if math.isfinite(v) else None for k, v in per_body.items()},
              "minimum_pair": minimum_pair, "conflicts": bad,
              "rod_containment": containment, "holder_collisions": "not_evaluated",
              "waist_side": "not_evaluated", "required_gap_m": required_gap_m,
              "numerical_tolerance_m": numerical_tolerance_m,
              "assumptions": ["10 mm is full axial extent; h=9.5 mm", "Uncoated side/back are opaque absorbing"]}
    if check_length_alternative:
        alternatives = [MirrorBody(b.center_m, b.rotation, b.mirror_id, b.radius_m,
                                   .01 if b.length_definition == "full_extent" else b.axial_height_m,
                                   "cut_center_to_back") for b in bodies]
        alternative = validate_bodies(alternatives, required_gap_m, numerical_tolerance_m, rod_regions, False)
        result["length_sensitivity"] = {"alternative": "10 mm from cut centre to back", "minimum_gap_m": alternative["minimum_gap_m"],
                                        "conflict_pairs": [p["pair"] for p in alternative["conflicts"]]}
    return result


def _relax_body_poses(bodies, required_gap_m=0., numerical_tolerance_m=1e-8,
                     max_iterations=60, starts=3, max_roll_rad=math.pi, progress=None,
                     line_directions=None, axial_sigma_m=None, axial_limit_sigma=3., config=None):
    """Resolve shaft collisions through the unobserved roll, with hard certificates.

Cut centres and optical normals remain fixed. Each body has one bounded SO(3)
roll variable; no independent axis/normal fitting or arbitrary centre shifts
are introduced. The objective selects the smallest change to the stated roll
prior *subject to* exact-support separation-plane inequalities. A finite set
of candidate planes is conservative: failure to find a plane is not proof of
collision. Every selected component is independently checked by GJK/interior
certificates, and an unsuccessful search is retained as a diagnostic.
"""
    original = [MirrorBody.from_dict(b) if isinstance(b, dict) else b for b in bodies]
    line_mode = line_directions is not None
    if line_mode and (axial_sigma_m is None or not math.isfinite(axial_sigma_m) or axial_sigma_m <= 0 or
                      not math.isfinite(axial_limit_sigma) or axial_limit_sigma <= 0):
        raise ValueError("Along-line centre uncertainty and bound require explicit positive scales")
    axial_bound = axial_limit_sigma * axial_sigma_m if line_mode else 0.
    source_models = _field_models(config)[0] if line_mode and config is not None else None
    if not (0 < max_roll_rad <= math.pi) or max_iterations < 1 or starts < 1:
        raise ValueError("Roll bounds, iterations and start count must be positive")
    output = list(original)
    # A sphere about the cut centre bounds the body for every possible roll.
    radii = [math.hypot(b.radius_m, b.axial_height_m) + axial_bound for b in original]
    adjacency = [set() for b in original]
    for i, a in enumerate(original):
        for j in range(i + 1, len(original)):
            if np.linalg.norm(a.center_m - original[j].center_m) <= radii[i] + radii[j] + required_gap_m:
                adjacency[i].add(j); adjacency[j].add(i)
    remaining, components = set(range(len(original))), []
    while remaining:
        seed = min(remaining); connected = {seed}; frontier = [seed]
        while frontier:
            current = frontier.pop()
            new = adjacency[current] - connected
            connected |= new; frontier.extend(sorted(new))
        remaining -= connected
        if len(connected) > 1 or line_mode:
            components.append(sorted(connected))
    component_reports = []
    # Deterministic covering directions, supplemented with the actual body's
    # normals, axes and cross axes at each iterate. Every positive sampled gap
    # is itself a valid analytic-support separation certificate.
    k = np.arange(96, dtype=float)
    zz = 1 - 2 * (k + .5) / len(k)
    phi = k * math.pi * (3 - math.sqrt(5))
    sphere = np.column_stack([np.sqrt(1 - zz * zz) * np.cos(phi),
                              np.sqrt(1 - zz * zz) * np.sin(phi), zz])

    for component_index, indices in enumerate(components):
        group = [original[i] for i in indices]
        n = len(group)
        pairs = list(itertools.combinations(range(n), 2))
        ia = np.array([p[0] for p in pairs], dtype=int); ib = np.array([p[1] for p in pairs], dtype=int)
        center = np.array([b.center_m for b in group])
        base = np.array([b.rotation for b in group])
        normal = np.array([b.normal for b in group])
        radius = np.array([b.radius_m for b in group])
        height = np.array([b.axial_height_m for b in group])
        outgoing = np.array([unit(line_directions[b.mirror_id]) for b in group]) if line_mode else None
        launch_maps, launch_offsets = [], []
        if source_models is not None:
            for b in group:
                models = source_models.get(b.mirror_id.split("_")[0], [])
                if len(models) != 1:
                    raise ValueError("Along-line relaxation needs one verified incident branch per body")
                model = models[0]
                projection = np.eye(3) - np.outer(model["direction"], model["direction"])
                mapping = np.linalg.pinv(projection @ model["map"][:2].T) @ projection
                launch_maps.append(mapping)
                launch_offsets.append(-mapping @ model["map"][2])
            launch_maps, launch_offsets = np.array(launch_maps), np.array(launch_offsets)
            source_radius = config.source.waist_radius * config.source.cutoff_ratio

        def rotations(parameters):
            return Rotation.from_rotvec(parameters[:n, None] * normal).as_matrix() @ base

        def centers(parameters):
            return center + parameters[n:, None] * axial_sigma_m * outgoing if line_mode else center

        def footprint(parameters):
            if source_models is None:
                return np.empty(0)
            launch_xy = np.einsum("ijk,ik->ij", launch_maps, centers(parameters)) + launch_offsets
            return 1. - np.linalg.norm(launch_xy, axis=1) / source_radius

        def prior(parameters):
            roll_cost = np.mean((parameters[:n] / max_roll_rad) ** 2)
            return float(roll_cost + (np.mean(parameters[n:] ** 2) if line_mode else 0.))

        def gaps(angles, return_normals=False):
            if not pairs:
                return (np.empty(0), np.empty((0, 3))) if return_normals else np.empty(0)
            pose = rotations(angles)
            position = centers(angles)
            delta = position[ib] - position[ia]
            axis = pose[:, :, 2]
            vectors = np.stack([delta, -delta, normal[ia], -normal[ia], normal[ib], -normal[ib],
                                axis[ia], -axis[ia], axis[ib], -axis[ib],
                                np.cross(axis[ia], axis[ib]), -np.cross(axis[ia], axis[ib])], axis=1)
            lengths = np.linalg.norm(vectors, axis=2)
            vectors /= np.maximum(lengths, 1e-30)[:, :, None]
            vectors = np.concatenate([np.broadcast_to(sphere, (len(pairs), len(sphere), 3)), vectors], axis=1)

            def support_local(directions, poses, r, h):
                local = np.einsum("pdk,pkj->pdj", directions, poses)
                return np.where(local[:, :, 2] >= 0,
                                r[:, None] * np.hypot(local[:, :, 0], local[:, :, 1]) + h[:, None] * local[:, :, 2],
                                r[:, None] * np.hypot(local[:, :, 0] + local[:, :, 2], local[:, :, 1]))

            value = (np.einsum("pdk,pk->pd", vectors, delta)
                     - support_local(vectors, pose[ia], radius[ia], height[ia])
                     - support_local(-vectors, pose[ib], radius[ib], height[ib]))
            # Zero-length cross-axis vectors are not unit separating normals.
            value[:, len(sphere):] = np.where(lengths > 1e-12, value[:, len(sphere):], -np.inf)
            argmax = np.argmax(value, axis=1)
            answer = value[np.arange(len(pairs)), argmax]
            return (answer, vectors[np.arange(len(pairs)), argmax]) if return_normals else answer

        initial_exact = validate_bodies(group, required_gap_m, numerical_tolerance_m, check_length_alternative=False)
        zero = np.zeros(n * (2 if line_mode else 1))
        initial_footprint = footprint(zero)
        if not initial_exact["conflicts"] and np.all(initial_footprint >= -1e-12):
            component_reports.append({"mirror_ids": [b.mirror_id for b in group], "status": "already_separated",
                                      "roll_angles_rad": [0.] * n, "starts": []})
            continue
        solutions, attempts, diagnostic_solutions = [], [], []
        original_violation_count = len(initial_exact["conflicts"]) + int(np.sum(initial_footprint < -1e-12))
        for start_index in range(starts):
            if start_index == 0:
                initial = np.zeros(n)
            elif start_index == 1:
                initial = np.where(np.arange(n) % 2 == 0, .5 * max_roll_rad, -.5 * max_roll_rad)
            else:
                initial = max_roll_rad * np.sin((np.arange(n) + 1) * (start_index + 1) * 2.399963229728653)
            if line_mode:
                shifts = np.zeros(n) if start_index == 0 else .1 * axial_limit_sigma * np.sin(np.arange(n) * 2.4 + start_index)
                initial = np.r_[initial, shifts]
            # Scale metres to mirror radii for well-conditioned constraints.
            margin = required_gap_m + 2 * numerical_tolerance_m
            bounds = [(-max_roll_rad, max_roll_rad)] * n + ([(-axial_limit_sigma, axial_limit_sigma)] * n if line_mode else [])
            solution = minimize(prior, initial,
                                method="SLSQP", bounds=bounds,
                                constraints={"type": "ineq", "fun": lambda a: np.r_[(gaps(a) - margin) / .0005, footprint(a)]},
                                options={"maxiter": int(max_iterations), "ftol": 1e-9})
            pose = rotations(solution.x)
            candidate = [MirrorBody(centers(solution.x)[i], pose[i], b.mirror_id, b.radius_m,
                                    b.axial_height_m, b.length_definition) for i, b in enumerate(group)]
            check = validate_bodies(candidate, required_gap_m, numerical_tolerance_m, check_length_alternative=False)
            sample_gaps, planes = gaps(solution.x, True)
            footprint_violations = int(np.sum(footprint(solution.x) < -1e-9))
            feasible = not check["conflicts"] and footprint_violations == 0
            attempts.append({"start": start_index, "optimizer_success": bool(solution.success),
                             "optimizer_message": str(solution.message), "iterations": int(solution.nit),
                             "certificate_validated": feasible, "conflict_count": len(check["conflicts"]),
                             "minimum_sampled_gap_m": float(sample_gaps.min()) if len(sample_gaps) else None,
                             "source_footprint_violations": footprint_violations,
                             "roll_angles_rad": solution.x[:n].tolist(),
                             "axial_shifts_m": (solution.x[n:] * axial_sigma_m).tolist() if line_mode else [0.] * n})
            if feasible:
                solutions.append((prior(solution.x), candidate, solution.x, planes))
            elif line_mode:
                count = len(check["conflicts"]) + footprint_violations
                if count < original_violation_count:
                    diagnostic_solutions.append((count, prior(solution.x), candidate, solution.x))
        if solutions:
            _, chosen, angles, planes = min(solutions, key=lambda s: s[0])
            for index, body in zip(indices, chosen):
                output[index] = body
            component_reports.append({"mirror_ids": [b.mirror_id for b in group], "status": "separated",
                                      "roll_angles_rad": angles[:n].tolist(), "starts": attempts,
                                      "axial_shifts_m": (angles[n:] * axial_sigma_m).tolist() if line_mode else [0.] * n,
                                      "separating_plane_directions": planes.tolist()})
        elif diagnostic_solutions:
            # Preserve a strictly lower-violation diagnostic candidate without
            # ever relabelling it feasible. It cannot replace accepted state.
            count, _, chosen, angles = min(diagnostic_solutions, key=lambda s: (s[0], s[1]))
            for index, body in zip(indices, chosen):
                output[index] = body
            component_reports.append({"mirror_ids": [b.mirror_id for b in group], "status": "improved_diagnostic_only",
                                      "remaining_violation_count": count, "initial_violation_count": original_violation_count,
                                      "roll_angles_rad": angles[:n].tolist(), "starts": attempts,
                                      "axial_shifts_m": (angles[n:] * axial_sigma_m).tolist()})
        else:
            # Never let a soft residual or success=True overwrite a candidate
            # with uncertified geometry. Keep the original failed component.
            component_reports.append({"mirror_ids": [b.mirror_id for b in group], "status": "no_certified_solution",
                                      "roll_angles_rad": [0.] * n, "starts": attempts})
        if progress is not None:
            progress(component_index + 1, len(components))
    final_check = validate_bodies(output, required_gap_m, numerical_tolerance_m, check_length_alternative=False)
    pending_diagnostics = any(c["status"] in {"improved_diagnostic_only", "no_certified_solution"} for c in component_reports)
    return output, {"status": "body_clearance_verified" if not final_check["conflicts"] and not pending_diagnostics else "no_global_feasible_solution",
                    "method": "bounded_along_line_centres_and_SO3_roll" if line_mode else "bounded_SO3_roll_multistart_with_support_plane_constraints",
                    "centers_fixed": not line_mode, "output_lines_fixed": True, "optical_normals_fixed": True,
                    "roll_prior": "minimum rotation about physical cut normal from supplied pose",
                    "max_roll_rad": max_roll_rad, "max_iterations": max_iterations, "starts": starts,
                    "axial_sigma_m": axial_sigma_m, "axial_limit_sigma": axial_limit_sigma if line_mode else None,
                    "source_footprint_constrained": source_models is not None,
                    "components": component_reports, "final_geometry": final_check}


def relax_body_rolls(bodies, required_gap_m=0., numerical_tolerance_m=1e-8,
                     max_iterations=60, starts=3, max_roll_rad=math.pi, progress=None):
    """Constrained unobserved roll relaxation with fixed centres and normals."""
    return _relax_body_poses(bodies, required_gap_m, numerical_tolerance_m,
                             max_iterations, starts, max_roll_rad, progress)


def relax_line_centers(config, bodies, outgoing_by_id, options, progress=None):
    """Use the unobserved centre coordinate along each fitted output line.

The optical field of the current scene is affine and direction-constant in
each verified planar branch; therefore fixing the output line also fixes the
physical cut normal. Non-planar/ambiguous fields are rejected, never silently
approximated. Full-source tracing remains the final, independent check.
"""
    sigma = options.get("axial_center_sigma_m", options.get("individual_center_sigma_m"))
    return _relax_body_poses(bodies, options.get("minimum_gap_m", 0.),
                             options.get("numerical_epsilon_m", 1e-8),
                             options.get("physical_max_iterations", 80), options.get("physical_starts", 3),
                             math.pi, progress, outgoing_by_id, sigma,
                             options.get("axial_center_limit_sigma", 3.), config)


_FIELD_CACHE = {}
_INCOMING_CACHE = {}


def _common_config(config):
    common = copy.copy(config)
    common.scene = Scene([e for e in config.scene.elements if not e.name.startswith("BUNDLE_")])
    return common


def _trace(config, rays):
    return RayTracer(config.scene, max_interactions=max(48, config.max_interactions),
                     max_time_s=config.integration_time_s, **config.tracer_options).trace(rays)


def _records(result):
    rows = {}
    for block in result.segments:
        for i, identifier in enumerate(block["ray_id"]):
            rows[int(identifier)] = {"id": int(identifier), "parent": int(block["parent_id"][i]),
                                     "surface": str(block["surface"][i]),
                                     "start": np.array([block[a + "0"][i] for a in "xyz"]),
                                     "end": np.array([block[a + "1"][i] for a in "xyz"]),
                                     "power": float(block["power"][i])}
    for row in rows.values():
        chain, current, seen = [], row, set()
        while current["parent"] in rows and current["id"] not in seen:
            seen.add(current["id"])
            current = rows[current["parent"]]
            chain.append(current["surface"])
        row["chain"] = list(reversed(chain))
        row["root"] = current["id"]
    return rows


def _record_indexes(rows):
    """Preserve trace insertion order, including equal-distance tie order."""
    by_branch, by_parent = {}, {}
    for row in rows.values():
        by_branch.setdefault((row["root"], tuple(row["chain"])), []).append(row)
        by_parent.setdefault(row["parent"], []).append(row)
    return by_branch, by_parent


def _field_models(config):
    common = _common_config(config)
    state = {"elements": [asdict(e) for e in common.scene.elements], "source": asdict(common.source),
             "time": common.integration_time_s, "trace": common.tracer_options,
             "max_interactions": max(48, common.max_interactions)}
    encoded = json.dumps(state, default=lambda v: to_numpy(v).tolist(), sort_keys=True)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    if digest in _FIELD_CACHE:
        return _FIELD_CACHE[digest], digest
    if any(not isinstance(s, PlaneSurface) for s in common.scene.build_surfaces() if not s.optics.detector):
        raise ValueError("Affine incident field requires planar common optics; nonlinear shooting is needed for this scene")
    launch = common.emit(19)
    rows = _records(_trace(common, launch))
    axis = unit(common.source.axis)
    helper = np.eye(3)[np.argmin(abs(axis))]
    u = unit(np.cross(axis, helper)); v = np.cross(axis, u)
    origin = np.asarray(common.source.waist_position)
    models = {}
    for rod in range(1, 5):
        samples = [row for row in rows.values() if row["chain"] and row["chain"][-1] == f"MC{rod}"]
        groups = {}
        for row in samples:
            groups.setdefault(tuple(row["chain"]), []).append(row)
        models[str(rod)] = []
        for chain, group in sorted(groups.items()):
            if len(group) < 3:
                continue
            directions = np.array([unit(row["end"] - row["start"]) for row in group])
            direction = unit(directions.mean(axis=0))
            direction_error = float(np.max(np.linalg.norm(directions - direction, axis=1)))
            if direction_error > 1e-9:
                continue
            initial = np.array([to_numpy(launch.position[row["root"]]) for row in group]) - origin
            design = np.column_stack([initial @ u, initial @ v, np.ones(len(group))])
            starts = np.array([row["start"] for row in group])
            coefficient, _, rank, _ = np.linalg.lstsq(design, starts, rcond=None)
            error = float(np.max(np.linalg.norm(design @ coefficient - starts, axis=1)))
            if rank < 3 or error > 1e-8:
                continue
            models[str(rod)].append({"direction": direction, "map": coefficient,
                                     "u": u, "v": v, "origin": origin, "chain": list(chain),
                                     "map_error_m": error, "direction_error": direction_error})
    if len(_FIELD_CACHE) >= 8:
        _FIELD_CACHE.pop(next(iter(_FIELD_CACHE)))
    _FIELD_CACHE[digest] = models
    return models, digest


def _launch_bundle(config, points):
    n = len(points)
    return RayBundle(np.asarray(points), np.tile(unit(config.source.axis), (n, 1)), np.ones(n),
                     np.full(n, config.source.wavelength_m), np.full(n, config.source.n_medium),
                     polarization=np.tile(config.source.polarization_reference, (n, 1)))


def incoming_at_points(config, points):
    """Full-optics incident field at specified cut centres, keyed r_w_j.

Returns unavailable/ambiguous explicitly. The centre-through-axis launch is
only a test of illumination; reconstruction may choose a different cut hit.
"""
    if not points:
        return {}
    models, digest = _field_models(config)
    # Exact coordinates and request order are significant: order determines
    # root IDs and therefore deterministic branch/tie ordering in the trace.
    point_key = tuple((identifier, np.asarray(center, dtype=float).shape,
                       np.asarray(center, dtype=float).tobytes()) for identifier, center in points.items())
    cache_key = (digest, point_key)
    if cache_key in _INCOMING_CACHE:
        return copy.deepcopy(_INCOMING_CACHE[cache_key])
    output, tests = {}, []
    radius = config.source.waist_radius * config.source.cutoff_ratio
    for identifier, center in points.items():
        center = np.asarray(center, dtype=float)
        rod = str(identifier).split("_")[0]
        alternatives = []
        for model in models.get(rod, []):
            d, mapping = model["direction"], model["map"]
            projection = np.eye(3) - np.outer(d, d)
            xy, _, rank, _ = np.linalg.lstsq(projection @ mapping[:2].T, projection @ (center - mapping[2]), rcond=None)
            start = np.r_[xy, 1.] @ mapping
            error = float(np.linalg.norm(projection @ (center - start)))
            launch = model["origin"] + xy[0] * model["u"] + xy[1] * model["v"]
            illuminated = rank == 2 and error < 1e-8 and np.linalg.norm(xy) <= radius + 1e-10 and np.dot(center - start, d) > 0
            item = {"direction": d.tolist(), "point_m": center.tolist(), "launch_point_m": launch.tolist(),
                    "field_origin_m": start.tolist(), "branch_id": "/".join(model["chain"]),
                    "interaction_chain": model["chain"], "status": "pending" if illuminated else "outside_illumination",
                    "affine_residual_m": error, "field_hash": digest,
                    "center_hit_assumption": "characteristic ray through cut centre; not inferred from output line"}
            alternatives.append(item)
            if illuminated:
                tests.append((identifier, item, launch))
        output[identifier] = {"status": "unavailable", "alternatives": alternatives, "field_hash": digest}
    if tests:
        rows = _records(_trace(_common_config(config), _launch_bundle(config, [t[2] for t in tests])))
        by_branch, _ = _record_indexes(rows)
        for index, (identifier, item, launch) in enumerate(tests):
            matches = by_branch.get((index, tuple(item["interaction_chain"])), [])
            errors = []
            for row in matches:
                actual = unit(row["end"] - row["start"])
                center = np.asarray(item["point_m"])
                t = float(np.dot(center - row["start"], actual))
                length = float(np.linalg.norm(row["end"] - row["start"]))
                miss = float(np.linalg.norm(center - row["start"] - t * actual))
                if t >= -1e-9 and t <= length + 1e-8:
                    errors.append((miss, actual, row))
            if errors:
                miss, actual, row = min(errors, key=lambda x: x[0])
                item.update(status="available" if miss <= 1e-8 else "verification_failed", direction=actual.tolist(),
                            full_trace_miss_m=miss, field_origin_m=row["start"].tolist())
            else:
                item["status"] = "branch_not_reached"
    for identifier, value in output.items():
        good = [a for a in value["alternatives"] if a["status"] == "available"]
        if len(good) == 1:
            output[identifier] = {**good[0], "alternatives": value["alternatives"]}
        elif good:
            output[identifier] = {"status": "ambiguous", "alternatives": good, "field_hash": digest}
        elif value["alternatives"]:
            # Retain a traced branch's direction for diagnostics outside its
            # footprint, while keeping the failed illumination status explicit.
            output[identifier] = {**value["alternatives"][0], "alternatives": value["alternatives"]}
    if len(_INCOMING_CACHE) >= 8:
        _INCOMING_CACHE.pop(next(iter(_INCOMING_CACHE)))
    # Neither the first result nor later hits expose the private cached value.
    _INCOMING_CACHE[cache_key] = copy.deepcopy(output)
    return output


def incoming_at_mirrors(config):
    from ..configurations import _bundle_surfaces
    return incoming_at_points(config, {key: to_numpy(surface.center) for key, surface in _bundle_surfaces(config)})


def install_bodies(config, bodies):
    from ..configurations import StaticElement
    bodies = [MirrorBody.from_dict(b) if isinstance(b, dict) else b for b in bodies]
    elements = [e for e in config.scene.elements if not e.name.startswith("BUNDLE_")]
    for body in bodies:
        surfaces = body.build_surfaces()
        elements.append(StaticElement(surfaces[0].name, surfaces))
    config.scene = Scene(elements)
    config.metadata["mirror_body_model"] = "cut_cylinder_45deg_v1"
    config.metadata["nonreflecting_surface_model"] = "opaque_absorbing_assumption"
    return config


def validate_reflections(config, bodies, physical_bodies=True):
    """Launch a targeted ray at each cut through the complete current scene.

Checks the first physical body hit, actual emitted daughter, reflection law,
and the downstream segment. An absence of random samples is never a verdict.
"""
    bodies = [MirrorBody.from_dict(b) if isinstance(b, dict) else b for b in bodies]
    if physical_bodies:
        # Validate the supplied candidate, never the historic bundle planes
        # still present in the caller's active configuration. Keep acceptance
        # atomic: diagnostics must not mutate that active state.
        candidate = copy.copy(config)
        candidate.metadata = copy.deepcopy(config.metadata)
        install_bodies(candidate, bodies)
        config = candidate
    field = incoming_at_points(config, {b.mirror_id: b.center_m for b in bodies})
    launched = [b for b in bodies if field[b.mirror_id]["status"] == "available"]
    records = {}
    trace_diagnostics = {}
    if launched:
        traced = _trace(config, _launch_bundle(config, [field[b.mirror_id]["launch_point_m"] for b in launched]))
        records = _records(traced)
        trace_diagnostics = traced.diagnostics
    reports = {}
    by_branch, by_parent = _record_indexes(records)
    for body in bodies:
        incoming = field[body.mirror_id]
        reports[body.mirror_id] = {"status": "unavailable", "incoming": incoming,
                                    "target_surface": body.build_surfaces()[0].name,
                                    "waist_side": "not_evaluated"}
    for root, body in enumerate(launched):
        incoming = field[body.mirror_id]
        branch = incoming["interaction_chain"]
        candidates = by_branch.get((root, tuple(branch)), [])
        report = reports[body.mirror_id]
        wanted = body.build_surfaces()[0].name
        if not candidates:
            report.update(status="infeasible", reason="Incoming branch not present in physical scene")
            continue
        row = min(candidates, key=lambda r: np.linalg.norm(r["end"] - body.center_m))
        report["incident_segment"] = dict(start_m=row["start"].tolist(), end_m=row["end"].tolist(),
                                          surface=row["surface"])
        if row["surface"] != wanted:
            report.update(status="infeasible", reason="Another physical surface is hit before the requested cut",
                          first_surface=row["surface"])
            continue
        din = unit(row["end"] - row["start"])
        expected = din - 2 * np.dot(din, body.normal) * body.normal
        children = by_parent.get(row["id"], [])
        valid_children = [r for r in children if np.linalg.norm(r["end"] - r["start"]) > 1e-12 and
                          np.linalg.norm(unit(r["end"] - r["start"]) - expected) < 1e-7]
        side = float(np.dot(din, body.normal))
        cut = body.on_cut(row["end"])
        paths, downstream_body = [], []
        pending = [(child, [], [row["end"].tolist()]) for child in valid_children]
        while pending:
            current, path, points = pending.pop()
            segment_vector = current["end"] - current["start"]
            segment = {"ray_id": current["id"], "parent_id": current["parent"],
                       "start_m": current["start"].tolist(), "end_m": current["end"].tolist(),
                       "direction": unit(segment_vector).tolist() if np.linalg.norm(segment_vector) > 1e-12 else None,
                       "surface": current["surface"], "power": current["power"]}
            next_path = path + [segment]
            next_points = points + [current["end"].tolist()]
            if current["surface"].startswith("BUNDLE_"):
                downstream_body.append(current["surface"])
            descendants = by_parent.get(current["id"], [])
            if descendants:
                pending.extend((child, next_path, next_points) for child in descendants)
            else:
                paths.append({"points_m": next_points, "segments": next_path,
                              "interaction_chain": [s["surface"] for s in next_path],
                              "terminal_surface": current["surface"]})
        valid = bool(valid_children) and side < -1e-8 and cut and not downstream_body
        report.update(status="validated" if valid else "infeasible", actual_reflected_child=bool(valid_children),
                      incident_dot_normal=side, outgoing_dot_normal=float(np.dot(expected, body.normal)),
                      hit_m=row["end"].tolist(), hit_on_ellipse=cut, outgoing_direction=expected.tolist(),
                      outgoing_blockers=sorted(set(downstream_body)), next_surfaces=[r["surface"] for r in valid_children],
                      reflected_trajectories=paths,
                      reflected_child={"point_m": row["end"].tolist(), "direction": expected.tolist(),
                                       "ray_ids": [r["id"] for r in valid_children]})
    return {"mirrors": reports, "validated_count": sum(r["status"] == "validated" for r in reports.values()),
            "failed_count": sum(r["status"] == "infeasible" for r in reports.values()),
            "unavailable_count": sum(r["status"] == "unavailable" for r in reports.values()),
            "trace_settings": {"collision_mode": "exhaustive", "max_interactions": max(48, config.max_interactions),
                               "integration_time_s": config.integration_time_s, "targeted_launch_count": len(launched)},
            "trace_diagnostics": trace_diagnostics,
            "method": "targeted_source_launch_and_actual_reflected_children"}
