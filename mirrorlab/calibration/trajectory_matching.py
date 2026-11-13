"""One-to-one geometric comparison, independent of ray and assembly labels."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _complete_observations(observations):
    complete, excluded = [], []
    for row in observations:
        screens = row.get("screens_m") or []
        if len(screens) != 2 or any(v is None for p in screens for v in p):
            excluded.append(dict(observation_id=row["id"], reason="two_complete_points_required"))
            continue
        points = np.asarray(screens, float)
        if points.shape != (2, 3) or not np.isfinite(points).all() or np.linalg.norm(points[1] - points[0]) < 1e-12:
            excluded.append(dict(observation_id=row["id"], reason="invalid_or_coincident_points"))
            continue
        complete.append((row, points))
    return complete, excluded


def _pair_metrics(row, points, candidate):
    measured = points[1] - points[0]
    measured /= np.linalg.norm(measured)
    origin = np.asarray(candidate["point_m"], float)
    direction = np.asarray(candidate["direction"], float)
    norm = np.linalg.norm(direction)
    if origin.shape != (3,) or direction.shape != (3,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid trajectory direction or origin")
    direction = direction / norm
    length = candidate.get("length_m", np.inf)
    if length <= 0 or np.isnan(length) or not np.isfinite(origin).all():
        raise ValueError("Invalid trajectory extent")
    along = (points - origin) @ direction
    closest = origin + np.clip(along, 0., length)[:, None] * direction
    distances = np.linalg.norm(points - closest, axis=1)
    angle = float(np.arctan2(np.linalg.norm(np.cross(measured, direction)), abs(measured @ direction)))
    return dict(observation_id=row["id"], trajectory_id=candidate["id"],
                point_distances_m=distances.tolist(), maximum_distance_m=float(distances.max()),
                angular_error_rad=angle)


def _summary(values):
    values = np.asarray(values, float)
    return dict(count=len(values), median=float(np.median(values)) if len(values) else None,
                p90=float(np.quantile(values, .9)) if len(values) else None,
                p95=float(np.quantile(values, .95)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def match_trajectories(observations, trajectories, *, distance_tolerance_m=.003,
                       angle_tolerance_rad=None):
    """Maximize matched complete observations, then minimize geometric error.

    Points must lie on the forward, actually traced segment (or forward ray for
    explicitly untraced predictions). Screen order does not encode propagation
    direction. Labels are returned for inspection, never used in the cost.
    """
    tolerances = [distance_tolerance_m] + ([angle_tolerance_rad] if angle_tolerance_rad is not None else [])
    if not all(np.isfinite(v) and v > 0 for v in tolerances):
        raise ValueError("Trajectory matching tolerances must be positive and finite")
    complete, excluded = _complete_observations(observations)
    n, m = len(complete), len(trajectories)
    # A lost match must cost more than every possible improvement in the other
    # n accepted pairs combined. Forbidden pairs cost more than all dummies.
    costs = np.full((n, m + n), float(n + 1))
    costs[:, :m] = (n + 1) * (m + n + 1)
    metrics = {}
    for i, (row, points) in enumerate(complete):
        for j, candidate in enumerate(trajectories):
            metric = _pair_metrics(row, points, candidate)
            distance, angle = metric["maximum_distance_m"], metric["angular_error_rad"]
            score = distance / distance_tolerance_m
            allowed = distance <= distance_tolerance_m
            if angle_tolerance_rad is not None:
                score = .5 * (score**2 + (angle / angle_tolerance_rad)**2)
                allowed = allowed and angle <= angle_tolerance_rad
            metrics[i, j] = dict(metric, within_tolerance=bool(allowed), normalized_cost=float(score))
            if allowed:
                costs[i, j] = score
    rows, columns = linear_sum_assignment(costs)
    pairs = [metrics[i, j] for i, j in zip(rows, columns) if j < m and metrics[i, j]["within_tolerance"]]
    used_rows = {i for i, j in zip(rows, columns) if j < m and metrics[i, j]["within_tolerance"]}
    used_columns = {j for i, j in zip(rows, columns) if j < m and metrics[i, j]["within_tolerance"]}
    unmatched = []
    for i, (row, _) in enumerate(complete):
        if i not in used_rows:
            eligible = [metrics[i, j] for j in range(m) if metrics[i, j]["within_tolerance"]]
            nearest = min(eligible or [metrics[i, j] for j in range(m)], key=lambda x: x["normalized_cost"], default=None)
            unmatched.append(dict(observation_id=row["id"], nearest=nearest,
                                  reason="one_to_one_conflict" if eligible
                                  else "no_trajectory_within_tolerances"))
    return dict(status="evaluated", method="maximum_cardinality_minimum_geometric_cost",
                numbering_ignored=True, distance_tolerance_m=distance_tolerance_m,
                angle_tolerance_rad=angle_tolerance_rad, observation_count=n, trajectory_count=m,
                matched_count=len(pairs), unmatched_observation_count=len(unmatched),
                unmatched_trajectory_count=m-len(used_columns), matches=pairs,
                unmatched_observations=unmatched,
                unmatched_trajectory_ids=[c["id"] for j, c in enumerate(trajectories) if j not in used_columns],
                excluded_observations=excluded,
                tolerance_basis="explicit comparison scenario, not certified metrology",
                extra_trajectory_note="No measured counterpart does not imply a spurious ray: observations are incomplete")


def best_fit_trajectories(observations, trajectories, *, distance_tolerance_m=.003):
    """Assign all possible pairs by minimum sum of maximum two-point errors.

    No pair is rejected or weighted by its angle. Summaries include the worst
    pairs, not only those inside the display tolerance. This is an in-sample
    geometric fit and is not a physical or withheld-data validation.
    """
    if not np.isfinite(distance_tolerance_m) or distance_tolerance_m <= 0:
        raise ValueError("Trajectory matching tolerance must be positive and finite")
    complete, excluded = _complete_observations(observations)
    metrics = {(i, j): _pair_metrics(row, points, candidate)
               for i, (row, points) in enumerate(complete) for j, candidate in enumerate(trajectories)}
    costs = np.empty((len(complete), len(trajectories)))
    for (i, j), metric in metrics.items():
        costs[i, j] = metric["maximum_distance_m"]
    rows, columns = linear_sum_assignment(costs)
    pairs = [dict(metrics[i, j], within_tolerance=metrics[i, j]["maximum_distance_m"] <= distance_tolerance_m)
             for i, j in zip(rows, columns)]
    used_rows, used_columns = set(rows), set(columns)
    outliers = [p for p in pairs if not p["within_tolerance"]]
    return dict(status="evaluated", method="minimum_sum_maximum_two_point_distance",
                numbering_ignored=True, angle_tolerance_rad=None, distance_tolerance_m=distance_tolerance_m,
                observation_count=len(complete), trajectory_count=len(trajectories), assigned_count=len(pairs),
                matched_count=sum(p["within_tolerance"] for p in pairs),
                counts_within_1mm=sum(p["maximum_distance_m"] <= .001 for p in pairs),
                counts_within_3mm=sum(p["maximum_distance_m"] <= .003 for p in pairs),
                position_error_mm=_summary([p["maximum_distance_m"] * 1000 for p in pairs]),
                angular_error_mrad=_summary([p["angular_error_rad"] * 1000 for p in pairs]),
                matches=pairs, outliers=outliers,
                unmatched_observation_count=len(complete)-len(pairs),
                unmatched_observations=[dict(observation_id=row["id"], reason="fewer_calculated_trajectories")
                                        for i, (row, _) in enumerate(complete) if i not in used_rows],
                unmatched_trajectory_ids=[p["id"] for j, p in enumerate(trajectories) if j not in used_columns],
                excluded_observations=excluded, physical_paths=False,
                statistical_scope="all_assigned_complete_observations_including_outliers",
                interpretation="in_sample_geometric_fit_not_physical_feasibility_or_missing_ray_validation")


def traced_output_trajectories(reflections, frame):
    """Extract continuous straight output paths, including transparent screens.

    Stop at a turn or the traced endpoint: never extrapolate through an opaque
    blocker. A blocked path can still reproduce points before its blocker.
    Repeated branch prefixes count once, not once per downstream descendant.
    """
    q, t = np.asarray(frame["rotation"], float), np.asarray(frame["translation_m"], float)
    unique = {}
    for mirror_id, report in reflections.get("mirrors", {}).items():
        for path in report.get("reflected_trajectories", []):
            segments = path.get("segments", [])
            if not segments:
                continue
            origin = np.asarray(segments[0]["start_m"], float)
            delta = np.asarray(segments[0]["end_m"], float) - origin
            length = float(np.linalg.norm(delta))
            if length <= 1e-12:
                continue
            direction = delta / length
            last = segments[0]
            stop_reason = "trace_endpoint"
            prefix = [last]
            for segment in segments[1:]:
                start, end = np.asarray(segment["start_m"], float), np.asarray(segment["end_m"], float)
                step = end - start
                size = np.linalg.norm(step)
                if (size <= 1e-12 or np.linalg.norm(step / size - direction) > 1e-7 or
                        np.linalg.norm(start - (origin + length * direction)) > 1e-6):
                    stop_reason = "direction_change_or_discontinuity"
                    break
                length = float((end - origin) @ direction)
                last = segment
                prefix.append(segment)
            key = tuple(np.round(np.r_[origin, direction], 9))
            if key not in unique or length > unique[key]["length_m"]:
                unique[key] = dict(id=f"path:{mirror_id}:{len(unique)}", mirror_id=mirror_id,
                                   point_m=(origin @ q.T + t).tolist(), direction=(direction @ q.T).tolist(),
                                   length_m=length,
                                   end_m=(np.asarray(last["end_m"]) @ q.T + t).tolist(),
                                   terminal_surface=last.get("surface"), stop_reason=stop_reason,
                                   segments=[dict(start_m=(np.asarray(s["start_m"]) @ q.T + t).tolist(),
                                                  end_m=(np.asarray(s["end_m"]) @ q.T + t).tolist(),
                                                  surface=s.get("surface")) for s in prefix])
    # Stable geometry order, independent of the input dictionary/old IDs.
    return [unique[k] for k in sorted(unique)]


def _dataset_context(dataset):
    observations = dataset.get("observations")
    if isinstance(observations, dict):
        observations = observations.get("records", observations.get("observations", []))
    frame = dataset.get("display_frame")
    if not observations:
        return None, None, "observations_unavailable"
    observations = [{**o, "id": o.get("id", o.get("observation_id"))} for o in observations]
    if frame is None:
        if dataset.get("coordinate_frame") != "survey":
            return None, None, "survey_coordinate_transform_unavailable"
        frame = dict(rotation=np.eye(3), translation_m=np.zeros(3))
    return observations, frame, None


def compare_dataset_geometric_fit(dataset, *, distance_tolerance_m=.003):
    observations, frame, reason = _dataset_context(dataset)
    if reason:
        return dict(status="not_evaluated", reason=reason)
    q, t = np.asarray(frame["rotation"], float), np.asarray(frame["translation_m"], float)
    trajectories = []
    for mirror in dataset.get("mirrors", []):
        # Do not substitute a traced direction for a missing fitted direction:
        # these reports answer different questions.
        if mirror.get("outgoing_direction") is None:
            continue
        trajectories.append(dict(id=f"fit:{mirror['mirror_id']}", mirror_id=mirror["mirror_id"],
            point_m=(np.asarray(mirror["center_m"], float) @ q.T + t).tolist(),
            direction=(np.asarray(mirror["outgoing_direction"], float) @ q.T).tolist()))
    if not trajectories:
        return dict(status="not_evaluated", reason="fitted_directions_unavailable")
    report = best_fit_trajectories(observations, trajectories, distance_tolerance_m=distance_tolerance_m)
    report.update(coordinate_frame="survey", source="stored_fitted_output_rays",
                  scope="all_complete_observations_on_this_side_without_assembly_or_ray_number_constraints",
                  trajectories=trajectories, physical_feasibility_is_separate=True)
    return report


def compare_dataset_trajectories(dataset, *, distance_tolerance_m=.003):
    observations, frame, reason = _dataset_context(dataset)
    if reason:
        return dict(status="not_evaluated", reason=reason)
    reflections = dataset.get("diagnostics", {}).get("reflections")
    if reflections is None:
        return dict(status="not_evaluated", reason="traced_reflections_unavailable")
    trajectories = traced_output_trajectories(reflections, frame)
    report = match_trajectories(observations, trajectories,
                               distance_tolerance_m=distance_tolerance_m)
    report.update(coordinate_frame="survey", source="stored_actual_reflection_segments",
                  scope="all_complete_observations_on_this_side_without_assembly_or_ray_number_constraints",
                  trajectories=trajectories, physical_feasibility_is_separate=True)
    return report


def dataset_comparison_report(dataset):
    """Recompute both reports from embedded data, without mutating the dataset."""
    from .trajectory_diagnostics import diagnose_trajectories
    fit = compare_dataset_geometric_fit(dataset)
    physical = compare_dataset_trajectories(dataset)
    return dict(direction=dataset["direction"], status=dataset["status"], diagnostics=dict(
        geometric_fit=fit, trajectory_matching=physical,
        trajectory_diagnostics=diagnose_trajectories(dataset, fit, physical)))
