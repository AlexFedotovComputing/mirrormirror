"""Leave-assembly-out, hide-ray validation with source-identity isolation."""
from __future__ import annotations

import numpy as np

from .empirical import angular_distance, learn_template, unit
from .observations import exclude_sources


def _summary(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(count=len(values), median=float(np.median(values)) if values else None,
                p90=float(np.quantile(values, .9)) if values else None,
                maximum=max(values) if values else None)


def screen_errors(point, direction, screens):
    """Real XY sections through recorded screen points, not XY line projections."""
    output = []
    for screen in screens:
        if any(v is None for v in screen):
            continue
        target = np.asarray(screen)
        if abs(direction[2]) > 1e-7:
            s = float((target[2] - point[2]) / direction[2])
            predicted = point + s * direction
            output.append(dict(z_m=float(target[2]), distance_m=float(np.linalg.norm(predicted[:2] - target[:2])),
                               predicted_point_m=predicted.tolist(), forward=s >= 0, method="intersection_with_real_z_section"))
        else:
            distance = np.linalg.norm(np.cross(target - point, direction))
            output.append(dict(z_m=float(target[2]), distance_m=float(distance), predicted_point_m=None,
                               forward=bool((target - point) @ direction >= 0), method="3D_line_distance_near_parallel_section"))
    return output


def validate_withheld(config, registry, options, progress=None):
    from .reconstruction import (_notify, group_observations, fit_assembly)
    groups = group_observations(registry["records"])
    folds = []
    planned = [(key, row) for key, group in sorted(groups.items())
               for row in sorted([r for r in group["records"] if r["complete_points"] == 2], key=lambda r: r["id"])[:int(options["holdout_per_assembly"])]]
    for index, (key, hidden) in enumerate(planned):
        _notify(progress, "оценка качества", completed=index, total=len(planned), assembly=key)
        target_ids = {r["independent_source_id"] for r in groups[key]["records"]}
        training_records = exclude_sources(registry, target_ids)
        visible_records = exclude_sources(registry, {hidden["independent_source_id"]})
        training_groups = group_observations(training_records)
        visible_groups = group_observations(visible_records)
        fold = dict(assembly=key, hidden_observation_id=hidden["id"],
                    training_observation_ids=[r["id"] for r in training_records],
                    target_visible_observation_ids=[r["id"] for r in visible_groups[key]["records"]],
                    excluded_independent_sources=sorted(target_ids),
                    provenance_policy="Entire target assembly excluded from template; survey coordinates used directly without any derived geometry or registration; all hidden-source copies excluded",
                    status="failed")
        try:
            frame = dict(rotation=np.eye(3).tolist(), translation_m=[0., 0., 0.])
            visible = visible_groups[key]
            model = learn_template(training_groups, iterations=options["template_iterations"], ambiguity_rad=options["assignment_resolution_rad"])
            # Historical snapshots are themselves reconstructed from these
            # surveys and therefore are NOT independent mechanical priors.
            # The primary benchmark uses a conditional point-fan model whose
            # origin is initialized exclusively by the remaining visible rays.
            centres = np.tile(np.asarray(visible["anchor"]), (7, 1))
            fold_options = {**options, "assembly_translation_sigma_m": 1.}
            relaxed = fit_assembly(model, visible, centres, frame, fold_options)
            rigid = fit_assembly(model, visible, centres, frame, fold_options, allow_individual=False)
            free = [j for j in range(7) if j not in relaxed["assigned"]]
            if not free:
                raise ValueError("No unmatched canonical place after withholding")
            q, t = np.asarray(frame["rotation"]), np.asarray(frame["translation_m"])
            # Freeze candidates and selection BEFORE accessing hidden truth. The
            # first canonical free place is a gauge decision, not best-of-truth.
            selected = free[0]
            frozen = [dict(slot=j+1, point=(relaxed["centres"][j] @ q.T + t).tolist(),
                           direction=(relaxed["directions"][j] @ q.T).tolist()) for j in free]
            truth_direction = unit(np.asarray(hidden["screens_m"][1]) - hidden["screens_m"][0])
            truth_point = np.asarray(hidden["screens_m"][0])
            visible_anchor = np.asarray(visible["anchor"]) @ q.T + t
            if truth_direction @ (truth_point - visible_anchor) < 0:
                truth_direction *= -1
            distances = [float(angular_distance(p["direction"], truth_direction)) for p in frozen]
            prediction = frozen[0]
            sections = screen_errors(np.asarray(prediction["point"]), np.asarray(prediction["direction"]), hidden["screens_m"])
            rigid_direction = rigid["directions"][selected] @ q.T
            sigma = max(options["individual_angle_sigma_rad"], model["statistics"][selected]["observed_predictive_sigma_rad"] or 0.)
            # All unassigned places AND comparable modes are retained. Interval
            # coverage is labelled set-valued; it cannot imply unique recovery.
            mode_directions = []
            for alternative in relaxed["alternatives"]:
                for j in range(7):
                    if j not in alternative["assigned"]:
                        d = np.asarray(model["directions_local"])[j] @ np.asarray(alternative["rotation"]).T @ q.T
                        mode_directions.append(d)
            mode_errors = [float(angular_distance(d, truth_direction)) for d in mode_directions]
            fold.update(status="predicted", selected_slot=selected+1, candidate_count=len(frozen), predictions=frozen,
                        selection_rule="First unmatched canonical place fixed before consulting withheld truth",
                        angular_error_mrad=distances[0] * 1000,
                        set_angular_error_mrad=min(mode_errors or distances) * 1000,
                        set_mode_count=len(mode_errors), set_metric_note="Nearest member evaluated after prediction; not a single-ray accuracy claim",
                        screen_sections=sections,
                        maximum_screen_error_m=max((s["distance_m"] for s in sections), default=None),
                        rigid_empirical_error_mrad=float(angular_distance(rigid_direction, truth_direction)) * 1000,
                        historical_comparator="ineligible_for_independent_score: historical geometry derives from the held-out survey",
                        two_point_current_method="No prediction for missing ray",
                        predictive_sigma_rad=sigma, covered_conditional_95=distances[0] <= 1.96 * sigma,
                        set_covered_conditional_95=min(mode_errors or distances) <= 1.96 * sigma,
                        registration_training_groups=[], coordinate_registration="not_required_raw_survey_frame",
                        center_prior="conditional point fan initialized from visible rays only; 1 m weak common translation prior",
                        physical_body_validation="not_evaluated_without_independent_body_layout",
                        excluded_target_assembly_from_template=True,
                        uncertainty_sensitivity=[dict(measurement_sigma_m=s, angular_sigma_rad=float(np.sqrt(sigma**2 + 2*s*s / max(np.linalg.norm(np.asarray(hidden['screens_m'][1])-hidden['screens_m'][0])**2, 1e-12))))
                                                 for s in options["uncertainty_scenarios_m"]])
        except (ValueError, np.linalg.LinAlgError) as error:
            fold["reason"] = str(error)
        folds.append(fold)
    good = [f for f in folds if f["status"] == "predicted"]
    return dict(status="evaluated", experiment="leave_assembly_out_then_hide_primary_ray",
                whole_assembly_prediction="not_evaluated_separate_experiment", folds=folds,
                planned=len(planned), predicted=len(good), failures=len(folds)-len(good),
                angular_error_mrad=_summary([f["angular_error_mrad"] for f in good]),
                set_angular_error_mrad=_summary([f["set_angular_error_mrad"] for f in good]),
                screen_error_m=_summary([f["maximum_screen_error_m"] for f in good]),
                rigid_empirical_error_mrad=_summary([f["rigid_empirical_error_mrad"] for f in good]),
                historical_comparator="Not independently scorable: historical snapshots were fitted to the original measurements",
                conditional_interval_coverage=float(np.mean([f["covered_conditional_95"] for f in good])) if good else None,
                conditional_set_coverage=float(np.mean([f["set_covered_conditional_95"] for f in good])) if good else None,
                intervals_certified=False,
                limitations=["Only 16 assembly clusters per side; interval coverage is a small-sample empirical diagnostic",
                             "Selected missing slot may be ambiguous; set-valued metrics include cardinality and are optimistic for unique prediction",
                             "Hyperparameters are fixed scenarios, not tuned on hidden observations",
                             "Primary score is angular/section prediction using a conditional point-fan model, not a fully physical body test",
                             "Historical body geometry and coordinate registration are excluded because they contain fitted survey information",
                             "Whole-assembly concealment is not the same experiment and is not claimed"])
