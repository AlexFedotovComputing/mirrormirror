"""Partially observed seven-place angular template with geometric assignments.

The seven angles are estimated, not specified by a fan opening. Rigid assembly
rotations are removed on SO(3). Every candidate subset is evaluated; unresolved
edge omissions remain separate assignment hypotheses.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def unit(values):
    values = np.asarray(values, float)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-15)


def angular_distance(first, second):
    return np.arctan2(np.linalg.norm(np.cross(first, second), axis=-1), np.sum(np.asarray(first) * second, axis=-1))


def rigid_rotation(source, target):
    """Proper Wahba rotation; no averages of Euler angles."""
    u, _, vt = np.linalg.svd(np.asarray(source).T @ target)
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    return vt.T @ correction @ u.T


def gauge_template(directions):
    directions = unit(directions)
    x = unit(directions.sum(axis=0))
    z = unit(np.cross(directions[0], directions[-1]))
    if np.linalg.norm(z) < .5:
        raise ValueError("Angular template is collinear")
    y = unit(np.cross(z, x))
    z = unit(np.cross(x, y))
    return unit(directions @ np.array([x, y, z]).T)


def local_fan(directions, anchor=None):
    """Canonical ordering by physical transverse angle, independent of old IDs."""
    directions = unit(directions)
    x = unit(directions.sum(axis=0))
    z = np.array([0., 0., 1.])
    y = unit(np.cross(z, x))
    if np.linalg.norm(y) < .5:
        _, _, vt = np.linalg.svd(directions)
        y = unit(vt[1])
    z = unit(np.cross(x, y))
    basis = np.array([x, y, z]).T
    local = directions @ basis
    order = np.argsort(np.arctan2(local[:, 1], local[:, 0]), kind="stable")
    return local[order], order, basis


def assign_fan(template, directions, *, ambiguity_rad=.002, max_alternatives=6, allow_outlier=True):
    """Enumerate monotone subsets and at most one explicitly excluded outlier."""
    template, directions = unit(template), unit(directions)
    if not 3 <= len(directions) <= 7:
        raise ValueError("A fan needs 3 to 7 complete rays")
    local, order, basis = local_fan(directions)
    candidates = []
    masks = [np.arange(len(local))]
    if allow_outlier and len(local) > 4:
        masks += [np.delete(np.arange(len(local)), k) for k in range(len(local))]
    for keep in masks:
        for slots in combinations(range(7), len(keep)):
            slots = np.asarray(slots, int)
            rotation = rigid_rotation(template[slots], local[keep])
            errors = angular_distance(template[slots] @ rotation.T, local[keep])
            # Removing a point requires a 5-sigma gross error, not an automatic match.
            cost = float(np.sum(np.minimum(errors**2, (10 * ambiguity_rad)**2))
                         + (len(local) - len(keep)) * (5 * ambiguity_rad)**2)
            global_rotation = basis @ rotation
            candidates.append(dict(slots=slots.tolist(), observation_indices=order[keep].tolist(),
                                   outliers=order[np.setdiff1d(np.arange(len(local)), keep)].tolist(),
                                   cost=cost, rotation=global_rotation.tolist(), angular_residual_rad=errors.tolist()))
    candidates.sort(key=lambda c: (c["cost"], c["slots"]))
    best = candidates[0]
    tolerance = len(directions) * ambiguity_rad**2
    alternatives = [c for c in candidates if c["cost"] <= best["cost"] + tolerance]
    return {**best, "alternatives": alternatives[:max_alternatives], "alternative_count": len(alternatives),
            "ambiguous": len(alternatives) > 1, "cost_is_probability": False,
            "unmatched_cost_rad2": (5 * ambiguity_rad)**2}


def learn_template(groups, *, iterations=8, ambiguity_rad=.002):
    """Block alternating robust spherical estimation from incomplete assemblies."""
    usable = {key: value for key, value in groups.items() if len(value["directions"]) >= 3}
    if len(usable) < 2:
        raise ValueError("At least two partially measured assemblies are needed")
    gaps = []
    for group in usable.values():
        local, _, _ = local_fan(group["directions"])
        angle = np.arctan2(local[:, 1], local[:, 0])
        gaps.extend(np.diff(angle).tolist())
    # A data-derived nearest-neighbour scale is initialization only; all seven
    # spherical directions are subsequently estimated without equal spacing.
    gaps = np.asarray(gaps)
    step = float(np.median(gaps[gaps < np.quantile(gaps, .7)]))
    initial = np.linspace(-3, 3, 7) * step
    starts = [initial, initial + .08 * step * np.sin(np.arange(7)), initial - .08 * step * np.sin(np.arange(7))]
    solutions = []
    for start in starts:
        template = np.column_stack([np.cos(start), np.sin(start), np.zeros(7)])
        history = []
        for _ in range(iterations):
            samples = [[] for _ in range(7)]
            assignments = {}
            for key, group in usable.items():
                fit = assign_fan(template, group["directions"], ambiguity_rad=ambiguity_rad)
                assignments[key] = fit
                aligned = np.asarray(group["directions"]) @ np.asarray(fit["rotation"])
                for j, k in zip(fit["slots"], fit["observation_indices"]):
                    samples[j].append(aligned[k])
            updated = []
            for j, sample in enumerate(samples):
                if not sample:
                    updated.append(template[j])
                    continue
                sample = np.asarray(sample)
                center = unit(np.median(sample, axis=0))
                errors = angular_distance(sample, center)
                weights = np.minimum(1., max(ambiguity_rad, float(np.median(errors)) * 1.4826) / np.maximum(errors, 1e-15))
                updated.append(unit(np.sum(sample * weights[:, None], axis=0)))
            updated = gauge_template(updated)
            change = float(np.max(angular_distance(updated, template)))
            history.append(dict(cost=float(sum(a["cost"] for a in assignments.values())), change_rad=change))
            template = updated
            if change < 1e-7:
                break
        final = {key: assign_fan(template, group["directions"], ambiguity_rad=ambiguity_rad) for key, group in usable.items()}
        solutions.append((sum(a["cost"] for a in final.values()), template, final, history))
    cost, template, assignments, history = min(solutions, key=lambda x: x[0])
    residuals = [[] for _ in range(7)]
    normal_samples = [[] for _ in range(7)]
    clusters = [set() for _ in range(7)]
    for key, fit in assignments.items():
        aligned = np.asarray(usable[key]["directions"]) @ np.asarray(fit["rotation"])
        for j, k in zip(fit["slots"], fit["observation_indices"]):
            residuals[j].append(aligned[k] - template[j])
            clusters[j].add(str(key).split("_")[0])
            incoming = usable[key].get("incoming_direction")
            if incoming is not None:
                n = unit(np.asarray(usable[key]["directions"])[k] - incoming)
                normal_samples[j].append(n @ np.asarray(fit["rotation"]))
    stats = []
    for j, errors in enumerate(residuals):
        count = len(errors)
        if count > 1:
            cov = np.cov(np.asarray(errors).T)
            # Small PSD block, no unidentifiable full spatial covariance.
            cov = .8 * cov + .2 * np.diag(np.diag(cov))
            predictive = float(np.sqrt(max(np.trace(cov), 0.)))
            # Assembly errors on the same rod are not independent. This is a
            # conservative effective-cluster scale, explicitly not a precise
            # small-sample confidence interval with only four rods available.
            mean = predictive / np.sqrt(max(1, len(clusters[j])))
        else:
            cov, predictive, mean = None, None, None
        stats.append(dict(canonical_slot=j + 1, independent_assemblies=count,
                          rod_clusters=len(clusters[j]),
                          covariance_direction=cov.tolist() if cov is not None else None,
                          observed_predictive_sigma_rad=predictive, mean_sigma_rad=mean,
                          status="data_supported" if count > 1 else "prior_dominated"))
    return dict(version=1, method="robust_partial_spherical_template_SO3_multistart",
                directions_local=template.tolist(), statistics=stats,
                independent_assemblies=len(usable), complete_assemblies=sum(len(g["directions"]) == 7 for g in usable.values()),
                assignments=assignments, convergence=history, starts=len(starts), cost=float(cost),
                initial_neighbour_gap_rad=step,
                gauge="Mean direction +X; ordered fan runs -Y to +Y; fan normal +Z; centres are mechanical priors",
                pooling="separate_sides_pending_independent_cross_validation",
                uncertainty_note="Empirical combined scatter; metrology noise and manufacturing variation are not separately identified",
                mean_uncertainty_method="predictive scatter divided by sqrt(number of contributing rod clusters); conditional scale, not calibrated confidence",
                physical_normals_local=[unit(np.sum(samples, axis=0)).tolist() if samples else None for samples in normal_samples],
                normal_observation_counts=[len(samples) for samples in normal_samples],
                normal_convention="n=(d_out-d_in)/norm; traced local incident branch; physical coating side",
                centre_observability="prior_dominated", axial_roll_observability="prior_dominated")
