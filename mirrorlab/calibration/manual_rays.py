"""Finite aperture path search shared by the application and numerical audit.

Samples establish existence of paths, not beam power or spot moments.
"""
from __future__ import annotations
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from .physical import _launch_bundle, _records, _record_indexes, _trace, incoming_at_points, validate_reflections
from .trajectory_matching import _pair_metrics, dataset_comparison_report, traced_output_trajectories
from ..engine.bodies import MirrorBody
from ..state import configuration_from_dataset, digest


def solid_entry(points, direction, body):
    """Independent ray/convex-solid interval; return first positive entry."""
    p = (points - body.center_m) @ body.rotation
    d = np.asarray(direction) @ body.rotation
    a = float(d[:2] @ d[:2])
    b = p[:, :2] @ d[:2]
    c = np.sum(p[:, :2] ** 2, axis=1) - body.radius_m ** 2
    if a < 1e-25:
        good = c <= 0
        lo, hi = np.full(len(p), -np.inf), np.full(len(p), np.inf)
    else:
        disc = b*b - a*c
        good = disc >= 0
        root = np.sqrt(np.maximum(disc, 0))
        lo, hi = (-b-root)/a, (-b+root)/a
    # z-x >= 0, h-z >= 0.
    for origin, slope in ((p[:, 2]-p[:, 0], d[2]-d[0]),
                          (body.axial_height_m-p[:, 2], -d[2])):
        if abs(slope) < 1e-15:
            good &= origin >= 0
        elif slope > 0:
            lo = np.maximum(lo, -origin/slope)
        else:
            hi = np.minimum(hi, -origin/slope)
    lo = np.maximum(lo, 1e-8)
    return np.where(good & (hi >= lo), lo, np.inf)


def first_body(points, direction, bodies, exclude):
    distance = np.full(len(points), np.inf)
    ids = np.full(len(points), "", dtype=object)
    for body in bodies:
        if body.mirror_id == exclude:
            continue
        candidate = solid_entry(points, direction, body)
        better = candidate < distance
        ids[better] = body.mirror_id
        distance = np.minimum(distance, candidate)
    return distance, ids


def aperture_samples(body):
    local = [[0., 0., 0.]]
    for r in np.linspace(.1, .98, 10) * body.radius_m:
        for theta in np.linspace(0, 2*np.pi, 48, endpoint=False):
            x, y = r*np.cos(theta), r*np.sin(theta)
            local.append([x, y, x])
    return np.asarray(local) @ body.rotation.T + body.center_m


def full_sample_trace(config, samples, bodies, frame):
    """Test locally surviving off-centre samples through the full optical scene."""
    targets = {s["key"]: s["point"] for s in samples}
    field = incoming_at_points(config, targets)
    launched = [s for s in samples if field[s["key"]]["status"] == "available"]
    if not launched:
        return {"tested": len(samples), "illuminated": 0, "reached_cut": 0, "paths": []}
    rows = _records(_trace(config, _launch_bundle(config,
                    [field[s["key"]]["launch_point_m"] for s in launched])))
    branch_index, children = _record_indexes(rows)
    reports, reached, details = {}, 0, []
    by_id = {b.mirror_id: b for b in bodies}
    for root, sample in enumerate(launched):
        identifier = sample["mirror_id"]
        body = by_id[identifier]
        wanted = body.build_surfaces()[0].name
        candidates = branch_index.get((root, tuple(field[sample["key"]]["interaction_chain"])), [])
        row = min(candidates, key=lambda r: np.linalg.norm(r["end"]-sample["point"]), default=None)
        if row is None or row["surface"] != wanted:
            details.append(dict(key=sample["key"], reached=False,
                                first_surface=row["surface"] if row else None))
            continue
        reached += 1
        din = (row["end"]-row["start"])/np.linalg.norm(row["end"]-row["start"])
        expected = din-2*np.dot(din,body.normal)*body.normal
        pending = [(c, []) for c in children.get(row["id"], [])
                   if np.linalg.norm(c["end"]-c["start"]) > 1e-12 and
                   np.linalg.norm((c["end"]-c["start"])/np.linalg.norm(c["end"]-c["start"])-expected) < 1e-7]
        paths = []
        while pending:
            current, path = pending.pop()
            path = path + [dict(start_m=current["start"].tolist(), end_m=current["end"].tolist(), surface=current["surface"])]
            descendant = children.get(current["id"], [])
            if descendant:
                pending.extend((c, path) for c in descendant)
            else:
                paths.append(dict(segments=path))
        reports[sample["key"]] = dict(reflected_trajectories=paths)
        details.append(dict(key=sample["key"], mirror_id=identifier, reached=True,
                            offset_m=float(np.linalg.norm(sample["point"]-body.center_m)),
                            launch_point_m=field[sample["key"]]["launch_point_m"],
                            hit_m=row["end"].tolist()))
    paths = traced_output_trajectories(dict(mirrors=reports), frame)
    for path in paths:
        path["sample_key"] = path["mirror_id"]
        path["mirror_id"] = next(s["mirror_id"] for s in samples if s["key"] == path["sample_key"])
    return dict(tested=len(samples), illuminated=len(launched), reached_cut=reached,
                paths=paths, details=details)


def one_to_one_aperture(comparison, sample_paths):
    """Choose a ray per measurement, with capacity ONE per physical mirror."""
    observations = [r for r in comparison["trajectory_diagnostics"]["observations"]
                    if r["status"] != "excluded"]
    paths = comparison["trajectory_matching"]["trajectories"] + sample_paths
    ids = sorted({p["mirror_id"] for p in paths})
    id_index = {identifier:j for j,identifier in enumerate(ids)}
    n, m = len(observations), len(ids)
    errors = np.full((n,m), np.inf)
    best = {}
    for i, observation in enumerate(observations):
        for path in paths:
            j = id_index[path["mirror_id"]]
            metric = _pair_metrics({"id":observation["observation_id"]},
                                  np.asarray(observation["screens_m"], float), path)
            if metric["maximum_distance_m"] < errors[i,j]:
                errors[i,j] = metric["maximum_distance_m"]
                best[i,j] = path
    results = {}
    for tolerance in [.001, .002, .003, .005]:
        cost = np.full((n,m+n), float(n+1))
        cost[:,:m] = np.where(errors <= tolerance, errors/tolerance, (n+1)*(m+n+1))
        rows, columns = linear_sum_assignment(cost)
        accepted = [(i,j) for i,j in zip(rows,columns) if j<m and errors[i,j]<=tolerance]
        used = {i for i,j in accepted}
        matches = [dict(observation_id=observations[i]["observation_id"],
                        mirror_id=ids[j], error_mm=float(errors[i,j]*1000),
                        sample_key=best[i,j].get("sample_key")) for i,j in accepted]
        assert len({p["mirror_id"] for p in matches}) == len(matches)
        results[str(tolerance)] = dict(count=len(matches), matches=matches,
            unmatched=[dict(observation_id=o["observation_id"],
                            best_error_mm=float(errors[i].min()*1000) if m else None)
                       for i,o in enumerate(observations) if i not in used])
    return results


def input_fingerprint(dataset):
    return digest(dict(bodies=[MirrorBody.from_dict(m).to_dict() for m in dataset['mirrors']],
                       optics=dataset['optics_snapshot'], frame=dataset['display_frame'],
                       observations=dataset['observations'],
                       directions=[m.get('target_outgoing_native', m.get('outgoing_direction'))
                                   for m in dataset['mirrors']]))


def current_analysis(dataset):
    analysis = dataset.get('manual_rays')
    return analysis if analysis and analysis.get('input_sha256') == input_fingerprint(dataset) else None


def manual_geometry_on_base(base, manual):
    """Place supplied manual bodies in the base coordinate system, retaining its optics."""
    result = copy.deepcopy(base)
    result.update(copy.deepcopy(manual))
    if not base.get('display_frame') or base.get('optics_snapshot') is None:
        raise ValueError('Для переноса ручных зеркал нужны общая оптика и привязка исходной конфигурации к геодезии.')
    source_frame, target_frame = manual['display_frame'], base['display_frame']
    qs, qt = np.asarray(source_frame['rotation']), np.asarray(target_frame['rotation'])
    ts, tt = np.asarray(source_frame['translation_m']), np.asarray(target_frame['translation_m'])
    transform = qt.T @ qs
    # Reflection of local y leaves the cylinder and its x=z cut invariant.
    parity = np.diag([1., -1. if np.linalg.det(transform) < 0 else 1., 1.])
    raw = manual['manual_references']['payload']
    sources = {m.get('source_manual_id', m.get('source_id')): m for m in raw['mirrors']}
    for mirror in result['mirrors']:
        source = sources[mirror['source_manual_id']]
        supplied = source.get('recommended_optimization_seed', {}).get('body_native', mirror)
        body = MirrorBody.from_dict(supplied)
        body.center_m = qt.T @ (qs @ body.center_m + ts - tt)
        body.rotation = transform @ body.rotation @ parity
        mirror.update(body.to_dict())
        mirror['target_outgoing_native'] = (transform @ np.asarray(mirror['target_outgoing_native'])).tolist()
        mirror['trajectory'] = None
    for key in ('optics_snapshot', 'optics_sha256', 'display_frame'):
        result[key] = copy.deepcopy(base[key])
    result['original_optics_snapshot'] = copy.deepcopy(base['optics_snapshot'])
    result['optics_origin'] = 'preserved_base_configuration'
    result['manual_ray_seed'] = 'manual_bodies_in_base_frame'
    result['manual_geometry_base'] = dict(optics_sha256=base['optics_sha256'],
                                        display_frame=copy.deepcopy(base['display_frame']),
                                        status=base['status'])
    for key in ('manual_rays', 'manual_refinement', 'physical_validation'):
        result.pop(key, None)
    return result


def add_manual_rays(dataset, progress=None):
    """Recompute audit paths on a copy; never fit or accept mirror geometry."""
    if not dataset.get('manual_references'):
        raise ValueError('Для добавления лучей нужна ручная реконструкция.')
    from ..manual_schema import manual_starting_geometry
    result = manual_starting_geometry(dataset)
    def notify(stage, fraction):
        if progress:
            progress(dict(stage=stage, fraction=fraction))
    notify('Подготовка ручной геометрии', .02)
    bodies = [MirrorBody.from_dict(m) for m in result['mirrors']]
    config = configuration_from_dataset(result)
    notify('Проверка центральных лучей', .1)
    reflections = validate_reflections(config, bodies)
    result['physical_validation'] = reflections
    result.setdefault('diagnostics', {})['reflections'] = reflections
    for m in result['mirrors']:
        m['outgoing_direction'] = m['target_outgoing_native']
        check = reflections['mirrors'][m['mirror_id']]
        m['optical_validation'] = check
        m['trajectory'] = ({**check['reflected_child'], 'paths': check['reflected_trajectories']}
                           if check.get('actual_reflected_child') else None)
    comparison = dataset_comparison_report(result)['diagnostics']
    samples = []
    for i, body in enumerate(bodies):
        check = reflections['mirrors'][body.mirror_id]
        if not check.get('actual_reflected_child'):
            continue
        direction = np.asarray(check['outgoing_direction'])
        distance, _ = first_body(body.center_m[None,:], direction, bodies, body.mirror_id)
        if not np.isfinite(distance[0]):
            continue
        points = aperture_samples(body)
        outward, _ = first_body(points, direction, bodies, body.mirror_id)
        backward, _ = first_body(points, -np.asarray(check['incoming']['direction']), bodies, body.mirror_id)
        upstream = np.linalg.norm(body.center_m - np.asarray(check['incoming']['field_origin_m']))
        clear = np.where((outward > 3.) & (backward > upstream))[0]
        nearest = sorted(clear, key=lambda j:np.linalg.norm(points[j]-body.center_m))[:3]
        samples.extend(dict(key=f'{body.mirror_id}_sample{j}', mirror_id=body.mirror_id, point=points[k])
                       for j,k in enumerate(nearest))
        notify('Поиск лучей на площадках зеркал', .3 + .35*i/len(bodies))
    notify('Полная трассировка дополнительных лучей', .7)
    traced = full_sample_trace(config, samples, bodies, result['display_frame'])
    matches = one_to_one_aperture(comparison, traced['paths'])['0.003']
    result['manual_rays'] = dict(version=1, input_sha256=input_fingerprint(result),
        coordinate_frame='survey', source='finite_aperture_full_scene_trace',
        baseline_count=comparison['trajectory_matching']['matched_count'],
        observation_count=comparison['trajectory_matching']['observation_count'],
        matched_count=matches['count'], matches=matches['matches'], unmatched=matches['unmatched'],
        distance_tolerance_m=.003, full_trace=traced,
        sampling=dict(points_per_cut=481, selected_per_blocked_mirror=3, local_exit_length_m=3.),
        power_fraction=None, spot_shape=None,
        interpretation='sampled_path_existence_not_power_or_spot_validation')
    notify('Лучи добавлены', 1.)
    return result


def scene_traces(dataset):
    analysis = current_analysis(dataset)
    if not analysis:
        return []
    traces = []
    for path in analysis['full_trace']['paths']:
        points = [path['point_m']] + [s['end_m'] for s in path['segments']]
        traces.append(dict(type='scatter3d', mode='lines',
            x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points],
            line=dict(color='#00b6b0', width=4), name=f"Дополнительный луч · {path['mirror_id']}",
            legendgroup='manual_aperture_rays', showlegend=False,
            meta=dict(layer='manual_rays', kind='manual_aperture_ray', mirror_id=path['mirror_id'],
                      trajectory_id=path['id']),
            hovertemplate=f"Зеркало {path['mirror_id']}<br>Луч через площадку<extra></extra>"))
    return traces


