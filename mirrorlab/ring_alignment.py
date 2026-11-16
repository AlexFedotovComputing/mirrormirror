"""Correct EAST's upstream ring while preserving its registered rod axes.

The drawing's vertical is native Y: MS1/MS4 form its top pair. Reflection
exchanges the four splitter outputs. MS1..4 reconnect those outputs to the
existing rods. When fitted cuts need it, MS/MC normals recenter the footprint
while retaining the exact outgoing direction. No MC centre, microassembly,
detector, survey coordinate or fitted body is moved.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .engine import PlaneSurface


RING_ALIGNMENT_REVISION = "east_ring_vertical_reflection_v1"
_RING_ELEMENTS = frozenset(("MP1", "MP2", "MP3", "One of many mirrors",
                            "ML1", "ML2", "ML3", "MR1", "MR2",
                            "PL1", "PL2", "PR1", "PR2", "PR3"))
_TOLERANCE_M = 1e-8
_DIRECTION_TOLERANCE = 1e-9


def _transform_plane(surface, matrix, pivot=None):
    if not isinstance(surface, PlaneSurface):
        raise ValueError("Ring correction requires planar upstream optics")
    pivot = np.zeros(3) if pivot is None else np.asarray(pivot, float)
    surface.center = pivot + matrix @ (np.asarray(surface.center) - pivot)
    surface.normal = matrix @ surface.normal
    surface.in_plane_reference = matrix @ surface._u
    # The basis rebuilt by PlaneSurface is right-handed. An improper map
    # reverses its local v axis, so triangle coordinates must reverse too.
    if surface.triangle_vertices is not None:
        surface.triangle_vertices = np.asarray(surface.triangle_vertices) * [1., np.linalg.det(matrix)]
    surface.__post_init__()


def _chief_rows(config):
    from .calibration.physical import _common_config, _launch_bundle, _records, _trace

    rows = _records(_trace(_common_config(config), _launch_bundle(config, [config.source.waist_position])))
    result = {}
    required = ("MP1", "MP2", "MP3", *(f"MS{i}" for i in range(1, 5)), *(f"MC{i}" for i in range(1, 5)))
    for name in required:
        matches = [row for row in rows.values() if row["surface"] == name]
        if len(matches) != 1:
            raise ValueError(f"Ring correction requires one verified chief path through {name}")
        result[name] = matches[0]
    return result


def _preserve_illumination(before, after):
    """Recenter a changed footprint using MS/MC, keeping every cut direction.

    Each two-plane relay is an isometry on the incident parallel ray field.
    Solve its footprint translation against the existing illuminated cuts;
    never enlarge the source or move fitted bodies to manufacture coverage.
    MC centres stay fixed. Their normals may change to keep the outgoing
    direction exactly fixed while the chief hit moves within the aperture.
    """
    from scipy.optimize import brentq, minimize
    from .calibration.physical import _field_models, incoming_at_mirrors, reflecting_normal, unit

    previous = incoming_at_mirrors(before)
    current = incoming_at_mirrors(after)
    affected = {int(k.split('_')[0]) for k, item in previous.items()
                if item['status'] == 'available' and current.get(k, {}).get('status') != 'available'}
    if not affected:
        return []
    models, _ = _field_models(after)
    chiefs = _chief_rows(after)
    surfaces = {s.name: s for s in after.scene.build_surfaces()}
    radius = after.source.waist_radius * after.source.cutoff_ratio
    reports = []

    def reflection(normal, center):
        matrix = np.eye(3)-2*np.outer(normal, normal)
        return matrix, 2*np.dot(normal, center)*normal

    for rod in sorted(affected):
        model = models[str(rod)][0]
        direction = model['direction']
        basis_u = unit(np.cross(direction, np.eye(3)[np.argmin(abs(direction))]))
        basis = np.column_stack((basis_u, np.cross(direction, basis_u)))
        identifiers = [k for k, item in previous.items()
                       if k.startswith(f'{rod}_') and item['status'] == 'available']
        points = np.array([previous[k]['point_m'] for k in identifiers])
        ms, mc = surfaces[f'MS{rod}'], surfaces[f'MC{rod}']
        old_ms_normal, old_mc_normal = ms.normal.copy(), mc.normal.copy()
        ms_hit, mc_hit = chiefs[ms.name]['end'], chiefs[mc.name]['end']
        incoming = unit(chiefs[ms.name]['end']-chiefs[ms.name]['start'])
        # Undo the previous relay on affine output lines to obtain the field
        # just before MS. Reflection acts on points as well as directions.
        mapping = model['map'].copy()
        for surface in (mc, ms):
            matrix, offset = reflection(surface.normal, surface.center)
            mapping = mapping @ matrix.T
            mapping[2] += offset

        def relay(shift_mm):
            base_hit = mc_hit + basis @ (np.asarray(shift_mm)*.001)

            def plane_constraint(t):
                hit = base_hit+t*direction
                return np.dot(unit(hit-ms_hit)-direction, hit-mc.center)

            axial = brentq(plane_constraint, -.05, .05, xtol=1e-14)
            hit = base_hit+axial*direction
            between = unit(hit-ms_hit)
            ms_normal = reflecting_normal(incoming, between)
            mc_normal = reflecting_normal(between, direction)
            transformed = mapping.copy()
            for normal, center in ((ms_normal, ms_hit), (mc_normal, mc.center)):
                matrix, offset = reflection(normal, center)
                transformed = transformed @ matrix.T
                transformed[2] += offset
            launches = np.linalg.solve((transformed[:2] @ basis).T,
                                       ((points-transformed[2]) @ basis).T).T
            return launches, hit, ms_normal, mc_normal

        # Millimetre-scaled variables keep finite differences well conditioned.
        def constraints(shift):
            launches = relay(shift)[0]
            return 1-np.sum(launches**2, axis=1)/(radius*.999)**2

        fit = minimize(lambda shift: float(np.dot(shift, shift)), np.zeros(2),
                       method='SLSQP', bounds=[(-5., 5.), (-5., 5.)],
                       constraints={'type': 'ineq', 'fun': constraints},
                       options={'ftol': 1e-12, 'maxiter': 200})
        # SLSQP can stop at a feasible boundary with a line-search warning.
        # Feasibility and the independent full trace below decide acceptance.
        if not np.isfinite(fit.x).all() or np.min(constraints(fit.x)) < -1e-8:
            raise ValueError(f'Cannot retain illumination in rod {rod} with the existing source and relay: '
                             f'{fit.message}; residual={np.min(constraints(fit.x)):.3g}; shift_mm={fit.x}')
        launches, hit, ms.normal, mc.normal = relay(fit.x)
        ms.center = ms_hit.copy()
        ms.in_plane_reference = np.cross([0., 0., 1.], ms.normal)
        mc.in_plane_reference = mc._u.copy()
        ms.__post_init__()
        mc.__post_init__()
        reports.append(dict(rod=rod, preserved_mirror_ids=identifiers,
                            chief_hit_before_m=mc_hit.tolist(), chief_hit_after_m=hit.tolist(),
                            transverse_shift_m=float(np.linalg.norm(fit.x)*.001),
                            ms_normal_before=old_ms_normal.tolist(), ms_normal_after=ms.normal.tolist(),
                            mc_normal_before=old_mc_normal.tolist(), mc_normal_after=mc.normal.tolist(),
                            max_launch_radius_m=float(np.max(np.linalg.norm(launches, axis=1))),
                            source_radius_m=radius))
    return reports


def _compare_fields(before, after, recentered=False):
    from .calibration.physical import _field_models, incoming_at_mirrors

    previous, _ = _field_models(before)
    current, _ = _field_models(after)
    fields = []
    for rod in range(1, 5):
        old, new = previous.get(str(rod), []), current.get(str(rod), [])
        if len(old) != 1 or len(new) != 1:
            raise ValueError(f"Ring correction requires one affine incident field in rod {rod}")
        old, new = old[0], new[0]
        projection = np.eye(3) - np.outer(old["direction"], old["direction"])
        offset = float(np.linalg.norm(projection @ (old["map"][2] - new["map"][2])))
        direction_error = float(np.linalg.norm(old["direction"] - new["direction"]))
        if (offset > _TOLERANCE_M and not recentered) or direction_error > _DIRECTION_TOLERANCE:
            raise ValueError(f"Ring correction changed the registered incident axis in rod {rod}")
        covariance = [projection @ (model["map"][:2].T @ model["map"][:2]) @ projection
                      for model in (old, new)]
        fields.append(dict(rod=rod, axis_offset_m=offset, direction_error=direction_error,
                           transverse_metric_before=np.linalg.eigvalsh(covariance[0])[-2:].tolist(),
                           transverse_metric_after=np.linalg.eigvalsh(covariance[1])[-2:].tolist()))
    old_incident, new_incident = incoming_at_mirrors(before), incoming_at_mirrors(after)
    lost = [key for key, item in old_incident.items()
            if item["status"] == "available" and new_incident.get(key, {}).get("status") != "available"]
    if lost:
        raise ValueError("Ring correction lost illumination at existing mirror centres: " + ", ".join(lost))
    errors = [float(np.linalg.norm(np.asarray(item["direction"]) - new_incident[key]["direction"]))
              for key, item in old_incident.items()
              if "direction" in item and "direction" in new_incident.get(key, {})]
    if errors and max(errors) > _DIRECTION_TOLERANCE:
        raise ValueError("Ring correction changed incident directions at fitted mirrors")
    return dict(rod_fields=fields, mirror_count=len(old_incident),
                illuminated_mirrors_before=sum(v["status"] == "available" for v in old_incident.values()),
                illuminated_mirrors_after=sum(v["status"] == "available" for v in new_incident.values()),
                max_mirror_direction_error=max(errors, default=0.))


def apply_ring_alignment(config):
    """Apply the correction atomically in place and return its numerical report.

Already corrected scenes are unchanged. Other routes are unchanged. Failed
geometry or illumination checks leave the supplied configuration untouched.
The circular source is retained, but exchanged refractive branches can change
the transverse intensity profile; that limitation is recorded explicitly.
"""
    if config.direction != "west_to_east":
        return None
    existing = config.metadata.get("ring_alignment")
    if existing is not None:
        if existing.get("revision") != RING_ALIGNMENT_REVISION:
            raise ValueError("Unsupported ring alignment revision")
        return copy.deepcopy(existing)

    from .calibration.physical import reflecting_normal, unit

    names = {element.name for element in config.scene.elements}
    required = _RING_ELEMENTS | {f"MS{i}" for i in range(1, 5)} | {f"MC{i}" for i in range(1, 5)}
    if not required <= names:
        raise ValueError("Ring correction is missing upstream elements: " + ", ".join(sorted(required - names)))
    before = _chief_rows(config)
    candidate = copy.copy(config)
    candidate.scene = copy.deepcopy(config.scene)
    candidate.source = copy.deepcopy(config.source)
    candidate.metadata = copy.deepcopy(config.metadata)
    reflection = np.diag([-1., 1., 1.])
    for element in candidate.scene.elements:
        if element.name in _RING_ELEMENTS:
            for surface in element.surfaces:
                _transform_plane(surface, reflection)
    candidate.source.waist_position = reflection @ np.asarray(config.source.waist_position)
    candidate.source.axis = reflection @ np.asarray(config.source.axis)
    candidate.source.polarization_reference = reflection @ np.asarray(config.source.polarization_reference)
    surfaces = {surface.name: surface for surface in candidate.scene.build_surfaces()}

    # A separate rigid rotation of the two-mirror periscope keeps its exit
    # chief line fixed. Rotating about the mechanical centre would miss MP3
    # because the archived source has small, intentional alignment offsets.
    pivot = reflection @ before["MP2"]["end"]
    down = (surfaces["MP1"].center - surfaces["MP2"].center)[:2]
    if np.linalg.norm(down) < _TOLERANCE_M or np.linalg.norm(np.asarray(candidate.source.axis)[:2]) > _DIRECTION_TOLERANCE:
        raise ValueError("Ring correction requires a nondegenerate axial periscope")
    angle = -math.pi / 2 - math.atan2(down[1], down[0])
    angle = math.atan2(math.sin(angle), math.cos(angle))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    for name in ("MP1", "MP2"):
        _transform_plane(surfaces[name], rotation, pivot)
    candidate.source.waist_position = pivot + rotation @ (candidate.source.waist_position - pivot)
    candidate.source.axis = rotation @ candidate.source.axis
    candidate.source.polarization_reference = rotation @ candidate.source.polarization_reference

    steering = []
    for rod in range(1, 5):
        partner = 5 - rod
        reflected_input = before[f"MS{partner}"]
        fixed_output = before[f"MC{rod}"]
        point = reflection @ reflected_input["end"]
        incoming = reflection @ unit(reflected_input["end"] - reflected_input["start"])
        target = fixed_output["start"]
        outgoing = unit(fixed_output["end"] - fixed_output["start"])
        parameters, _, rank, _ = np.linalg.lstsq(np.column_stack([incoming, -outgoing]), target - point, rcond=None)
        intersection = point + parameters[0] * incoming
        miss = float(np.linalg.norm(intersection - target - parameters[1] * outgoing))
        if rank != 2 or miss > _TOLERANCE_M:
            raise ValueError(f"Reflected splitter path cannot meet the unchanged MC{rod} axis")
        surface = surfaces[f"MS{rod}"]
        if not isinstance(surface, PlaneSurface):
            raise ValueError("Ring steering requires planar MS mirrors")
        old_center, old_normal = surface.center.copy(), surface.normal.copy()
        surface.center = intersection
        surface.normal = reflecting_normal(incoming, outgoing)
        # Preserve the original aperture dimensions: wide in XY, short in Z.
        surface.in_plane_reference = np.cross([0., 0., 1.], surface.normal)
        surface.__post_init__()
        steering.append(dict(mirror=surface.name, reflected_branch=partner,
                             center_before_m=old_center.tolist(), center_after_m=intersection.tolist(),
                             shift_m=float(np.linalg.norm(intersection - old_center)),
                             normal_before=old_normal.tolist(), normal_after=surface.normal.tolist(),
                             line_intersection_miss_m=miss))

    after = _chief_rows(candidate)
    chief_checks = []
    for rod in range(1, 5):
        old, new = before[f"MC{rod}"], after[f"MC{rod}"]
        miss = float(np.linalg.norm(old["end"] - new["end"]))
        error = float(np.linalg.norm(unit(old["end"] - old["start"]) - unit(new["end"] - new["start"])))
        if miss > _TOLERANCE_M or error > _DIRECTION_TOLERANCE:
            raise ValueError(f"Ring correction failed chief-ray preservation at MC{rod}")
        chief_checks.append(dict(rod=rod, hit_shift_m=miss, incoming_direction_error=error))
    recentering = _preserve_illumination(config, candidate)
    field_report = _compare_fields(config, candidate, recentered=bool(recentering))
    report = dict(revision=RING_ALIGNMENT_REVISION,
                  reflection_matrix=reflection.tolist(), reflection_center_m=[0., 0., 0.],
                  drawing_vertical="native +Y; MS1/MS4 are the upper pair",
                  periscope_rotation_rad=angle, periscope_pivot_m=pivot.tolist(),
                  splitter_output_to_rod={str(5 - i): i for i in range(1, 5)},
                  steering=steering, chief_checks_before_recentering=chief_checks,
                  footprint_recentering=recentering, **field_report,
                  preserved=["MC centres", "microassembly geometry and fitted body poses",
                             "detectors", "survey frame", "incident directions and previously illuminated cuts",
                             "source waist radius and cutoff"],
                  limitations=["Exchanged refractive branches change transverse beam profiles; per-mirror power is not invariant.",
                               "Geometric alignment is preserved; this is not a new photometric calibration."])
    candidate.metadata["ring_alignment"] = report
    config.scene, config.source, config.metadata = candidate.scene, candidate.source, candidate.metadata
    return copy.deepcopy(report)
