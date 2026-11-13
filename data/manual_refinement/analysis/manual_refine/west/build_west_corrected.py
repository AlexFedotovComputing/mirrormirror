"""Export WEST manual reconstruction in the current project's survey frame.

The original plane and the operational seed plane are kept separate: the
former was reconstructed with the opposite incident-direction convention.
Nothing in this script changes primary observations or fits the 112 centres.

Run in this workspace:
  python3 build_west_corrected.py
Portable use:
  python3 build_west_corrected.py --source SOURCE.txt --reference REFERENCE.json \
      --repo MIRRORMIRROR --out OUTPUT_DIRECTORY

REFERENCE is the previously evaluated historical registration artifact; its
matrix and the 83 assignments are bundled into the produced JSON verbatim.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[3]
COMMIT = '4ab80dfba048e11cf5a3b0bef7dc6832b298540b'


def load_points(path):
    rows = {}
    for row in csv.reader(path.read_text(encoding='utf-8-sig').splitlines(), delimiter='\t'):
        if not row or not any(v.strip() for v in row):
            continue
        key = tuple(map(int, row[0].split('.')))
        points = np.array([ast.literal_eval(v) for v in row[1:]], float)
        if key in rows or points.shape != (4, 3) or not np.isfinite(points).all():
            raise ValueError(f'Invalid input row: {row[0]}')
        rows[key] = points
    assert set(rows) == set(itertools.product(range(1, 5), range(1, 5), range(1, 8)))
    return rows


def unit(value):
    value = np.asarray(value, float)
    return value / np.linalg.norm(value)


def angle_unsigned(a, b):
    a, b = unit(a), unit(b)
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), abs(a @ b)))


def fixed_ellipse_original_plane(points):
    """Fix opposite diameters to 1 and sqrt(2) mm, preserving axes and plane."""
    center = points.mean(axis=0)
    d0, d1 = points[2] - points[0], points[3] - points[1]
    major_first = np.linalg.norm(d0) > np.linalg.norm(d1)
    desired = [.001 * np.sqrt(2), .001] if major_first else [.001, .001 * np.sqrt(2)]
    axes = [unit(d0), unit(d1)]
    # Restore exactly perpendicular axes inside the fitted face plane.
    _, _, vt = np.linalg.svd(points - center)
    normal = vt[-1]
    axes[0] = unit(axes[0] - (axes[0] @ normal) * normal)
    axes[1] = unit(np.cross(normal, axes[0])) * np.sign(np.cross(normal, axes[0]) @ axes[1])
    return np.array([center - axes[0] * desired[0] / 2,
                     center - axes[1] * desired[1] / 2,
                     center + axes[0] * desired[0] / 2,
                     center + axes[1] * desired[1] / 2])


def json_write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', '--input', type=Path, default=WORKSPACE / 'upload/Координаты_микрозеркал_из+к-.txt')
    local_reference = Path(__file__).resolve().parent / 'west_historical_registration_reference.json'
    parser.add_argument('--reference', type=Path, default=local_reference if local_reference.exists() else WORKSPACE / 'outputs/manual_reconstruction_review_2026-11-12/review_data/west_reference_registration_hypotheses.json')
    parser.add_argument('--repo', type=Path, default=WORKSPACE / 'analysis/manual_review/repo')
    parser.add_argument('--out', type=Path, default=WORKSPACE / 'analysis/manual_refine/west')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo.resolve()))
    from mirrorlab.configurations import load_configuration
    from mirrorlab.coordinates import survey_frames
    from mirrorlab.calibration.physical import incoming_at_points, reflecting_normal, rotation_from_normal
    from mirrorlab.engine.bodies import MirrorBody

    points = load_points(args.source)
    reference = json.loads(args.reference.read_text(encoding='utf8'))
    reference_by_id = {r['manual_id']: r for r in reference['mirrors']}
    historical = reference['historical_registration']
    Qh, th = np.array(historical['rotation']), np.array(historical['translation_m'])
    frame = survey_frames()['east_to_west']
    Qc, tc = np.array(frame.rotation), np.array(frame.translation)
    mirrors = []
    input_points_native = {}
    for (rod, window, mirror), values in sorted(points.items()):
        sid = f'{rod}_{window}_{mirror}'
        rec = reference_by_id[sid]
        shift = np.array(rec['removed_sector_translation_m'])
        survey_points = (values - shift) @ Qh.T + th
        survey_center = survey_points.mean(axis=0)
        native_points = (survey_points - tc) @ Qc
        native_center = native_points.mean(axis=0)
        nid = f'{rod}_{5-window}_{mirror}'
        input_points_native[nid] = native_center
        n_source = unit(rec['historical_plane_normal_survey'])
        d_target = unit(rec['historical_output_line_direction_survey'])
        fixed = fixed_ellipse_original_plane(survey_points)
        mirrors.append({
            'source_manual_id': sid,
            'source_manual_label': f'{rod}.{window}.{mirror}',
            'survey_bundle_id': f'{rod}_{window}',
            'native_stable_seed_id': nid,
            'native_bundle_id': f'{rod}_{5-window}',
            'native_seed_index_is_physical_number': False,
            'matched_primary_survey_id': rec['matched_primary_survey_id'],
            'identity_mapping_status': 'geometrically_assigned_to_full_survey' if rec['matched_primary_survey_id'] else 'not_measured_or_unassigned',
            'source_kind': 'manual_reconstructed_surface',
            'used_as_primary_measurement': False,
            'quadrilateral_point_order_is_source_order': True,
            'infer_coated_side_from_point_winding': False,
            'source_face_points_legacy_m': values.tolist(),
            'removed_sector_translation_legacy_m': shift.tolist(),
            'survey_center_m': survey_center.tolist(),
            'survey_edge_points_as_transformed_m': survey_points.tolist(),
            'survey_original_plane_normal': n_source.tolist(),
            'survey_repaired_45deg_ellipse_in_original_plane_m': fixed.tolist(),
            'ellipse_repair_max_tip_displacement_m': float(np.max(np.linalg.norm(fixed-survey_points,axis=1))),
            'survey_target_outgoing': d_target.tolist(),
            'target_direction_provenance': 'historical_manual_plane_reflection_with_traced_old_native_incident_branch; historical_registration_applied',
            'native_center_m': native_center.tolist(),
            'native_original_plane_normal': (Qc.T @ n_source).tolist(),
            'native_edge_points_as_transformed_m': native_points.tolist(),
            'native_target_outgoing': (Qc.T @ d_target).tolist(),
            'historical_alignment_rms_mm': rec['training_alignment_rms_mm'],
            'historical_direction_residual_mrad': rec['primary_direction_difference_mrad'],
        })

    print('Tracing current ideal common optics at all corrected centres', flush=True)
    config = load_configuration('east_to_west', 'ideal')
    field = incoming_at_points(config, input_points_native)
    J = np.diag([1., -1., 1.]) if np.linalg.det(Qc) < 0 else np.eye(3)
    physical_bodies = []
    local_tips = np.array([[.0005, 0., .0005], [0., .0005, 0.], [-.0005, 0., -.0005], [0., -.0005, 0.]])
    reflection_residuals = []
    for rec in mirrors:
        f = field[rec['native_stable_seed_id']]
        rec['current_common_optics_incidence'] = f
        if 'direction' not in f:
            rec['operational_optical_seed'] = None
            continue
        incoming = unit(f['direction'])
        outgoing = np.array(rec['native_target_outgoing'])
        normal = reflecting_normal(incoming, outgoing)
        rotation = rotation_from_normal(normal, axis_hint=incoming)
        body = MirrorBody(rec['native_center_m'], rotation, rec['native_stable_seed_id'])
        physical_bodies.append(body)
        normal_world = Qc @ normal
        body_world_rotation = Qc @ rotation @ J
        body_world = MirrorBody(rec['survey_center_m'], body_world_rotation, rec['source_manual_id'])
        expected = incoming - 2*(incoming@normal)*normal
        residual = float(np.linalg.norm(expected-outgoing))
        reflection_residuals.append(residual)
        rec['operational_optical_seed'] = {
            'status': 'candidate_requires_body_and_holder_validation' if f['status']=='available' else 'conditional_direction_only_not_illuminated',
            'normal_semantics': 'corrected_incident_branch_convention; independent rederivation, not small pose fit',
            'survey_incident_direction': (Qc@incoming).tolist(),
            'survey_reflecting_normal': normal_world.tolist(),
            'native_reflecting_normal': normal.tolist(),
            'plane_change_relative_to_transformed_original_deg': float(np.degrees(angle_unsigned(normal_world, rec['survey_original_plane_normal']))),
            'survey_correct_45deg_face_points_m': (local_tips @ body_world_rotation.T + np.array(rec['survey_center_m'])).tolist(),
            'native_correct_45deg_face_points_m': (local_tips @ rotation.T + np.array(rec['native_center_m'])).tolist(),
            'body_native': body.to_dict(),
            'body_survey': body_world.to_dict(),
            'incident_dot_coated_outward_normal': float(incoming@normal),
            'reflection_vector_error': residual,
            'body_roll_prior': 'axis closest to incident direction; roll not observed',
            'body_fit_or_collision_optimization_performed': False,
        }
    counts = {status:sum(f['status']==status for f in field.values()) for status in sorted({f['status'] for f in field.values()})}
    roundtrip = max(float(np.max(np.linalg.norm((np.array(r['native_edge_points_as_transformed_m'])@Qc.T+tc)-r['survey_edge_points_as_transformed_m'],axis=1))) for r in mirrors)
    center_errors = [np.linalg.norm(np.array(r['survey_center_m'])-reference_by_id[r['source_manual_id']]['center_historical_registered_survey_m']) for r in mirrors]
    report = {
        'count': len(mirrors), 'matched_full_primary_count': sum(r['matched_primary_survey_id'] is not None for r in mirrors),
        'primary_ids_unique': len({r['matched_primary_survey_id'] for r in mirrors if r['matched_primary_survey_id']})==83,
        'source_id_and_native_seed_id_unique': len({r['source_manual_id'] for r in mirrors})==len({r['native_stable_seed_id'] for r in mirrors})==112,
        'max_survey_native_roundtrip_error_m': roundtrip,
        'max_center_difference_from_historical_registration_m': float(max(center_errors)),
        'survey_z_range_m': [min(r['survey_center_m'][2] for r in mirrors),max(r['survey_center_m'][2] for r in mirrors)],
        'incidence_status_counts': counts,
        'max_algebraic_reflection_vector_error': max(reflection_residuals) if reflection_residuals else None,
        'physical_body_count':len(physical_bodies),
        'original_historical_fit_metrics': {k:historical[k] for k in ['n','rms_mm','median_mm','max_mm','angle_median_mrad']},
        'fit_metrics_are_independent_uncertainty': False,
    }
    package = {
        'format':'mirrorlab.manual-reconstruction-input', 'schema_version':1,
        'application_native_physical_configuration': False,
        'purpose':'manual_reference_and_optimization_seed',
        'canonical_coordinate_frame':'survey', 'units':'m', 'injection_side':'WEST','route':'east_to_west',
        'center_definition':'reflecting_cut_ellipse_center',
        'project_commit':COMMIT,
        'source_file':args.source.name,'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'source_epoch':None,
        'coordinate_policy':{
            'world_is_original_geodesy_xyz':True,
            'numerical_z_sign_is_not_derived_from_EAST_WEST_label':True,
            'manual_to_survey':'p_survey = Q_historical @ (p_supplied - rod_translation) + t_historical',
            'survey_to_current_native':'p_native = Q_current.T @ (p_survey - t_current)',
            'historical_rotation':Qh.tolist(),'historical_translation_m':th.tolist(),
            'current_native_to_survey_rotation':Qc.tolist(),'current_native_to_survey_translation_m':tc.tolist(),
            'current_frame_determinant':float(np.linalg.det(Qc)),
            'body_rotation_conversion':'R_survey = Q_current @ R_native @ diag(1,-1,1) for det(Q_current)<0; local y reflection leaves this body invariant',
            'source_manual_window_to_native_window':{str(w):5-w for w in range(1,5)},
            'snapshot_bound_native_coordinates':True,
        },
        'corrections':[
            'Remove four exact per-rod legacy translations that degraded registration to current survey.',
            'Apply selected historical rigid registration to all face points and manual output directions.',
            'Preserve source manual IDs and all original face points; store83 geometric survey correspondences.',
            'Offer exact45 ellipse in original plane separately; it is not the operational physical plane.',
            'Recompute operational coated normal from current traced incident branch and desired directed output.',
            'Rebuild proper SO(3) cut-cylinder pose; do not apply an improper frame matrix as a physical rotation.'
        ],
        'limitations':[
            'Historical source planes used opposite incident-direction convention; do not use them as a strong orientation prior for operational optics.',
            'Manual coordinates and directions are derived data and must not enter the primary measurement likelihood.',
            'No source epoch or independently measured holder envelope supplied.',
            'Centre reconstruction is retained; no physical position optimisation has been performed by this exporter.',
            'Operational seed requires collision, full-path, rod/holder and waist checks; unavailable illumination is a blocking condition.',
            'Native coordinates and directions apply only to the embedded current frame; prefer survey as durable storage.'
        ],
        'validation':report,
        'mirrors':mirrors,
    }
    json_write(args.out/'west_manual_corrected_survey.json', package)
    json_write(args.out/'west_current_optical_seed_bodies_native.json', {'format':'mirrorlab.body-seed-list','schema_version':1,'route':'east_to_west','units':'m','native_to_survey':package['coordinate_policy'],'bodies':[b.to_dict() for b in physical_bodies],'activation_allowed':False})
    json_write(args.out/'west_coordinate_validation.json', report)

    for frame_name, ckey, nkey in [('survey','survey_center_m','survey_reflecting_normal'),('native','native_center_m','native_reflecting_normal')]:
        filename = args.out/f'west_manual_centers_{frame_name}_m.csv'
        with filename.open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.writer(stream)
            writer.writerow(['source_manual_id','matched_survey_id','native_stable_seed_id','x_m','y_m','z_m','operational_nx','operational_ny','operational_nz','incidence_status','normal_is_measured'])
            for r in mirrors:
                s=r['operational_optical_seed']
                writer.writerow([r['source_manual_id'],r['matched_primary_survey_id'] or '',r['native_stable_seed_id'],*r[ckey],*(s[nkey] if s else ['','','']),r['current_common_optics_incidence']['status'],False])
    with (args.out/'west_manual_face_points_survey_m.txt').open('w',encoding='utf8') as stream:
        stream.write('# Frame: current project survey/world; units: metres. Centre geometry corrected; source planes preserved.\n')
        stream.write('# Source manual ID\tP1\tP2\tP3\tP4; this is an agent input, not the existing legacy-import format.\n')
        for r in mirrors:
            stream.write(r['source_manual_label']+'\t'+'\t'.join('('+', '.join(format(v,'.14g') for v in p)+')' for p in r['survey_edge_points_as_transformed_m'])+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__=='__main__':
    main()
