"""Conditional EAST mirror recovery from manual rays and original survey.

Reproducible analysis, not an application preset or an independently calibrated
measurement. The manual file is retained as a prior, never placed in the primary
measurement ledger. A source-footprint bound is an optical constraint, not a
manufacturing drawing of a rod or holder.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation


def scalar(x):
    return float(str(x).replace(',', '.'))


def unit(a):
    a = np.asarray(a, float)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-30)


def serial(x):
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    if isinstance(x, np.ndarray):
        return serial(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x


def stats(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), median=np.median(a), rms=np.sqrt(np.mean(a*a)),
                max=np.max(a), p90=np.quantile(a, .9)) if len(a) else dict(n=0)


def save(path, value):
    path.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2), encoding='utf-8')


def fan_basis(d):
    x = unit(np.asarray(d).mean(axis=0))
    y = unit(np.cross([0., 0., 1.], x))
    z = unit(np.cross(x, y))
    return np.column_stack([x, y, z])


def wahba(source, target):
    u, _, vt = np.linalg.svd(np.asarray(source).T @ target)
    return vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T


def angular_distance(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1))


def common_anchor(points, directions):
    directions = unit(directions)
    pp = np.eye(3) - directions[:, :, None]*directions[:, None, :]
    return np.linalg.lstsq(pp.reshape(-1, 3), np.einsum('nij,nj->ni', pp, points).ravel(), rcond=None)[0]


def empirical_angles(groups, exclude=None):
    training = {k: g for k, g in groups.items() if k != exclude}
    steps = []
    for g in training.values():
        idx = np.asarray(g['full_slots'])
        d = np.asarray(g['full_d'])
        b = fan_basis(d)
        a = np.arctan2((d@b)[:, 1], (d@b)[:, 0])
        order = np.argsort(idx)
        steps.extend((np.diff(a[order])/np.diff(idx[order])).tolist())
    step = float(np.median(steps))
    a = np.arange(-3, 4)*step
    template = np.column_stack([np.cos(a), np.sin(a), np.zeros(7)])
    for iteration in range(20):
        samples = [[] for _ in range(7)]
        for g in training.values():
            s = np.asarray(g['full_slots'])
            d = np.asarray(g['full_d'])
            r = wahba(template[s], d)
            for j, vector in zip(s, d@r):
                samples[j].append(vector)
        updated = unit([np.median(v, axis=0) if v else template[j] for j, v in enumerate(samples)])
        updated = updated@fan_basis(updated)
        change = np.max(angular_distance(updated, template))
        template = updated
        if change < 1e-9:
            break
    errors = [[] for _ in range(7)]
    for g in training.values():
        s = np.asarray(g['full_slots']); d = np.asarray(g['full_d'])
        r = wahba(template[s], d)
        for j, e in zip(s, angular_distance(template[s]@r.T, d)):
            errors[j].append(float(e))
    sigmas = np.array([max(.006, np.sqrt(np.mean(np.asarray(e)**2))) if e else .02 for e in errors])
    return dict(directions=template, sigma_rad=sigmas, training_groups=list(training),
                error_rad=errors, iterations=iteration+1,
                method='robust spherical mean after SO3 registration; fixed explicit geometric slots',
                uncertainty='combined observed scatter, not separately metrology/manufacturing and not calibrated CI')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--west-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--skip-reflection-trace', action='store_true')
    args = parser.parse_args()
    args.repo = args.repo.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))
    from mirrorlab.calibration.observations import observation_registry
    from mirrorlab.calibration.physical import (_field_models, incoming_at_points, rotation_from_normal,
                                               validate_bodies, validate_reflections)
    from mirrorlab.configurations import load_configuration
    from mirrorlab.coordinates import survey_frames
    from mirrorlab.engine.bodies import MirrorBody
    from mirrorlab.state import snapshot, digest

    manual = {}
    for text in args.input.read_text(encoding='utf-8-sig').splitlines()[1:]:
        if not text.strip():
            continue
        t = text.split()
        manual[t[0].replace('.', '_')] = np.array([scalar(x) for x in t[1:]]).reshape(2, 3)
    assert len(manual) == 112
    registry = observation_registry(args.repo, 'west_to_east')
    records = {row['old_id']: row for row in registry['records']}
    config = load_configuration('west_to_east', 'real')
    frame = survey_frames()['west_to_east']
    q, t = np.asarray(frame.rotation), np.asarray(frame.translation)
    field_models, field_hash = _field_models(config)
    radius = config.source.waist_radius * config.source.cutoff_ratio

    groups, id_maps = {}, []
    for rod in range(1, 5):
        for window in range(1, 5):
            name = f'{rod}_{window}'
            ids = [f'{name}_{j}' for j in range(1, 8)]
            md = unit([manual[k][0]-manual[k][1] for k in ids])
            basis = fan_basis(md)
            order = np.argsort(np.arctan2((md@basis)[:, 1], (md@basis)[:, 0]))
            slot_ids = [ids[j] for j in order]
            if name == '3_2':
                slot_ids = ['3_2_1', '3_2_2', '3_2_3', '3_2_4', '3_2_7', '3_2_5', '3_2_6']
            full = [k for k in slot_ids if records[k]['complete_points'] == 2]
            points = np.array([records[k]['screens_m'][0] for k in full])
            fd = unit(np.array([records[k]['screens_m'][1] for k in full])-points)
            anchor = common_anchor(points, fd)
            fd[np.sum((points-anchor)*fd, axis=1) < 0] *= -1
            model = field_models[str(rod)][0]
            dn = np.asarray(model['direction']); dw = q@dn
            origin = q@np.asarray(model['map'])[2]+t
            beam_anchor = origin + (anchor[2]-origin[2])/dw[2]*dw
            projection = np.eye(3)-np.outer(dn, dn)
            mapping = np.linalg.pinv(projection@np.asarray(model['map'])[:2].T)@projection
            launch_matrix = mapping@q.T
            launch_offset = -launch_matrix@t-mapping@np.asarray(model['map'])[2]
            groups[name] = dict(name=name, rod=rod, window=window, slot_ids=slot_ids,
                                full_slots=[slot_ids.index(k) for k in full], full_d=fd,
                                original_anchor_m=anchor, beam_anchor_m=beam_anchor,
                                incoming_world=dw, launch_matrix=launch_matrix, launch_offset=launch_offset,
                                source_radius_m=radius, native_window=5-window)
            for j, source_id in enumerate(slot_ids):
                id_maps.append(dict(source_id=source_id, canonical_slot=j+1,
                                    survey_group=name, native_id=f'{rod}_{5-window}_{j+1}',
                                    basis='user_accepted_internal_gap' if name == '3_2' else 'geometric ordering of manual forward directions',
                                    hardware_number_verified=False))

    # Empirical spatial prior uses complete manual WEST bundles, not external33° geometry.
    west = json.loads(args.west_reference.read_text())
    samples, spatial_groups = [], []
    for rod in range(1, 5):
        for window in range(1, 5):
            name = f'{rod}_{window}'
            rows = [r for r in west['mirrors'] if r['manual_id'].startswith(name+'_')]
            c = np.array([r['center_historical_registered_survey_m'] for r in rows])
            d = unit([r['historical_output_line_direction_survey'] for r in rows])
            basis = fan_basis(d)
            order = np.argsort(np.arctan2((d@basis)[:, 1], (d@basis)[:, 0]))
            samples.append((c[order]-c.mean(axis=0))@basis*1000)
            spatial_groups.append(dict(group=name, ordered_manual_ids=[rows[j]['manual_id'] for j in order]))
    samples = np.array(samples)
    spatial_mean_mm = np.median(samples, axis=0)
    spatial_mean_mm -= spatial_mean_mm.mean(axis=0)
    spatial_std_mm = np.maximum(.3, np.sqrt(np.mean((samples-spatial_mean_mm)**2, axis=0)))
    spatial_prior = dict(mean_offsets_local_mm=spatial_mean_mm, conditional_scatter_mm=spatial_std_mm,
                         source_file=args.west_reference.name,
                         source_sha256=hashlib.sha256(args.west_reference.read_bytes()).hexdigest(),
                         source_type='manual_derived_spatial_prior', training_assemblies=spatial_groups,
                         notes=['16 complete manually reconstructed WEST bundles; not independent manufacturing measurements',
                                'Rigid orientation removed with outgoing fan basis; no assumed33degree angular template',
                                'Cross-side transfer is an explicit conditional mechanical prior',
                                'Componentwise median and RMS scatter;0.3mm numerical/prior floor, not instrument uncertainty'])

    all_bodies_native = []
    mirrors, group_results, raw_candidates, angle_templates = [], [], [], {}
    start_time = time.monotonic()
    for name, group in groups.items():
        print('fitting', name, flush=True)
        template = empirical_angles(groups, exclude=name)
        angle_templates[name] = template
        dt = template['directions']
        r0 = wahba(dt[group['full_slots']], group['full_d'])
        d0 = dt@r0.T
        base_offsets = spatial_mean_mm@r0.T
        anchor = group['beam_anchor_m']; din = group['incoming_world']
        slot_ids = group['slot_ids']
        measured = [records[k] for k in slot_ids]
        tangents = []
        for d in d0:
            aa = unit(np.cross(d, [0, 0, 1.])); tangents.append([aa, unit(np.cross(d, aa))])
        tangents = np.array(tangents)
        # Parameters: assembly translation(mm), individual positions(mm), direction tangent(mrad), roll(deg).
        count = 3+21+14+7
        pair_i, pair_j = np.array(list(itertools.combinations(range(7), 2))).T
        # Support-plane candidates include transverse and spherical directions.
        ph = np.linspace(0, 2*np.pi, 96, endpoint=False)
        transverse = np.column_stack([np.cos(ph), np.sin(ph), np.zeros(len(ph))])
        kk = np.arange(64); zz=1-2*(kk+.5)/len(kk); phi=kk*np.pi*(3-np.sqrt(5))
        sphere=np.column_stack([np.sqrt(1-zz*zz)*np.cos(phi), np.sqrt(1-zz*zz)*np.sin(phi), zz])
        plane_dirs=np.r_[transverse, sphere]

        def unpack(x, poses=False):
            dc = x[3:24].reshape(7, 3).copy()
            dc -= dc.mean(axis=0)
            c = anchor + (x[:3]+base_offsets+dc)/1000
            a = x[24:38].reshape(7, 2)/1000
            d = unit(d0+np.einsum('ni,nij->nj', a, tangents))
            if not poses:
                return c, d, dc
            normal = unit(d-din)
            rotation = np.array([rotation_from_normal(n, axis_hint=din) for n in normal])
            rotation = Rotation.from_rotvec(np.deg2rad(x[38:45])[:, None]*normal).as_matrix()@rotation
            return c, d, dc, normal, rotation

        def data_residual(c, d):
            result = []
            for j, row in enumerate(measured):
                for screen in row['screens_m']:
                    mask = np.array([v is not None for v in screen])
                    if not mask.any():
                        continue
                    target=np.array([v for v in screen if v is not None])
                    ds,cs=d[j,mask],c[j,mask]
                    s=max(0., float(ds@(target-cs)/(ds@ds)))
                    result.extend((cs+s*ds-target)*1000)
            return np.array(result)

        def residual(x):
            c,d,dc=unpack(x)
            #1mm per-coordinate metrology scenario; known-false manual line for3.2.7 excluded.
            res=[data_residual(c,d), x[:3]/10., (dc@r0/spatial_std_mm).ravel(),
                 x[3:24].reshape(7,3).mean(axis=0)/.3,
                 (x[24:38].reshape(7,2)/(template['sigma_rad'][:,None]*1000)).ravel(), x[38:45]/5.]
            for j,k in enumerate(slot_ids):
                if k=='3_2_7' or records[k]['complete_points']==2:
                    continue
                manual_d=unit(manual[k][0]-manual[k][1])
                # Derived prior weak enough to allow primary points/compactness to override it.
                res.append(np.cross(d[j],manual_d)/.05)
            return np.concatenate(res)

        def physical_constraints(x):
            c,d,dc,normals,poses=unpack(x,True)
            launches=c@group['launch_matrix'].T+group['launch_offset']
            beam=(radius-np.linalg.norm(launches,axis=1))*1000
            delta=c[pair_j]-c[pair_i]
            axes=poses[:,:,2]
            extra=np.stack([unit(delta),-unit(delta),normals[pair_i],-normals[pair_i],
                            normals[pair_j],-normals[pair_j],axes[pair_i],-axes[pair_i],
                            axes[pair_j],-axes[pair_j]],axis=1)
            vv=np.concatenate([np.broadcast_to(plane_dirs,(len(pair_i),len(plane_dirs),3)),extra],axis=1)
            def support(v,p):
                local=np.einsum('pdk,pkj->pdj',v,p)
                return np.where(local[:,:,2]>=0,
                                .0005*np.hypot(local[:,:,0],local[:,:,1])+.0095*local[:,:,2],
                                .0005*np.hypot(local[:,:,0]+local[:,:,2],local[:,:,1]))
            gap=np.max(np.einsum('pdk,pk->pd',vv,delta)-support(vv,poses[pair_i])-support(-vv,poses[pair_j]),axis=1)
            # Positive exact support-plane gap certifies nonintersection.1um target+20nm tolerance margin.
            return np.r_[beam, (gap-.00000102)*1000, 1.5-np.linalg.norm(dc,axis=1)]

        x0=np.zeros(count)
        fit_ls=least_squares(residual,x0,max_nfev=120,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        x_initial=fit_ls.x.copy()
        # Keep initial shared position in finite illumination, then exact constrained optimization.
        x_initial[:2]=np.clip(x_initial[:2],-6.,6.)
        x_initial[3:24]=np.clip(x_initial[3:24],-1.,1.)
        x_initial[24:38]=np.clip(x_initial[24:38],-40.,40.)
        bounds=[(-10,10),(-10,10),(-5,5)]+[(-1.5,1.5)]*21+[(-50,50)]*14+[(-10,10)]*7
        fit=minimize(lambda x: float(residual(x)@residual(x)),x_initial,method='SLSQP',bounds=bounds,
                     constraints={'type':'ineq','fun':physical_constraints},
                     options={'maxiter':args.iterations,'ftol':1e-8})
        c,d,dc,normal,poses=unpack(fit.x,True)
        # Independent exact body validation, not optimizer.success alone.
        bodies_world=[MirrorBody(c[j],poses[j],f"{group['rod']}_{group['native_window']}_{j+1}") for j in range(7)]
        body_check=validate_bodies(bodies_world,required_gap_m=1e-6,numerical_tolerance_m=1e-8,check_length_alternative=False)
        launch=c@group['launch_matrix'].T+group['launch_offset']
        check_margin=float(physical_constraints(fit.x).min())
        before_full=[];after_full=[];before_single=[];after_single=[]
        for j,k in enumerate(slot_ids):
            row=records[k]; pts=[np.array(p,float) for p in row['screens_m'] if all(v is not None for v in p)]
            md=unit(manual[k][0]-manual[k][1])
            manual_dist=np.linalg.norm(np.cross(np.array(pts)-manual[k][1],md),axis=1)*1000 if pts else np.array([])
            recovered_dist=np.linalg.norm(np.cross(np.array(pts)-c[j],d[j]),axis=1)*1000 if pts else np.array([])
            (before_full if len(pts)==2 else before_single).extend(manual_dist)
            (after_full if len(pts)==2 else after_single).extend(recovered_dist)
            cn=q.T@(c[j]-t); rn=q.T@poses[j]
            body_native=MirrorBody(cn,rn,bodies_world[j].mirror_id)
            all_bodies_native.append(body_native)
            # Rawmanual only: nearest point along original line to primary anchor; diagnostic degeneracy.
            ac=group['original_anchor_m']; rawc=manual[k][1]+np.dot(ac-manual[k][1],md)*md
            raw_candidates.append(dict(source_id=k,center_survey_m=rawc,
                                       method='orthogonal projection of original complete-line fan anchor onto manual infinite line',
                                       point_on_raw_line=True, not_unique=True,
                                       bundle_anchor_distance_mm=np.linalg.norm(rawc-ac)*1000,
                                       use_for_training=False))
            eigs=[]
            if len(pts)==2:
                baseline=np.linalg.norm(pts[1]-pts[0])
                theta_sigma=np.sqrt(2)*.001/baseline
            else:
                theta_sigma=float(template['sigma_rad'][j])
            mirrors.append(dict(source_id=k,source_type='manual_derived_then_primary_constrained_reconstruction',
                                source_manual_points_survey_m=manual[k],
                                survey_group=name,canonical_slot=j+1,physical_mirror_number_verified=False,
                                native_id=body_native.mirror_id,center_definition='reflecting_ellipse_center',
                                center_survey_m=c[j],center_native_m=cn,
                                outgoing_survey=d[j],outgoing_native=q.T@d[j],
                                incoming_survey=din,incoming_native=q.T@din,
                                normal_survey=normal[j],normal_native=q.T@normal[j],
                                rotation_body_to_survey=poses[j],rotation_body_to_native=rn,
                                cylinder=body_native.to_dict(),
                                ellipse_vertices_survey_m=(np.array([[.0005,0,.0005],[0,.0005,0],[-.0005,0,-.0005],[0,-.0005,0]])@poses[j].T+c[j]),
                                primary_observation_id=row['id'],primary_complete_points=len(pts),
                                manual_primary_point_residual_mm=manual_dist,
                                fitted_primary_point_residual_mm=recovered_dist,
                                manual_direction_change_mrad=float(angular_distance(d[j],md)*1000),
                                template_direction_change_mrad=float(angular_distance(d[j],d0[j])*1000),
                                individual_center_delta_mm=dc[j], roll_from_axis_hint_deg=fit.x[38+j],
                                source_launch_radius_mm=np.linalg.norm(launch[j])*1000,
                                accepted_assignment='internal_slot5_for_source3.2.7' if k=='3_2_7' else 'ordered_fan',
                                raw_manual_direction_used_as_prior=k!='3_2_7' and row['complete_points']!=2,
                                confidence=dict(status='conditional_regularized_candidate',
                                                measured_point_sigma_scenario_m=.001,
                                                angular_sigma_scenario_rad=theta_sigma,
                                                center_sigma_estimate_m=None,
                                                unobserved=['longitudinal_center_without_mechanics','body_roll','within_aperture_hit_offset'],
                                                note='Mechanical scales are priors, not posterior confidence intervals'),
                                physics=dict(body_group_status=body_check['status'],finite_source_footprint=bool(np.linalg.norm(launch[j])<=radius+1e-8),
                                             reflection_law_residual=np.linalg.norm(din-2*np.dot(din,normal[j])*normal[j]-d[j]),
                                             incident_dot_normal=np.dot(din,normal[j]),
                                             holder_and_rod_containment='not_evaluated_no_independent_dimensions',
                                             waist_side='not_evaluated_no_physical_waist'),
                                used_as_primary_measurement=False))
        report=dict(group=name,raw_primary_anchor_m=group['original_anchor_m'],beam_anchor_m=anchor,
                    anchor_to_current_beam_axis_mm=np.linalg.norm(group['original_anchor_m']-anchor)*1000,
                    center_mean_survey_m=c.mean(0),bundle_radius_mm=np.max(np.linalg.norm(c-c.mean(0),axis=1))*1000,
                    solver_success=fit.success,solver_message=fit.message,iterations=fit.nit,
                    objective=float(fit.fun),minimum_constraint_margin_mm=check_margin,
                    candidate_feasible_for_enforced_constraints=check_margin>=-1e-5 and not body_check['conflicts'],
                    before_full_points_mm=stats(before_full),after_full_points_mm=stats(after_full),
                    before_single_points_mm=stats(before_single),after_single_points_mm=stats(after_single),
                    individual_position_rms_mm=np.sqrt(np.mean(np.sum(dc*dc,axis=1))),
                    max_individual_position_mm=np.max(np.linalg.norm(dc,axis=1)),
                    body_validation=body_check,
                    angular_training_groups=template['training_groups'])
        group_results.append(report)
        save(args.output/'east_recovery_progress.json',dict(groups=group_results,mirrors=mirrors))
        print(name,'rms_mm',round(report['after_full_points_mm']['rms'],3),'success',fit.success,
              'margin_mm',round(check_margin,6),'body_conflicts',len(body_check['conflicts']),flush=True)

    print('full optics validation',flush=True)
    field=incoming_at_points(config,{b.mirror_id:b.center_m for b in all_bodies_native})
    all_body_report=validate_bodies(all_bodies_native,required_gap_m=1e-6,numerical_tolerance_m=1e-8,check_length_alternative=False)
    optical_report=validate_reflections(config,all_bodies_native) if not args.skip_reflection_trace else None
    for mirror in mirrors:
        key=mirror['native_id']; mirror['incoming_field_check']=field[key]
        if optical_report:
            mirror['physics']['full_optical_path']=optical_report['mirrors'][key]
    full_before=[v for m in mirrors if m['primary_complete_points']==2 for v in m['manual_primary_point_residual_mm']]
    full_after=[v for m in mirrors if m['primary_complete_points']==2 for v in m['fitted_primary_point_residual_mm']]
    singles_before=[v for m in mirrors if m['primary_complete_points']==1 for v in m['manual_primary_point_residual_mm']]
    singles_after=[v for m in mirrors if m['primary_complete_points']==1 for v in m['fitted_primary_point_residual_mm']]
    result=dict(format='analysis.manual-reconstruction-seeds',schema_version=1,application_preset=False,
                route='west_to_east',side='EAST',units='m',coordinate_frame='survey',
                current_project_snapshot='4ab80dfba048e11cf5a3b0bef7dc6832b298540b',
                optics_snapshot=snapshot(config),optics_sha256=digest(snapshot(config)),
                native_to_survey=dict(rotation=q,translation_m=t,window_map={str(w):5-w for w in range(1,5)}),
                source_manual_file=args.input.name,source_manual_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                primary_measurement_ledger=registry,primary_ledger_is_immutable=True,
                accepted_decisions={'3_2':'source1,2,3,4,5,6 -> slots1,2,3,4,6,7; source7 reconstructed atslot5; its prior direction excluded'},
                method='cross_fitted_empirical_angular_template_and_manual_WEST_spatial_prior_with_finite_source_and_exact_support_body_constraints',
                spatial_prior=spatial_prior,angular_templates=angle_templates,
                conditional_parameters=dict(point_sigma_m=.001,individual_center_limit_m=.0015,
                                            roll_limit_deg=10,direction_tangent_limit_rad=.05,
                                            minimum_body_gap_m=1e-6,full_cylinder_extent_m=.01,diameter_m=.001,
                                            note='Exploratory conditional scales, not manufacturing tolerances'),
                summary=dict(mirror_count=len(mirrors),before_full_points_mm=stats(full_before),after_full_points_mm=stats(full_after),
                             before_single_points_mm=stats(singles_before),after_single_points_mm=stats(singles_after),
                             incoming_status_counts=dict(Counter(v['status'] for v in field.values())),
                             body_conflicts=len(all_body_report['conflicts']),
                             validated_reflections=optical_report['validated_count'] if optical_report else None,
                             elapsed_seconds=time.monotonic()-start_time),
                groups=group_results,mirrors=mirrors,source_to_slot_mapping=id_maps,
                body_validation=all_body_report,reflection_validation=optical_report,
                validation_status='conditional_candidate_not_certified',
                limitations=['Unknown metrology covariance;1mm is an explicit scenario',
                             'Current common optics held fixed; data/illumination conflicts retained',
                             'Actual rod/holder solid dimensions and physical waists are unavailable',
                             'Spatial prior transferred from manual WEST bundles; not independent metrology',
                             'No unknown ray is relabelled a measurement; no production activation is implied'])
    save(args.output/'east_recovered_centers.json',result)
    save(args.output/'east_raw_manual_projected_centers.json',dict(format='analysis.raw-manual-center-projection',
         note='Not a unique mirror recovery; direct manual-line preserving comparison only',mirrors=raw_candidates))
    save(args.output/'east_recovery_summary.json',dict(summary=result['summary'],groups=[{k:v for k,v in g.items() if k!='body_validation'} for g in group_results]))
    with (args.output/'east_recovered_centers.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['source_id','canonical_slot','native_id','x_survey_m','y_survey_m','z_survey_m','x_native_m','y_native_m','z_native_m',
                'nx_survey','ny_survey','nz_survey','dx_out_survey','dy_out_survey','dz_out_survey','primary_complete_points',
                'manual_residual_rms_mm','fitted_residual_rms_mm','source_launch_radius_mm','incoming_status','optical_status']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for m in mirrors:
            row=dict(source_id=m['source_id'],canonical_slot=m['canonical_slot'],native_id=m['native_id'],
                     primary_complete_points=m['primary_complete_points'],
                     manual_residual_rms_mm=stats(m['manual_primary_point_residual_mm']).get('rms'),
                     fitted_residual_rms_mm=stats(m['fitted_primary_point_residual_mm']).get('rms'),
                     source_launch_radius_mm=m['source_launch_radius_mm'],incoming_status=m['incoming_field_check']['status'],
                     optical_status=m['physics'].get('full_optical_path',{}).get('status'))
            row.update(dict(zip(['x_survey_m','y_survey_m','z_survey_m'],m['center_survey_m'])))
            row.update(dict(zip(['x_native_m','y_native_m','z_native_m'],m['center_native_m'])))
            row.update(dict(zip(['nx_survey','ny_survey','nz_survey'],m['normal_survey'])))
            row.update(dict(zip(['dx_out_survey','dy_out_survey','dz_out_survey'],m['outgoing_survey'])))
            writer.writerow(row)
    print(json.dumps(serial(result['summary']),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
