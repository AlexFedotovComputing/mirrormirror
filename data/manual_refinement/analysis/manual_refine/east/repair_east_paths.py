"""Repair observed path occlusions with fixed optical directions and body poses.

Each new inequality separates a full straight line from an exact convex mirror
body by a support plane perpendicular to that line. It is conservative (the
physical path is only a half-line/segment) but every positive margin certifies
that this blocker is absent. Actual complete optical traces are run afterward.
"""
from pathlib import Path
import argparse
import csv
import itertools
import json
import re
import sys

import numpy as np
from scipy.optimize import minimize

from recover_east import serial, save, stats, unit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--iterations',type=int,default=6)
    args=ap.parse_args()
    sys.path.insert(0,str(args.repo.resolve()))
    from mirrorlab.calibration.physical import _field_models, validate_bodies, validate_reflections, incoming_at_points
    from mirrorlab.configurations import load_configuration
    from mirrorlab.engine.bodies import MirrorBody
    from mirrorlab.state import snapshot, digest
    data=json.loads(args.candidate.read_text())
    data['pre_path_repair_summary']=data['summary'].copy()
    q=np.array(data['native_to_survey']['rotation']);t=np.array(data['native_to_survey']['translation_m'])
    config=load_configuration('west_to_east','real');models,_=_field_models(config)
    data['optics_snapshot']=snapshot(config)
    data['optics_sha256']=digest(data['optics_snapshot'])
    radius=config.source.waist_radius*config.source.cutoff_ratio
    primary={r['old_id']:r for r in data['primary_measurement_ledger']['records']}
    by_native={r['native_id']:r for r in data['mirrors']}
    original={k:np.array(r['center_survey_m']) for k,r in by_native.items()}
    history=[];known={}

    def to_bodies():
        return [MirrorBody(np.array(r['center_native_m']),np.array(r['rotation_body_to_native']),r['native_id']) for r in data['mirrors']]

    def body_world(k):
        r=by_native[k]
        return MirrorBody(np.array(r['center_survey_m']),np.array(r['rotation_body_to_survey']),k)

    def parse_name(name):
        m=re.match(r'BUNDLE_(\d+)_(\d+) Mirror (\d+)',name or '')
        return '_'.join(m.groups()) if m else None

    for iteration in range(args.iterations):
        optics=data['reflection_validation'] if iteration==0 else validate_reflections(config,to_bodies())
        print('iteration',iteration,'validated',optics['validated_count'],'failed',optics['failed_count'],flush=True)
        if optics['validated_count']==112:break
        new=[]
        for target,report in optics['mirrors'].items():
            obstacle=parse_name(report.get('first_surface'))
            if obstacle and obstacle!=target:
                new.append((target,obstacle,'incoming'))
            for name in report.get('outgoing_blockers',[]):
                obstacle=parse_name(name)
                if obstacle and obstacle!=target:
                    new.append((target,obstacle,'outgoing'))
        for target,obstacle,kind in new:
            key=(target,obstacle,kind)
            if key in known:continue
            row=by_native[target]
            direction=np.array(row['incoming_survey'] if kind=='incoming' else row['outgoing_survey'])
            e1=unit(np.cross(direction,np.eye(3)[np.argmin(np.abs(direction))]));e2=np.cross(direction,e1)
            theta=np.linspace(0,2*np.pi,720,endpoint=False)
            vv=np.cos(theta)[:,None]*e1+np.sin(theta)[:,None]*e2
            obstacle_body=body_world(obstacle);target_c=np.array(row['center_survey_m'])
            support=np.array([obstacle_body.support(v) for v in vv])
            gaps=vv@target_c-support;best=int(np.argmax(gaps))
            normal=vv[best];offset=support[best]-normal@obstacle_body.center_m
            known[key]=dict(target=target,obstacle=obstacle,kind=kind,normal=normal,body_support_offset_m=offset,
                            initial_gap_m=float(gaps[best]))
        group_logs=[]
        for group in sorted(data['groups'],key=lambda x:tuple(map(int,x['group'].split('_')))):
            name=group['group'];rows=sorted([m for m in data['mirrors'] if m['survey_group']==name],key=lambda x:x['canonical_slot'])
            ids=[r['native_id'] for r in rows]
            active=[v for v in known.values() if v['target'] in ids]
            if not active:continue
            rod=int(name.split('_')[0])
            c=np.array([r['center_survey_m'] for r in rows]);d=np.array([r['outgoing_survey'] for r in rows]);poses=np.array([r['rotation_body_to_survey'] for r in rows])
            old=c.copy();ref=c.mean(axis=0);offsets=(c-ref)*1000-np.array([r['individual_center_delta_mm'] for r in rows])
            model=models[str(rod)][0];dn=np.array(model['direction']);projection=np.eye(3)-np.outer(dn,dn)
            mm=np.linalg.pinv(projection@np.array(model['map'])[:2].T)@projection
            launch_matrix=mm@q.T;launch_offset=-launch_matrix@t-mm@np.array(model['map'])[2]
            ki,kj=np.array(list(itertools.combinations(range(7),2))).T
            aa=np.linspace(0,2*np.pi,720,endpoint=False);vv0=np.column_stack([np.cos(aa),np.sin(aa),np.zeros_like(aa)])
            kk=np.arange(128);zz=1-2*(kk+.5)/len(kk);phi=kk*np.pi*(3-np.sqrt(5))
            vv0=np.r_[vv0,np.column_stack([np.sqrt(1-zz*zz)*np.cos(phi),np.sqrt(1-zz*zz)*np.sin(phi),zz])]
            planes=[];separation=[]
            for i,j in zip(ki,kj):
                b1,b2=body_world(ids[i]),body_world(ids[j]);vv=np.r_[vv0,unit(c[j]-c[i])[None,:]]
                widths=np.array([b1.support(v)-v@c[i]+b2.support(-v)+v@c[j] for v in vv])
                gap=vv@(c[j]-c[i])-widths;k=int(np.argmax(gap));planes.append(vv[k]);separation.append(widths[k]+2e-6)
            planes=np.array(planes);separation=np.array(separation)

            def unpack(x):return ref+x.reshape(7,3)/1000

            def constraints(x):
                cc=unpack(x)
                body_gap=(np.sum(planes*(cc[kj]-cc[ki]),axis=1)-separation)*1000
                source=(radius-.5e-6-np.linalg.norm(cc@launch_matrix.T+launch_offset,axis=1))*1000
                delta=(cc-cc.mean(axis=0))*1000-offsets
                paths=[]
                for row in active:
                    j=ids.index(row['target']);other=row['obstacle']
                    obstacle_c=cc[ids.index(other)] if other in ids else np.array(by_native[other]['center_survey_m'])
                    paths.append((row['normal']@(cc[j]-obstacle_c)-row['body_support_offset_m']-10e-6)*1000)
                return np.r_[body_gap,source,1.5-np.linalg.norm(delta,axis=1),paths]

            def objective(x):
                cc=unpack(x);res=[]
                for j,r in enumerate(rows):
                    for screen in primary[r['source_id']]['screens_m']:
                        mask=np.array([v is not None for v in screen])
                        if not mask.any():continue
                        target=np.array([v for v in screen if v is not None]);ds=d[j,mask];cs=cc[j,mask]
                        distance=max(0.,float(ds@(target-cs)/(ds@ds)))
                        res.extend((cs+distance*ds-target)*1000)
                #Minimal whole-group repair; current optics unchanged.
                res.extend(((cc-old)*1000/.1).ravel())
                return float(np.array(res)@np.array(res))

            fit=minimize(objective,(c-ref).ravel()*1000,method='SLSQP',
                         constraints={'type':'ineq','fun':constraints},options={'maxiter':400,'ftol':1e-10})
            cc=unpack(fit.x);margin=float(constraints(fit.x).min())
            log=dict(group=name,success=fit.success,message=fit.message,minimum_margin_mm=margin,
                     max_shift_mm=float(np.linalg.norm(cc-old,axis=1).max()*1000),path_constraints=len(active))
            if margin < -1e-5:
                log['accepted']=False
                group_logs.append(log)
                print('unresolved',name,log,flush=True)
                continue
            log['accepted']=True
            for j,r in enumerate(rows):
                r['center_survey_m']=cc[j].tolist();r['center_native_m']=(q.T@(cc[j]-t)).tolist()
                r['individual_center_delta_mm']=((cc[j]-cc.mean(axis=0))*1000-offsets[j]).tolist()
            group_logs.append(log)
            print('repaired',name,'shift_mm',round(log['max_shift_mm'],4),flush=True)
        history.append(dict(iteration=iteration,validated_before=optics['validated_count'],new_blocker_count=len(new),groups=group_logs))

    allbodies=to_bodies()
    optics=validate_reflections(config,allbodies)
    field=incoming_at_points(config,{b.mirror_id:b.center_m for b in allbodies})
    body_check=validate_bodies(allbodies,required_gap_m=1e-6,numerical_tolerance_m=1e-8,check_length_alternative=False)
    from collections import Counter
    for r,b in zip(data['mirrors'],allbodies):
        k=r['native_id'];c=np.array(r['center_survey_m']);d=np.array(r['outgoing_survey']);poses=np.array(r['rotation_body_to_survey'])
        r['cylinder']=b.to_dict();r['path_repair_delta_m']=(c-original[k]).tolist()
        r['ellipse_vertices_survey_m']=(np.array([[.0005,0,.0005],[0,.0005,0],[-.0005,0,-.0005],[0,-.0005,0]])@poses.T+c).tolist()
        r['incoming_field_check']=field[k];r['physics']['full_optical_path']=optics['mirrors'][k]
        model=models[k.split('_')[0]][0];dn=np.array(model['direction']);p=np.eye(3)-np.outer(dn,dn)
        mapping=np.linalg.pinv(p@np.array(model['map'])[:2].T)@p
        launch=mapping@(b.center_m-np.array(model['map'])[2])
        r['source_launch_radius_mm']=float(np.linalg.norm(launch)*1000)
        r['physics']['finite_source_footprint']=field[k]['status']=='available'
        pts=[p for p in primary[r['source_id']]['screens_m'] if all(v is not None for v in p)]
        r['fitted_primary_point_residual_mm']=(np.linalg.norm(np.cross(np.array(pts)-c,d),axis=1)*1000).tolist() if pts else []
    for g in data['groups']:
        rows=[r for r in data['mirrors'] if r['survey_group']==g['group']]
        c=np.array([r['center_survey_m'] for r in rows]);g['center_mean_survey_m']=c.mean(axis=0).tolist()
        g['bundle_radius_mm']=float(np.linalg.norm(c-c.mean(axis=0),axis=1).max()*1000)
        for kind,n in [('full',2),('single',1)]:
            g[f'after_{kind}_points_mm']=stats([v for r in rows if r['primary_complete_points']==n for v in r['fitted_primary_point_residual_mm']])
        g['optical_validated_count']=sum(r['physics']['full_optical_path']['status']=='validated' for r in rows)
        g['body_validation']=validate_bodies([b for b in allbodies if b.mirror_id in {r['native_id'] for r in rows}],required_gap_m=1e-6,numerical_tolerance_m=1e-8,check_length_alternative=False)
        g['individual_position_rms_mm']=float(np.sqrt(np.mean([np.sum(np.array(r['individual_center_delta_mm'])**2) for r in rows])))
        g['max_individual_position_mm']=float(max(np.linalg.norm(r['individual_center_delta_mm']) for r in rows))
    data['body_validation']=body_check;data['reflection_validation']=optics
    data['path_repair']=dict(method='fixed_direction_pose_conservative_support_plane_line_body_separation_then_actual_full_trace',history=history,
                             constraints=list(known.values()),line_body_margin_m=10e-6,
                             all_optical_paths_validated=optics['validated_count']==112)
    for kind,n in [('full',2),('single',1)]:
        data['summary'][f'after_{kind}_points_mm']=stats([v for r in data['mirrors'] if r['primary_complete_points']==n for v in r['fitted_primary_point_residual_mm']])
    data['summary']['incoming_status_counts']=dict(Counter(v['status'] for v in field.values()))
    data['summary']['body_conflicts']=len(body_check['conflicts']);data['summary']['minimum_body_gap_m']=body_check['minimum_gap_m']
    data['summary']['validated_reflections']=optics['validated_count'];data['summary']['failed_reflections']=optics['failed_count']
    data['summary']['unavailable_reflections']=optics['unavailable_count']
    data['summary']['maximum_path_repair_shift_mm']=float(max(np.linalg.norm(np.array(r['path_repair_delta_m'])) for r in data['mirrors'])*1000)
    add_consistency_annotations(data)
    save(args.output,data)
    save(args.output.with_name('east_recovery_summary.json'),dict(summary=data['summary'],groups=[{k:v for k,v in g.items() if k!='body_validation'} for g in data['groups']]))
    with args.output.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['source_id','canonical_slot','native_id','x_survey_m','y_survey_m','z_survey_m','x_native_m','y_native_m','z_native_m',
                'nx_survey','ny_survey','nz_survey','dx_out_survey','dy_out_survey','dz_out_survey','primary_complete_points',
                'manual_residual_rms_mm','fitted_residual_rms_mm','source_launch_radius_mm','incoming_status','optical_status','data_consistency_status','group_data_consistency_status']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for m in data['mirrors']:
            row=dict(source_id=m['source_id'],canonical_slot=m['canonical_slot'],native_id=m['native_id'],primary_complete_points=m['primary_complete_points'],
                     manual_residual_rms_mm=stats(m['manual_primary_point_residual_mm']).get('rms'),fitted_residual_rms_mm=stats(m['fitted_primary_point_residual_mm']).get('rms'),
                     source_launch_radius_mm=m['source_launch_radius_mm'],incoming_status=m['incoming_field_check']['status'],optical_status=m['physics']['full_optical_path']['status'],
                     data_consistency_status=m['data_consistency']['status'],group_data_consistency_status=m['data_consistency']['group_status'])
            row.update(dict(zip(['x_survey_m','y_survey_m','z_survey_m'],m['center_survey_m'])));row.update(dict(zip(['x_native_m','y_native_m','z_native_m'],m['center_native_m'])))
            row.update(dict(zip(['nx_survey','ny_survey','nz_survey'],m['normal_survey'])));row.update(dict(zip(['dx_out_survey','dy_out_survey','dz_out_survey'],m['outgoing_survey'])))
            w.writerow(row)
    print(json.dumps(serial(data['summary']),ensure_ascii=False,indent=2),flush=True)


def add_consistency_annotations(data):
    """Keep optical feasibility distinct from metrological consistency."""
    conflicts=[]
    for g in data['groups']:
        if 'initial_nonlinear_pose_fit' not in g:
            fields=['solver_success','solver_message','iterations','objective','minimum_constraint_margin_mm']
            g['initial_nonlinear_pose_fit']={k:g.pop(k) for k in fields if k in g}
            g['initial_nonlinear_pose_fit']['diagnostic_stage']='before_clearance_and_optical_path_repairs'
        rms=g['after_full_points_mm']['rms']
        conflict=rms>5.
        g['data_consistency']=dict(status='conflict_with_primary_under_fixed_current_optics' if conflict else 'conditional_fit_to_primary',
                                   measured_point_sigma_scenario_m=.001,diagnostic_group_rms_threshold_mm=5.,
                                   threshold_is_instrument_specification=False,
                                   optical_validation_does_not_establish_measurement_accuracy=True)
        g['known_physical_constraints_passed']=g.get('optical_validated_count')==7 and not g['body_validation']['conflicts']
        g['candidate_feasible_for_enforced_constraints']=g['known_physical_constraints_passed']
        g['final_validation']=dict(diagnostic_stage='after_clearance_and_optical_path_repairs',
                                    optical_validated_count=g.get('optical_validated_count'),
                                    cylinder_conflict_count=len(g['body_validation']['conflicts']),
                                    minimum_body_gap_m=g['body_validation']['minimum_gap_m'],
                                    individual_center_limit_m=.0015,
                                    max_individual_center_delta_m=g['max_individual_position_mm']/1000,
                                    all_missing_mechanical_constraints_verified=False)
        if conflict:conflicts.append(g['group'])
        for r in [m for m in data['mirrors'] if m['survey_group']==g['group']]:
            n=r['primary_complete_points'];rr=r['fitted_primary_point_residual_mm']
            rm=float(np.sqrt(np.mean(np.array(rr)**2))) if rr else None
            status=('no_primary_point_prior_dominated' if n==0 else
                    'inconsistent_with_1mm_point_scenario' if rm>5 else
                    'single_point_compatible_direction_prior_dominated' if n==1 else
                    'conditional_fit_to_two_primary_points')
            r['data_consistency']=dict(status=status,group_status=g['data_consistency']['status'],
                                       point_rms_mm=rm,metrology_sigma_known=False,used_as_primary=False)
            r['physics']['known_constraints_passed']=r['physics']['full_optical_path']['status']=='validated' and not g['body_validation']['conflicts']
    good=[v for m in data['mirrors'] if m['survey_group'] not in conflicts and m['primary_complete_points']==2 for v in m['fitted_primary_point_residual_mm']]
    data['summary']['primary_conflict_groups']=conflicts
    data['summary']['nonconflicting_groups_full_point_residual_mm']=stats(good)
    data['validation_status']='conditional_optical_body_validation_passed_primary_data_conflicts_remain'
    data['automatic_activation_allowed']=False
    data['ready_for_agent_refinement']=True
    data['activation_blockers']=['primary_data_conflicts:'+','.join(conflicts),
                                'actual_rod_holder_containment_not_evaluated','physical_waist_side_not_evaluated',
                                'uncertainty_not_metrologically_calibrated']


if __name__=='__main__':main()
