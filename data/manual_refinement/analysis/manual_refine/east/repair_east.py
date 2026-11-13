"""Conservative position repair of EAST candidate, followed by full ray tracing.

Directions, optical normals, and body rotations are fixed. Exact analytic
support planes make the position subproblem convex except the small relative
bundle-deformation ball (it is convex too). This avoids relying on convergence
of the preceding max-of-support-plane nonlinear pose search.
"""
from pathlib import Path
import argparse
import csv
import itertools
import json
import sys

import numpy as np
from scipy.optimize import minimize

from recover_east import serial, save, stats, unit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    sys.path.insert(0,str(args.repo.resolve()))
    from mirrorlab.calibration.physical import _field_models, validate_bodies, validate_reflections, incoming_at_points
    from mirrorlab.configurations import load_configuration
    from mirrorlab.engine.bodies import MirrorBody
    data=json.loads(args.candidate.read_text())
    data['pre_clearance_repair_summary']=data['summary'].copy()
    q=np.array(data['native_to_survey']['rotation']);t=np.array(data['native_to_survey']['translation_m'])
    config=load_configuration('west_to_east','real')
    models,_=_field_models(config)
    radius=config.source.waist_radius*config.source.cutoff_ratio
    primary={r['old_id']:r for r in data['primary_measurement_ledger']['records']}
    allbodies=[]
    repair_reports=[]
    for group in data['groups']:
        name=group['group'];rod=int(name.split('_')[0])
        rows=sorted([m for m in data['mirrors'] if m['survey_group']==name],key=lambda x:x['canonical_slot'])
        c=np.array([m['center_survey_m'] for m in rows]);d=np.array([m['outgoing_survey'] for m in rows])
        poses=np.array([m['rotation_body_to_survey'] for m in rows])
        oldc=c.copy();reference=c.mean(axis=0)
        olddc=np.array([m['individual_center_delta_mm'] for m in rows])
        template_offsets_mm=(c-c.mean(axis=0))*1000-olddc
        previous_bodies=[MirrorBody(c[j],poses[j],rows[j]['native_id']) for j in range(7)]
        model=models[str(rod)][0]
        dn=np.array(model['direction']);proj=np.eye(3)-np.outer(dn,dn)
        a=np.linalg.pinv(proj@np.array(model['map'])[:2].T)@proj
        launch_matrix=a@q.T;launch_offset=-launch_matrix@t-a@np.array(model['map'])[2]
        ki,kj=np.array(list(itertools.combinations(range(7),2))).T
        kk=np.arange(256);zz=1-2*(kk+.5)/len(kk);ph=kk*np.pi*(3-np.sqrt(5))
        dirs=np.column_stack([np.sqrt(1-zz*zz)*np.cos(ph),np.sqrt(1-zz*zz)*np.sin(ph),zz])
        aa=np.linspace(0,2*np.pi,720,endpoint=False)
        dirs=np.r_[dirs,np.column_stack([np.cos(aa),np.sin(aa),np.zeros_like(aa)])]
        chosen=[];bounds=[]
        for i,j in zip(ki,kj):
            vv=np.r_[dirs,unit(c[j]-c[i])[None,:],poses[i].T,-poses[i].T,poses[j].T,-poses[j].T]
            offa=np.array([previous_bodies[i].support(v)-v@c[i] for v in vv])
            offb=np.array([previous_bodies[j].support(-v)+v@c[j] for v in vv])
            gap=vv@(c[j]-c[i])-offa-offb
            best=int(np.argmax(gap));chosen.append(vv[best]);bounds.append((offa[best]+offb[best])*1000+.002)
        chosen=np.array(chosen);bounds=np.array(bounds)

        def position(x):
            return reference+x.reshape(7,3)/1000

        def constraints(x):
            cc=position(x);delta=(cc[kj]-cc[ki])*1000
            gap=np.sum(chosen*delta,axis=1)-bounds
            footprint=(radius-.0000005-np.linalg.norm(cc@launch_matrix.T+launch_offset,axis=1))*1000
            defo=(cc-cc.mean(axis=0))*1000-template_offsets_mm
            return np.r_[gap,footprint,1.5-np.linalg.norm(defo,axis=1)]

        def objective(x):
            cc=position(x);res=[]
            for j,row in enumerate(rows):
                for screen in primary[row['source_id']]['screens_m']:
                    mask=np.array([v is not None for v in screen])
                    if not mask.any():continue
                    target=np.array([v for v in screen if v is not None]);ds=d[j,mask];cs=cc[j,mask]
                    distance=max(0.,float(ds@(target-cs)/(ds@ds)))
                    res.extend((cs+distance*ds-target)*1000)
            # Minimal explicit repair;0.05mm scale keeps perturbations small.
            res.extend(((cc-oldc)*1000/.05).ravel())
            return float(np.asarray(res)@np.asarray(res))

        initial=(c-reference).ravel()*1000
        fit=minimize(objective,initial,method='SLSQP',constraints={'type':'ineq','fun':constraints},
                     options={'maxiter':400,'ftol':1e-10})
        if constraints(fit.x).min() < -1e-5:
            print('repair_unresolved',name,fit.message,float(constraints(fit.x).min()),flush=True)
        c=position(fit.x)
        # Retain diagnostic result even if failed; never relabel as feasible.
        bodies=[MirrorBody(c[j],poses[j],rows[j]['native_id']) for j in range(7)]
        check=validate_bodies(bodies,required_gap_m=1e-6,numerical_tolerance_m=1e-8,check_length_alternative=False)
        repair=dict(group=name,solver_success=fit.success,message=fit.message,
                    min_margin_mm=float(constraints(fit.x).min()),maximum_center_shift_mm=np.max(np.linalg.norm(c-oldc,axis=1))*1000,
                    exact_conflicts=len(check['conflicts']),gap_m=check['minimum_gap_m'])
        repair_reports.append(repair)
        group['before_position_repair']=dict(minimum_constraint_margin_mm=group['minimum_constraint_margin_mm'],
                                            candidate_feasible_for_enforced_constraints=group['candidate_feasible_for_enforced_constraints'])
        group['position_repair']=repair
        group['body_validation']=check
        group['center_mean_survey_m']=c.mean(axis=0).tolist()
        group['bundle_radius_mm']=float(np.max(np.linalg.norm(c-c.mean(axis=0),axis=1))*1000)
        group['candidate_feasible_for_enforced_constraints']=bool(constraints(fit.x).min()>=-1e-5 and not check['conflicts'])
        afterfull=[];aftersingle=[]
        for j,row in enumerate(rows):
            cn=q.T@(c[j]-t);rn=q.T@poses[j]
            body=MirrorBody(cn,rn,row['native_id']);allbodies.append(body)
            row['center_before_clearance_repair_survey_m']=row['center_survey_m']
            row['clearance_repair_delta_m']=(c[j]-oldc[j]).tolist()
            row['center_survey_m']=c[j].tolist();row['center_native_m']=cn.tolist();row['cylinder']=body.to_dict()
            row['individual_center_delta_mm']=((c[j]-c.mean(axis=0))*1000-template_offsets_mm[j]).tolist()
            row['ellipse_vertices_survey_m']=(np.array([[.0005,0,.0005],[0,.0005,0],[-.0005,0,-.0005],[0,-.0005,0]])@poses[j].T+c[j]).tolist()
            pts=[p for p in primary[row['source_id']]['screens_m'] if all(v is not None for v in p)]
            rr=np.linalg.norm(np.cross(np.array(pts)-c[j],d[j]),axis=1)*1000 if pts else np.array([])
            row['fitted_primary_point_residual_mm']=rr.tolist()
            (afterfull if len(pts)==2 else aftersingle).extend(rr)
            row['source_launch_radius_mm']=float(np.linalg.norm(c[j]@launch_matrix.T+launch_offset)*1000)
            row['physics']['body_group_status']=check['status']
            row['physics']['finite_source_footprint']=bool(row['source_launch_radius_mm']<=radius*1000+1e-5)
        group['after_full_points_mm']=stats(afterfull);group['after_single_points_mm']=stats(aftersingle)
        print(name,'repair_um',round(repair['maximum_center_shift_mm']*1000,2),'conflicts',repair['exact_conflicts'],flush=True)

    print('validating full112 optics',flush=True)
    field=incoming_at_points(config,{b.mirror_id:b.center_m for b in allbodies})
    body_check=validate_bodies(allbodies,required_gap_m=1e-6,numerical_tolerance_m=1e-8,check_length_alternative=False)
    optics=validate_reflections(config,allbodies)
    from collections import Counter
    for row in data['mirrors']:
        row['incoming_field_check']=field[row['native_id']]
        row['physics']['full_optical_path']=optics['mirrors'][row['native_id']]
    data['body_validation']=body_check;data['reflection_validation']=optics
    data['position_repair']=dict(method='fixed_rotations_minimum_position_change_with_conservative_exact_support_planes',
                                 target_gap_m=2e-6,source_boundary_margin_m=.5e-6,groups=repair_reports)
    for kind,n in [('full',2),('single',1)]:
        rr=[v for m in data['mirrors'] if m['primary_complete_points']==n for v in m['fitted_primary_point_residual_mm']]
        data['summary'][f'after_{kind}_points_mm']=stats(rr)
    data['summary']['incoming_status_counts']=dict(Counter(v['status'] for v in field.values()))
    data['summary']['body_conflicts']=len(body_check['conflicts'])
    data['summary']['minimum_body_gap_m']=body_check['minimum_gap_m']
    data['summary']['validated_reflections']=optics['validated_count']
    data['summary']['failed_reflections']=optics['failed_count']
    data['summary']['unavailable_reflections']=optics['unavailable_count']
    data['summary']['maximum_clearance_repair_shift_mm']=max(r['maximum_center_shift_mm'] for r in repair_reports)
    save(args.output,data)
    summary_path=args.output.with_name('east_recovery_summary.json')
    save(summary_path,dict(summary=data['summary'],groups=[{k:v for k,v in g.items() if k!='body_validation'} for g in data['groups']]))
    csv_path=args.output.with_suffix('.csv')
    fields=['source_id','canonical_slot','native_id','x_survey_m','y_survey_m','z_survey_m','x_native_m','y_native_m','z_native_m',
            'nx_survey','ny_survey','nz_survey','dx_out_survey','dy_out_survey','dz_out_survey','primary_complete_points',
            'manual_residual_rms_mm','fitted_residual_rms_mm','source_launch_radius_mm','incoming_status','optical_status']
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for m in data['mirrors']:
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
    print(json.dumps(serial(data['summary']),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
