"""Attach the separately labelled Z-relaxed diagnostic WEST candidate."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=ROOT/'analysis/manual_review/repo')
    parser.add_argument('--input',type=Path,default=ROOT/'analysis/manual_refine/west/west_manual_corrected_survey.json')
    parser.add_argument('--candidate-bodies',type=Path,default=ROOT/'analysis/manual_refine/frame/west_exit_z_candidate_bodies_native.json')
    parser.add_argument('--full-trace-report',type=Path,default=ROOT/'analysis/manual_refine/frame/west_exit_z_full_reflections.json')
    parser.add_argument('--residual-report',type=Path,default=ROOT/'analysis/manual_refine/frame/west_exit_z_primary_residuals.json')
    parser.add_argument('--out',type=Path,default=ROOT/'analysis/manual_refine/west')
    args=parser.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    from mirrorlab.engine.bodies import MirrorBody
    from mirrorlab.configurations import load_configuration,configuration_ray_references
    packet=json.loads(args.input.read_text());candidate=json.loads(args.candidate_bodies.read_text())
    trace=json.loads(args.full_trace_report.read_text());residual=json.loads(args.residual_report.read_text())
    rms_by_id={r['manual_id']:r for r in residual['records']}
    bodies={b['mirror_id']:MirrorBody.from_dict(b) for b in candidate['bodies']}
    Q=np.array(packet['coordinate_policy']['current_native_to_survey_rotation']);t=np.array(packet['coordinate_policy']['current_native_to_survey_translation_m'])
    S=np.diag([1.,-1.,1.]) if np.linalg.det(Q)<0 else np.eye(3)
    local_tips=np.array([[.0005,0,.0005],[0,.0005,0],[-.0005,0,-.0005],[0,-.0005,0]])
    for rec in packet['mirrors']:
        b=bodies[rec['native_stable_seed_id']];bw=MirrorBody(Q@b.center_m+t,Q@b.rotation@S,rec['source_manual_id'])
        result=trace['mirrors'][b.mirror_id];base=rec['recommended_optimization_seed']
        rec['alternative_exit_refinement_seed']={
            'candidate_id':'west_mc_slide_roll_and_bounded_z_for_exit_v1',
            'survey_center_m':bw.center_m.tolist(),'native_center_m':b.center_m.tolist(),
            'center_shift_from_exact_manual_survey_m':(bw.center_m-rec['survey_center_m']).tolist(),
            'additional_z_shift_survey_m':float(bw.center_m[2]-base['survey_center_m'][2]),
            'survey_reflecting_normal':bw.normal.tolist(),'native_reflecting_normal':b.normal.tolist(),
            'survey_correct_45deg_face_points_m':(local_tips@bw.rotation.T+bw.center_m).tolist(),
            'native_correct_45deg_face_points_m':(local_tips@b.rotation.T+b.center_m).tolist(),
            'body_native':b.to_dict(),'body_survey':bw.to_dict(),
            'incoming':result['incoming'],
            'full_scene_forward_reflection_status':result['status'],
            'full_scene_validation':{k:result.get(k) for k in ['status','waist_side','actual_reflected_child','incident_dot_normal','outgoing_dot_normal','hit_m','hit_on_ellipse','outgoing_direction','outgoing_blockers','next_surfaces']},
            'primary_residual':rms_by_id.get(rec['source_manual_id']),
            'output_line_translated':bool(np.linalg.norm(bw.center_m-base['survey_center_m'])>1e-12),
            'activation_allowed':False,
        }
    packet['alternative_exit_refinement_candidate']={
        'id':'west_mc_slide_roll_and_bounded_z_for_exit_v1',
        'status':'infeasible_due_to_remaining_outgoing_occlusions',
        'base_candidate':packet['recommended_optimization_candidate']['id'],
        'calibrated_optics_ref':'recommended_optimization_candidate.calibrated_optics_snapshot',
        'fixed_outgoing_directions':True,'fixed_outgoing_lines':False,
        'additional_z_bound_m':candidate['additional_z_shift_bound_m'],
        'maximum_additional_z_shift_m':max(abs(r['alternative_exit_refinement_seed']['additional_z_shift_survey_m']) for r in packet['mirrors']),
        'bundle_mean_z_preserved':candidate['bundle_mean_z_preserved'],
        'full_scene_validated_count':trace['validated_count'],'full_scene_blocked_count':trace['failed_count'],
        'incoming_unavailable_count':trace['unavailable_count'],
        'body_pair_conflicts':0,
        'primary_residuals':{k:v for k,v in residual.items() if k!='records'},
        'bounds_are_measured_tolerances':False,'activation_allowed':False,
    }
    # Quantify deviation from current ideal anchors without asserting an
    # unspecified rod radius or silently moving the survey data back to ideal.
    refs=configuration_ray_references(load_configuration('east_to_west','ideal'))
    offsets=[]
    for rod in range(1,5):
        for w in range(1,5):
            selected=[r for r in packet['mirrors'] if r['survey_bundle_id']==f'{rod}_{w}']
            measured=np.array([r['native_center_m'] for r in selected]).mean(axis=0)
            ideal=np.array([refs[f'{rod}_{5-w}_{j}']['point_m'] for j in range(1,8)]).mean(axis=0)
            offsets.append({'survey_bundle_id':f'{rod}_{w}','native_bundle_id':f'{rod}_{5-w}',
                'manual_minus_ideal_anchor_native_m':(measured-ideal).tolist(),
                'distance_m':float(np.linalg.norm(measured-ideal))})
    packet['mechanical_and_identifiability_limits']={
        'rod_and_holder_envelope':'not_independently_supplied',
        'source_manual_centers_unique_from_optical_lines':False,
        'body_roll_observed':False,
        'reported_residual_is_independent_coordinate_uncertainty':False,
        'historical_registration_covariance':None,
        'maximum_bundle_anchor_deviation_from_current_ideal_m':max(r['distance_m'] for r in offsets),
        'bundle_anchor_deviations':offsets,
        'explanation':'Corrected manual geometry remains10–20mm away from some ideal bundle anchors; a holder/rod envelope or independent rod-pose registration is required before certifying containment. Do not erase this discrepancy by moving primary survey data.'
    }
    args.input.write_text(json.dumps(packet,ensure_ascii=False,indent=2,allow_nan=False))
    with (args.out/'west_exit_refinement_centers_survey_m.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['source_manual_id','matched_survey_id','native_seed_id','x_m','y_m','z_m','normal_x','normal_y','normal_z','additional_z_shift_mm','full_forward_path_status','outgoing_blockers','activation_allowed'])
        for r in packet['mirrors']:
            c=r['alternative_exit_refinement_seed'];v=c['full_scene_validation']
            writer.writerow([r['source_manual_id'],r['matched_primary_survey_id'] or '',r['native_stable_seed_id'],*c['survey_center_m'],*c['survey_reflecting_normal'],1000*c['additional_z_shift_survey_m'],v['status'],';'.join(v['outgoing_blockers'] or []),False])
    print(json.dumps({'alternative':packet['alternative_exit_refinement_candidate'],
        'max_ideal_anchor_offset_mm':1000*packet['mechanical_and_identifiability_limits']['maximum_bundle_anchor_deviation_from_current_ideal_m']},indent=2),flush=True)


if __name__=='__main__':main()
