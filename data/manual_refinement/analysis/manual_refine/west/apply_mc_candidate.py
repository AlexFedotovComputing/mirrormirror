"""Attach optional MC steering and bounded body-roll seed to WEST input JSON.

Positions and outgoing lines stay fixed. The candidate is never activated.
The selected calibration snapshot and all diagnostics are embedded.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=ROOT/'analysis/manual_review/repo')
    parser.add_argument('--input',type=Path,default=ROOT/'analysis/manual_refine/west/west_manual_corrected_survey.json')
    parser.add_argument('--optical-candidate',type=Path,default=ROOT/'analysis/manual_refine/frame/west_mc_steering_candidate.json')
    parser.add_argument('--out',type=Path,default=ROOT/'analysis/manual_refine/west')
    args=parser.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    from mirrorlab.engine.bodies import MirrorBody
    from mirrorlab.calibration.physical import rotation_from_normal,validate_bodies,relax_body_rolls
    packet=json.loads(args.input.read_text());cal=json.loads(args.optical_candidate.read_text())
    records={r['source_manual_id']:r for r in cal['records']}
    bodies=[]
    for r in packet['mirrors']:
        c=records[r['source_manual_id']]
        assert np.linalg.norm(np.array(c['incoming']['point_m'])-r['native_center_m'])<1e-12
        bodies.append(MirrorBody(r['native_center_m'],rotation_from_normal(c['physical_normal_native'],c['incoming']['direction']),r['native_stable_seed_id']))
    baseline=validate_bodies(bodies,required_gap_m=1e-6,numerical_tolerance_m=1e-8)
    print('MC baseline conflicts',len(baseline['conflicts']),flush=True)
    fitted,details=relax_body_rolls(bodies,required_gap_m=1e-6,numerical_tolerance_m=1e-8,
        max_iterations=60,starts=3,max_roll_rad=math.radians(5),
        progress=lambda i,n:print('roll',i,n,flush=True))
    by_id={b.mirror_id:b for b in fitted};Q=np.array(packet['coordinate_policy']['current_native_to_survey_rotation'])
    S=np.diag([1.,-1.,1.]) if np.linalg.det(Q)<0 else np.eye(3)
    local_tips=np.array([[.0005,0,.0005],[0,.0005,0],[-.0005,0,-.0005],[0,-.0005,0]])
    for r in packet['mirrors']:
        c=records[r['source_manual_id']];b=by_id[r['native_stable_seed_id']]
        bw=MirrorBody(r['survey_center_m'],Q@b.rotation@S,r['source_manual_id'])
        r['recommended_optimization_seed']={
            'candidate_id':'west_minimal_mc_steering_and_fixed_center_roll_v1',
            'normal_semantics':'current physical incident branch; historic opposite-incident convention corrected',
            'survey_center_m':r['survey_center_m'],'native_center_m':r['native_center_m'],
            'original_manual_center_retained':True,
            'incoming':c['incoming'],
            'survey_incident_direction':c['incident_survey'],
            'survey_reflecting_normal':bw.normal.tolist(),
            'native_reflecting_normal':b.normal.tolist(),
            'survey_correct_45deg_face_points_m':(local_tips@bw.rotation.T+bw.center_m).tolist(),
            'native_correct_45deg_face_points_m':(local_tips@b.rotation.T+b.center_m).tolist(),
            'body_native':b.to_dict(),'body_survey':bw.to_dict(),
            'reflection_vector_error':c['reflection_residual'],
            'illumination_status':c['incoming']['status'],
            'full_scene_forward_reflection_status':'pending_separate_validation',
            'activation_allowed':False,
        }
    packet['recommended_optimization_candidate']={
        'id':'west_minimal_mc_steering_and_fixed_center_roll_v1',
        'status':'candidate_pending_full_scene_and_holder_checks',
        'source_manual_centers_fixed':True,'desired_outgoing_lines_fixed':True,
        'coordinate_frame_fixed':True,'source_and_rod_locations_fixed':True,
        'MC_normal_adjustment_bound_mrad':cal['maximum_mc_normal_tilt_bound_mrad'],
        'MC_steering':cal['steering'],
        'calibrated_optics_snapshot':cal['calibrated_optics_snapshot'],
        'calibrated_optics_sha256':cal['calibrated_optics_sha256'],
        'current_unadjusted_incidence_status_counts':packet['validation']['incidence_status_counts'],
        'calibrated_incidence_status_counts':cal['field_status_counts'],
        'body_roll_limit_deg':5.,'requested_body_gap_m':1e-6,
        'body_geometry_status':details['status'],
        'baseline_collision_count':len(baseline['conflicts']),
        'remaining_collision_count':len(details['final_geometry']['conflicts']),
        'source_calibration_is_independently_measured':False,
        'activation_allowed':False,
    }
    outputs={
        args.input:packet,
        args.out/'west_mc_roll_seed_bodies_native.json':{'format':'mirrorlab.body-seed-list','schema_version':1,'route':'east_to_west','units':'m','bodies':[b.to_dict() for b in fitted],'activation_allowed':False},
        args.out/'west_mc_roll_validation.json':{'baseline':baseline,'roll_relaxation':details},
    }
    for path,value in outputs.items():path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'body_status':details['status'],'baseline_conflicts':len(baseline['conflicts']),
        'remaining_conflicts':len(details['final_geometry']['conflicts']),
        'max_roll_deg':max(abs(a)*180/math.pi for c in details['components'] for a in c['roll_angles_rad'])},indent=2),flush=True)


if __name__=='__main__':main()
