"""One bounded conditional relaxation, retaining the corrected manual input.

Only along-line centre displacement (<=0.5 mm) and unobserved body roll
(<=5 degrees) are varied. These bounds are working priors, not independently
measured mechanical tolerances. No output line or primary observation moves.
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
    parser.add_argument('--out',type=Path,default=ROOT/'analysis/manual_refine/west')
    args=parser.parse_args();sys.path.insert(0,str(args.repo.resolve()))
    from mirrorlab.engine.bodies import MirrorBody
    from mirrorlab.calibration.physical import _relax_body_poses,rotation_from_normal
    from mirrorlab.state import configuration_from_dataset
    packet=json.loads(args.input.read_text());cal=packet['recommended_optimization_candidate']
    # Start each body from its axial prior, so the5degree bound is relative
    # to the same stated natural pose rather than accumulating multiple fits.
    bodies=[MirrorBody(r['native_center_m'],rotation_from_normal(
        r['recommended_optimization_seed']['native_reflecting_normal'],
        r['recommended_optimization_seed']['incoming']['direction']),r['native_stable_seed_id']) for r in packet['mirrors']]
    config=configuration_from_dataset({'optics_snapshot':cal['calibrated_optics_snapshot'],
        'direction':'east_to_west','status':'diagnostic','mirrors':[b.to_dict() for b in bodies]})
    outgoing={r['native_stable_seed_id']:r['native_target_outgoing'] for r in packet['mirrors']}
    fitted,details=_relax_body_poses(bodies,required_gap_m=1e-6,numerical_tolerance_m=1e-8,
        max_iterations=80,starts=3,max_roll_rad=math.radians(5),line_directions=outgoing,
        axial_sigma_m=.00025,axial_limit_sigma=2.,config=config,
        progress=lambda i,n:print('bounded slide',i,n,flush=True))
    by_id={b.mirror_id:b for b in fitted};Q=np.array(packet['coordinate_policy']['current_native_to_survey_rotation']);t=np.array(packet['coordinate_policy']['current_native_to_survey_translation_m'])
    S=np.diag([1.,-1.,1.]) if np.linalg.det(Q)<0 else np.eye(3)
    local_tips=np.array([[.0005,0,.0005],[0,.0005,0],[-.0005,0,-.0005],[0,-.0005,0]])
    for r in packet['mirrors']:
        b=by_id[r['native_stable_seed_id']]
        world_center=Q@b.center_m+t
        bw=MirrorBody(world_center,Q@b.rotation@S,r['source_manual_id'])
        old=r['recommended_optimization_seed'];r['fixed_center_calibrated_optical_seed']=old
        r['recommended_optimization_seed']={**old,
            'candidate_id':'west_mc_steering_bounded_slide_and_roll_v1',
            'survey_center_m':world_center.tolist(),'native_center_m':b.center_m.tolist(),
            'original_manual_center_retained':bool(np.linalg.norm(b.center_m-r['native_center_m'])<1e-14),
            'center_shift_from_manual_survey_m':(world_center-r['survey_center_m']).tolist(),
            'center_shift_from_manual_norm_m':float(np.linalg.norm(b.center_m-r['native_center_m'])),
            'survey_correct_45deg_face_points_m':(local_tips@bw.rotation.T+bw.center_m).tolist(),
            'native_correct_45deg_face_points_m':(local_tips@b.rotation.T+b.center_m).tolist(),
            'body_native':b.to_dict(),'body_survey':bw.to_dict(),
            'incoming_record_sampled_at_original_center':True,
            'full_scene_forward_reflection_status':'pending_separate_validation',
        }
    packet['fixed_center_optimization_candidate']=cal
    packet['recommended_optimization_candidate']={**cal,
        'id':'west_mc_steering_bounded_slide_and_roll_v1',
        'status':'candidate_pending_full_scene_and_holder_checks',
        'source_manual_centers_fixed':False,'raw_corrected_manual_coordinates_preserved_separately':True,
        'along_line_shift_bound_m':.0005,'along_line_prior_sigma_m':.00025,
        'working_bounds_are_manufacturing_tolerances':False,
        'body_geometry_status':details['status'],
        'remaining_collision_count':len(details['final_geometry']['conflicts']),
        'max_center_shift_m':max(r['recommended_optimization_seed']['center_shift_from_manual_norm_m'] for r in packet['mirrors']),
    }
    outputs={args.input:packet,
        args.out/'west_mc_slide_roll_seed_bodies_native.json':{'format':'mirrorlab.body-seed-list','schema_version':1,'route':'east_to_west','units':'m','bodies':[b.to_dict() for b in fitted],'activation_allowed':False},
        args.out/'west_mc_slide_roll_validation.json':details}
    for path,value in outputs.items():path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'body_status':details['status'],'remaining_conflicts':len(details['final_geometry']['conflicts']),
        'max_shift_mm':1000*packet['recommended_optimization_candidate']['max_center_shift_m'],
        'max_roll_deg':max(abs(a)*180/math.pi for c in details['components'] for a in c['roll_angles_rad'])},indent=2),flush=True)


if __name__=='__main__':main()
