"""Attach full-scene validation to the line-preserving WEST seed.

This retains failed outgoing paths instead of certifying the candidate merely
because all solids are disjoint and every incoming ray reaches its cut.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'analysis/manual_refine/west/west_manual_corrected_survey.json')
    parser.add_argument('--full-trace-report',type=Path,default=ROOT/'analysis/manual_refine/frame/west_mc_slide_roll_full_reflections.json')
    parser.add_argument('--out',type=Path,default=ROOT/'analysis/manual_refine/west')
    args=parser.parse_args();packet=json.loads(args.input.read_text());report=json.loads(args.full_trace_report.read_text())
    metadata=packet['recommended_optimization_candidate']
    metadata['status']='infeasible_due_to_outgoing_occlusions' if report['failed_count'] else 'partial_pending_independent_mechanical_constraints'
    metadata['full_scene_validation']={k:report[k] for k in ['validated_count','failed_count','unavailable_count','trace_settings','trace_diagnostics','method']}
    metadata['activation_allowed']=False
    for rec in packet['mirrors']:
        item=report['mirrors'][rec['native_stable_seed_id']]
        seed=rec['recommended_optimization_seed']
        seed['incoming']=item['incoming']
        seed['incoming_record_sampled_at_original_center']=False
        seed['full_scene_forward_reflection_status']=item['status']
        seed['full_scene_validation']={k:item.get(k) for k in ['status','waist_side','actual_reflected_child','incident_dot_normal','outgoing_dot_normal','hit_m','hit_on_ellipse','outgoing_direction','outgoing_blockers','next_surfaces']}
        seed['activation_allowed']=False
    packet['validation']['recommended_seed_full_scene']={
        'correct_cut_reflections':sum(bool(r.get('actual_reflected_child')) and bool(r.get('hit_on_ellipse')) for r in report['mirrors'].values()),
        'forward_paths_unblocked':report['validated_count'],'forward_paths_blocked':report['failed_count'],
        'incoming_unavailable':report['unavailable_count'],
        'body_pair_conflicts':metadata['remaining_collision_count'],
        'holder_and_waist_status':'not_evaluated',
        'fully_feasible_configuration':False,
    }
    note='The line-preserving MC+body candidate has112 correct cut reflections but35 rays blocked by neighbouring bodies; collision-free solids alone do not validate the optical system.'
    if note not in packet['limitations']:packet['limitations'].append(note)
    args.input.write_text(json.dumps(packet,ensure_ascii=False,indent=2,allow_nan=False))
    for frame in ['survey','native']:
        path=args.out/f'west_refined_seed_centers_{frame}_m.csv'
        with path.open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.writer(stream)
            writer.writerow(['source_manual_id','matched_survey_id','native_stable_seed_id','x_m','y_m','z_m','coated_normal_x','coated_normal_y','coated_normal_z','shift_from_exact_manual_um','incoming_status','full_forward_path_status','outgoing_blockers','activation_allowed'])
            for rec in packet['mirrors']:
                seed=rec['recommended_optimization_seed'];v=seed['full_scene_validation']
                writer.writerow([rec['source_manual_id'],rec['matched_primary_survey_id'] or '',rec['native_stable_seed_id'],
                    *seed[f'{frame}_center_m'],*seed[f'{frame}_reflecting_normal'],
                    seed['center_shift_from_manual_norm_m']*1e6,seed['incoming']['status'],v['status'],
                    ';'.join(v['outgoing_blockers'] or []),False])
    print(json.dumps(packet['validation']['recommended_seed_full_scene'],indent=2))


if __name__=='__main__':main()
