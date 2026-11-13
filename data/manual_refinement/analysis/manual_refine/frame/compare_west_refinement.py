import sys,json,collections
from pathlib import Path
import numpy as np
import argparse
_parser = argparse.ArgumentParser(description="Reproduce manual-reconstruction analysis using an explicit repository and data workspace.")
_parser.add_argument("--repo", required=True, type=Path, help="Complete mirrormirror checkout (pinned analysis commit: 4ab80dfba048e11cf5a3b0bef7dc6832b298540b)")
_parser.add_argument("--workspace", required=True, type=Path, help="Data workspace containing analysis/manual_refine/{west,frame} and documented input files")
_args = _parser.parse_args()
B = _args.workspace.expanduser().resolve()
REPO = _args.repo.expanduser().resolve()
sys.path.insert(0, str(REPO))
from mirrorlab.calibration.io import read_survey,FILES
p=json.loads((B/'analysis/manual_refine/west/west_manual_corrected_survey.json').read_text())
cal=json.loads((B/'analysis/manual_refine/frame/west_mc_steering_candidate.json').read_text());fr=cal['display_frame'];Q=np.array(fr['rotation']);t=np.array(fr['translation_m'])
variant=json.loads((B/'analysis/manual_refine/frame/west_exit_z_candidate_bodies_native.json').read_text());bs={b['mirror_id']:b for b in variant['bodies']}
rows,_=read_survey(REPO/FILES['east_to_west']);obs={r.id:r for r in rows}
records=[]
for m in p['mirrors']:
 source=m['matched_primary_survey_id']
 if source is None:continue
 measured=np.array(obs[source].points);d=np.array(m['survey_target_outgoing']);original=np.array(m['survey_center_m']);new=Q@np.array(bs[m['native_stable_seed_id']]['center_m'])+t
 def dist(c):v=measured-c;return np.linalg.norm(v-np.outer(v@d,d),axis=1)*1000
 a=dist(original);b=dist(new)
 records.append(dict(manual_id=m['source_manual_id'],survey_id=source,rms_before_mm=float(np.sqrt(np.mean(a*a))),rms_after_mm=float(np.sqrt(np.mean(b*b))),point_distances_before_mm=a.tolist(),point_distances_after_mm=b.tolist(),centre_shift_survey_mm=((new-original)*1000).tolist()))
a=np.array([r['point_distances_before_mm'] for r in records]);b=np.array([r['point_distances_after_mm'] for r in records])
report=dict(n=len(records),measurement_points=int(a.size),method='Unchanged directed target ray with changed line origin; primary two-point perpendicular distances. Not independentaccuracy.',rms_before_mm=float(np.sqrt(np.mean(a*a))),rms_after_mm=float(np.sqrt(np.mean(b*b))),max_line_rms_before_mm=max(r['rms_before_mm'] for r in records),max_line_rms_after_mm=max(r['rms_after_mm'] for r in records),records=records)
(B/'analysis/manual_refine/frame/west_exit_z_primary_residuals.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print({k:v for k,v in report.items() if k!='records'})
