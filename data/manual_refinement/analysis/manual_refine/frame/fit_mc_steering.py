"""Optional common-optics calibration: smallest final-mirror tilts to illuminate fixed West centres.
Keeps source aperture, upstream optics, mirror locations and all manual centers fixed.
Does not validate unknown holders or reflect paths through reconstructed microbodies.
"""
import sys,json,copy,collections
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import argparse
_parser = argparse.ArgumentParser(description="Reproduce manual-reconstruction analysis using an explicit repository and data workspace.")
_parser.add_argument("--repo", required=True, type=Path, help="Complete mirrormirror checkout (pinned analysis commit: 4ab80dfba048e11cf5a3b0bef7dc6832b298540b)")
_parser.add_argument("--workspace", required=True, type=Path, help="Data workspace containing analysis/manual_refine/{west,frame} and documented input files")
_args = _parser.parse_args()
B = _args.workspace.expanduser().resolve()
REPO = _args.repo.expanduser().resolve()
sys.path.insert(0, str(REPO))
from mirrorlab.configurations import load_configuration
from mirrorlab.calibration.physical import incoming_at_points,_field_models,unit,reflecting_normal
from mirrorlab.coordinates import survey_frames
from mirrorlab.state import snapshot,digest
outdir=B/'analysis/manual_refine/frame'
route='east_to_west';config=load_configuration(route,'real');frame=survey_frames()[route]
data=json.loads((B/'analysis/manual_refine/west/west_manual_corrected_survey.json').read_text())
points={r['native_stable_seed_id']:np.asarray(r['native_center_m']) for r in data['mirrors']}
old_models,_=_field_models(config);old_snapshot=snapshot(config);report=[]
for rod in range(1,5):
 mc=next(s for s in config.scene.build_surfaces() if s.name==f'MC{rod}')
 model=old_models[str(rod)][0];m0=np.array(model['map']);d0=np.array(model['direction']);n0=np.array(mc.normal);c0=np.array(mc.center)
 din=d0-2*np.dot(d0,n0)*n0
 helper=np.eye(3)[np.argmin(np.abs(n0))];u=unit(np.cross(n0,helper));v=np.cross(n0,u)
 group={k:p for k,p in points.items() if k.startswith(f'{rod}_')};centers=np.array(list(group.values()))
 def geometry(params):
  n=Rotation.from_rotvec((params[0]*u+params[1]*v)*.001).apply(n0)
  d=unit(din-2*np.dot(din,n)*n)
  projection=np.eye(3)-np.outer(din,n)/(din@n)
  m=m0@projection.T;m[2]+=din*(n@c0)/(din@n)
  return n,d,m
 def launch(params):
  n,d,m=geometry(params);P=np.eye(3)-np.outer(d,d)
  lm=np.linalg.pinv(P@m[:2].T)@P
  return (centers-m[2])@lm.T
 target=.009 # one millimetre source-footprint margin
 def cons(params):return (target-np.linalg.norm(launch(params),axis=1))/.001
 # First find feasible minimum-square footprint; then minimise angular deviation.
 warm=minimize(lambda p:np.mean(np.sum(launch(p)**2,axis=1))/.001**2,[0.,0.],method='BFGS',options={'gtol':1e-9,'maxiter':150})
 fit=minimize(lambda p:float(p@p),warm.x,method='SLSQP',constraints={'type':'ineq','fun':cons},bounds=[(-20,20),(-20,20)],options={'ftol':1e-11,'maxiter':200})
 if min(cons(fit.x)) < -1e-6:raise RuntimeError(f'No feasible 9mm-footprint steering rod{rod}: {fit.message} {min(cons(fit.x))}')
 n,d,m=geometry(fit.x)
 oldref=np.array(mc.in_plane_reference);newref=oldref-(oldref@n)*n
 mc.normal=n;mc.in_plane_reference=newref;mc.__post_init__()
 row=dict(rod=rod,parameter_mrad=fit.x.tolist(),normal_before=n0.tolist(),normal_after=n.tolist(),normal_tilt_mrad=float(np.linalg.norm(fit.x)),outgoing_before_native=d0.tolist(),outgoing_after_native=d.tolist(),outgoing_after_survey=frame.vectors(d).tolist(),outgoing_tilt_mrad=float(np.arctan2(np.linalg.norm(np.cross(d0,d)),d0@d)*1000),source_footprint_radius_before_max_mm=float(np.max(np.linalg.norm(launch([0.,0.]),axis=1))*1000),source_footprint_radius_after_max_mm=float(np.max(np.linalg.norm(launch(fit.x),axis=1))*1000),optimizer_success=bool(fit.success),optimizer_message=str(fit.message),MC_center_fixed_native_m=c0.tolist())
 report.append(row);print(rod,round(row['normal_tilt_mrad'],6),round(row['outgoing_tilt_mrad'],6),row['source_footprint_radius_after_max_mm'],flush=True)
field=incoming_at_points(config,points);new_snapshot=snapshot(config)
records=[]
for r in data['mirrors']:
 key=r['native_stable_seed_id'];inc=field[key];d=np.asarray(inc.get('direction',inc['alternatives'][0]['direction']));dout=np.asarray(r['survey_target_outgoing']);doutnative=frame.rotation.T@dout
 n=reflecting_normal(d,doutnative)
 records.append(dict(source_manual_id=r['source_manual_id'],native_id=key,incoming=inc,incident_survey=frame.vectors(d).tolist(),physical_normal_native=n.tolist(),physical_normal_survey=frame.vectors(n).tolist(),target_outgoing_native=doutnative.tolist(),target_outgoing_survey=dout.tolist(),reflection_residual=float(np.linalg.norm(d-2*(d@n)*n-doutnative))))
packet=dict(format='mirrorlab.manual-optical-calibration-candidate',schema_version=1,status='incident_illumination_verified',route=route,coordinate_frame='survey',notes=['Optional calibrated optical candidate, not an independently measured pose of common MC mirrors.','All manual cut centres fixed; source10mmradius and upstream optics unchanged. Only MC1..MC4 normals tilted.','Full source-to-centre common-optics traversal independently verifies each available incident branch.','Microbody clearance, reflected-path occlusion, rod containment and holder/waist constraints require separate validation.'],maximum_mc_normal_tilt_bound_mrad=20.,required_source_radius_m=.01,target_maximum_launch_radius_m=target,steering=report,field_status_counts=dict(collections.Counter(x['incoming']['status'] for x in records)),display_frame=frame.metadata(),original_optics_snapshot=old_snapshot,original_optics_sha256=digest(old_snapshot),calibrated_optics_snapshot=new_snapshot,calibrated_optics_sha256=digest(new_snapshot),records=records)
(outdir/'west_mc_steering_candidate.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2))
print('fulltrace',packet['field_status_counts'],flush=True)
