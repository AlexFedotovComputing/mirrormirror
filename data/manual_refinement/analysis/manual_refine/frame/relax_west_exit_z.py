"""Bounded cut-Z shifts to clear both incoming/outgoing rays, preserving rotations.
Exact convex support halfspace witnesses; fullbody and opticaltrace finalchecks.
"""
import sys,json,collections,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import argparse
_parser = argparse.ArgumentParser(description="Reproduce manual-reconstruction analysis using an explicit repository and data workspace.")
_parser.add_argument("--repo", required=True, type=Path, help="Complete mirrormirror checkout (pinned analysis commit: 4ab80dfba048e11cf5a3b0bef7dc6832b298540b)")
_parser.add_argument("--workspace", required=True, type=Path, help="Data workspace containing analysis/manual_refine/{west,frame} and documented input files")
_args = _parser.parse_args()
B = _args.workspace.expanduser().resolve()
REPO = _args.repo.expanduser().resolve()
sys.path.insert(0, str(REPO))
from mirrorlab.calibration.physical import unit,body_pair_clearance,validate_bodies,validate_reflections
from mirrorlab.engine.bodies import MirrorBody
from mirrorlab.state import configuration_from_dataset
outdir=B/'analysis/manual_refine/frame'
base=json.loads((B/'analysis/manual_refine/west/west_mc_slide_roll_seed_bodies_native.json').read_text())
cal=json.loads((outdir/'west_mc_steering_candidate.json').read_text())
bodies=[MirrorBody.from_dict(b) for b in base['bodies']];lookup={r['native_id']:r for r in cal['records']}
config=configuration_from_dataset(dict(optics_snapshot=cal['calibrated_optics_snapshot'],direction=cal['route'],status='partial',mirrors=base['bodies']))
groups=collections.defaultdict(list)
for b in bodies:groups['_'.join(b.mirror_id.split('_')[:2])].append(b)
updated=[];reports=[];epsilon_m=1e-6
for group,bs in sorted(groups.items()):
 n=len(bs);rayterms=[];pairterms=[]
 for i,a in enumerate(bs):
  directions=[np.asarray(lookup[a.mirror_id]['target_outgoing_native']),-np.asarray(lookup[a.mirror_id]['incoming']['direction'])]
  for kind,d in zip(['outgoing','incoming_reverse'],directions):
   d=unit(d);hu=np.eye(3)[np.argmin(abs(d))];u=unit(np.cross(d,hu));v=np.cross(d,u)
   angles=np.linspace(0,2*np.pi,360,endpoint=False);circle=np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v
   for j,b in enumerate(bs):
    if i==j:continue
    candidates=[circle,-d[None,:]]
    for vv in [b.center_m-a.center_m,b.normal,-b.normal,b.axis,-b.axis]:
     vv=np.array(vv);vv=vv-d*max(0.,vv@d)
     if np.linalg.norm(vv)>1e-14:candidates.append(unit(vv)[None,:])
    normals=np.concatenate(candidates);constants=np.array([-b.support(-nn)-a.center_m@nn for nn in normals])*1000
    rayterms.append((i,j,kind,normals[:,2],constants))
 for i,j in itertools.combinations(range(n),2):
  cert=body_pair_clearance(bs[i],bs[j],1e-8,epsilon_m)
  if cert['status']!='separated':raise RuntimeError(f'Baseline notseparated {group} {i} {j}')
  nn=np.asarray(cert['separating_direction']);gap=cert['gap_lower_bound_m']
  pairterms.append((i,j,nn[2],gap*1000))
 def cons(z,gradient=False):
  vals=[];grads=[]
  for i,j,k,nz,c in rayterms:
   arr=c+nz*(z[j]-z[i]);idx=np.argmax(arr);vals.append(arr[idx]-epsilon_m*1000)
   grad=np.zeros(n);grad[j]=nz[idx];grad[i]=-nz[idx];grads.append(grad)
  for i,j,nz,c in pairterms:
   vals.append(c+nz*(z[j]-z[i])-epsilon_m*1000)
   grad=np.zeros(n);grad[j]=nz;grad[i]=-nz;grads.append(grad)
  return np.array(grads) if gradient else np.array(vals)
 outcomes=[]
 for start in range(3):
  z0=np.zeros(n) if start==0 else .15*np.sin(np.arange(n)*2.4+start);z0-=z0.mean()
  fit=minimize(lambda z:float(z@z),z0,jac=lambda z:2*z,method='SLSQP',bounds=[(-.5,.5)]*n,
   constraints=[{'type':'ineq','fun':cons,'jac':lambda z:cons(z,True)},{'type':'eq','fun':lambda z:z.sum(),'jac':lambda z:np.ones(n)}],options={'ftol':1e-12,'maxiter':300})
  outcomes.append((float(min(cons(fit.x))),float(fit.x@fit.x),fit))
  if outcomes[-1][0]>-1e-7:break
 feasible=[r for r in outcomes if r[0]>-1e-7]
 chosen=min(feasible,key=lambda r:r[1]) if feasible else max(outcomes,key=lambda r:r[0])
 margin,cost,fit=chosen;accepted=bool(feasible)
 zz=fit.x if accepted else np.zeros(n)
 for b,dz in zip(bs,zz):updated.append(MirrorBody(b.center_m+np.array([0,0,dz/1000]),b.rotation,b.mirror_id,b.radius_m,b.axial_height_m,b.length_definition))
 report=dict(bundle_native_id=group,status='support_certificates_passed' if accepted else 'no_solution_within_bounds',z_shifts_native_mm=zz.tolist(),candidate_z_shifts_native_mm=fit.x.tolist(),minimum_constraint_margin_mm=margin,optimizer_success=bool(fit.success),optimizer_message=str(fit.message),max_abs_shift_mm=float(max(abs(zz))),ray_constraints=len(rayterms),body_constraints=len(pairterms))
 reports.append(report);print(group,report['status'],'maxmm',report['max_abs_shift_mm'],'margin',margin,flush=True)
packet=dict(format='mirrorlab.manual-body-candidate',schema_version=1,route=cal['route'],units='m',coordinate_frame='native',source_variant='west_mc_slide_roll_seed_bodies_native',body_parameters_unchanged=True,additional_z_shift_bound_m=.0005,bundle_mean_z_preserved=True,required_ray_clearance_m=epsilon_m,notes=['Individual additionalcut-Z shifts preserve normals/outgoingdirections buttranslateoutgoinglines.','Bound is an explicit numerical prior, not experimentally established mechanical tolerance.','Original transformedmanualcoordinates retained separately.'],groups=reports,bodies=[b.to_dict() for b in updated],activation_allowed=False)
(outdir/'west_exit_z_candidate_bodies_native.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2))
check=validate_bodies(updated,epsilon_m,1e-8,check_length_alternative=False)
(outdir/'west_exit_z_body_validation.json').write_text(json.dumps(check,ensure_ascii=False,indent=2))
print('bodyconflicts',len(check['conflicts']),flush=True)
check=validate_reflections(config,updated)
(outdir/'west_exit_z_full_reflections.json').write_text(json.dumps(check,ensure_ascii=False,indent=2))
print('optics',{k:check[k] for k in ['validated_count','failed_count','unavailable_count']},flush=True)
