import json,sys,collections,argparse
from pathlib import Path
import argparse
_parser = argparse.ArgumentParser(description="Reproduce manual-reconstruction analysis using an explicit repository and data workspace.")
_parser.add_argument("--repo", required=True, type=Path, help="Complete mirrormirror checkout (pinned analysis commit: 4ab80dfba048e11cf5a3b0bef7dc6832b298540b)")
_parser.add_argument("--workspace", required=True, type=Path, help="Data workspace containing analysis/manual_refine/{west,frame} and documented input files")
_parser.add_argument("body_file", help="Native body JSON; absolute path or relative to current working directory")
_parser.add_argument("output_file", help="Output JSON; absolute path or relative to current working directory")
_args = _parser.parse_args()
B = _args.workspace.expanduser().resolve()
REPO = _args.repo.expanduser().resolve()
sys.path.insert(0, str(REPO))
from mirrorlab.state import configuration_from_dataset
from mirrorlab.calibration.physical import validate_reflections
a = _args
opt=json.loads((B/'analysis/manual_refine/frame/west_mc_steering_candidate.json').read_text())
bs=json.loads(Path(a.body_file).read_text());bodies=bs.get('bodies',bs.get('mirrors',[]))
c=configuration_from_dataset(dict(optics_snapshot=opt['calibrated_optics_snapshot'],direction=opt['route'],status='partial',mirrors=bodies))
r=validate_reflections(c,bodies)
Path(a.output_file).write_text(json.dumps(r,ensure_ascii=False,indent=2))
print({k:v for k,v in r.items() if k not in ['mirrors','trace_diagnostics']})
print('reasons',collections.Counter(x.get('reason','optical_check') for x in r['mirrors'].values() if x['status']!='validated'))
print('failures',[(k,v.get('first_surface'),v.get('outgoing_blockers'),v.get('incident_dot_normal'),v.get('hit_on_ellipse')) for k,v in r['mirrors'].items() if v['status']!='validated'])
