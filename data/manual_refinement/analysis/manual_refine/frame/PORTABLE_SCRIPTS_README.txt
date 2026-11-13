PORTABLE ANALYSIS SCRIPTS

These copies replace the original machine-specific repository/workspace paths with:
  --repo ABSOLUTE_PATH_TO_MIRRORMIRROR_CHECKOUT
  --workspace ABSOLUTE_PATH_TO_EXTRACTED_DATA_WORKSPACE

The mathematical calculations and original output formats are unchanged. These are
reproduction/diagnostic scripts, not new production features of mirrormirror.
The analysed commit is 4ab80dfba048e11cf5a3b0bef7dc6832b298540b. A complete checkout
is required, including its static configurations and original survey XLSX files.
Use the Python environment required by that checkout (NumPy, SciPy, openpyxl, etc.).

The workspace must preserve the analysis/manual_refine directory layout. It may be
the root of the extracted handoff package. The scripts write output JSON into
WORKSPACE/analysis/manual_refine/frame; keep a copy of the supplied baseline
results if scientific reruns are required. No calculations were rerun while
making these portable copies; only compilation and --help were checked.

CORE SCRIPTS AND REQUIRED DATA

1. fit_mc_steering.py
   Reads analysis/manual_refine/west/west_manual_corrected_survey.json.
   Uses the checkout's current/common optics and survey_frames().
   Writes analysis/manual_refine/frame/west_mc_steering_candidate.json.
   Preserves fixed manual centres and source radius; fits four MC normal tilts.

2. validate_west_optical_candidate.py
   Reads analysis/manual_refine/frame/west_mc_steering_candidate.json.
   Takes two additional positional arguments: body_file and output_file.
   Those two arguments may be absolute paths or paths relative to the current
   working directory; they are not implicitly relative to --workspace.
   Runs full source-to-body-to-output tracing of the supplied body candidate.

3. relax_west_exit_z.py
   Reads analysis/manual_refine/west/west_mc_slide_roll_seed_bodies_native.json
   and analysis/manual_refine/frame/west_mc_steering_candidate.json.
   Writes west_exit_z_candidate_bodies_native.json,
   west_exit_z_body_validation.json and west_exit_z_full_reflections.json
   into analysis/manual_refine/frame.
   This is the restricted diagnostic Z trial, not a general optimizer.

4. compare_west_refinement.py
   Reads west/west_manual_corrected_survey.json,
   frame/west_mc_steering_candidate.json and
   frame/west_exit_z_candidate_bodies_native.json under analysis/manual_refine.
   Reads primary West survey observations from the checkout.
   Writes frame/west_exit_z_primary_residuals.json.

OPTIONAL EARLIER AUDIT SCRIPT

5. check_mapped_incoming.py
   Also requires the prior-review file:
   WORKSPACE/outputs/manual_reconstruction_review_2026-11-12/review_data/
   west_reference_registration_hypotheses.json
   This earlier reference is not required by the four core scripts above.
   If it is not included in the handoff archive, omit this optional script from
   the run sequence. The prior review-data ZIP contains that file.
   Writes analysis/manual_refine/frame/mapped_incoming.json.

EXAMPLE (run from the extracted workspace root)

python analysis/manual_refine/frame/portable/fit_mc_steering.py --repo /path/to/mirrormirror --workspace /path/to/workspace

python analysis/manual_refine/frame/portable/validate_west_optical_candidate.py --repo /path/to/mirrormirror --workspace /path/to/workspace analysis/manual_refine/west/west_mc_slide_roll_seed_bodies_native.json analysis/manual_refine/frame/west_mc_slide_roll_full_reflections.json

python analysis/manual_refine/frame/portable/relax_west_exit_z.py --repo /path/to/mirrormirror --workspace /path/to/workspace

python analysis/manual_refine/frame/portable/compare_west_refinement.py --repo /path/to/mirrormirror --workspace /path/to/workspace

Each script supports --help. No repository code is edited by these scripts.
The required physical inputs that remain unknown (holder/rod envelopes, real
waist positions, measurement covariances) are not synthesized by these scripts.
