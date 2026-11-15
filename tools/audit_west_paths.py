"""Read-only numerical audit of saved and manual WEST paths.

Writes evidence under artifacts/expert-audit; does not modify model inputs.
The independent intersection check clips a ray against a cylinder and two
half-spaces, rather than calling the engine's surface intersection routines.
The aperture experiment is a feasibility sample, NOT transmitted power.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mirrorlab.calibration.physical import (
    _launch_bundle, _records, _record_indexes, _trace, incoming_at_points,
    validate_reflections,
)
from mirrorlab.calibration.trajectory_matching import (
    _pair_metrics, dataset_comparison_report, traced_output_trajectories,
)
from mirrorlab.calibration.manual_rays import solid_entry, first_body, aperture_samples, full_sample_trace, one_to_one_aperture
from mirrorlab.engine.bodies import MirrorBody
from mirrorlab.manual_schema import import_manual
from mirrorlab.state import configuration_from_dataset

OUT = ROOT / "artifacts/expert-audit"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def plot_aperture(label, body, bodies, report, sample_report, source_manual_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points = aperture_samples(body)
    outgoing, _ = first_body(points, report["outgoing_direction"], bodies, body.mirror_id)
    backward, _ = first_body(points, -np.array(report["incoming"]["direction"]), bodies, body.mirror_id)
    upstream = np.linalg.norm(body.center_m-np.array(report["incoming"]["field_origin_m"]))
    clear = (outgoing > 3.) & (backward > upstream)
    local = body.local(points)
    x, y = local[:,0]*np.sqrt(2)*1000, local[:,1]*1000
    fig,ax = plt.subplots(figsize=(7.4,5.9),layout="constrained")
    ax.scatter(x[~clear],y[~clear],s=15,c="#c76060",label="Локально перекрыто")
    ax.scatter(x[clear],y[clear],s=15,c="#299b8f",label="Локально свободно")
    ax.scatter([0],[0],s=110,c="black",marker="x",linewidths=2.2,label="Центральный луч перекрыт")
    sample = next(s for s in sample_report["details"] if s.get("reached") and s["mirror_id"]==body.mirror_id)
    loc = body.local(np.array(sample["hit_m"]))
    ax.scatter([loc[0]*np.sqrt(2)*1000],[loc[1]*1000],s=200,c="#f2be35",edgecolors="black",marker="*",
               label="Проверен полной трассировкой",zorder=5)
    ax.set(xlabel="Координата на площадке, мм",ylabel="Координата на площадке, мм",
           title=f"WEST: ручное зеркало {source_manual_id}\nЦентр закрыт, часть площадки пропускает лучи",aspect="equal")
    ax.grid(alpha=.15)
    ax.legend(fontsize=9,loc="upper center",bbox_to_anchor=(.5,-.16),ncol=2)
    fig.savefig(OUT/(label+"-aperture.png"),dpi=170)
    plt.close(fig)


def audit(label, dataset):
    print("audit", label, flush=True)
    comparison = dataset_comparison_report(dataset)["diagnostics"]
    bodies = [MirrorBody.from_dict(m) for m in dataset["mirrors"]]
    config = configuration_from_dataset(dataset)
    original = dataset["diagnostics"]["reflections"]
    retraced = validate_reflections(config, bodies)
    write(label+"-retraced.json", retraced)
    original_status = {k: v["status"] for k, v in original["mirrors"].items()}
    retraced_status = {k: v["status"] for k, v in retraced["mirrors"].items()}
    failures, samples = [], []
    for body in bodies:
        report = retraced["mirrors"][body.mirror_id]
        if not report.get("actual_reflected_child"):
            continue
        direction = np.array(report["outgoing_direction"])
        distance, blocker = first_body(body.center_m[None,:], direction, bodies, body.mirror_id)
        if not np.isfinite(distance[0]):
            continue
        points = aperture_samples(body)
        outgoing, _ = first_body(points, direction, bodies, body.mirror_id)
        backward, _ = first_body(points, -np.array(report["incoming"]["direction"]), bodies, body.mirror_id)
        upstream_length = np.linalg.norm(body.center_m-np.array(report["incoming"]["field_origin_m"]))
        clear = (outgoing > 3.) & (backward > upstream_length)
        saved_segment = report["reflected_trajectories"][0]["segments"][0]
        traced_distance = np.linalg.norm(np.array(saved_segment["end_m"])-body.center_m)
        failures.append(dict(mirror_id=body.mirror_id, blocker_id=blocker[0],
            center_distance_mm=float(distance[0]*1000),
            traced_distance_mm=float(traced_distance*1000),
            distance_difference_m=float(abs(traced_distance-distance[0])),
            sampled_clear_count=int(clear.sum()), sampled_total=len(points),
            local_test="incoming from MC and outgoing for 3 m against all other solid bodies"))
        eligible = np.where(clear)[0]
        eligible = sorted(eligible, key=lambda i: np.linalg.norm(points[i]-body.center_m))[:3]
        for j,i in enumerate(eligible):
            samples.append(dict(key=body.mirror_id+f"_sample{j}", mirror_id=body.mirror_id, point=points[i]))
    sample_report = full_sample_trace(config, samples, bodies, dataset["display_frame"])
    observations = dataset["observations"]
    if isinstance(observations, dict):
        observations = observations["records"]
    # A sample ray has a separate trajectory, so also report unique covered
    # observations using the ORIGINAL fitted mirror assignment. Do not claim
    # 3 samples from one mirror are 3 independently recovered channels.
    obs_by_id = {r["id"]: r for r in observations}
    fit_by_obs = {p["observation_id"]: p for p in comparison["geometric_fit"].get("matches", [])}
    fit_mirrors = {p["id"]: p["mirror_id"] for p in comparison["geometric_fit"].get("trajectories", [])}
    rescued = []
    for row in comparison["trajectory_diagnostics"]["observations"]:
        if row["status"] != "unmatched" or row["observation_id"] not in fit_by_obs:
            continue
        identifier = row["observation_id"]
        body_id = fit_mirrors[fit_by_obs[identifier]["trajectory_id"]]
        candidates = [p for p in sample_report["paths"] if p["mirror_id"] == body_id]
        metrics = [_pair_metrics(obs_by_id[identifier], np.asarray(obs_by_id[identifier]["screens_m"]), p) for p in candidates]
        best = min(metrics, key=lambda m: m["maximum_distance_m"], default=None)
        if best and best["maximum_distance_m"] <= .003:
            rescued.append(dict(observation_id=identifier, mirror_id=body_id, **{k:v for k,v in best.items() if k != "observation_id"}))
    write(label+"-comparison.json", comparison)
    write(label+"-aperture.json", dict(blocked=failures, full_trace=sample_report, rescued_observations=rescued))
    one_to_one = one_to_one_aperture(comparison, sample_report["paths"])
    write(label+"-one-to-one-aperture.json", one_to_one)
    if label == "manual-west" and rescued:
        eligible = [f for f in failures if f["mirror_id"] in {r["mirror_id"] for r in rescued}]
        example = min(eligible, key=lambda f:abs(f["sampled_clear_count"]-f["sampled_total"]/2))
        body = next(b for b in bodies if b.mirror_id==example["mirror_id"])
        mirror = next(m for m in dataset["mirrors"] if m["mirror_id"]==body.mirror_id)
        plot_aperture(label, body, bodies, retraced["mirrors"][body.mirror_id], sample_report,
                      mirror.get("source_manual_id",body.mirror_id))
    summary = dict(
        complete_observations=comparison["geometric_fit"].get("observation_count"),
        fitted_within_3mm=comparison["geometric_fit"].get("matched_count"),
        finite_paths_within_3mm=comparison["trajectory_matching"].get("matched_count"),
        reasons=comparison["trajectory_diagnostics"]["reason_counts"],
        retraced_validated=retraced["validated_count"],
        retraced_status_changes=[k for k in original_status if original_status[k] != retraced_status[k]],
        independently_blocked=len(failures),
        independent_distance_max_difference_m=max((f["distance_difference_m"] for f in failures),default=None),
        locally_clear_offcenter_mirrors=sum(f["sampled_clear_count"] > 0 for f in failures),
        sampled_sources_tested=sample_report["tested"],
        sampled_sources_illuminated=sample_report["illuminated"],
        sampled_sources_reached=sample_report["reached_cut"],
        rescued_observation_count=len(rescued),
        rescued_mirrors=sorted({r["mirror_id"] for r in rescued}),
        one_to_one_aperture_counts={k:v["count"] for k,v in one_to_one.items()},
        aperture_interpretation="existence of sampled rays only; not integrated power or all-aperture proof",
    )
    print(json.dumps({label:summary}), flush=True)
    return summary


def main():
    saved_path = "mirrorlab/saved restored/mirrorlab-configuration-both1.json"
    manual_path = "data/manual_refinement/west_manual_corrected_survey.json"
    saved = read(saved_path)["datasets"]["east_to_west"]
    manual = import_manual(ROOT/manual_path)["datasets"]["east_to_west"]
    raw = manual["manual_references"]["payload"]
    manual["optics_snapshot"] = raw["recommended_optimization_candidate"]["calibrated_optics_snapshot"]
    manual["optics_sha256"] = raw["recommended_optimization_candidate"]["calibrated_optics_sha256"]
    for m,r in zip(manual["mirrors"],raw["mirrors"]):
        m.update(copy.deepcopy(r["recommended_optimization_seed"]["body_native"]))
        # Explicit adapter for this audit: current UI comparison expects these
        # fields, but the manual import/refine path stores different names.
        m["outgoing_direction"] = m["target_outgoing_native"]
    manual["diagnostics"]["reflections"] = read("data/manual_refinement/analysis/manual_refine/frame/west_mc_slide_roll_full_reflections.json")
    summaries = {label:audit(label,d) for label,d in (("saved-west",saved),("manual-west",manual))}
    summaries["input_sha256"] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (saved_path,manual_path)}
    write("summary.json",summaries)


if __name__ == "__main__":
    main()
