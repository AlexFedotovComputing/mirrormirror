"""Independent serialization, frame, reflection and primary-residual checks.

This does not replace the full body/ray tracing reports included in the package.
Run: python validate_delivery.py --package PATH_TO_EXTRACTED_PACKAGE
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def load(path):
    def reject(x):
        raise ValueError(f"Non-finite JSON constant: {x}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def v(x):
    a = np.asarray(x, float)
    assert np.isfinite(a).all()
    return a


def proper(r):
    r = v(r)
    assert r.shape == (3, 3)
    err = max(np.max(np.abs(r.T @ r - np.eye(3))), abs(np.linalg.det(r)-1))
    assert err < 1e-10
    return float(err)


def ellipse(points, center, normal):
    p, c, n = v(points), v(center), v(normal)
    assert p.shape == (4, 3)
    assert np.linalg.norm(p.mean(axis=0)-c) < 1e-10
    assert np.max(np.abs((p-c) @ n)) < 1e-10
    diameters = sorted([np.linalg.norm(p[2]-p[0]), np.linalg.norm(p[3]-p[1])])
    assert np.max(np.abs(v(diameters)-[.001, np.sqrt(2)*.001])) < 1e-10


def rms(a):
    return float(np.sqrt(np.mean(v(a)**2)))


def verify(package):
    w = load(package / "west_manual_corrected_survey.json")
    e = load(package / "east_recovered_centers.json")
    expected = {f"{r}_{b}_{j}" for r in range(1, 5) for b in range(1, 5) for j in range(1, 8)}
    assert len(w["mirrors"]) == len(e["mirrors"]) == 112
    assert {m["source_manual_id"] for m in w["mirrors"]} == expected
    assert {m["source_id"] for m in e["mirrors"]} == expected
    assert w["units"] == e["units"] == "m"
    assert w["injection_side"] == "WEST" and e["side"] == "EAST"
    assert w["route"] == "east_to_west" and e["route"] == "west_to_east"
    wf = w["coordinate_policy"]
    q, t = v(wf["current_native_to_survey_rotation"]), v(wf["current_native_to_survey_translation_m"])
    qh, th = v(wf["historical_rotation"]), v(wf["historical_translation_m"])
    assert abs(np.linalg.det(q)+1) < 1e-10
    j = np.diag([1., -1., 1.])
    west_round, west_source, body_errors, west_reflection = [], [], [], []
    for m in w["mirrors"]:
        c = v(m["survey_center_m"])
        west_round.append(np.linalg.norm(q @ v(m["native_center_m"])+t-c))
        points = (v(m["source_face_points_legacy_m"])-v(m["removed_sector_translation_legacy_m"])) @ qh.T+th
        west_source.append(np.max(np.abs(points-v(m["survey_edge_points_as_transformed_m"]))))
        assert np.linalg.norm(points.mean(axis=0)-c) < 1e-10
        assert m["used_as_primary_measurement"] is False
        ellipse(m["survey_repaired_45deg_ellipse_in_original_plane_m"], c, m["survey_original_plane_normal"])
        for name in ("recommended_optimization_seed", "alternative_exit_refinement_seed"):
            s = m[name]
            rn, rw = v(s["body_native"]["rotation"]), v(s["body_survey"]["rotation"])
            body_errors.extend([proper(rn), proper(rw)])
            assert np.max(np.abs(rw-q @ rn @ j)) < 1e-10
            assert np.linalg.norm(q @ v(s["native_center_m"])+t-v(s["survey_center_m"])) < 1e-10
            assert s["activation_allowed"] is False
            ellipse(s["survey_correct_45deg_face_points_m"], s["survey_center_m"], s["survey_reflecting_normal"])
            di, n, do = v(s["incoming"]["direction"]), v(s["native_reflecting_normal"]), v(m["native_target_outgoing"])
            assert di @ n < 0
            west_reflection.append(np.linalg.norm(di-2*(di@n)*n-do))
    assert max(west_round+west_source+west_reflection) < 1e-9
    matched = [m["matched_primary_survey_id"] for m in w["mirrors"] if m["matched_primary_survey_id"]]
    assert len(matched) == len(set(matched)) == 83

    ef = e["native_to_survey"]
    qe, te = v(ef["rotation"]), v(ef["translation_m"])
    assert abs(np.linalg.det(qe)-1) < 1e-10
    raw = {r["old_id"]:r for r in e["primary_measurement_ledger"]["records"]}
    dist, good_dist, east_round, east_reflection = [], [], [], []
    group_residuals = {}
    for m in e["mirrors"]:
        c, d = v(m["center_survey_m"]), v(m["outgoing_survey"])
        east_round.append(np.linalg.norm(qe @ v(m["center_native_m"])+te-c))
        rn, rw = v(m["rotation_body_to_native"]), v(m["rotation_body_to_survey"])
        body_errors.extend([proper(rn), proper(rw)])
        assert np.max(np.abs(rw-qe@rn)) < 1e-10
        n, di = v(m["normal_survey"]), v(m["incoming_survey"])
        assert di @ n < 0
        assert abs(np.linalg.norm(d)-1) < 1e-10
        east_reflection.append(np.linalg.norm(di-2*(di@n)*n-d))
        ellipse(m["ellipse_vertices_survey_m"], c, n)
        assert m["used_as_primary_measurement"] is False
        r = raw[m["source_id"]]
        if r["complete_points"] == 2:
            points = v(r["screens_m"])
            offset = points-c
            residual = np.linalg.norm(offset-(offset@d)[:,None]*d, axis=1)*1000
            assert np.max(np.abs(residual-v(m["fitted_primary_point_residual_mm"]))) < 1e-7
            dist.extend(residual)
            group_residuals.setdefault(m["survey_group"], []).extend(residual)
            if m["survey_group"] not in {"2_1", "2_2", "3_4"}:
                good_dist.extend(residual)
    assert max(east_round+east_reflection) < 1e-9
    assert len(dist) == 170 and len(good_dist) == 140
    assert abs(rms(dist)-e["summary"]["after_full_points_mm"]["rms"]) < 1e-7
    lookup = {m["source_id"]:m for m in e["mirrors"]}
    for source, slot in zip(range(1, 8), [1, 2, 3, 4, 6, 7, 5]):
        m = lookup[f"3_2_{source}"]
        assert m["canonical_slot"] == slot
    assert all(m["physics"]["full_optical_path"]["status"] == "validated" for m in e["mirrors"])
    result = {
        "status":"passed_serialization_frame_and_algebra_checks",
        "not_a_new_full_ray_trace":True,
        "west_count":112, "east_count":112, "west_unique_primary_mappings":83,
        "west_max_roundtrip_error_m":max(west_round),
        "west_max_exact_point_transform_error_m":max(west_source),
        "east_max_roundtrip_error_m":max(east_round),
        "max_body_SO3_error":max(body_errors),
        "west_max_reflection_vector_error":max(west_reflection),
        "east_max_reflection_vector_error":max(east_reflection),
        "east_recomputed_primary_170_points_rms_mm":rms(dist),
        "east_recomputed_13_groups_140_points_rms_mm":rms(good_dist),
        "east_recomputed_13_groups_140_points_max_mm":float(max(good_dist)),
        "east_primary_rms_by_group_mm":{k:rms(x) for k,x in sorted(group_residuals.items())},
        "east_3_2_internal_slot_5_verified":True,
        "east_reported_full_path_status_counts":dict(Counter(m["physics"]["full_optical_path"]["status"] for m in e["mirrors"])),
    }
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package",required=True,type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    result=verify(a.package)
    if a.output:
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
