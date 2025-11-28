from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

import reconstruction_case as rc


Array = np.ndarray


def _world_geometry(bundle: rc.BundleConfig, R: Array, t: Array) -> Tuple[Array, Array]:
    normals = []
    centers = []
    for m in bundle.mirrors or []:
        n_loc, c_loc = rc.mirror_geometry(m)
        normals.append(n_loc)
        centers.append(c_loc)
    normals = np.stack(normals)
    centers = np.stack(centers)
    normals_w = (R @ normals.T).T
    centers_w = (R @ centers.T).T + t
    return normals_w, centers_w


def _observed_rays(rays: Iterable[Tuple[Array, Array]]) -> Tuple[Array, Array]:
    dirs = []
    anchors = []
    for p0, p1 in rays:
        d = p1 - p0
        d = d / np.linalg.norm(d)
        dirs.append(d)
        anchors.append(p0)
    return np.stack(dirs), np.stack(anchors)


def compute_pair_metrics(
    bundle: rc.BundleConfig,
    rays,
    R: Array,
    t: Array,
    match: Iterable[Tuple[int, int]],
):
    """Compute per-pair angular and positional errors for a reconstructed pose."""
    normals_w, centers_w = _world_geometry(bundle, R, t)
    d_obs, anchors = _observed_rays(rays)

    results = []
    for mi, rj in match:
        n = normals_w[mi]
        c = centers_w[mi]
        d = d_obs[rj]
        a = anchors[rj]

        d_pred = rc.D_IN - 2 * np.dot(n, rc.D_IN) * n
        d_pred = d_pred / np.linalg.norm(d_pred)
        if np.dot(d_pred, d) < 0:
            d = -d

        cos_ang = float(np.clip(np.dot(d_pred, d), -1.0, 1.0))
        ang_deg = float(np.degrees(np.arccos(cos_ang)))

        dist = rc.distance_point_to_line(c, a, d)

        rel = a - c
        along = float(np.dot(rel, d))

        s = np.dot(d, c - a)
        p_closest = a + s * d

        results.append(
            {
                "mirror_index": int(mi),
                "ray_index": int(rj),
                "angle_deg": ang_deg,
                "distance": dist,
                "along": along,
                "hit_point": p_closest,
                "center_world": c,
            }
        )
    return results


def summarize_metrics(per_pair: List[dict]) -> dict:
    if not per_pair:
        return {}
    angles = np.array([p["angle_deg"] for p in per_pair], dtype=float)
    dists = np.array([p["distance"] for p in per_pair], dtype=float)
    along = np.array([p["along"] for p in per_pair], dtype=float)
    return {
        "mean_angle_deg": float(angles.mean()),
        "max_angle_deg": float(angles.max()),
        "mean_distance": float(dists.mean()),
        "max_distance": float(dists.max()),
        "min_along": float(along.min()),
    }


def evaluate_real_case(
    bundle_path: Path | str = rc.BUNDLE_CONFIG,
    rays_path: Path | str = rc.RAYS_CONFIG,
):
    bundle = rc.load_bundle_config(Path(bundle_path))
    rays = rc.load_rays_config(Path(rays_path))
    total, match, R, t = rc.solve_pose(
        bundle, rays, w_ang=1.0, w_pos=1.0, w_div=10.0
    )
    per_pair = compute_pair_metrics(bundle, rays, R, t, match)
    summary = summarize_metrics(per_pair)
    return {
        "total_cost": float(total),
        "summary": summary,
        "pairs": per_pair,
        "R": R,
        "t": t,
        "match": match,
    }


def _format_summary(summary: dict) -> str:
    return (
        f"  mean angle (deg): {summary['mean_angle_deg']:.3f}\n"
        f"  max angle (deg):  {summary['max_angle_deg']:.3f}\n"
        f"  mean distance:    {summary['mean_distance']:.3f}\n"
        f"  max distance:     {summary['max_distance']:.3f}\n"
        f"  min along:        {summary['min_along']:.3f}"
    )


def main() -> None:
    result = evaluate_real_case()
    print("Real data reconstruction metrics:")
    print(f"  total cost:       {result['total_cost']:.6f}")
    print(_format_summary(result["summary"]))


if __name__ == "__main__":
    main()
