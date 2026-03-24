from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import itertools as it

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation


BUNDLE_CONFIG = Path("bundle_config.txt")
RAYS_CONFIG = Path("rays_config.txt")

# Incoming ray: along -Z, hits the center of each mirror (toward -Z)
D_IN = np.array([0.0, 0.0, -1.0], dtype=float)


@dataclass
class Mirror:
    name: str
    cut_deg: float
    tilt_deg: float
    height: float
    x: float
    y: float


@dataclass
class BundleConfig:
    fiber_diameter: float = 1.0
    pitch: float = 1.4
    label_offset: float = 0.6
    mirrors: List[Mirror] | None = None


def _parse_float(value: str, field: str, raw: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Failed to read {field} in line: {raw}") from exc


def load_bundle_config(path: Path) -> BundleConfig:
    mirrors: List[Mirror] = []
    globals_cfg = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        clean = raw.split("#", 1)[0].strip()
        if not clean:
            continue
        if clean.startswith("mirror"):
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            params = {}
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k.strip()] = v.strip()
            name = params.get("mirror") or params.get("name")
            cut = _parse_float(params.get("cut", ""), "cut", raw)
            tilt = _parse_float(params.get("tilt", "0"), "tilt", raw)
            height = _parse_float(params.get("height", "0"), "height", raw)
            x = _parse_float(params.get("x", "0"), "x", raw)
            y = _parse_float(params.get("y", "0"), "y", raw)
            mirrors.append(
                Mirror(
                    name=name,
                    cut_deg=cut,
                    tilt_deg=tilt,
                    height=height,
                    x=x,
                    y=y,
                )
            )
        else:
            if "=" not in clean:
                continue
            k, v = clean.split("=", 1)
            globals_cfg[k.strip()] = float(v.strip())
    return BundleConfig(
        fiber_diameter=float(globals_cfg.get("fiber_diameter", 1.0)),
        pitch=float(globals_cfg.get("pitch", globals_cfg.get("pitch_mm", 1.4))),
        label_offset=float(globals_cfg.get("label_offset", 0.6)),
        mirrors=mirrors,
    )


def load_rays_config(path: Path):
    rays = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        clean = raw.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = [p.strip() for p in clean.split(",") if p.strip()]
        p0 = p1 = None
        for p in parts:
            if p.startswith("p0="):
                vals = p.split("=", 1)[1].strip().split()
                p0 = np.array([float(v) for v in vals], dtype=float)
            if p.startswith("p1="):
                vals = p.split("=", 1)[1].strip().split()
                p1 = np.array([float(v) for v in vals], dtype=float)
        if p0 is None or p1 is None:
            raise ValueError(f"Missing p0/p1 in line: {raw}")
        rays.append((p0, p1))
    return rays


def mirror_geometry(m: Mirror) -> Tuple[np.ndarray, np.ndarray]:
    cut_rad = np.deg2rad(m.cut_deg)
    tilt_rad = np.deg2rad(m.tilt_deg)
    n_local = np.array(
        [
            np.sin(cut_rad) * np.cos(tilt_rad),
            np.sin(cut_rad) * np.sin(tilt_rad),
            np.cos(cut_rad),
        ],
        dtype=float,
    )
    c_local = np.array([m.x, m.y, m.height], dtype=float)
    # Orient normal so that its projection in XY points outward from the bundle center.
    if np.linalg.norm(c_local[:2]) > 1e-6:
        n_xy = n_local[:2]
        c_xy = c_local[:2]
        if np.dot(n_xy, c_xy) < 0:
            n_local = -n_local
    else:
        # For the central mirror (no XY radius), fall back to a convention
        # that the reflective side faces the incoming beam (dot(D_IN, n) < 0).
        if np.dot(D_IN, n_local) > 0:
            n_local = -n_local
    return n_local / np.linalg.norm(n_local), c_local


def kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Accept either (N,3) or (3,N) inputs
    if A.shape[1] == 3 and B.shape[1] == 3:
        H = A.T @ B
    else:
        H = A @ B.T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def distance_point_to_line(p: np.ndarray, a: np.ndarray, d: np.ndarray) -> float:
    diff = p - a
    proj = np.dot(diff, d) * d
    return float(np.linalg.norm(diff - proj))


def normalize(v: np.ndarray, *, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def distance_point_to_ray_sq(point: np.ndarray, ray_p0: np.ndarray, ray_dir: np.ndarray) -> float:
    d = normalize(ray_dir)
    diff = point - ray_p0
    proj = diff - np.dot(diff, d) * d
    return float(np.dot(proj, proj))


def solve_pose(
    bundle: BundleConfig,
    rays,
    *,
    w_normals: float = 1.0,
    w_centers: float = 0.1,
    max_iter: int = 50,
    tol_angle_deg: float = 1e-3,
    verbose: bool = False,
):
    """Iterative pose/assignment solver (Hungarian) using normals + center distances."""
    normals_local = []
    centers_local = []
    for m in bundle.mirrors or []:
        n_loc, c_loc = mirror_geometry(m)
        normals_local.append(n_loc)
        centers_local.append(c_loc)
    normals_local = np.stack(normals_local)
    centers_local = np.stack(centers_local)

    d_obs = []
    ray_anchors = []
    for p0, p1 in rays:
        d = p1 - p0
        d = d / np.linalg.norm(d)
        d_obs.append(d)
        ray_anchors.append(p0)
    d_obs = np.stack(d_obs)
    ray_anchors = np.stack(ray_anchors)

    m = len(bundle.mirrors or [])
    k = len(rays)
    if k > m:
        raise ValueError("Number of rays exceeds number of mirrors")

    # Observed normals from reflection; orient against D_IN
    n_est = normalize(d_obs - D_IN, axis=1)
    dot_in = np.sum(n_est * D_IN, axis=1)
    mask = dot_in > 0
    n_est[mask] = -n_est[mask]

    mirrors_idx = list(range(m))
    best = None
    I = np.eye(3)

    for mirrors_subset in it.combinations(mirrors_idx, k):
        A_loc = normals_local[list(mirrors_subset)]
        C_loc = centers_local[list(mirrors_subset)]

        R = np.eye(3)
        assignment = np.arange(k)

        for _ in range(max_iter):
            # estimate t for current R and assignment
            A_rows = []
            b_rows = []
            for i in range(k):
                j = assignment[i]
                c_rot = R @ C_loc[j]
                d = d_obs[i]
                P = I - np.outer(d, d)
                A_rows.append(P)
                b_rows.append(P @ (ray_anchors[i] - c_rot))
            A = np.concatenate(A_rows, axis=0)
            b = np.concatenate(b_rows, axis=0)
            t, *_ = np.linalg.lstsq(A, b, rcond=None)

            centers_pred = (R @ C_loc.T).T + t
            normals_pred = (R @ A_loc.T).T

            # cost matrix
            normal_cost = 1.0 - n_est @ normals_pred.T  # minimize
            center_cost = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    center_cost[i, j] = distance_point_to_ray_sq(
                        centers_pred[j], ray_anchors[i], d_obs[i]
                    )
            cost = w_normals * normal_cost + w_centers * center_cost
            rows, cols = linear_sum_assignment(cost)
            assert np.all(rows == np.arange(k))
            new_assignment = cols

            # update R via Kabsch on matched normals
            R_new = kabsch(A_loc[new_assignment], n_est)
            try:
                R_delta = R_new @ R.T
                tr = float(np.trace(R_delta))
                tr = max(-1.0, min(3.0, tr))
                angle = float(np.degrees(np.arccos((tr - 1.0) / 2.0)))
            except ValueError:
                angle = 180.0

            changed = np.any(new_assignment != assignment)
            R = R_new
            assignment = new_assignment
            if angle < tol_angle_deg and not changed:
                break

        # final t for this subset
        A_rows = []
        b_rows = []
        for i in range(k):
            j = assignment[i]
            c_rot = R @ C_loc[j]
            d = d_obs[i]
            P = I - np.outer(d, d)
            A_rows.append(P)
            b_rows.append(P @ (ray_anchors[i] - c_rot))
        A = np.concatenate(A_rows, axis=0)
        b = np.concatenate(b_rows, axis=0)
        t, *_ = np.linalg.lstsq(A, b, rcond=None)

        # score
        centers_pred = (R @ C_loc.T).T + t
        normals_pred = (R @ A_loc.T).T
        total = 0.0
        for i in range(k):
            j = assignment[i]
            d_pred = D_IN - 2 * np.dot(normals_pred[j], D_IN) * normals_pred[j]
            d_pred = d_pred / np.linalg.norm(d_pred)
            ang = 1.0 - np.dot(d_pred, d_obs[i])
            dist = distance_point_to_line(centers_pred[j], ray_anchors[i], d_obs[i])
            total += w_normals * ang + w_centers * dist

        full_assignment = list(zip([list(mirrors_subset)[idx] for idx in assignment], list(range(k))))
        if best is None or total < best[0]:
            best = (total, full_assignment, R, t)

    if best is None:
        raise RuntimeError("Assignment not found")
    return best


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-8
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.rad2deg(np.array([x, y, z], dtype=float))


def generate_rays_from_bundle(
    bundle: BundleConfig, R_true: np.ndarray, t_true: np.ndarray, ray_len: float = 30.0
):
    normals = []
    centers = []
    for m in bundle.mirrors or []:
        n_loc, c_loc = mirror_geometry(m)
        normals.append(n_loc)
        centers.append(c_loc)
    normals = np.stack(normals)
    centers = np.stack(centers)
    normals_w = (R_true @ normals.T).T
    centers_w = (R_true @ centers.T).T + t_true
    rays = []
    for c, n in zip(centers_w, normals_w):
        d_out = D_IN - 2 * np.dot(n, D_IN) * n
        d_out = d_out / np.linalg.norm(d_out)
        p0 = c.copy()
        p1 = c + d_out * ray_len
        rays.append((p0, p1))
    return rays


def rotation_error_deg(R_true: np.ndarray, R_hat: np.ndarray) -> float:
    R_err = R_hat @ R_true.T
    trace = float(np.trace(R_err))
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(np.rad2deg(np.arccos(cos_angle)))


def main():
    bundle = load_bundle_config(BUNDLE_CONFIG)

    theta = np.deg2rad(10.0)
    R_true = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    t_true = np.array([5.0, -3.0, 10.0], dtype=float)

    rays = generate_rays_from_bundle(bundle, R_true, t_true, ray_len=30.0)

    rng = np.random.default_rng(0)
    idx = np.arange(len(rays))
    rng.shuffle(idx)
    rays_shuffled = [rays[i] for i in idx]

    total, match, R_hat, t_hat = solve_pose(
        bundle, rays_shuffled, w_normals=1.0, w_centers=0.1
    )

    rot_err = rotation_error_deg(R_true, R_hat)
    trans_err = float(np.linalg.norm(t_true - t_hat))

    euler_true = rotation_matrix_to_euler(R_true)
    euler_hat = rotation_matrix_to_euler(R_hat)

    print("Synthetic reconstruction case:")
    print(f"  Total cost: {total:.6f}")
    print(f"  Rotation true (deg XYZ): {euler_true}")
    print(f"  Rotation est  (deg XYZ): {euler_hat}")
    print(f"  Rotation error (deg):   {rot_err:.6e}")
    print(f"  Translation true:       {t_true}")
    print(f"  Translation est:        {t_hat}")
    print(f"  Translation error:      {trans_err:.6e}")
    print("  Assignment (mirror idx -> ray idx):", match)

    print()
    print("Real data case (rays_config.txt):")
    rays_real = load_rays_config(RAYS_CONFIG)
    total_r, match_r, R_r, t_r = solve_pose(
        bundle, rays_real, w_normals=1.0, w_centers=0.1
    )
    euler_r = rotation_matrix_to_euler(R_r)
    print(f"  Total cost: {total_r:.6f}")
    print(f"  Rotation (deg XYZ): {euler_r}")
    print(f"  Translation:        {t_r}")
    print(f"  Assignment:         {match_r}")


if __name__ == "__main__":
    main()
