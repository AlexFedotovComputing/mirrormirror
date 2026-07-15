from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from raytrace import RayTracer, to_numpy
from vizual import (
    _CYLINDRICAL_BUNDLE_MAX_DIAMETER_M,
    _CYLINDRICAL_BUNDLE_MIN_RAY_COUNT_FRACTION,
    _collect_detector_hit_blocks,
    _cylindrical_bundle_metrics,
    _cylindrical_unwrap_coordinates,
    _finite_weights,
    _screen_b_bundle_rows,
    _spot_max_extent_m,
    write_cylindrical_unwrap_view,
)


AngleKey = Tuple[str, str]


TARGET_RANGES_DEG: Dict[AngleKey, Tuple[float, float]] = {
    ("MP1", "x"): (-0.03000, 0.03000),
    ("MP1", "z"): (-0.05000, 0.03000),
    ("MP2", "x"): (-0.04000, 0.04000),
    ("MP2", "z"): (-0.04000, 0.05000),
    ("MS1", "x"): (-0.50000, 0.50000),
    ("MS1", "z"): (-0.07000, 0.07000),
}

TARGET_MIRRORS = ("MP1", "MP2", "MS1")
TARGET_AXES = ("x", "z")
DEFAULT_OUTDIR = Path("scene_gaussian_35ns_output") / "screen_b1_angle_search_current"


def _tick(angle_deg: float, step_deg: float) -> int:
    return int(round(float(angle_deg) / float(step_deg)))


def _angle(tick: int, step_deg: float) -> float:
    return float(tick) * float(step_deg)


def _surface_spec(surface: object) -> Dict[str, object]:
    return {
        "name": surface.name,
        "center": surface.center,
        "axis": surface.axis,
        "radius": float(surface.radius),
        "length": float(surface.length),
    }


def _weighted_kmeans_2d(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    cluster_count: int,
    max_iterations: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(points.shape[0])
    k = min(max(1, int(cluster_count)), n)
    order = np.lexsort((points[:, 1], points[:, 0]))
    seed_positions = np.linspace(0, n - 1, k, dtype=int)
    centers = points[order[seed_positions]].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(int(max_iterations)):
        distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            mask = labels == idx
            if not np.any(mask):
                farthest_idx = int(np.argmax(np.min(distances, axis=1)))
                centers[idx] = points[farthest_idx]
                labels[farthest_idx] = idx
                continue
            centers[idx] = np.average(points[mask], axis=0, weights=weights[mask])

    center_order = np.lexsort((centers[:, 1], centers[:, 0]))
    remap = np.zeros(k, dtype=int)
    remap[center_order] = np.arange(k)
    return remap[labels], centers[center_order]


def _unwrap_arc_about_cluster_center(
    arc: np.ndarray,
    weights: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    if arc.size == 0 or float(radius) <= 0.0:
        return arc
    angles = np.asarray(arc, dtype=float) / float(radius)
    weights = _finite_weights(weights)
    center_angle = math.atan2(
        float(np.sum(weights * np.sin(angles))),
        float(np.sum(weights * np.cos(angles))),
    )
    shifted = (angles - center_angle + np.pi) % (2.0 * np.pi) - np.pi
    return shifted * float(radius)


def _compact_screen_bundle_counts(
    result: object,
    surface: Mapping[str, object],
    *,
    cluster_count: int = 7,
) -> List[int]:
    data = _collect_detector_hit_blocks(result, str(surface["name"]))
    positions = data["position"]
    if positions.size == 0:
        return [0] * int(cluster_count)

    intensity = data["intensity"]
    power = data["power"]
    arc, axial = _cylindrical_unwrap_coordinates(
        positions,
        center=surface["center"],
        axis=surface["axis"],
        radius=float(surface["radius"]),
    )
    valid = np.isfinite(arc) & np.isfinite(axial) & np.isfinite(intensity)
    if int(np.count_nonzero(valid)) < int(cluster_count):
        return [0] * int(cluster_count)

    arc = arc[valid]
    axial = axial[valid]
    intensity = intensity[valid]
    power = power[valid]
    weights = _finite_weights(intensity)
    arc = arc - float(np.average(arc, weights=weights))
    axial = axial - float(np.average(axial, weights=weights))

    points = np.column_stack((arc, axial))
    centered = points - np.average(points, axis=0, weights=weights)
    cov = (centered.T * weights) @ centered / max(float(np.sum(weights)), 1e-30)
    eigvals, eigvecs = np.linalg.eigh(cov)
    primary = eigvecs[:, int(np.argmax(eigvals))]
    secondary = np.array((-primary[1], primary[0]), dtype=float)

    candidate_axes = (
        (arc, axial),
        (axial, arc),
        (centered @ primary, centered @ secondary),
        (centered @ secondary, centered @ primary),
    )
    best_counts = [0] * int(cluster_count)
    best_score = (-1, -1)
    for candidate_u, candidate_v in candidate_axes:
        rows = _cylindrical_bundle_metrics(
            candidate_u,
            candidate_v,
            intensity,
            power,
            cluster_count=int(cluster_count),
            name_prefix=str(surface["name"]),
        )
        counts = [int(row.get("ray_count", 0)) for row in rows]
        counts.extend([0] * (int(cluster_count) - len(counts)))
        counts = counts[: int(cluster_count)]
        visible_count = sum(1 for count in counts if count > 0)
        min_count = min(counts) if counts else 0
        score = (visible_count, min_count)
        if score > best_score:
            best_score = score
            best_counts = counts
    return best_counts


def _screen_surface_attrs(screen_rods: str) -> Tuple[str, ...]:
    text = str(screen_rods).strip().lower()
    rods = (1, 2, 3, 4) if text == "all" else tuple(int(item) for item in text.split(",") if item.strip())
    attrs: List[str] = []
    for rod in rods:
        if rod < 1 or rod > 4:
            raise ValueError(f"Unsupported screen rod index: {rod}")
        for assembly in range(1, 5):
            surface_index = (rod - 1) * 4 + assembly
            attrs.append(f"CYLINDRICAL_SURFACE_{surface_index}")
    return tuple(attrs)


class ScreenB1AngleEvaluator:
    def __init__(
        self,
        *,
        target_count: int,
        backend: str,
        max_interactions: int | None,
        screen_rods: str = "1",
    ) -> None:
        self.scene_module = importlib.import_module("scene_gaussian_35ns")
        self.backend = backend
        self.max_interactions = (
            int(max_interactions)
            if max_interactions is not None
            else int(self.scene_module.MAX_SECONDARY_RAY_GENERATIONS)
        )
        self.default_angles_deg = {
            mirror: {
                axis: float(angle_deg)
                for axis, angle_deg in rotations.items()
            }
            for mirror, rotations in self.scene_module.DEFAULT_ADJUSTABLE_MIRROR_ROTATIONS_DEG.items()
        }
        source = self.scene_module.build_initial_source(backend)
        self.rays = self.scene_module.emit_initial_rays(source, target_count=int(target_count))
        self.screen_specs = [
            _surface_spec(getattr(self.scene_module, attr))
            for attr in _screen_surface_attrs(screen_rods)
        ]

    def evaluate(self, angles_deg: Mapping[AngleKey, float]) -> Dict[str, object]:
        result = self.trace(angles_deg)
        return self.metrics(result)

    def trace(self, angles_deg: Mapping[AngleKey, float]):
        for mirror, rotations in self.default_angles_deg.items():
            self.scene_module.ADJUSTABLE_MIRROR_ROTATIONS_DEG[mirror].update(rotations)
        for (mirror, axis), angle_deg in angles_deg.items():
            self.scene_module.ADJUSTABLE_MIRROR_ROTATIONS_DEG[mirror][axis] = float(angle_deg)

        tracer = RayTracer(
            scene=self.scene_module.build_initial_scene(),
            backend=self.backend,
            max_interactions=self.max_interactions,
            max_time_s=self.scene_module.INTEGRATION_TIME_S,
            record_segments=True,
            bundle_clip_inner_radius_m=self.scene_module.BUNDLE_RAY_INNER_CYLINDER_RADIUS_M,
            bundle_clip_outer_radius_m=self.scene_module.BUNDLE_RAY_OUTER_CYLINDER_RADIUS_M,
            bundle_clip_z_min_m=self.scene_module.BUNDLE_RAY_Z_MIN_M,
            bundle_clip_z_max_m=self.scene_module.BUNDLE_RAY_Z_MAX_M,
            skip_repeated_bundle_reflections=True,
        )
        return tracer.trace(self.rays)

    def metrics(self, result: object) -> Dict[str, object]:
        parent_by_ray: Dict[int, int] = {}
        surface_by_ray: Dict[int, str] = {}
        for block in result.segments:
            ray_ids = to_numpy(block["ray_id"]).astype(int)
            parent_ids = to_numpy(block["parent_id"]).astype(int)
            surfaces = [str(surface) for surface in block["surface"]]
            for ray_id, parent_id, surface in zip(ray_ids, parent_ids, surfaces):
                parent_by_ray[int(ray_id)] = int(parent_id)
                surface_by_ray[int(ray_id)] = surface

        def bundle_mirror_for_ray(ray_id: int, expected_bundle: str) -> int | None:
            expected_pattern = re.compile(rf"^{re.escape(expected_bundle)} Mirror ([1-7])$")
            current_ray_id = int(ray_id)
            for _ in range(100):
                surface = surface_by_ray.get(current_ray_id, "")
                match = expected_pattern.match(surface)
                if match:
                    return int(match.group(1))
                parent_id = parent_by_ray.get(current_ray_id)
                if parent_id is None or parent_id == current_ray_id:
                    return None
                current_ray_id = parent_id
            return None

        row: Dict[str, object] = {}
        all_visible = True
        min_ray_count = None
        for surface in self.screen_specs:
            screen_name = str(surface["name"])
            ray_counts = [0] * 7
            screen_match = re.match(r"^screen_b_(\d+)_(\d+)$", screen_name)
            group_positions: Dict[int, List[np.ndarray]] = {idx: [] for idx in range(1, 8)}
            group_intensities: Dict[int, List[float]] = {idx: [] for idx in range(1, 8)}
            if screen_match:
                expected_bundle = f"BUNDLE_{screen_match.group(1)}_{screen_match.group(2)}"
                for block in result.detector_hits:
                    if str(block["surface"]) != screen_name:
                        continue
                    ray_ids = to_numpy(block["ray_id"]).astype(int)
                    positions = np.asarray(to_numpy(block["position"]), dtype=float).reshape(-1, 3)
                    intensities = np.asarray(to_numpy(block["intensity"]), dtype=float).reshape(-1)
                    for ray_id, position, intensity in zip(ray_ids, positions, intensities):
                        mirror_idx = bundle_mirror_for_ray(int(ray_id), expected_bundle)
                        if mirror_idx is None:
                            continue
                        group_positions[mirror_idx].append(position)
                        group_intensities[mirror_idx].append(float(intensity))

            for mirror_idx in range(1, 8):
                positions = np.asarray(group_positions[mirror_idx], dtype=float).reshape(-1, 3)
                if positions.size == 0:
                    continue
                intensities = np.asarray(group_intensities[mirror_idx], dtype=float).reshape(-1)
                arc, axial = _cylindrical_unwrap_coordinates(
                    positions,
                    center=surface["center"],
                    axis=surface["axis"],
                    radius=float(surface["radius"]),
                )
                arc = _unwrap_arc_about_cluster_center(
                    arc,
                    intensities,
                    radius=float(surface["radius"]),
                )
                max_extent = _spot_max_extent_m(arc, axial, intensities)
                if np.isfinite(max_extent) and float(max_extent) > _CYLINDRICAL_BUNDLE_MAX_DIAMETER_M:
                    continue
                ray_counts[mirror_idx - 1] = int(positions.shape[0])
            visible_count = sum(1 for count in ray_counts if count > 0)
            all_visible = all_visible and visible_count == 7
            screen_min = min(ray_counts) if ray_counts else 0
            min_ray_count = screen_min if min_ray_count is None else min(min_ray_count, screen_min)
            row[f"{screen_name}_visible_bundles"] = visible_count
            row[f"{screen_name}_min_ray_count"] = screen_min
            row[f"{screen_name}_ray_counts"] = " ".join(str(count) for count in ray_counts)

        row["ok"] = int(all_visible)
        row["min_ray_count"] = int(min_ray_count or 0)
        return row


def _case_slug(mirror: str, axis: str, bound: str, angle_deg: float) -> str:
    angle_text = f"{abs(float(angle_deg)):.6f}".replace(".", "p")
    sign = "minus" if float(angle_deg) < 0.0 else "plus"
    return f"{mirror.lower()}_{axis}_{bound}_{sign}_{angle_text}"


def _html_index(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    link_rows = []
    for row in rows:
        case_dir = str(row["case_dir"])
        links = " ".join(
            f'<a href="{case_dir}/screen_b_1_{idx}_unwrap.html">screen_b_1_{idx}</a>'
            for idx in range(1, 5)
        )
        link_rows.append(
            "      <tr>"
            f"<td>{row['mirror']}</td>"
            f"<td>{row['axis']}</td>"
            f"<td>{row['bound']}</td>"
            f"<td>{row['angle_deg']}</td>"
            f"<td>{row['ok']}</td>"
            f"<td>{row['min_ray_count']}</td>"
            f"<td>{links}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>screen_b_1_i sections</title>
  <style>
    body {{
      margin: 0;
      padding: 28px;
      font-family: "Segoe UI", Arial, sans-serif;
      color: #183052;
      background: #f4f7fb;
    }}
    h1 {{
      margin: 0 0 18px;
      font-size: 28px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      box-shadow: 0 10px 30px rgba(32, 58, 96, 0.12);
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #d9e2ef;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #dfe8f5;
    }}
    a {{
      display: inline-block;
      margin: 0 10px 6px 0;
      color: #1f5d9e;
      font-weight: 600;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <h1>Сечения screen_b_1_i для предельных одиночных поворотов</h1>
  <table>
    <thead>
      <tr>
        <th>Зеркало</th>
        <th>Ось</th>
        <th>Граница</th>
        <th>Угол, deg</th>
        <th>7 пучков</th>
        <th>min ray count</th>
        <th>Сечения</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(link_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _contiguous_limits(ok_by_tick: Mapping[int, bool], lo_tick: int, hi_tick: int) -> Tuple[int, int]:
    low = 0
    for tick in range(-1, lo_tick - 1, -1):
        if ok_by_tick.get(tick, False):
            low = tick
            continue
        break

    high = 0
    for tick in range(1, hi_tick + 1):
        if ok_by_tick.get(tick, False):
            high = tick
            continue
        break

    return low, high


def _row_for_angles(
    *,
    angles: Mapping[AngleKey, float],
    metrics: Mapping[str, object],
    extra: Mapping[str, object],
) -> Dict[str, object]:
    row: Dict[str, object] = dict(extra)
    for (mirror, axis), angle_deg in angles.items():
        row[f"{mirror}_{axis}_deg"] = f"{float(angle_deg):.6f}"
    row.update(metrics)
    return row


def _selected_ranges(only_items: Sequence[str] | None) -> Dict[AngleKey, Tuple[float, float]]:
    if not only_items:
        return dict(TARGET_RANGES_DEG)
    only = set()
    for item in only_items:
        mirror, axis = item.split(":", 1)
        only.add((mirror.upper(), axis.lower()))
    return {key: value for key, value in TARGET_RANGES_DEG.items() if key in only}


def _evaluate_single_axis(
    evaluator: ScreenB1AngleEvaluator,
    mirror: str,
    axis: str,
    tick: int,
    step: float,
    *,
    phase: str,
) -> Dict[str, object]:
    angle_deg = _angle(tick, step)
    metrics = evaluator.evaluate({(mirror, axis): angle_deg})
    return _row_for_angles(
        angles={(mirror, axis): angle_deg},
        metrics=metrics,
        extra={
            "mirror": mirror,
            "axis": axis,
            "phase": phase,
            "angle_deg": f"{angle_deg:.6f}",
            "angle_tick": tick,
        },
    )


def _find_direction_limit(
    evaluator: ScreenB1AngleEvaluator,
    mirror: str,
    axis: str,
    bound_tick: int,
    step: float,
    *,
    progress: bool,
) -> Tuple[int, int | None, List[Dict[str, object]]]:
    if bound_tick == 0:
        return 0, None, []

    rows: List[Dict[str, object]] = []
    bound_row = _evaluate_single_axis(
        evaluator,
        mirror,
        axis,
        bound_tick,
        step,
        phase="bound",
    )
    rows.append(bound_row)
    if progress:
        print(
            f"{mirror}:{axis} bound={_angle(bound_tick, step):.6f} "
            f"ok={bound_row['ok']} min_ray_count={bound_row['min_ray_count']}",
            flush=True,
        )
    if bool(int(bound_row["ok"])):
        return bound_tick, None, rows

    pass_tick = 0
    fail_tick = bound_tick
    while abs(fail_tick - pass_tick) > 1:
        mid_tick = (pass_tick + fail_tick) // 2
        mid_row = _evaluate_single_axis(
            evaluator,
            mirror,
            axis,
            mid_tick,
            step,
            phase="bisect",
        )
        rows.append(mid_row)
        if progress:
            print(
                f"{mirror}:{axis} test={_angle(mid_tick, step):.6f} "
                f"ok={mid_row['ok']} min_ray_count={mid_row['min_ray_count']}",
                flush=True,
            )
        if bool(int(mid_row["ok"])):
            pass_tick = mid_tick
        else:
            fail_tick = mid_tick

    return pass_tick, fail_tick, rows


def run_limits(args: argparse.Namespace) -> None:
    evaluator = ScreenB1AngleEvaluator(
        target_count=args.target_count,
        backend=args.backend,
        max_interactions=args.max_interactions,
        screen_rods=args.screen_rods,
    )
    outdir = Path(args.outdir)
    step = float(args.step_deg)
    start = time.perf_counter()
    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    zero_metrics = evaluator.evaluate({})
    detail_rows.append(
        _row_for_angles(
            angles={},
            metrics=zero_metrics,
            extra={"mirror": "", "axis": "", "phase": "zero", "angle_deg": "0.000000", "angle_tick": 0},
        )
    )
    if not bool(int(zero_metrics["ok"])):
        raise RuntimeError("The zero-angle reference does not show 7 bundles on every screen_b_1_i.")

    for (mirror, axis), (lo, hi) in _selected_ranges(args.only).items():
        lo_tick = _tick(lo, step)
        hi_tick = _tick(hi, step)
        low_tick, low_fail_tick, low_rows = _find_direction_limit(
            evaluator,
            mirror,
            axis,
            lo_tick,
            step,
            progress=args.progress,
        )
        high_tick, high_fail_tick, high_rows = _find_direction_limit(
            evaluator,
            mirror,
            axis,
            hi_tick,
            step,
            progress=args.progress,
        )
        detail_rows.extend(low_rows)
        detail_rows.extend(high_rows)
        summary_rows.append(
            {
                "mirror": mirror,
                "axis": axis,
                "range_min_deg": f"{lo:.6f}",
                "range_max_deg": f"{hi:.6f}",
                "contiguous_min_deg": f"{_angle(low_tick, step):.6f}",
                "contiguous_max_deg": f"{_angle(high_tick, step):.6f}",
                "first_fail_below_deg": f"{_angle(low_fail_tick, step):.6f}" if low_fail_tick is not None else "",
                "first_fail_above_deg": f"{_angle(high_fail_tick, step):.6f}" if high_fail_tick is not None else "",
                "target_count": int(args.target_count),
                "step_deg": f"{step:.6f}",
            }
        )
        if args.progress:
            print(
                f"{mirror}:{axis} limit=[{_angle(low_tick, step):.6f}, "
                f"{_angle(high_tick, step):.6f}]",
                flush=True,
            )

    _write_csv(outdir / "limit_search_details.csv", detail_rows)
    _write_csv(outdir / "individual_summary.csv", summary_rows)
    print(f"Wrote {outdir / 'individual_summary.csv'}")
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")


def run_individual(args: argparse.Namespace) -> None:
    evaluator = ScreenB1AngleEvaluator(
        target_count=args.target_count,
        backend=args.backend,
        max_interactions=args.max_interactions,
        screen_rods=args.screen_rods,
    )
    outdir = Path(args.outdir)
    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    step = float(args.step_deg)
    start = time.perf_counter()

    ranges = _selected_ranges(args.only)

    for (mirror, axis), (lo, hi) in ranges.items():
        lo_tick = _tick(lo, step)
        hi_tick = _tick(hi, step)
        ticks = list(range(lo_tick, hi_tick + 1))
        axis_rows: List[Dict[str, object]] = []
        ok_by_tick: Dict[int, bool] = {}

        for index, tick in enumerate(ticks, start=1):
            angle_deg = _angle(tick, step)
            metrics = evaluator.evaluate({(mirror, axis): angle_deg})
            ok_by_tick[tick] = bool(metrics["ok"])
            row: Dict[str, object] = {
                "mirror": mirror,
                "axis": axis,
                "angle_deg": f"{angle_deg:.6f}",
                "angle_tick": tick,
            }
            row.update(metrics)
            axis_rows.append(row)
            all_rows.append(row)
            if args.progress:
                print(
                    f"{mirror}:{axis} {index}/{len(ticks)} "
                    f"angle={angle_deg:.6f} ok={metrics['ok']} "
                    f"min_ray_count={metrics['min_ray_count']}",
                    flush=True,
                )

        low_tick, high_tick = _contiguous_limits(ok_by_tick, lo_tick, hi_tick)
        pass_ticks = [tick for tick, ok in ok_by_tick.items() if ok]
        summary_rows.append(
            {
                "mirror": mirror,
                "axis": axis,
                "range_min_deg": f"{lo:.6f}",
                "range_max_deg": f"{hi:.6f}",
                "contiguous_min_deg": f"{_angle(low_tick, step):.6f}",
                "contiguous_max_deg": f"{_angle(high_tick, step):.6f}",
                "any_pass_min_deg": f"{_angle(min(pass_ticks), step):.6f}" if pass_ticks else "",
                "any_pass_max_deg": f"{_angle(max(pass_ticks), step):.6f}" if pass_ticks else "",
                "failed_inside_contiguous_rule": int(
                    any(not ok_by_tick.get(tick, False) for tick in range(low_tick, high_tick + 1))
                ),
                "target_count": int(args.target_count),
                "step_deg": f"{step:.6f}",
            }
        )
        _write_csv(outdir / f"{mirror.lower()}_{axis}_individual.csv", axis_rows)

    _write_csv(outdir / "individual_sweep.csv", all_rows)
    _write_csv(outdir / "individual_summary.csv", summary_rows)
    print(f"Wrote {outdir / 'individual_summary.csv'}")
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")


def _read_individual_limits(path: Path) -> Dict[AngleKey, Tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    limits: Dict[AngleKey, Tuple[float, float]] = {}
    for row in rows:
        key = (str(row["mirror"]).upper(), str(row["axis"]).lower())
        limits[key] = (float(row["contiguous_min_deg"]), float(row["contiguous_max_deg"]))
    return limits


def run_corners(args: argparse.Namespace) -> None:
    evaluator = ScreenB1AngleEvaluator(
        target_count=args.target_count,
        backend=args.backend,
        max_interactions=args.max_interactions,
        screen_rods=args.screen_rods,
    )
    outdir = Path(args.outdir)
    limits_path = Path(args.limits_csv) if args.limits_csv else outdir / "individual_summary.csv"
    limits = _read_individual_limits(limits_path)
    keys = [key for key in TARGET_RANGES_DEG if key in limits]
    rows: List[Dict[str, object]] = []
    start = time.perf_counter()

    for index, bits in enumerate(itertools.product((0, 1), repeat=len(keys)), start=1):
        angles = {
            key: limits[key][int(bit)]
            for key, bit in zip(keys, bits)
        }
        metrics = evaluator.evaluate(angles)
        row: Dict[str, object] = {
            "corner_index": index,
            "sign_bits": "".join(str(bit) for bit in bits),
        }
        for mirror, axis in keys:
            row[f"{mirror}_{axis}_deg"] = f"{angles[(mirror, axis)]:.6f}"
        row.update(metrics)
        rows.append(row)
        if args.progress:
            print(
                f"corner {index}/{2 ** len(keys)} bits={row['sign_bits']} "
                f"ok={metrics['ok']} min_ray_count={metrics['min_ray_count']}",
                flush=True,
            )

    _write_csv(outdir / "individual_limit_corners.csv", rows)
    ok_count = sum(int(row["ok"]) for row in rows)
    print(f"Wrote {outdir / 'individual_limit_corners.csv'}")
    print(f"Passing corners: {ok_count}/{len(rows)}")
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")


def run_bounds(args: argparse.Namespace) -> None:
    evaluator = ScreenB1AngleEvaluator(
        target_count=args.target_count,
        backend=args.backend,
        max_interactions=args.max_interactions,
        screen_rods=args.screen_rods,
    )
    outdir = Path(args.outdir)
    rows: List[Dict[str, object]] = []
    start = time.perf_counter()

    zero_metrics = evaluator.evaluate({})
    rows.append(
        _row_for_angles(
            angles={},
            metrics=zero_metrics,
            extra={"mirror": "", "axis": "", "bound": "zero", "angle_deg": "0.000000"},
        )
    )

    for (mirror, axis), (lo, hi) in _selected_ranges(args.only).items():
        for bound_name, angle_deg in (("min", lo), ("max", hi)):
            metrics = evaluator.evaluate({(mirror, axis): angle_deg})
            rows.append(
                _row_for_angles(
                    angles={(mirror, axis): angle_deg},
                    metrics=metrics,
                    extra={
                        "mirror": mirror,
                        "axis": axis,
                        "bound": bound_name,
                        "angle_deg": f"{angle_deg:.6f}",
                    },
                )
            )
            if args.progress:
                print(
                    f"{mirror}:{axis} {bound_name}={angle_deg:.6f} "
                    f"ok={metrics['ok']} min_ray_count={metrics['min_ray_count']}",
                    flush=True,
                )

    _write_csv(outdir / "range_bound_checks.csv", rows)
    print(f"Wrote {outdir / 'range_bound_checks.csv'}")
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")


def run_sections(args: argparse.Namespace) -> None:
    evaluator = ScreenB1AngleEvaluator(
        target_count=args.target_count,
        backend=args.backend,
        max_interactions=args.max_interactions,
        screen_rods=args.screen_rods,
    )
    outdir = Path(args.outdir)
    rows: List[Dict[str, object]] = []
    start = time.perf_counter()

    for (mirror, axis), (lo, hi) in _selected_ranges(args.only).items():
        for bound_name, angle_deg in (("min", lo), ("max", hi)):
            angles = {(mirror, axis): angle_deg}
            result = evaluator.trace(angles)
            metrics = evaluator.metrics(result)
            slug = _case_slug(mirror, axis, bound_name, angle_deg)
            case_dir = outdir / slug
            for screen_idx, surface in enumerate(evaluator.screen_specs, start=1):
                screen_name = str(surface["name"])
                write_cylindrical_unwrap_view(
                    case_dir / f"{screen_name}_unwrap.html",
                    result,
                    surface=surface,
                    title=f"{screen_name}: {mirror} {axis} {angle_deg:.6f} deg",
                )
            row: Dict[str, object] = {
                "mirror": mirror,
                "axis": axis,
                "bound": bound_name,
                "angle_deg": f"{angle_deg:.6f}",
                "case_dir": slug,
            }
            row.update(metrics)
            rows.append(row)
            if args.progress:
                print(
                    f"{mirror}:{axis} {bound_name}={angle_deg:.6f} "
                    f"ok={metrics['ok']} min_ray_count={metrics['min_ray_count']} "
                    f"wrote={case_dir}",
                    flush=True,
                )

    _write_csv(outdir / "sections_summary.csv", rows)
    _html_index(outdir / "index.html", rows)
    print(f"Wrote {outdir / 'index.html'}")
    print(f"Wrote {outdir / 'sections_summary.csv'}")
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")


def run_check(args: argparse.Namespace) -> None:
    angles: Dict[AngleKey, float] = {}
    for item in args.angle:
        key_text, value_text = item.split("=", 1)
        mirror, axis = key_text.split(":", 1)
        angles[(mirror.upper(), axis.lower())] = float(value_text)

    evaluator = ScreenB1AngleEvaluator(
        target_count=args.target_count,
        backend=args.backend,
        max_interactions=args.max_interactions,
        screen_rods=args.screen_rods,
    )
    metrics = evaluator.evaluate(angles)
    for key in sorted(angles):
        print(f"{key[0]}:{key[1]}={angles[key]:.6f}")
    print(f"ok={metrics['ok']} min_ray_count={metrics['min_ray_count']}")
    for key, value in metrics.items():
        if str(key).startswith("screen_b_"):
            print(f"{key}={value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find MP1/MP2/MS1 x/z angle limits that keep 7 bundles visible on screen_b_1_i."
    )
    parser.add_argument("--mode", choices=("limits", "individual", "corners", "bounds", "sections", "check"), default="bounds")
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--target-count", type=int, default=2000)
    parser.add_argument("--max-interactions", type=int, default=None)
    parser.add_argument("--step-deg", type=float, default=0.0005)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--screen-rods", default="1", help='Screen rods to check: "1", "1,2", or "all".')
    parser.add_argument("--limits-csv", default="")
    parser.add_argument("--only", action="append", help="Limit individual sweep to MIRROR:axis, for example MP1:x.")
    parser.add_argument("--angle", action="append", default=[], help="For check mode: MIRROR:axis=value_deg.")
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "limits":
        run_limits(args)
    elif args.mode == "individual":
        run_individual(args)
    elif args.mode == "corners":
        run_corners(args)
    elif args.mode == "bounds":
        run_bounds(args)
    elif args.mode == "sections":
        run_sections(args)
    elif args.mode == "check":
        run_check(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
