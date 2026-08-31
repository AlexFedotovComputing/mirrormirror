from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from find_screen_b1_angle_limits import AngleKey, ScreenB1AngleEvaluator


FINAL_SCREEN_NAMES = ("Screen 1", "Screen 2", "Screen 3", "Screen 4")
DEFAULT_OUTPUT = Path("Чувствительность_зеркал_на_круглых_экранах.csv")


def _screen_centroids(result: object) -> Dict[str, Tuple[np.ndarray, int, float]]:
    positions: Dict[str, list[np.ndarray]] = {name: [] for name in FINAL_SCREEN_NAMES}
    intensities: Dict[str, list[np.ndarray]] = {name: [] for name in FINAL_SCREEN_NAMES}

    for block in result.detector_hits:
        screen_name = str(block["surface"])
        if screen_name not in positions:
            continue
        positions[screen_name].append(np.asarray(block["position"], dtype=float).reshape(-1, 3))
        intensities[screen_name].append(np.asarray(block["intensity"], dtype=float).reshape(-1))

    centroids: Dict[str, Tuple[np.ndarray, int, float]] = {}
    for screen_name in FINAL_SCREEN_NAMES:
        if not positions[screen_name]:
            raise RuntimeError(f"No rays reached {screen_name}.")
        screen_positions = np.concatenate(positions[screen_name], axis=0)
        screen_intensities = np.concatenate(intensities[screen_name], axis=0)
        total_intensity = float(np.sum(screen_intensities))
        if total_intensity <= 0.0:
            raise RuntimeError(f"Non-positive total intensity on {screen_name}.")
        centroid = np.average(screen_positions, axis=0, weights=screen_intensities)
        centroids[screen_name] = (centroid, int(screen_positions.shape[0]), total_intensity)
    return centroids


def _control_angles(control: str, axis: str, deviation_deg: float) -> Dict[AngleKey, float]:
    if control == "MS1-MS4":
        return {(mirror, axis): deviation_deg for mirror in ("MS1", "MS2", "MS3", "MS4")}
    return {(control, axis): deviation_deg}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(*, target_count: int, delta_deg: float, output: Path) -> None:
    if delta_deg <= 0.0:
        raise ValueError("delta_deg must be positive.")

    evaluator = ScreenB1AngleEvaluator(
        target_count=target_count,
        backend="numpy",
        max_interactions=None,
        screen_rods="all",
    )
    baseline = _screen_centroids(evaluator.trace({}))
    controls = ("MP1", "MP2", "MS1", "MS2", "MS3", "MS4", "MS1-MS4")
    rows: list[Dict[str, object]] = []

    for control in controls:
        for axis in ("x", "z"):
            minus = _screen_centroids(
                evaluator.trace(_control_angles(control, axis, -delta_deg))
            )
            plus = _screen_centroids(
                evaluator.trace(_control_angles(control, axis, delta_deg))
            )
            for screen_name in FINAL_SCREEN_NAMES:
                baseline_point, baseline_hits, baseline_intensity = baseline[screen_name]
                minus_point, minus_hits, minus_intensity = minus[screen_name]
                plus_point, plus_hits, plus_intensity = plus[screen_name]
                derivative_m_per_deg = (plus_point - minus_point) / (2.0 * delta_deg)
                sensitivity_mm_per_deg = 1000.0 * derivative_m_per_deg[:2]
                magnitude_mm_per_deg = float(np.linalg.norm(sensitivity_mm_per_deg))
                rows.append(
                    {
                        "control": control,
                        "axis": axis,
                        "screen": screen_name,
                        "delta_deg": f"{delta_deg:.9f}",
                        "baseline_x_m": f"{baseline_point[0]:.12g}",
                        "baseline_y_m": f"{baseline_point[1]:.12g}",
                        "dx_mm_per_deg": f"{sensitivity_mm_per_deg[0]:.9g}",
                        "dy_mm_per_deg": f"{sensitivity_mm_per_deg[1]:.9g}",
                        "magnitude_mm_per_deg": f"{magnitude_mm_per_deg:.9g}",
                        "magnitude_m_per_rad": f"{magnitude_mm_per_deg * 1e-3 * 180.0 / math.pi:.9g}",
                        "baseline_hits": baseline_hits,
                        "minus_hits": minus_hits,
                        "plus_hits": plus_hits,
                        "minus_to_baseline_intensity_ratio": f"{minus_intensity / baseline_intensity:.9g}",
                        "plus_to_baseline_intensity_ratio": f"{plus_intensity / baseline_intensity:.9g}",
                    }
                )
            print(f"Calculated {control} {axis}", flush=True)

    _write_csv(output, rows)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate final round-screen spot-centroid sensitivity to mirror rotations."
    )
    parser.add_argument("--target-count", type=int, default=10000)
    parser.add_argument("--delta-deg", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(target_count=args.target_count, delta_deg=args.delta_deg, output=args.output)


if __name__ == "__main__":
    main()
