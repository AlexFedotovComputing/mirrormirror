from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from comsol_like_raytrace import GaussianBeamSource, RayTracer, build_demo_scene, to_numpy


def flatten_segment_blocks(blocks: List[Dict[str, np.ndarray]]) -> Iterable[Dict[str, object]]:
    for block in blocks:
        n = len(block["ray_id"])
        for i in range(n):
            yield {
                "ray_id": int(block["ray_id"][i]),
                "parent_id": int(block["parent_id"][i]),
                "depth": int(block["depth"][i]),
                "surface": str(block["surface"][i]),
                "x0": float(block["x0"][i]),
                "y0": float(block["y0"][i]),
                "z0": float(block["z0"][i]),
                "x1": float(block["x1"][i]),
                "y1": float(block["y1"][i]),
                "z1": float(block["z1"][i]),
                "power": float(block["power"][i]),
                "intensity": float(block["intensity"][i]),
            }


def flatten_detector_hits(blocks: List[Dict[str, np.ndarray]]) -> Iterable[Dict[str, object]]:
    for block in blocks:
        n = len(block["ray_id"])
        for i in range(n):
            pos = block["position"][i]
            d = block["direction"][i]
            yield {
                "surface": str(block["surface"]),
                "ray_id": int(block["ray_id"][i]),
                "parent_id": int(block["parent_id"][i]),
                "depth": int(block["depth"][i]),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "dx": float(d[0]),
                "dy": float(d[1]),
                "dz": float(d[2]),
                "power": float(block["power"][i]),
                "intensity": float(block["intensity"][i]),
                "u": float(block["local_u"][i]),
                "v": float(block["local_v"][i]),
            }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_line_trace(fig: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray, name: str, width: float) -> None:
    fig.add_scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"width": width},
        name=name,
        hoverinfo="skip",
    )


def write_plotly_trajectories(path: Path, result: Any) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    for block in result.segments:
        x0 = to_numpy(block["x0"])
        y0 = to_numpy(block["y0"])
        z0 = to_numpy(block["z0"])
        x1 = to_numpy(block["x1"])
        y1 = to_numpy(block["y1"])
        z1 = to_numpy(block["z1"])
        powers = to_numpy(block["power"])
        surface = str(block["surface"])
        n = len(x0)
        stride = max(1, n // 250)
        for i in range(0, n, stride):
            width = 2.0 if powers[i] > 0 else 1.0
            add_line_trace(
                fig,
                np.array([x0[i], x1[i]], dtype=float),
                np.array([y0[i], y1[i]], dtype=float),
                np.array([z0[i], z1[i]], dtype=float),
                name=surface,
                width=width,
            )

    if result.detector_hits:
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        labels: List[str] = []
        for block in result.detector_hits:
            surface = str(block["surface"])
            points = to_numpy(block["position"])
            for point in points:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
                zs.append(float(point[2]))
                labels.append(surface)
        fig.add_scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker={"size": 3, "color": "#d62728"},
            name="Detector hits",
            text=labels,
            hovertemplate="surface=%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        )

    fig.update_layout(
        title="COMSOL-like Gaussian ray tracing demo",
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        showlegend=False,
    )
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a COMSOL-like Gaussian ray-tracing demo.")
    parser.add_argument("--backend", default="numpy", choices=["numpy", "cupy"], help="Array backend")
    parser.add_argument("--n-glass", type=float, default=1.50, help="Placeholder refractive index of fused silica")
    parser.add_argument("--max-interactions", type=int, default=8)
    parser.add_argument("--outdir", default="demo_output", help="Directory for CSV outputs")
    parser.add_argument("--plot", action="store_true", help="Also save an interactive 3D Plotly plot of trajectories")
    args = parser.parse_args()

    source = GaussianBeamSource(
        waist_position=(-0.34448, -0.80321, 3.3199),
        axis=(0.0, 0.0, -1.0),
        waist_radius=0.01,
        peak_intensity=8.49e10,
        wavelength_m=266e-9,
        polarization_reference=(0.0, 1.0, 0.0),
        radial_positions=15,
        cutoff_ratio=1.0,
        n_medium=1.0,
        backend=args.backend,
    )
    rays = source.emit()
    scene = build_demo_scene(n_glass=args.n_glass)
    tracer = RayTracer(scene=scene, backend=args.backend, max_interactions=args.max_interactions)
    result = tracer.trace(rays)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_csv(outdir / "segments.csv", flatten_segment_blocks(result.segments))
    write_csv(outdir / "detector_hits.csv", flatten_detector_hits(result.detector_hits))

    summary = result.detector_power_summary()
    print("Surface list:")
    for name in result.surface_names:
        print(f"  - {name}")

    print("\nDetector power summary:")
    if not summary:
        print("  <no detector hits>")
    else:
        for key, value in summary.items():
            print(f"  {key}: {value:.6e}")

    print(f"\nFinal ray count: {result.final_rays.n_rays}")
    print(f"Wrote: {outdir / 'segments.csv'}")
    print(f"Wrote: {outdir / 'detector_hits.csv'}")

    if args.plot:
        plot_path = outdir / "trajectories.html"
        write_plotly_trajectories(plot_path, result)
        print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
