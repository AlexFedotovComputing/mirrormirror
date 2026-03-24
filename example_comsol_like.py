from __future__ import annotations

import argparse
import csv
import webbrowser
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from comsol_like_raytrace import GaussianBeamSource, RayTracer, build_demo_scene
from raytrace_plotly import write_plotly_trajectories


def flatten_segment_blocks(blocks: List[Dict[str, np.ndarray]]) -> Iterable[Dict[str, object]]:
    for block in blocks:
        n = len(block["ray_id"])
        for i in range(n):
            row = {
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
            if "t0_s" in block:
                row["t0_s"] = float(block["t0_s"][i])
            if "t1_s" in block:
                row["t1_s"] = float(block["t1_s"][i])
            yield row


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a COMSOL-like Gaussian ray-tracing demo.")
    parser.add_argument("--backend", default="numpy", choices=["numpy", "cupy"], help="Array backend")
    parser.add_argument("--n-glass", type=float, default=1.50, help="Placeholder refractive index of fused silica")
    parser.add_argument("--max-interactions", type=int, default=8)
    parser.add_argument("--outdir", default="demo_output", help="Directory for CSV outputs")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Skip saving the Plotly trajectories plot")
    parser.add_argument("--no-open-plot", dest="open_plot", action="store_false", help="Do not open the saved plot in a browser")
    parser.set_defaults(plot=True, open_plot=True)
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
        write_plotly_trajectories(plot_path, result, title="COMSOL-like Gaussian ray tracing demo")
        print(f"Wrote: {plot_path}")
        if args.open_plot:
            webbrowser.open(plot_path.resolve().as_uri())


if __name__ == "__main__":
    main()
