from __future__ import annotations

import argparse
import csv
import webbrowser
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from comsol_like_raytrace import Detector, GaussianBeamSource, PlaneMirror, RayTracer, Scene, to_numpy
from raytrace_plotly import make_circle_outline, make_rectangle_outline, write_plotly_trajectories


INTEGRATION_TIME_S = 35e-9

SOURCE_CONFIG = {
    "waist_position": (-0.34448, -0.80321, 3.3199),
    "axis": (0.0, 0.0, -1.0),
    "waist_radius": 0.01,
    "peak_intensity": 8.49e10,
    "wavelength_m": 266e-9,
    "polarization_reference": (0.0, 1.0, 0.0),
    "radial_positions": 15,
    "cutoff_ratio": 1.0,
    "n_medium": 1.0,
}

DETECTOR_CONFIG = {
    "name": "Detector_direct",
    "center": (-0.34448, -0.80321, 2.60),
    "normal": (0.0, 0.0, 1.0),
    "shape": "rectangle",
    "width": 0.08,
    "height": 0.08,
    "in_plane_reference": (1.0, 0.0, 0.0),
}

SMALL_MIRROR_CONFIG = {
    "name": "SmallMirror",
    "center": (-0.06948408707129516, -0.32689692816523785, 0.436),
    "normal": (0.0, 0.7071067811865476, 0.7071067811865476),
    "shape": "rectangle",
    "width": 0.03,
    "height": 0.03,
    "in_plane_reference": (1.0, 0.0, 0.0),
    "reflectance": 1.0,
}


def build_initial_source(backend: str = "numpy") -> GaussianBeamSource:
    return GaussianBeamSource(backend=backend, **SOURCE_CONFIG)


def build_initial_scene() -> Scene:
    scene = Scene()
    scene.add(
        PlaneMirror(**SMALL_MIRROR_CONFIG),
        Detector(**DETECTOR_CONFIG),
    )
    return scene


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


def detector_energy_summary(result: object, integration_time_s: float) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for hit in result.detector_hits:
        name = str(hit["surface"])
        energy = float(np.sum(to_numpy(hit["power"]))) * integration_time_s
        summary[name] = summary.get(name, 0.0) + energy
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Initial 35 ns scene with the same Gaussian source as the demo.")
    parser.add_argument("--backend", default="numpy", choices=["numpy", "cupy"], help="Array backend")
    parser.add_argument("--max-interactions", type=int, default=1, help="Number of ray interactions")
    parser.add_argument("--outdir", default="scene_gaussian_35ns_output", help="Directory for outputs")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Skip saving the Plotly trajectories plot")
    parser.add_argument("--no-open-plot", dest="open_plot", action="store_false", help="Do not open the saved plot in a browser")
    parser.set_defaults(plot=True, open_plot=True)
    args = parser.parse_args()

    source = build_initial_source(args.backend)
    rays = source.emit()
    scene = build_initial_scene()
    tracer = RayTracer(scene=scene, backend=args.backend, max_interactions=args.max_interactions)
    result = tracer.trace(rays)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_csv(outdir / "segments.csv", flatten_segment_blocks(result.segments))
    write_csv(outdir / "detector_hits.csv", flatten_detector_hits(result.detector_hits))

    source_power = float(np.sum(to_numpy(rays.power)))
    power_summary = result.detector_power_summary()
    energy_summary = detector_energy_summary(result, INTEGRATION_TIME_S)

    print("Initial scene: Gaussian source with scene scaffold")
    print(f"Integration time: {INTEGRATION_TIME_S * 1e9:.3f} ns")
    print(f"Source total power: {source_power:.6e} W")
    print("Scene elements:")
    print(f"  - {SMALL_MIRROR_CONFIG['name']}")
    print(f"  - {DETECTOR_CONFIG['name']}")
    print("Detector power summary:")
    if not power_summary:
        print("  <no detector hits>")
    else:
        for name, power in power_summary.items():
            print(f"  {name}: {power:.6e} W")
    print("Detector energy summary:")
    if not energy_summary:
        print("  <no detector hits>")
    else:
        for name, energy in energy_summary.items():
            print(f"  {name}: {energy:.6e} J")
    print(f"Final ray count: {result.final_rays.n_rays}")
    print(f"Wrote: {outdir / 'segments.csv'}")
    print(f"Wrote: {outdir / 'detector_hits.csv'}")

    if args.plot:
        plot_path = outdir / "scene_gaussian_35ns.html"
        overlays = [
            make_circle_outline(
                name="Source waist",
                center=SOURCE_CONFIG["waist_position"],
                normal=SOURCE_CONFIG["axis"],
                radius=float(SOURCE_CONFIG["waist_radius"]),
                color="#1f77b4",
                in_plane_reference=SOURCE_CONFIG["polarization_reference"],
            ),
            make_rectangle_outline(
                name=SMALL_MIRROR_CONFIG["name"],
                center=SMALL_MIRROR_CONFIG["center"],
                normal=SMALL_MIRROR_CONFIG["normal"],
                width=float(SMALL_MIRROR_CONFIG["width"]),
                height=float(SMALL_MIRROR_CONFIG["height"]),
                color="#ff7f0e",
                in_plane_reference=SMALL_MIRROR_CONFIG["in_plane_reference"],
            ),
            make_rectangle_outline(
                name=DETECTOR_CONFIG["name"],
                center=DETECTOR_CONFIG["center"],
                normal=DETECTOR_CONFIG["normal"],
                width=float(DETECTOR_CONFIG["width"]),
                height=float(DETECTOR_CONFIG["height"]),
                color="#2ca02c",
                in_plane_reference=DETECTOR_CONFIG["in_plane_reference"],
            ),
        ]
        write_plotly_trajectories(
            plot_path,
            result,
            title="Initial 35 ns scene",
            overlays=overlays,
        )
        print(f"Wrote: {plot_path}")
        if args.open_plot:
            webbrowser.open(plot_path.resolve().as_uri())


if __name__ == "__main__":
    main()
