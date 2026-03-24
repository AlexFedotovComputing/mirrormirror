from __future__ import annotations

import argparse
import copy
import csv
import webbrowser
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from comsol_like_raytrace import GaussianBeamSource, PlaneMirror, RayTracer, Scene, to_numpy
from raytrace_plotly import (
    make_circle_outline,
    make_rectangle_outline,
    write_plotly_trajectories,
)


INTEGRATION_TIME_S = 35e-9
BEAM_RADIAL_POSITIONS = 55
BEAM_CUTOFF_RATIO = 1.0


SOURCE_TEMPLATE = GaussianBeamSource(
    waist_position=(-0.3444840871, -0.8032109003, 3.31987),
    axis=(0.0, 0.0, -1.0),
    waist_radius=0.01,
    peak_intensity=8.49e10,
    wavelength_m=266e-9,
    polarization_reference=(0.0, 1.0, 0.0),
    radial_positions=BEAM_RADIAL_POSITIONS,
    cutoff_ratio=BEAM_CUTOFF_RATIO,
    n_medium=1.0,
    backend="numpy",
)

PERISCOPE_MIRROR_2 = PlaneMirror(
    name="Periscope Mirror 2",
    center=(-0.06948408707129516, -0.32689692816523785, 0.436),
    normal=(-0.35355339059327373, -0.6123724356957946, -0.7071067811865476),
    shape="rectangle",
    width=0.03,
    height=0.03,
    in_plane_reference=(0.8660254037844387, -0.5, 0.0),
    reflectance=1.0,
)

PERISCOPE_MIRROR_1 = PlaneMirror(
    name="Periscope Mirror 1",
    center=(-0.3444820035901876, -0.8032092682588535, 0.436),
    normal=(0.35355339059327373, 0.6123724356957946, 0.7071067811865476),
    shape="rectangle",
    width=0.03,
    height=0.03,
    in_plane_reference=(0.8660254037844387, -0.5, 0.0),
    reflectance=1.0,
)

ONE_OF_MANY_MIRRORS = PlaneMirror(
    name="One of many mirrors",
    center=(0.2168904, 0.2218567, 0.0),
    normal=(0.0, 0.0, 1.0),
    shape="rectangle",
    width=0.044,
    height=0.0081,
    in_plane_reference=(-0.03612, -0.99935, 0.0),
    reflectance=1.0,
)

ON_ENTER_BEAMSPLITTER = PlaneMirror(
    name="On enter beamsplitter",
    center=(-0.06948408707129516, -0.32689692816523785, 0.025),
    normal=(-0.566592014759144, 0.42305258397884055, 0.7071067811865476),
    shape="rectangle",
    width=0.03,
    height=0.03,
    in_plane_reference=(0.5665920147591441, -0.4230525839788406, 0.7071067811865475),
    reflectance=1.0,
)

def build_initial_source(backend: str = "numpy") -> GaussianBeamSource:
    source = copy.deepcopy(SOURCE_TEMPLATE)
    source.backend = backend
    return source


def build_initial_scene() -> Scene:
    scene = Scene()
    scene.add(
        copy.deepcopy(PERISCOPE_MIRROR_1),
        copy.deepcopy(PERISCOPE_MIRROR_2),
        copy.deepcopy(ONE_OF_MANY_MIRRORS),
        copy.deepcopy(ON_ENTER_BEAMSPLITTER),
    )
    return scene

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


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> bool:
    rows = list(rows)
    if not rows:
        if path.exists():
            path.unlink()
        return False
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return True


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
    parser.add_argument("--max-interactions", type=int, default=4, help="Number of ray interactions")
    parser.add_argument("--outdir", default="scene_gaussian_35ns_output", help="Directory for outputs")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Skip saving the Plotly trajectories plot")
    parser.add_argument("--no-open-plot", dest="open_plot", action="store_false", help="Do not open the saved plot in a browser")
    parser.set_defaults(plot=True, open_plot=True)
    args = parser.parse_args()

    source = build_initial_source(args.backend)
    rays = source.emit()
    scene = build_initial_scene()
    tracer = RayTracer(
        scene=scene,
        backend=args.backend,
        max_interactions=args.max_interactions,
        max_time_s=INTEGRATION_TIME_S,
    )
    result = tracer.trace(rays)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    segments_path = outdir / "segments.csv"
    detector_hits_path = outdir / "detector_hits.csv"
    wrote_segments = write_csv(segments_path, flatten_segment_blocks(result.segments))
    wrote_detector_hits = write_csv(detector_hits_path, flatten_detector_hits(result.detector_hits))

    source_power = float(np.sum(to_numpy(rays.power)))
    power_summary = result.detector_power_summary()
    energy_summary = detector_energy_summary(result, INTEGRATION_TIME_S)

    print("Initial scene: Gaussian source with scene scaffold")
    print(f"Integration time: {INTEGRATION_TIME_S * 1e9:.3f} ns")
    print(f"Source total power: {source_power:.6e} W")
    print("Scene elements:")
    print(f"  - {PERISCOPE_MIRROR_1.name}")
    print(f"  - {PERISCOPE_MIRROR_2.name}")
    print(f"  - {ONE_OF_MANY_MIRRORS.name}")
    print(f"  - {ON_ENTER_BEAMSPLITTER.name}")
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
    if wrote_segments:
        print(f"Wrote: {segments_path}")
    else:
        print("No ray-surface intersections were recorded.")
    if wrote_detector_hits:
        print(f"Wrote: {detector_hits_path}")
    else:
        print("No detector hits were recorded.")

    if args.plot:
        plot_path = outdir / "scene_gaussian_35ns.html"
        overlays = [
            make_circle_outline(
                name="Source waist",
                center=SOURCE_TEMPLATE.waist_position,
                normal=SOURCE_TEMPLATE.axis,
                radius=float(SOURCE_TEMPLATE.waist_radius),
                color="#1f77b4",
                in_plane_reference=SOURCE_TEMPLATE.polarization_reference,
            ),
            make_rectangle_outline(
                name=PERISCOPE_MIRROR_1.name,
                center=PERISCOPE_MIRROR_1.center,
                normal=PERISCOPE_MIRROR_1.normal,
                width=float(PERISCOPE_MIRROR_1.width),
                height=float(PERISCOPE_MIRROR_1.height),
                color="#ff7f0e",
                in_plane_reference=PERISCOPE_MIRROR_1.in_plane_reference,
            ),
            make_rectangle_outline(
                name=PERISCOPE_MIRROR_2.name,
                center=PERISCOPE_MIRROR_2.center,
                normal=PERISCOPE_MIRROR_2.normal,
                width=float(PERISCOPE_MIRROR_2.width),
                height=float(PERISCOPE_MIRROR_2.height),
                color="#2ca02c",
                in_plane_reference=PERISCOPE_MIRROR_2.in_plane_reference,
            ),
            make_rectangle_outline(
                name=ONE_OF_MANY_MIRRORS.name,
                center=ONE_OF_MANY_MIRRORS.center,
                normal=ONE_OF_MANY_MIRRORS.normal,
                width=float(ONE_OF_MANY_MIRRORS.width),
                height=float(ONE_OF_MANY_MIRRORS.height),
                color="#d62728",
                in_plane_reference=ONE_OF_MANY_MIRRORS.in_plane_reference,
            ),
            make_rectangle_outline(
                name=ON_ENTER_BEAMSPLITTER.name,
                center=ON_ENTER_BEAMSPLITTER.center,
                normal=ON_ENTER_BEAMSPLITTER.normal,
                width=float(ON_ENTER_BEAMSPLITTER.width),
                height=float(ON_ENTER_BEAMSPLITTER.height),
                color="#9467bd",
                in_plane_reference=ON_ENTER_BEAMSPLITTER.in_plane_reference,
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
