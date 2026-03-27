from __future__ import annotations

import argparse
import copy
import csv
import webbrowser
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from comsol_like_raytrace import AIR, BlockMirror, GaussianBeamSource, PlaneMirror, RayTracer, Scene, SemiTransparentMirror, TriangularPrism, to_numpy
from raytrace_plotly import (
    make_circle_outline,
    make_rectangle_outline,
    make_rectangular_prism_overlays,
    make_triangular_prism_overlays,
    write_plotly_trajectories,
)


INTEGRATION_TIME_S = 35e-9
BEAM_RADIAL_POSITIONS = 55
BEAM_CUTOFF_RATIO = 1.0
# Limits the number of secondary-ray generations via RayTracer.max_interactions.
MAX_SECONDARY_RAY_GENERATIONS = 15

PLANE_MIRROR_COLOR = "#9467bd"
SEMI_TRANSPARENT_MIRROR_COLOR = "#17becf"
BLOCK_MIRROR_COLOR = "#1f77b4"
TRIANGULAR_PRISM_COLOR = "#e377c2"


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

BLOCK_MIRROR_1 = BlockMirror(
    name="BlockMirror_44_9deg",
    center=(0.216890, 0.221857, 0.025),
    normal=(0.0, 0.0, 1.0),
    width=0.044,
    height=0.0081,
    thickness=0.05,
    in_plane_reference=(-0.036121, -0.999348, 0.0),
    reflectance=1.0,
)

BLOCK_MIRROR_2 = BlockMirror(
    name="BlockMirror_315_4deg",
    center=(-0.215102, 0.222493, 0.025),
    normal=(0.0, 0.0, 1.0),
    width=0.044,
    height=0.0081,
    thickness=0.05,
    in_plane_reference=(-0.068884, 0.997624, 0.0),
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

SMALL_REFLECTIVE_MIRROR = PlaneMirror(
    name="Turning round mirror 1",
    center=(-0.7661097252, 1.017574008, 0.02510104076),
    normal=(0.677282781278, 0.287360519850, 0.677282781278),
    shape="disk",
    radius=0.025,
    in_plane_reference=(0.707106781187, 0.0, -0.707106781187),
    reflectance=1.0,
)

TURNING_ROUND_MIRROR_2 = PlaneMirror(
    name="Turning round mirror 2",
    center=(-1.017574008, -0.7661097252, 0.02510104076),
    normal=(0.33523098, -0.62259151, -0.70710678),
    shape="disk",
    radius=0.025,
    in_plane_reference=(0.88047735, 0.47408821, 0.0),
    reflectance=1.0,
)

TURNING_ROUND_MIRROR_3 = PlaneMirror(
    name="Turning round mirror 3",
    center=(0.7661097252, -1.017574008, 0.02510104076),
    normal=(0.62259151, 0.33523098, -0.70710678),
    shape="disk",
    radius=0.025,
    in_plane_reference=(-0.47408821, 0.88047735, 0.0),
    reflectance=1.0,
)

TURNING_ROUND_MIRROR_4 = PlaneMirror(
    name="Turning round mirror 4",
    center=(1.017574008, 0.7661097252, 0.02510104076),
    normal=(-0.33523098, 0.62259151, -0.70710678),
    shape="disk",
    radius=0.025,
    in_plane_reference=(-0.88047735, -0.47408821, 0.0),
    reflectance=1.0,
)

TURNING_SQUARE_MIRROR_1 = PlaneMirror(
    name="Turning square mirror 1",
    center=(-0.9305, 0.9305, 0.02425),
    normal=(0.9893994401, -0.1452196698, 0.0),
    shape="rectangle",
    width=0.0521000001,
    height=0.024,
    in_plane_reference=(0.1452196698, 0.9893994401, 0.0),
    reflectance=1.0,
)

TURNING_SQUARE_MIRROR_2 = PlaneMirror(
    name="Turning square mirror 2",
    center=(-0.9305, -0.9305, 0.02425),
    normal=(0.1452196698, 0.9893994401, 0.0),
    shape="rectangle",
    width=0.0521000001,
    height=0.024,
    in_plane_reference=(-0.9893994401, 0.1452196698, 0.0),
    reflectance=1.0,
)

TURNING_SQUARE_MIRROR_3 = PlaneMirror(
    name="Turning square mirror 3",
    center=(0.9305, -0.9305, 0.02425),
    normal=(-0.9893994401, 0.1452196698, 0.0),
    shape="rectangle",
    width=0.0521000001,
    height=0.024,
    in_plane_reference=(-0.1452196698, -0.9893994401, 0.0),
    reflectance=1.0,
)

TURNING_SQUARE_MIRROR_4 = PlaneMirror(
    name="Turning square mirror 4",
    center=(0.9305, 0.9305, 0.02425),
    normal=(-0.1452196698, -0.9893994401, 0.0),
    shape="rectangle",
    width=0.0521000001,
    height=0.024,
    in_plane_reference=(0.9893994401, -0.1452196698, 0.0),
    reflectance=1.0,
)

SEMI_MIRROR_LEFT_1 = SemiTransparentMirror(
    name="SemiMirror_Left_1",
    center=(-0.1388869881677073, -0.2755569348737447, 0.025),
    normal=(0.0, 0.0, 1.0),
    thickness=0.05,
    n_glass=1.501,
    n_outside=AIR,
    front_reflectance=0.5,
    front_transmittance=0.5,
    back_reflectance=0.0,
    back_transmittance=1.0,
    shape="rectangle",
    width=0.044,
    height=0.0081,
    in_plane_reference=(-0.35836794954530027, -0.9335804264972017, 0.0),
)

SEMI_MIRROR_NEW = SemiTransparentMirror(
    name="SemiMirror_New_225deg",
    center=(-0.218148, -0.213819, 0.025),
    normal=(0.0, 0.0, 1.0),
    thickness=0.05,
    n_glass=1.501,
    n_outside=AIR,
    front_reflectance=0.5,
    front_transmittance=0.5,
    back_reflectance=0.0,
    back_transmittance=1.0,
    shape="rectangle",
    width=0.044,
    height=0.0081,
    in_plane_reference=(-0.997441, -0.071497, 0.0),
)

PRISM_1 = TriangularPrism(
    name="Prism_72_5deg",
    center=(0.311735, 0.098289, 0.025),
    normal=(0.0, 0.0, 1.0),
    in_plane_reference=(0.300706, -0.953717, 0.0),
    vertices_2d=[
        (-0.026645, -0.013363),
        (0.026645, -0.013363),
        (0.0, 0.026725)
    ],
    thickness=0.05,
    n_glass=1.5,
    n_outside=AIR,
    side_reflectances=[0.5, 0.0, 0.5],
    side_transmittances=[0.5, 1.0, 0.5],
)

PRISM_2 = TriangularPrism(
    name="Prism_158deg",
    center=(0.117201, -0.290082, 0.025),
    normal=(0.0, 0.0, 1.0),
    in_plane_reference=(-0.927184, -0.374607, 0.0),
    vertices_2d=[
        (-0.026645, -0.013363),
        (0.026645, -0.013363),
        (0.0, 0.026725)
    ],
    thickness=0.05,
    n_glass=1.501,
    n_outside=AIR,
    side_reflectances=[0.5, 0.0, 0.5],
    side_transmittances=[0.5, 1.0, 0.5],
)

PRISM_3 = TriangularPrism(
    name="Prism_114_5deg",
    center=(0.293337, -0.133680, 0.025),
    normal=(0.0, 0.0, 1.0),
    in_plane_reference=(-0.414693, -0.909961, 0.0),
    vertices_2d=[
        (-0.026645, -0.013362),
        (0.026645, -0.013362),
        (0.0, 0.026725)
    ],
    thickness=0.05,
    n_glass=1.501,
    n_outside=AIR,
    side_reflectances=[0.5, 0.0, 0.5],
    side_transmittances=[0.5, 1.0, 0.5],
)

PRISM_4 = TriangularPrism(
    name="Prism_240deg",
    center=(-0.275797, -0.159231, 0.025),
    normal=(0.0, 0.0, 1.0),
    in_plane_reference=(-0.5, 0.866025, 0.0),
    vertices_2d=[
        (-0.026645, -0.013363),
        (0.026645, -0.013363),
        (0.0, 0.026726)
    ],
    thickness=0.05,
    n_glass=1.501,
    n_outside=AIR,
    side_reflectances=[0.5, 0.5, 0.0],
    side_transmittances=[0.5, 0.5, 1.0],
)

PRISM_5 = TriangularPrism(
    name="Prism_285_1deg",
    center=(-0.309301, 0.083456, 0.025),
    normal=(0.0, 0.0, 1.0),
    in_plane_reference=(0.260505, 0.965471, 0.0),
    vertices_2d=[
        (-0.026645, -0.013363),
        (0.026645, -0.013363),
        (0.0, 0.026726)
    ],
    thickness=0.05,
    n_glass=1.501,
    n_outside=AIR,
    side_reflectances=[0.5, 0.5, 0.0],
    side_transmittances=[0.5, 0.5, 1.0],
)

SEMI_MIRROR_3 = SemiTransparentMirror(
    name="SemiMirror_134_5deg",
    center=(0.221274, -0.213235, 0.025),
    normal=(0.0, 0.0, 1.0),
    thickness=0.05,
    n_glass=1.501,
    n_outside=AIR,
    front_reflectance=0.5,
    front_transmittance=0.5,
    back_reflectance=0.0,
    back_transmittance=1.0,
    shape="rectangle",
    width=0.044,
    height=0.0081,
    in_plane_reference=(-0.999159, 0.041003, 0.0),
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
        copy.deepcopy(BLOCK_MIRROR_1),
        copy.deepcopy(BLOCK_MIRROR_2),
        copy.deepcopy(ON_ENTER_BEAMSPLITTER),
        copy.deepcopy(SMALL_REFLECTIVE_MIRROR),
        copy.deepcopy(TURNING_ROUND_MIRROR_2),
        copy.deepcopy(TURNING_ROUND_MIRROR_3),
        copy.deepcopy(TURNING_ROUND_MIRROR_4),
        copy.deepcopy(TURNING_SQUARE_MIRROR_1),
        copy.deepcopy(TURNING_SQUARE_MIRROR_2),
        copy.deepcopy(TURNING_SQUARE_MIRROR_3),
        copy.deepcopy(TURNING_SQUARE_MIRROR_4),
        copy.deepcopy(SEMI_MIRROR_LEFT_1),
        copy.deepcopy(SEMI_MIRROR_NEW),
        copy.deepcopy(PRISM_1),
        copy.deepcopy(PRISM_2),
        copy.deepcopy(PRISM_3),
        copy.deepcopy(PRISM_4),
        copy.deepcopy(PRISM_5),
        copy.deepcopy(SEMI_MIRROR_3),
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
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=MAX_SECONDARY_RAY_GENERATIONS,
        help="Maximum number of ray interactions / secondary-ray generations",
    )
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
    print(f"  - {BLOCK_MIRROR_1.name}")
    print(f"  - {BLOCK_MIRROR_2.name}")
    print(f"  - {ON_ENTER_BEAMSPLITTER.name}")
    print(f"  - {SMALL_REFLECTIVE_MIRROR.name}")
    print(f"  - {TURNING_ROUND_MIRROR_2.name}")
    print(f"  - {TURNING_ROUND_MIRROR_3.name}")
    print(f"  - {TURNING_ROUND_MIRROR_4.name}")
    print(f"  - {TURNING_SQUARE_MIRROR_1.name}")
    print(f"  - {TURNING_SQUARE_MIRROR_2.name}")
    print(f"  - {TURNING_SQUARE_MIRROR_3.name}")
    print(f"  - {TURNING_SQUARE_MIRROR_4.name}")
    print(f"  - {SEMI_MIRROR_LEFT_1.name}")
    print(f"  - {SEMI_MIRROR_NEW.name}")
    print(f"  - {PRISM_1.name}")
    print(f"  - {PRISM_2.name}")
    print(f"  - {PRISM_3.name}")
    print(f"  - {PRISM_4.name}")
    print(f"  - {PRISM_5.name}")
    print(f"  - {SEMI_MIRROR_3.name}")
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
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=PERISCOPE_MIRROR_1.in_plane_reference,
            ),
            make_rectangle_outline(
                name=PERISCOPE_MIRROR_2.name,
                center=PERISCOPE_MIRROR_2.center,
                normal=PERISCOPE_MIRROR_2.normal,
                width=float(PERISCOPE_MIRROR_2.width),
                height=float(PERISCOPE_MIRROR_2.height),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=PERISCOPE_MIRROR_2.in_plane_reference,
            ),
            make_rectangle_outline(
                name=ONE_OF_MANY_MIRRORS.name,
                center=ONE_OF_MANY_MIRRORS.center,
                normal=ONE_OF_MANY_MIRRORS.normal,
                width=float(ONE_OF_MANY_MIRRORS.width),
                height=float(ONE_OF_MANY_MIRRORS.height),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=ONE_OF_MANY_MIRRORS.in_plane_reference,
            ),
            *make_rectangular_prism_overlays(
                name=BLOCK_MIRROR_1.name,
                center=BLOCK_MIRROR_1.center,
                normal=BLOCK_MIRROR_1.normal,
                width=float(BLOCK_MIRROR_1.width),
                height=float(BLOCK_MIRROR_1.height),
                thickness=float(BLOCK_MIRROR_1.thickness),
                color=BLOCK_MIRROR_COLOR,
                in_plane_reference=BLOCK_MIRROR_1.in_plane_reference,
            ),
            *make_rectangular_prism_overlays(
                name=BLOCK_MIRROR_2.name,
                center=BLOCK_MIRROR_2.center,
                normal=BLOCK_MIRROR_2.normal,
                width=float(BLOCK_MIRROR_2.width),
                height=float(BLOCK_MIRROR_2.height),
                thickness=float(BLOCK_MIRROR_2.thickness),
                color=BLOCK_MIRROR_COLOR,
                in_plane_reference=BLOCK_MIRROR_2.in_plane_reference,
            ),
            make_rectangle_outline(
                name=ON_ENTER_BEAMSPLITTER.name,
                center=ON_ENTER_BEAMSPLITTER.center,
                normal=ON_ENTER_BEAMSPLITTER.normal,
                width=float(ON_ENTER_BEAMSPLITTER.width),
                height=float(ON_ENTER_BEAMSPLITTER.height),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=ON_ENTER_BEAMSPLITTER.in_plane_reference,
            ),
            make_circle_outline(
                name=SMALL_REFLECTIVE_MIRROR.name,
                center=SMALL_REFLECTIVE_MIRROR.center,
                normal=SMALL_REFLECTIVE_MIRROR.normal,
                radius=float(SMALL_REFLECTIVE_MIRROR.radius),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=SMALL_REFLECTIVE_MIRROR.in_plane_reference,
            ),
            make_circle_outline(
                name=TURNING_ROUND_MIRROR_2.name,
                center=TURNING_ROUND_MIRROR_2.center,
                normal=TURNING_ROUND_MIRROR_2.normal,
                radius=float(TURNING_ROUND_MIRROR_2.radius),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=TURNING_ROUND_MIRROR_2.in_plane_reference,
            ),
            make_circle_outline(
                name=TURNING_ROUND_MIRROR_3.name,
                center=TURNING_ROUND_MIRROR_3.center,
                normal=TURNING_ROUND_MIRROR_3.normal,
                radius=float(TURNING_ROUND_MIRROR_3.radius),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=TURNING_ROUND_MIRROR_3.in_plane_reference,
            ),
            make_circle_outline(
                name=TURNING_ROUND_MIRROR_4.name,
                center=TURNING_ROUND_MIRROR_4.center,
                normal=TURNING_ROUND_MIRROR_4.normal,
                radius=float(TURNING_ROUND_MIRROR_4.radius),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=TURNING_ROUND_MIRROR_4.in_plane_reference,
            ),
            make_rectangle_outline(
                name=TURNING_SQUARE_MIRROR_1.name,
                center=TURNING_SQUARE_MIRROR_1.center,
                normal=TURNING_SQUARE_MIRROR_1.normal,
                width=float(TURNING_SQUARE_MIRROR_1.width),
                height=float(TURNING_SQUARE_MIRROR_1.height),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=TURNING_SQUARE_MIRROR_1.in_plane_reference,
            ),
            make_rectangle_outline(
                name=TURNING_SQUARE_MIRROR_2.name,
                center=TURNING_SQUARE_MIRROR_2.center,
                normal=TURNING_SQUARE_MIRROR_2.normal,
                width=float(TURNING_SQUARE_MIRROR_2.width),
                height=float(TURNING_SQUARE_MIRROR_2.height),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=TURNING_SQUARE_MIRROR_2.in_plane_reference,
            ),
            make_rectangle_outline(
                name=TURNING_SQUARE_MIRROR_3.name,
                center=TURNING_SQUARE_MIRROR_3.center,
                normal=TURNING_SQUARE_MIRROR_3.normal,
                width=float(TURNING_SQUARE_MIRROR_3.width),
                height=float(TURNING_SQUARE_MIRROR_3.height),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=TURNING_SQUARE_MIRROR_3.in_plane_reference,
            ),
            make_rectangle_outline(
                name=TURNING_SQUARE_MIRROR_4.name,
                center=TURNING_SQUARE_MIRROR_4.center,
                normal=TURNING_SQUARE_MIRROR_4.normal,
                width=float(TURNING_SQUARE_MIRROR_4.width),
                height=float(TURNING_SQUARE_MIRROR_4.height),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=TURNING_SQUARE_MIRROR_4.in_plane_reference,
            ),
            *make_rectangular_prism_overlays(
                name=SEMI_MIRROR_LEFT_1.name,
                center=SEMI_MIRROR_LEFT_1.center,
                normal=SEMI_MIRROR_LEFT_1.normal,
                width=float(SEMI_MIRROR_LEFT_1.width),
                height=float(SEMI_MIRROR_LEFT_1.height),
                thickness=float(SEMI_MIRROR_LEFT_1.thickness),
                color=SEMI_TRANSPARENT_MIRROR_COLOR,
                in_plane_reference=SEMI_MIRROR_LEFT_1.in_plane_reference,
            ),
            *make_rectangular_prism_overlays(
                name=SEMI_MIRROR_NEW.name,
                center=SEMI_MIRROR_NEW.center,
                normal=SEMI_MIRROR_NEW.normal,
                width=float(SEMI_MIRROR_NEW.width),
                height=float(SEMI_MIRROR_NEW.height),
                thickness=float(SEMI_MIRROR_NEW.thickness),
                color=SEMI_TRANSPARENT_MIRROR_COLOR,
                in_plane_reference=SEMI_MIRROR_NEW.in_plane_reference,
            ),
            *make_triangular_prism_overlays(
                name=PRISM_1.name,
                center=PRISM_1.center,
                normal=PRISM_1.normal,
                vertices_2d=PRISM_1.vertices_2d,
                thickness=float(PRISM_1.thickness),
                color=TRIANGULAR_PRISM_COLOR,
                in_plane_reference=PRISM_1.in_plane_reference,
            ),
            *make_triangular_prism_overlays(
                name=PRISM_2.name,
                center=PRISM_2.center,
                normal=PRISM_2.normal,
                vertices_2d=PRISM_2.vertices_2d,
                thickness=float(PRISM_2.thickness),
                color=TRIANGULAR_PRISM_COLOR,
                in_plane_reference=PRISM_2.in_plane_reference,
            ),
            *make_triangular_prism_overlays(
                name=PRISM_3.name,
                center=PRISM_3.center,
                normal=PRISM_3.normal,
                vertices_2d=PRISM_3.vertices_2d,
                thickness=float(PRISM_3.thickness),
                color=TRIANGULAR_PRISM_COLOR,
                in_plane_reference=PRISM_3.in_plane_reference,
            ),
            *make_triangular_prism_overlays(
                name=PRISM_4.name,
                center=PRISM_4.center,
                normal=PRISM_4.normal,
                vertices_2d=PRISM_4.vertices_2d,
                thickness=float(PRISM_4.thickness),
                color=TRIANGULAR_PRISM_COLOR,
                in_plane_reference=PRISM_4.in_plane_reference,
            ),
            *make_triangular_prism_overlays(
                name=PRISM_5.name,
                center=PRISM_5.center,
                normal=PRISM_5.normal,
                vertices_2d=PRISM_5.vertices_2d,
                thickness=float(PRISM_5.thickness),
                color=TRIANGULAR_PRISM_COLOR,
                in_plane_reference=PRISM_5.in_plane_reference,
            ),
            *make_rectangular_prism_overlays(
                name=SEMI_MIRROR_3.name,
                center=SEMI_MIRROR_3.center,
                normal=SEMI_MIRROR_3.normal,
                width=float(SEMI_MIRROR_3.width),
                height=float(SEMI_MIRROR_3.height),
                thickness=float(SEMI_MIRROR_3.thickness),
                color=SEMI_TRANSPARENT_MIRROR_COLOR,
                in_plane_reference=SEMI_MIRROR_3.in_plane_reference,
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
