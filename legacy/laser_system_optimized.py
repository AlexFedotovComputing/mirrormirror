from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import math
import subprocess
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from itertools import repeat
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np

from raytrace import (
    AIR,
    BlockMirror,
    CylindricalScreen,
    Detector,
    GaussianBeamSource,
    MirrorArrayBundle,
    PlaneMirror,
    RayTracer,
    Scene,
    SemiTransparentMirror,
    TriangularPrism,
    to_numpy,
)
from vizual import (
    make_circle_outline,
    make_cylindrical_surface_overlays,
    make_disk_overlays,
    make_rectangle_outline,
    make_rectangular_prism_overlays,
    make_triangular_prism_overlays,
    write_plotly_trajectories,
)

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

INTEGRATION_TIME_S = 50e-9
BEAM_RADIAL_POSITIONS = 55
INITIAL_RAY_COUNT = 20_000
BEAM_CUTOFF_RATIO = 1.0
MAX_SECONDARY_RAY_GENERATIONS = 20

PLANE_MIRROR_COLOR = "#9467bd"
SEMI_TRANSPARENT_MIRROR_COLOR = "#17becf"
BLOCK_MIRROR_COLOR = "#1f77b4"
TRIANGULAR_PRISM_COLOR = "#e377c2"
SCREEN_COLOR = "#2ca02c"
CYLINDRICAL_SCREEN_COLOR = "#7f7f7f"

# ---------------------------------------------------------------------------
# Bundle geometry
# ---------------------------------------------------------------------------

BUNDLE_1_1_RADIUS_M = math.sqrt(0.5) * 1e-3 + 1e-9
BUNDLE_1_1_ROTATION_DEG = 0.0
BUNDLE_1_1_ROTATION_CENTER = (-0.7649020959, 1.026102123, -2.11135)

BUNDLE_1_2_Z_SHIFT_M = -0.33
BUNDLE_1_3_Z_SHIFT_M = -0.66
BUNDLE_1_4_Z_SHIFT_M = -0.99

BUNDLE_2_CENTER_X = -1.01917119
BUNDLE_2_CENTER_Y = -0.765885541
BUNDLE_2_ROTATION_DEG = 90.0
BUNDLE_2_ROTATION_CENTER = (
    BUNDLE_2_CENTER_X,
    BUNDLE_2_CENTER_Y,
    BUNDLE_1_1_ROTATION_CENTER[2],
)
BUNDLE_2_DX = BUNDLE_2_CENTER_X - BUNDLE_1_1_ROTATION_CENTER[0]
BUNDLE_2_DY = BUNDLE_2_CENTER_Y - BUNDLE_1_1_ROTATION_CENTER[1]

BUNDLE_3_CENTER_X = 0.7658855413
BUNDLE_3_CENTER_Y = -1.01917119
BUNDLE_3_ROTATION_DEG = -180.0
BUNDLE_3_ROTATION_CENTER = (
    BUNDLE_3_CENTER_X,
    BUNDLE_3_CENTER_Y,
    BUNDLE_1_1_ROTATION_CENTER[2],
)
BUNDLE_3_DX = BUNDLE_3_CENTER_X - BUNDLE_1_1_ROTATION_CENTER[0]
BUNDLE_3_DY = BUNDLE_3_CENTER_Y - BUNDLE_1_1_ROTATION_CENTER[1]

BUNDLE_4_CENTER_X = 1.017417566
BUNDLE_4_CENTER_Y = 0.752082817
BUNDLE_4_ROTATION_DEG = -90.0
BUNDLE_4_ROTATION_CENTER = (
    BUNDLE_4_CENTER_X,
    BUNDLE_4_CENTER_Y,
    BUNDLE_1_1_ROTATION_CENTER[2],
)
BUNDLE_4_DX = BUNDLE_4_CENTER_X - BUNDLE_1_1_ROTATION_CENTER[0]
BUNDLE_4_DY = BUNDLE_4_CENTER_Y - BUNDLE_1_1_ROTATION_CENTER[1]

SCREEN_1 = Detector(
    name="Screen 1",
    center=(-0.7658855413, 1.01917119, -3.4505),
    normal=(0.0, 0.0, 1.0),
    shape="disk",
    radius=0.025,
    in_plane_reference=(1.0, 0.0, 0.0),
)

SCREEN_2 = Detector(
    name="Screen 2",
    center=(-1.01917119, -0.7658855413, -3.4505),
    normal=(0.0, 0.0, 1.0),
    shape="disk",
    radius=0.025,
    in_plane_reference=(1.0, 0.0, 0.0),
)

SCREEN_3 = Detector(
    name="Screen 3",
    center=(0.7658855413, -1.01917119, -3.4505),
    normal=(0.0, 0.0, 1.0),
    shape="disk",
    radius=0.025,
    in_plane_reference=(1.0, 0.0, 0.0),
)

SCREEN_4 = Detector(
    name="Screen 4",
    center=(1.01917119, 0.7658855413, -3.4505),
    normal=(0.0, 0.0, 1.0),
    shape="disk",
    radius=0.025,
    in_plane_reference=(1.0, 0.0, 0.0),
)

CYLINDRICAL_SCREEN_1 = CylindricalScreen(
    name="Cylindrical Screen 1",
    center=(0.0, 0.0, -1.7655),
    axis=(0.0, 0.0, 1.0),
    radius=0.5,
    length=3.33,
    detector=True,
)

BUNDLE_RAY_INNER_CYLINDER_RADIUS_M = 0.35
BUNDLE_RAY_OUTER_CYLINDER_RADIUS_M = 1.35
BUNDLE_RAY_Z_MIN_M = (
    float(CYLINDRICAL_SCREEN_1.center[2])
    - 0.5 * float(CYLINDRICAL_SCREEN_1.length)
)
BUNDLE_RAY_Z_MAX_M = (
    float(CYLINDRICAL_SCREEN_1.center[2])
    + 0.5 * float(CYLINDRICAL_SCREEN_1.length)
)

_BUNDLE_1_1_BASE_DATA = [
    {
        "name": "BUNDLE_1_1 Mirror 1",
        "phi": 31.424112270736885,
        "center": (-0.7647815417, 1.0251094165, -2.1122),
        "radius": BUNDLE_1_1_RADIUS_M,
    },
    {
        "name": "BUNDLE_1_1 Mirror 2",
        "phi": 25.924090268233982,
        "center": (-0.76570152805, 1.025501367, -2.11135),
        "radius": BUNDLE_1_1_RADIUS_M,
    },
    {
        "name": "BUNDLE_1_1 Mirror 3",
        "phi": 20.42413750245636,
        "center": (-0.7658220822, 1.02649407375, -2.1105),
        "radius": BUNDLE_1_1_RADIUS_M,
    },
    {
        "name": "BUNDLE_1_1 Mirror 4",
        "phi": 53.42408141233929,
        "center": (-0.76502265, 1.02709483, -2.1105),
        "radius": BUNDLE_1_1_RADIUS_M,
    },
    {
        "name": "BUNDLE_1_1 Mirror 5",
        "phi": 47.92405613649733,
        "center": (-0.764102663675, 1.0267028795, -2.11135),
        "radius": BUNDLE_1_1_RADIUS_M,
    },
    {
        "name": "BUNDLE_1_1 Mirror 6",
        "phi": 42.42405433213915,
        "center": (-0.7639821095, 1.02571017275, -2.1122),
        "radius": BUNDLE_1_1_RADIUS_M,
    },
    {
        "name": "BUNDLE_1_1 Mirror 7",
        "phi": 36.92407312003851,
        "center": (-0.764902095875, 1.026102123, -2.11135),
        "radius": BUNDLE_1_1_RADIUS_M,
    },
]


def _rotate_xy_point(
    point: tuple[float, float, float],
    pivot: tuple[float, float, float],
    angle_deg: float,
) -> tuple[float, float, float]:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = point[0] - pivot[0]
    dy = point[1] - pivot[1]
    return (
        pivot[0] + cos_a * dx - sin_a * dy,
        pivot[1] + sin_a * dx + cos_a * dy,
        point[2],
    )


def _rotate_bundle_data(
    data: List[Dict[str, object]],
    pivot: tuple[float, float, float],
    angle_deg: float,
) -> List[Dict[str, object]]:
    rotated: List[Dict[str, object]] = []
    for item in data:
        center = tuple(float(v) for v in item["center"])
        rotated.append(
            {
                **item,
                "center": _rotate_xy_point(center, pivot, angle_deg),
                "phi": float(item["phi"]) + angle_deg,
            }
        )
    return rotated


def _translate_bundle_data(
    data: List[Dict[str, object]],
    *,
    name_prefix: str,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> List[Dict[str, object]]:
    translated: List[Dict[str, object]] = []
    for idx, item in enumerate(data, start=1):
        center = tuple(float(v) for v in item["center"])
        translated.append(
            {
                **item,
                "name": f"{name_prefix} Mirror {idx}",
                "center": (
                    center[0] + dx,
                    center[1] + dy,
                    center[2] + dz,
                ),
            }
        )
    return translated


def _center_of_bundle_data(
    data: List[Dict[str, object]],
) -> tuple[float, float, float]:
    centers = np.asarray([item["center"] for item in data], dtype=float)
    return tuple(float(v) for v in np.mean(centers, axis=0))


def _recenter_bundle_data(
    data: List[Dict[str, object]],
    *,
    name_prefix: str,
    center: tuple[float, float, float],
) -> List[Dict[str, object]]:
    current_center = _center_of_bundle_data(data)
    return _translate_bundle_data(
        data,
        name_prefix=name_prefix,
        dx=float(center[0]) - current_center[0],
        dy=float(center[1]) - current_center[1],
        dz=float(center[2]) - current_center[2],
    )


BUNDLE_1_1_DATA = _rotate_bundle_data(
    _BUNDLE_1_1_BASE_DATA,
    BUNDLE_1_1_ROTATION_CENTER,
    BUNDLE_1_1_ROTATION_DEG,
)

BUNDLE_1_2_DATA = _translate_bundle_data(
    BUNDLE_1_1_DATA,
    name_prefix="BUNDLE_1_2",
    dz=BUNDLE_1_2_Z_SHIFT_M,
)
BUNDLE_1_3_DATA = _translate_bundle_data(
    BUNDLE_1_1_DATA,
    name_prefix="BUNDLE_1_3",
    dz=BUNDLE_1_3_Z_SHIFT_M,
)
BUNDLE_1_4_DATA = _translate_bundle_data(
    BUNDLE_1_1_DATA,
    name_prefix="BUNDLE_1_4",
    dz=BUNDLE_1_4_Z_SHIFT_M,
)


def _make_bundle_2_data(
    name_prefix: str, dz: float = 0.0
) -> List[Dict[str, object]]:
    return _rotate_bundle_data(
        _translate_bundle_data(
            BUNDLE_1_1_DATA,
            name_prefix=name_prefix,
            dx=BUNDLE_2_DX,
            dy=BUNDLE_2_DY,
            dz=dz,
        ),
        BUNDLE_2_ROTATION_CENTER,
        BUNDLE_2_ROTATION_DEG,
    )


def _make_bundle_3_data(
    name_prefix: str, dz: float = 0.0
) -> List[Dict[str, object]]:
    return _rotate_bundle_data(
        _translate_bundle_data(
            BUNDLE_1_1_DATA,
            name_prefix=name_prefix,
            dx=BUNDLE_3_DX,
            dy=BUNDLE_3_DY,
            dz=dz,
        ),
        BUNDLE_3_ROTATION_CENTER,
        BUNDLE_3_ROTATION_DEG,
    )


def _make_bundle_4_data(
    name_prefix: str, dz: float = 0.0
) -> List[Dict[str, object]]:
    return _rotate_bundle_data(
        _translate_bundle_data(
            BUNDLE_1_1_DATA,
            name_prefix=name_prefix,
            dx=BUNDLE_4_DX,
            dy=BUNDLE_4_DY,
            dz=dz,
        ),
        BUNDLE_4_ROTATION_CENTER,
        BUNDLE_4_ROTATION_DEG,
    )


BUNDLE_2_1_DATA = _make_bundle_2_data("BUNDLE_2_1")
BUNDLE_2_2_DATA = _make_bundle_2_data("BUNDLE_2_2", dz=BUNDLE_1_2_Z_SHIFT_M)
BUNDLE_2_3_DATA = _make_bundle_2_data("BUNDLE_2_3", dz=BUNDLE_1_3_Z_SHIFT_M)
BUNDLE_2_4_DATA = _make_bundle_2_data("BUNDLE_2_4", dz=BUNDLE_1_4_Z_SHIFT_M)

BUNDLE_3_1_DATA = _make_bundle_3_data("BUNDLE_3_1")
BUNDLE_3_2_DATA = _make_bundle_3_data("BUNDLE_3_2", dz=BUNDLE_1_2_Z_SHIFT_M)
BUNDLE_3_3_DATA = _make_bundle_3_data("BUNDLE_3_3", dz=BUNDLE_1_3_Z_SHIFT_M)
BUNDLE_3_4_DATA = _make_bundle_3_data("BUNDLE_3_4", dz=BUNDLE_1_4_Z_SHIFT_M)

BUNDLE_4_1_DATA = _make_bundle_4_data("BUNDLE_4_1")
BUNDLE_4_2_DATA = _make_bundle_4_data("BUNDLE_4_2", dz=BUNDLE_1_2_Z_SHIFT_M)
BUNDLE_4_3_DATA = _make_bundle_4_data("BUNDLE_4_3", dz=BUNDLE_1_3_Z_SHIFT_M)
BUNDLE_4_4_DATA = _make_bundle_4_data("BUNDLE_4_4", dz=BUNDLE_1_4_Z_SHIFT_M)

_BUNDLE_TARGET_CENTERS = {
    (1, 1): (-0.7649020959, 1.026102123, -2.11135),
    (1, 2): (-0.7728164745, 1.020154635, -2.44135),
    (1, 3): (-0.7589546081, 1.018187745, -2.77135),
    (1, 4): (-0.7668689867, 1.012240257, -3.10135),
    (2, 1): (-1.026102123, -0.7649020959, -2.11135),
    (2, 2): (-1.020154635, -0.7728164745, -2.44135),
    (2, 3): (-1.018187745, -0.7589546081, -2.77135),
    (2, 4): (-1.012240257, -0.7668689867, -3.10135),
    (3, 1): (0.7649020959, -1.026102123, -2.11135),
    (3, 2): (0.7728164745, -1.020154635, -2.44135),
    (3, 3): (0.7589546081, -1.018187745, -2.77135),
    (3, 4): (0.7668689867, -1.012240257, -3.10135),
    (4, 1): (1.026102123, 0.7649020959, -2.11135),
    (4, 2): (1.020154635, 0.7728164745, -2.44135),
    (4, 3): (1.018187745, 0.7589546081, -2.77135),
    (4, 4): (1.012240257, 0.7668689867, -3.10135),
}

for _sector in range(1, 5):
    for _layer in range(1, 5):
        _var = f"BUNDLE_{_sector}_{_layer}_DATA"
        globals()[_var] = _recenter_bundle_data(
            globals()[_var],
            name_prefix=f"BUNDLE_{_sector}_{_layer}",
            center=_BUNDLE_TARGET_CENTERS[(_sector, _layer)],
        )


def make_bundle(name: str, data: List[Dict[str, object]]) -> MirrorArrayBundle:
    return MirrorArrayBundle(
        data,
        name=name,
        reflectance=1.0,
        transmittance=0.0,
        reflect_from_minus_side=True,
        reflect_from_plus_side=False,
    )


def make_bundle_by_index(sector: int, layer: int) -> MirrorArrayBundle:
    name = f"BUNDLE_{sector}_{layer}"
    return make_bundle(name, globals()[f"{name}_DATA"])


# Compatibility wrappers kept because external scripts/tests may import them.
def make_bundle_1_1() -> MirrorArrayBundle: return make_bundle_by_index(1, 1)
def make_bundle_1_2() -> MirrorArrayBundle: return make_bundle_by_index(1, 2)
def make_bundle_1_3() -> MirrorArrayBundle: return make_bundle_by_index(1, 3)
def make_bundle_1_4() -> MirrorArrayBundle: return make_bundle_by_index(1, 4)
def make_bundle_2_1() -> MirrorArrayBundle: return make_bundle_by_index(2, 1)
def make_bundle_2_2() -> MirrorArrayBundle: return make_bundle_by_index(2, 2)
def make_bundle_2_3() -> MirrorArrayBundle: return make_bundle_by_index(2, 3)
def make_bundle_2_4() -> MirrorArrayBundle: return make_bundle_by_index(2, 4)
def make_bundle_3_1() -> MirrorArrayBundle: return make_bundle_by_index(3, 1)
def make_bundle_3_2() -> MirrorArrayBundle: return make_bundle_by_index(3, 2)
def make_bundle_3_3() -> MirrorArrayBundle: return make_bundle_by_index(3, 3)
def make_bundle_3_4() -> MirrorArrayBundle: return make_bundle_by_index(3, 4)
def make_bundle_4_1() -> MirrorArrayBundle: return make_bundle_by_index(4, 1)
def make_bundle_4_2() -> MirrorArrayBundle: return make_bundle_by_index(4, 2)
def make_bundle_4_3() -> MirrorArrayBundle: return make_bundle_by_index(4, 3)
def make_bundle_4_4() -> MirrorArrayBundle: return make_bundle_by_index(4, 4)


# ---------------------------------------------------------------------------
# Reconstructed microassembly geometry
# ---------------------------------------------------------------------------

RECONSTRUCTED_MICROASSEMBLIES_PATH = Path(__file__).with_name(
    "Восстановленные_микросборки.txt"
)


def _normalized_reconstructed_vector(
    vector: np.ndarray,
) -> tuple[float, float, float]:
    length = float(np.linalg.norm(vector))
    if length <= 1e-15:
        raise ValueError(
            "Cannot normalize a zero-length reconstructed geometry vector."
        )
    return tuple(float(value) for value in vector / length)


def _reconstructed_mirror_config(
    points: Sequence[Sequence[float]],
) -> Dict[str, object]:
    point_arrays = [np.asarray(point, dtype=float) for point in points]
    if len(point_arrays) != 4:
        raise ValueError(
            f"A reconstructed mirror must have four points, got {len(point_arrays)}."
        )

    p0, p1, _, p3 = point_arrays
    center = np.mean(np.stack(point_arrays, axis=0), axis=0)
    in_plane_reference = _normalized_reconstructed_vector(p1 - p0)

    normal_vector = np.cross(p1 - p0, p3 - p0)
    if normal_vector[2] > 0.0:
        normal_vector = -normal_vector
    normal = _normalized_reconstructed_vector(normal_vector)

    return {
        "center": tuple(float(value) for value in center),
        "phi": math.degrees(
            math.atan2(p1[1] - p3[1], p1[0] - p3[0])
        ),
        "normal": normal,
        "in_plane_reference": in_plane_reference,
        "reconstructed_points": [
            tuple(float(value) for value in point)
            for point in point_arrays
        ],
    }


def _load_reconstructed_microassemblies(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Reconstructed mirror positions file not found: {path}"
        )

    loaded: Dict[str, Dict[int, Dict[str, object]]] = {}

    with path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(
            csv.reader(handle, delimiter="\t"),
            start=1,
        ):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != 5:
                raise ValueError(
                    f"Unexpected row {row_number} in {path.name}: {row!r}"
                )

            key_parts = row[0].strip().split(".")
            if len(key_parts) != 3:
                raise ValueError(
                    f"Invalid mirror key at row {row_number}: {row[0]!r}"
                )

            sector, layer, mirror_index = (
                int(value) for value in key_parts
            )
            bundle_key = f"{sector}.{layer}"
            points = [
                ast.literal_eval(cell.strip())
                for cell in row[1:]
            ]
            loaded.setdefault(bundle_key, {})[
                mirror_index
            ] = _reconstructed_mirror_config(points)

    expected_bundle_keys = {
        f"{sector}.{layer}"
        for sector in range(1, 5)
        for layer in range(1, 5)
    }

    if set(loaded) != expected_bundle_keys:
        missing = sorted(expected_bundle_keys - set(loaded))
        extra = sorted(set(loaded) - expected_bundle_keys)
        raise ValueError(
            "Unexpected reconstructed bundle set; "
            f"missing={missing}, extra={extra}"
        )

    for sector in range(1, 5):
        for layer in range(1, 5):
            bundle_key = f"{sector}.{layer}"
            items = loaded[bundle_key]

            if set(items) != set(range(1, 8)):
                raise ValueError(
                    f"Bundle {bundle_key} must contain mirror indices 1..7."
                )

            bundle_variable = f"BUNDLE_{sector}_{layer}_DATA"
            original_items = globals()[bundle_variable]

            globals()[bundle_variable] = [
                {
                    **original_items[mirror_index - 1],
                    **items[mirror_index],
                    "name": (
                        f"BUNDLE_{sector}_{layer} Mirror {mirror_index}"
                    ),
                }
                for mirror_index in range(1, 8)
            ]


_load_reconstructed_microassemblies(RECONSTRUCTED_MICROASSEMBLIES_PATH)

BUNDLE_SECTOR_TRANSLATIONS_M = {
    1: (-0.0198703276795, -0.0026805251862, 0.0),
    2: (0.0025362321414, -0.0193196587392, 0.0),
    3: (0.0160000000000, 0.0040000000000, 0.0),
    4: (-0.0032595754172, 0.0181154172432, 0.0),
}


def _apply_bundle_sector_translations() -> None:
    for sector, translation in BUNDLE_SECTOR_TRANSLATIONS_M.items():
        offset = np.asarray(translation, dtype=float)

        for layer in range(1, 5):
            bundle_data = globals()[f"BUNDLE_{sector}_{layer}_DATA"]

            for mirror in bundle_data:
                mirror["center"] = tuple(
                    float(value)
                    for value in (
                        np.asarray(mirror["center"], dtype=float) + offset
                    )
                )

                reconstructed_points = mirror.get("reconstructed_points")
                if reconstructed_points is not None:
                    mirror["reconstructed_points"] = [
                        tuple(
                            float(value)
                            for value in (
                                np.asarray(point, dtype=float) + offset
                            )
                        )
                        for point in reconstructed_points
                    ]


_apply_bundle_sector_translations()

# ---------------------------------------------------------------------------
# Source and optical elements
# ---------------------------------------------------------------------------

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
    in_plane_reference=(-0.03917316153986984, -0.9992330134457985, 0.0),
    reflectance=1.0,
)

BLOCK_MIRROR_2 = BlockMirror(
    name="BlockMirror_315_4deg",
    center=(-0.215102, 0.222493, 0.025),
    normal=(0.0, 0.0, 1.0),
    width=0.044,
    height=0.0081,
    thickness=0.05,
    in_plane_reference=(-0.0710603122788671, 0.9974713443757819, 0.0),
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
    normal=(-0.62259151, -0.33523098, -0.70710678),
    shape="disk",
    radius=0.025,
    in_plane_reference=(0.47408821, -0.88047735, 0.0),
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
    n_glass=1.5,
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
    n_glass=1.5,
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
    center=(0.3112809928, 0.09698887063, 0.025),
    normal=(0.0, 0.0, 1.0),
    in_plane_reference=(0.26723837659979455, -0.9636304524781525, 0.0),
    vertices_2d=[
        (-0.026644999988928504, -0.013362922497475945),
        (0.02664499999877028, -0.013362922497475945),
        (-9.841807056820695e-12, 0.026725844994951786),
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
    in_plane_reference=(-0.9242132772115091, -0.3818774574794407, 0.0),
    vertices_2d=[
        (-0.026645, -0.013363),
        (0.026645, -0.013363),
        (0.0, 0.026725),
    ],
    thickness=0.05,
    n_glass=1.5,
    n_outside=AIR,
    side_reflectances=[0.5, 0.0, 0.5],
    side_transmittances=[0.5, 1.0, 0.5],
)

PRISM_3 = TriangularPrism(
    name="Prism_114_5deg",
    center=(0.29286935123333335, -0.13470485113333333, 0.025),
    normal=(0.0, 0.0, 1.0),
    in_plane_reference=(0.5253231491917432, -0.8509028134563072, 0.0),
    vertices_2d=[
        (0.003620019968861894, -0.029587483460877778),
        (0.022257942427050502, 0.014793741799430887),
        (-0.02587796239591241, 0.01479374166144684),
    ],
    thickness=0.05,
    n_glass=1.5,
    n_outside=AIR,
    side_reflectances=[0.0, 0.5, 0.5],
    side_transmittances=[1.0, 0.5, 0.5],
)

PRISM_4 = TriangularPrism(
    name="Prism_240deg",
    center=(-0.275797, -0.159231, 0.025),
    normal=(0.0, 0.0, 1.0),
    in_plane_reference=(-0.5, 0.866025, 0.0),
    vertices_2d=[
        (-0.026645, -0.013363),
        (0.026645, -0.013363),
        (0.0, 0.026726),
    ],
    thickness=0.05,
    n_glass=1.5,
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
        (0.0, 0.026726),
    ],
    thickness=0.05,
    n_glass=1.5,
    n_outside=AIR,
    side_reflectances=[0.5, 0.5, 0.0],
    side_transmittances=[0.5, 0.5, 1.0],
)

SEMI_MIRROR_3 = SemiTransparentMirror(
    name="SemiMirror_134_5deg",
    center=(0.221274, -0.213235, 0.025),
    normal=(0.0, 0.0, 1.0),
    thickness=0.05,
    n_glass=1.5,
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

BASE_SCENE_OBJECTS = (
    PERISCOPE_MIRROR_1,
    PERISCOPE_MIRROR_2,
    ONE_OF_MANY_MIRRORS,
    BLOCK_MIRROR_1,
    BLOCK_MIRROR_2,
    ON_ENTER_BEAMSPLITTER,
    SMALL_REFLECTIVE_MIRROR,
    TURNING_ROUND_MIRROR_2,
    TURNING_ROUND_MIRROR_3,
    TURNING_ROUND_MIRROR_4,
    TURNING_SQUARE_MIRROR_1,
    TURNING_SQUARE_MIRROR_2,
    TURNING_SQUARE_MIRROR_3,
    TURNING_SQUARE_MIRROR_4,
    SEMI_MIRROR_LEFT_1,
    SEMI_MIRROR_NEW,
    PRISM_1,
    PRISM_2,
    PRISM_3,
    PRISM_4,
    PRISM_5,
    SEMI_MIRROR_3,
    SCREEN_1,
    SCREEN_2,
    SCREEN_3,
    SCREEN_4,
    CYLINDRICAL_SCREEN_1,
)

# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------


def build_initial_source(backend: str = "numpy") -> GaussianBeamSource:
    source = copy.deepcopy(SOURCE_TEMPLATE)
    source.backend = backend
    return source


def _hexagonal_ray_count(radial_positions: int) -> int:
    """Number of rays emitted by the source for a hexagonal ring layout."""
    n = int(radial_positions)
    if n < 0:
        raise ValueError("radial_positions must be non-negative")
    return 1 + 3 * n * (n + 1)


def _minimum_radial_positions_for_count(target_count: int) -> int:
    """Smallest n for which 1 + 3*n*(n+1) >= target_count."""
    if target_count <= 1:
        return 0

    estimate = math.ceil(
        (-3.0 + math.sqrt(12.0 * target_count - 3.0)) / 6.0
    )

    # Guard against floating-point rounding at exact boundaries.
    while _hexagonal_ray_count(estimate) < target_count:
        estimate += 1
    while estimate > 0 and _hexagonal_ray_count(estimate - 1) >= target_count:
        estimate -= 1

    return estimate


def emit_initial_rays(
    source: GaussianBeamSource,
    target_count: int = INITIAL_RAY_COUNT,
):
    """
    Emit exactly target_count rays while preserving the original sampling.

    The old implementation repeatedly called source.emit() while incrementing
    radial_positions. For 20,000 rays it performed 28 full emissions and
    generated 405,412 temporary rays.

    The source layout has the exact count N(n) = 1 + 3*n*(n+1), so the required
    radial_positions can be calculated before the single emission.

    max(current_n, required_n) is deliberate: it preserves the old behaviour
    for target counts smaller than the SOURCE_TEMPLATE's initial 55 rings.
    """
    if target_count <= 0:
        raise ValueError("target_count must be positive")

    required_positions = _minimum_radial_positions_for_count(target_count)
    source.radial_positions = max(
        int(source.radial_positions),
        required_positions,
    )

    rays = source.emit()

    if rays.n_rays < target_count:
        raise RuntimeError(
            "GaussianBeamSource emitted fewer rays than predicted by the "
            "hexagonal-ring count formula."
        )

    if rays.n_rays == target_count:
        return rays

    # Keep exactly the same deterministic trimming rule as the original code.
    keep_idx = np.floor(
        np.linspace(
            0,
            rays.n_rays,
            target_count,
            endpoint=False,
        )
    ).astype(np.int64)

    return rays.subset(
        rays.xp.asarray(keep_idx, dtype=np.int64)
    )


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def build_initial_scene() -> Scene:
    scene = Scene()

    # Preserve the original physical surface order.
    scene.add(*(copy.deepcopy(obj) for obj in BASE_SCENE_OBJECTS))

    for sector in range(1, 5):
        for layer in range(1, 5):
            bundle = make_bundle_by_index(sector, layer)
            scene.add(*bundle.build_surfaces())

    return scene


def build_bundle_overlays(
    bundle: MirrorArrayBundle,
) -> List[Dict[str, object]]:
    overlays: List[Dict[str, object]] = []

    for config, mirror in zip(
        bundle.configs,
        bundle.build_surfaces(),
    ):
        if isinstance(config, dict) and "reconstructed_points" in config:
            points = [
                tuple(float(value) for value in point)
                for point in config["reconstructed_points"]
            ]
            points.append(points[0])

            overlays.append(
                {
                    "x": [point[0] for point in points],
                    "y": [point[1] for point in points],
                    "z": [point[2] for point in points],
                    "mode": "lines+markers",
                    "name": mirror.name,
                    "line": {
                        "color": SEMI_TRANSPARENT_MIRROR_COLOR,
                        "width": 5,
                    },
                    "marker": {
                        "color": SEMI_TRANSPARENT_MIRROR_COLOR,
                        "size": 3,
                    },
                    "hovertemplate": (
                        f"{mirror.name}<br>"
                        "x=%{x:.6f}<br>"
                        "y=%{y:.6f}<br>"
                        "z=%{z:.6f}<extra></extra>"
                    ),
                    "showlegend": False,
                }
            )
            continue

        overlays.append(
            make_circle_outline(
                name=mirror.name,
                center=mirror.center,
                normal=mirror.normal,
                radius=float(mirror.radius),
                color=SEMI_TRANSPARENT_MIRROR_COLOR,
                in_plane_reference=mirror.in_plane_reference,
            )
        )

    return overlays


# ---------------------------------------------------------------------------
# Interactive mirror controls
# ---------------------------------------------------------------------------

CONTROLLED_MIRRORS = [
    ("Круглое MC1", SMALL_REFLECTIVE_MIRROR),
    ("Круглое MC2", TURNING_ROUND_MIRROR_2),
    ("Круглое MC3", TURNING_ROUND_MIRROR_3),
    ("Круглое MC4", TURNING_ROUND_MIRROR_4),
    ("Квадратное MS1", TURNING_SQUARE_MIRROR_1),
    ("Квадратное MS2", TURNING_SQUARE_MIRROR_2),
    ("Квадратное MS3", TURNING_SQUARE_MIRROR_3),
    ("Квадратное MS4", TURNING_SQUARE_MIRROR_4),
]


def _empty_controlled_mirror_angles() -> Dict[str, Dict[str, float]]:
    return {
        mirror.name: {"x": 0.0, "y": 0.0, "z": 0.0}
        for _, mirror in CONTROLLED_MIRRORS
    }


def _parse_controlled_mirror_angles(
    items: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    angles = _empty_controlled_mirror_angles()

    aliases = {
        f"MC{i + 1}": mirror.name
        for i, (_, mirror) in enumerate(CONTROLLED_MIRRORS[:4])
    }
    aliases.update(
        {
            f"MS{i + 1}": mirror.name
            for i, (_, mirror) in enumerate(CONTROLLED_MIRRORS[4:])
        }
    )
    aliases.update(
        {
            mirror.name: mirror.name
            for _, mirror in CONTROLLED_MIRRORS
        }
    )

    for item in items:
        key, value_text = item.rsplit("=", 1)
        mirror_key, axis = key.rsplit(":", 1)
        mirror_name = aliases.get(mirror_key)
        axis = axis.lower()
        value = float(value_text)

        if (
            mirror_name is None
            or axis not in ("x", "y", "z")
            or not math.isfinite(value)
            or abs(value) > 180.0
        ):
            raise ValueError(
                f"Invalid mirror angle override: {item!r}"
            )

        angles[mirror_name][axis] = value

    return angles


def _rotate_mirror_vector(
    vector: Sequence[float],
    axis: Sequence[float],
    angle_deg: float,
) -> tuple[float, float, float]:
    value = np.asarray(vector, dtype=float)
    axis_value = np.asarray(axis, dtype=float)
    axis_value /= np.linalg.norm(axis_value)

    angle = math.radians(float(angle_deg))

    result = (
        value * math.cos(angle)
        + np.cross(axis_value, value) * math.sin(angle)
        + axis_value
        * float(np.dot(axis_value, value))
        * (1.0 - math.cos(angle))
    )

    return tuple(
        float(component)
        for component in result / np.linalg.norm(result)
    )


def _apply_controlled_mirror_angles(
    angles: Dict[str, Dict[str, float]],
) -> None:
    axes = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }

    for _, mirror in CONTROLLED_MIRRORS:
        normal = tuple(mirror.normal)
        reference = tuple(mirror.in_plane_reference)

        for axis in ("x", "y", "z"):
            angle = angles[mirror.name][axis]

            if abs(angle) > 1e-15:
                normal = _rotate_mirror_vector(
                    normal,
                    axes[axis],
                    angle,
                )
                reference = _rotate_mirror_vector(
                    reference,
                    axes[axis],
                    angle,
                )

        mirror.normal = normal
        mirror.in_plane_reference = reference


def _add_mirror_angle_controls(
    path: Path,
    angles: Dict[str, Dict[str, float]],
) -> None:
    config = {
        mirror.name: {
            "label": label,
            "center": list(mirror.center),
        }
        for label, mirror in CONTROLLED_MIRRORS
    }

    sections = []

    for index, (label, mirror) in enumerate(CONTROLLED_MIRRORS):
        rows = "".join(
            f'<label><i>{axis}</i>'
            f'<button class="angle-step" data-mirror="{index}" '
            f'data-axis="{axis}" data-delta="-0.005">−</button>'
            f'<input type="number" '
            f'value="{angles[mirror.name][axis]:.6f}" '
            f'step="0.005" data-mirror="{index}" data-axis="{axis}">'
            f'<button class="angle-step" data-mirror="{index}" '
            f'data-axis="{axis}" data-delta="0.005">+</button></label>'
            for axis in ("x", "y", "z")
        )

        sections.append(
            f"<details><summary>{label}</summary>"
            f'<div class="axis-rows">{rows}</div>'
            f'<button class="reset-one" data-mirror="{index}">'
            "Сбросить</button></details>"
        )

    panel = f"""
<style>
body{{margin:0;overflow:hidden;font-family:Arial,sans-serif}}.plotly-graph-div{{width:calc(100vw - 286px)!important;height:100vh!important}}
#mirror-panel{{position:fixed;right:0;top:0;z-index:1000;box-sizing:border-box;width:286px;height:100vh;overflow-y:auto;padding:10px;background:#f8fafcf7;border-left:1px solid #cbd5e1;box-shadow:-3px 0 12px #0f17221f;font-size:12px;color:#1f2937}}
#mirror-panel h3{{margin:0 0 4px;font-size:14px}}#mirror-panel>p{{margin:0 0 8px;color:#64748b}}#mirror-panel details{{margin-bottom:5px;border:1px solid #d7dee8;border-radius:5px;background:white}}#mirror-panel summary{{padding:6px 7px;cursor:pointer;font-weight:700}}
.axis-rows{{display:grid;gap:4px;padding:0 7px 5px}}.axis-rows label{{display:grid;grid-template-columns:12px 26px 1fr 26px;align-items:center;gap:4px}}.axis-rows input{{box-sizing:border-box;min-width:0;width:100%;padding:3px;border:1px solid #cbd5e1;border-radius:3px;font-size:11px}}
.angle-step{{height:24px;padding:0;border:1px solid #94a3b8;border-radius:4px;background:white;font-size:16px;cursor:pointer}}.angle-step:active{{background:#e2e8f0}}.reset-one{{margin:0 7px 7px;padding:3px 7px;border:1px solid #94a3b8;border-radius:4px;background:#f8fafc;cursor:pointer}}
#reset-all,#recalculate{{width:100%;margin-top:6px;padding:7px;border-radius:4px;font-weight:700;cursor:pointer}}#reset-all{{border:1px solid #94a3b8;background:#f8fafc}}#recalculate{{border:0;background:#2563eb;color:white}}#recalculate:disabled{{background:#94a3b8;cursor:wait}}#status{{min-height:16px;margin:6px 0 0}}
</style>
<aside id="mirror-panel"><h3>Отклонение зеркал, °</h3><p>Относительно текущего положения по осям x, y, z.</p>{''.join(sections)}<button id="reset-all">Сбросить все</button><button id="recalculate">Пересчитать и показать</button><p id="status"></p></aside>
<script>
(()=>{{const config={json.dumps(config, ensure_ascii=False)},names=Object.keys(config),rendered={json.dumps(angles, ensure_ascii=False)},states=names.map(n=>({{...rendered[n]}}));
function rotate(p,c,a){{let x=p[0]-c[0],y=p[1]-c[1],z=p[2]-c[2],r=a.x*Math.PI/180,ny=y*Math.cos(r)-z*Math.sin(r),nz=y*Math.sin(r)+z*Math.cos(r);y=ny;z=nz;r=a.y*Math.PI/180;let nx=x*Math.cos(r)+z*Math.sin(r);nz=-x*Math.sin(r)+z*Math.cos(r);x=nx;z=nz;r=a.z*Math.PI/180;return[x*Math.cos(r)-y*Math.sin(r)+c[0],x*Math.sin(r)+y*Math.cos(r)+c[1],z+c[2]]}}
function init(){{const plot=document.querySelector('.plotly-graph-div');if(!plot||!plot.data){{setTimeout(init,50);return}}names.forEach(n=>{{const i=plot.data.findIndex(t=>t.name===n),t=i>=0?plot.data[i]:null;if(t){{config[n].i=i;config[n].points=t.x.map((x,j)=>[Number(x),Number(t.y[j]),Number(t.z[j])])}}}});
function preview(i){{const n=names[i],c=config[n];if(c.i===undefined)return;const a={{x:states[i].x-rendered[n].x,y:states[i].y-rendered[n].y,z:states[i].z-rendered[n].z}},p=c.points.map(v=>rotate(v,c.center,a));Plotly.restyle(plot,{{x:[p.map(v=>v[0])],y:[p.map(v=>v[1])],z:[p.map(v=>v[2])]}},[c.i])}}
document.querySelectorAll('#mirror-panel input').forEach(input=>input.addEventListener('input',()=>{{const i=Number(input.dataset.mirror);states[i][input.dataset.axis]=Number(input.value)||0;preview(i)}}));
document.querySelectorAll('.angle-step').forEach(button=>button.addEventListener('click',()=>{{const i=Number(button.dataset.mirror),axis=button.dataset.axis;states[i][axis]=Math.round((states[i][axis]+Number(button.dataset.delta))*1000)/1000;const input=document.querySelector(`input[data-mirror="${{i}}"][data-axis="${{axis}}"]`);input.value=states[i][axis].toFixed(3);preview(i)}}));
document.querySelectorAll('.reset-one').forEach(button=>button.addEventListener('click',()=>{{const i=Number(button.dataset.mirror);states[i]={{x:0,y:0,z:0}};document.querySelectorAll(`input[data-mirror="${{i}}"]`).forEach(v=>v.value='0');preview(i)}}));
async function recalc(){{const b=document.getElementById('recalculate'),s=document.getElementById('status'),payload={{}};names.forEach((n,i)=>payload[n]=states[i]);b.disabled=true;s.textContent='Выполняется трассировка лучей…';try{{const r=await fetch('/recalculate',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{angles:payload}})}}),v=await r.json();if(!r.ok)throw new Error(v.error||'Ошибка пересчёта');s.textContent='Готово. Обновляю визуализацию…';location.href='/scene_gaussian_35ns.html?v='+Date.now()}}catch(e){{s.textContent=e.message;b.disabled=false}}}}
document.getElementById('recalculate').addEventListener('click',recalc);document.getElementById('reset-all').addEventListener('click',()=>{{states.forEach((_,i)=>{{states[i]={{x:0,y:0,z:0}};preview(i)}});document.querySelectorAll('#mirror-panel input').forEach(v=>v.value='0');recalc()}});addEventListener('resize',()=>Plotly.Plots.resize(plot));Plotly.Plots.resize(plot)}}init()}})();
</script>"""

    html = path.read_text(encoding="utf-8")
    if "</body>" not in html:
        raise ValueError(
            f"Cannot inject mirror controls into {path}: </body> not found."
        )
    path.write_text(
        html.replace("</body>", panel + "</body>", 1),
        encoding="utf-8",
    )


def _serve_interactive_plot(
    *,
    outdir: Path,
    backend: str,
    max_interactions: int,
) -> None:
    """
    Current safe interactive mode.

    It intentionally still launches a fresh subprocess for a physical
    recalculation. This is slower than a future persistent worker, but preserves
    the current process-isolation behaviour and avoids introducing an
    unbenchmarked change to simulation state.
    """
    script_path = Path(__file__).resolve()
    allowed = {
        mirror.name
        for _, mirror in CONTROLLED_MIRRORS
    }

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                directory=str(outdir.resolve()),
                **kwargs,
            )

        def do_POST(self) -> None:
            if self.path != "/recalculate":
                self.send_error(404)
                return

            try:
                length = int(
                    self.headers.get("Content-Length", "0")
                )
                payload = json.loads(
                    self.rfile.read(length).decode("utf-8")
                )
                angles = payload["angles"]

                if set(angles) != allowed:
                    raise ValueError("Unexpected mirror set.")

                command = [
                    sys.executable,
                    str(script_path),
                    "--backend",
                    backend,
                    "--max-interactions",
                    str(max_interactions),
                    "--outdir",
                    str(outdir.resolve()),
                    "--no-open-plot",
                    "--skip-csv",
                ]

                for name in sorted(angles):
                    if set(angles[name]) != {"x", "y", "z"}:
                        raise ValueError(
                            f"Incomplete angles for {name}."
                        )

                    for axis in ("x", "y", "z"):
                        value = float(angles[name][axis])

                        if (
                            not math.isfinite(value)
                            or abs(value) > 180.0
                        ):
                            raise ValueError(
                                f"Invalid {name}:{axis} angle: {value}"
                            )

                        command.extend(
                            (
                                "--mirror-angle",
                                f"{name}:{axis}={value}",
                            )
                        )

                completed = subprocess.run(
                    command,
                    cwd=str(script_path.parent),
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )

                if completed.returncode:
                    raise RuntimeError(
                        completed.stderr.strip()
                        or completed.stdout.strip()
                        or "Recalculation failed."
                    )

                response = json.dumps({"ok": True}).encode()
                self.send_response(200)

            except Exception as exc:
                response = json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(400)

            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8",
            )
            self.send_header(
                "Content-Length",
                str(len(response)),
            )
            self.end_headers()
            self.wfile.write(response)

        def log_message(
            self,
            fmt: str,
            *args: object,
        ) -> None:
            print(f"Interactive plot: {fmt % args}")

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        Handler,
    )
    host, port = server.server_address
    url = f"http://{host}:{port}/scene_gaussian_35ns.html"

    print(f"Interactive plot: {url}")
    print(
        "Keep this process running for recalculation; "
        "press Ctrl+C to stop."
    )

    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Interactive plot stopped.")
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

SEGMENT_BASE_FIELDS = (
    "ray_id",
    "parent_id",
    "depth",
    "surface",
    "x0",
    "y0",
    "z0",
    "x1",
    "y1",
    "z1",
    "power",
    "intensity",
)

DETECTOR_HIT_FIELDS = (
    "surface",
    "ray_id",
    "parent_id",
    "depth",
    "x",
    "y",
    "z",
    "dx",
    "dy",
    "dz",
    "power",
    "intensity",
    "u",
    "v",
)


def _first_nonempty_block(
    blocks: Sequence[Dict[str, object]],
) -> Dict[str, object] | None:
    for block in blocks:
        if len(block["ray_id"]) > 0:
            return block
    return None


def _as_numpy(value):
    """Convert a NumPy/CuPy array once per block before text export."""
    return to_numpy(value)


def write_segment_blocks_csv(
    path: Path,
    blocks: Sequence[Dict[str, object]],
) -> bool:
    """
    Direct block CSV writer.

    No per-row dicts and no list(rows). Columns are converted once and zipped
    lazily into csv.writer. This is the architecture benchmarked as the chosen
    CSV optimization.
    """
    first = _first_nonempty_block(blocks)

    if first is None:
        if path.exists():
            path.unlink()
        return False

    include_t0 = "t0_s" in first
    include_t1 = "t1_s" in first

    fieldnames = list(SEGMENT_BASE_FIELDS)
    if include_t0:
        fieldnames.append("t0_s")
    if include_t1:
        fieldnames.append("t1_s")

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)

        for block in blocks:
            n = len(block["ray_id"])
            if n == 0:
                continue

            # Preserve DictWriter's old "extra keys" behaviour if a later
            # block unexpectedly introduces time columns absent from header.
            if "t0_s" in block and not include_t0:
                raise ValueError(
                    "A later segment block contains t0_s although the "
                    "first non-empty block did not."
                )
            if "t1_s" in block and not include_t1:
                raise ValueError(
                    "A later segment block contains t1_s although the "
                    "first non-empty block did not."
                )

            columns: list[Iterable[object]] = [
                map(int, _as_numpy(block["ray_id"])),
                map(int, _as_numpy(block["parent_id"])),
                map(int, _as_numpy(block["depth"])),
                map(str, _as_numpy(block["surface"])),
                map(float, _as_numpy(block["x0"])),
                map(float, _as_numpy(block["y0"])),
                map(float, _as_numpy(block["z0"])),
                map(float, _as_numpy(block["x1"])),
                map(float, _as_numpy(block["y1"])),
                map(float, _as_numpy(block["z1"])),
                map(float, _as_numpy(block["power"])),
                map(float, _as_numpy(block["intensity"])),
            ]

            if include_t0:
                columns.append(
                    map(float, _as_numpy(block["t0_s"]))
                    if "t0_s" in block
                    else repeat("", n)
                )

            if include_t1:
                columns.append(
                    map(float, _as_numpy(block["t1_s"]))
                    if "t1_s" in block
                    else repeat("", n)
                )

            writer.writerows(zip(*columns))

    return True


def write_detector_hits_csv(
    path: Path,
    blocks: Sequence[Dict[str, object]],
) -> bool:
    first = _first_nonempty_block(blocks)

    if first is None:
        if path.exists():
            path.unlink()
        return False

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(DETECTOR_HIT_FIELDS)

        for block in blocks:
            n = len(block["ray_id"])
            if n == 0:
                continue

            position = _as_numpy(block["position"])
            direction = _as_numpy(block["direction"])
            surface = str(block["surface"])

            columns = [
                repeat(surface, n),
                map(int, _as_numpy(block["ray_id"])),
                map(int, _as_numpy(block["parent_id"])),
                map(int, _as_numpy(block["depth"])),
                map(float, position[:, 0]),
                map(float, position[:, 1]),
                map(float, position[:, 2]),
                map(float, direction[:, 0]),
                map(float, direction[:, 1]),
                map(float, direction[:, 2]),
                map(float, _as_numpy(block["power"])),
                map(float, _as_numpy(block["intensity"])),
                map(float, _as_numpy(block["local_u"])),
                map(float, _as_numpy(block["local_v"])),
            ]

            writer.writerows(zip(*columns))

    return True


# Compatibility helpers retained for external code/tests.
# Main no longer uses them because direct block writers are faster and use
# dramatically less memory.
def flatten_segment_blocks(
    blocks: List[Dict[str, np.ndarray]],
) -> Iterator[Dict[str, object]]:
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


def flatten_detector_hits(
    blocks: List[Dict[str, np.ndarray]],
) -> Iterator[Dict[str, object]]:
    for block in blocks:
        n = len(block["ray_id"])
        for i in range(n):
            pos = block["position"][i]
            direction = block["direction"][i]
            yield {
                "surface": str(block["surface"]),
                "ray_id": int(block["ray_id"][i]),
                "parent_id": int(block["parent_id"][i]),
                "depth": int(block["depth"][i]),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "dx": float(direction[0]),
                "dy": float(direction[1]),
                "dz": float(direction[2]),
                "power": float(block["power"][i]),
                "intensity": float(block["intensity"][i]),
                "u": float(block["local_u"][i]),
                "v": float(block["local_v"][i]),
            }


def write_csv(
    path: Path,
    rows: Iterable[Dict[str, object]],
) -> bool:
    """
    Streaming compatibility writer.

    Unlike the old implementation, this never materializes list(rows).
    Main uses the faster direct block writers above.
    """
    iterator = iter(rows)

    try:
        first = next(iterator)
    except StopIteration:
        if path.exists():
            path.unlink()
        return False

    fieldnames = list(first.keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)

    return True


# ---------------------------------------------------------------------------
# Summaries and plotting
# ---------------------------------------------------------------------------


def detector_energy_summary(
    result: object,
    integration_time_s: float,
) -> Dict[str, float]:
    summary: Dict[str, float] = {}

    for hit in result.detector_hits:
        name = str(hit["surface"])
        energy = (
            float(np.sum(to_numpy(hit["power"])))
            * integration_time_s
        )
        summary[name] = summary.get(name, 0.0) + energy

    return summary


def write_bundle_plot(
    path: Path,
    result: object,
    bundle: MirrorArrayBundle,
) -> None:
    bundle_result = copy.copy(result)
    bundle_result.segments = []
    bundle_result.detector_hits = []

    write_plotly_trajectories(
        path,
        bundle_result,
        title=bundle.name,
        overlays=build_bundle_overlays(bundle),
    )


def _build_plot_overlays() -> List[Dict[str, object]]:
    overlays: List[Dict[str, object]] = []

    overlays.append(
        make_circle_outline(
            name="Source waist",
            center=SOURCE_TEMPLATE.waist_position,
            normal=SOURCE_TEMPLATE.axis,
            radius=float(SOURCE_TEMPLATE.waist_radius),
            color="#1f77b4",
            in_plane_reference=SOURCE_TEMPLATE.polarization_reference,
        )
    )

    for mirror in (
        PERISCOPE_MIRROR_1,
        PERISCOPE_MIRROR_2,
        ONE_OF_MANY_MIRRORS,
    ):
        overlays.append(
            make_rectangle_outline(
                name=mirror.name,
                center=mirror.center,
                normal=mirror.normal,
                width=float(mirror.width),
                height=float(mirror.height),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=mirror.in_plane_reference,
            )
        )

    for mirror in (BLOCK_MIRROR_1, BLOCK_MIRROR_2):
        overlays.extend(
            make_rectangular_prism_overlays(
                name=mirror.name,
                center=mirror.center,
                normal=mirror.normal,
                width=float(mirror.width),
                height=float(mirror.height),
                thickness=float(mirror.thickness),
                color=BLOCK_MIRROR_COLOR,
                in_plane_reference=mirror.in_plane_reference,
            )
        )

    overlays.append(
        make_rectangle_outline(
            name=ON_ENTER_BEAMSPLITTER.name,
            center=ON_ENTER_BEAMSPLITTER.center,
            normal=ON_ENTER_BEAMSPLITTER.normal,
            width=float(ON_ENTER_BEAMSPLITTER.width),
            height=float(ON_ENTER_BEAMSPLITTER.height),
            color=PLANE_MIRROR_COLOR,
            in_plane_reference=ON_ENTER_BEAMSPLITTER.in_plane_reference,
        )
    )

    for mirror in (
        SMALL_REFLECTIVE_MIRROR,
        TURNING_ROUND_MIRROR_2,
        TURNING_ROUND_MIRROR_3,
        TURNING_ROUND_MIRROR_4,
    ):
        overlays.append(
            make_circle_outline(
                name=mirror.name,
                center=mirror.center,
                normal=mirror.normal,
                radius=float(mirror.radius),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=mirror.in_plane_reference,
            )
        )

    for sector in range(1, 5):
        for layer in range(1, 5):
            overlays.extend(
                build_bundle_overlays(
                    make_bundle_by_index(sector, layer)
                )
            )

    for screen in (
        SCREEN_1,
        SCREEN_2,
        SCREEN_3,
        SCREEN_4,
    ):
        overlays.extend(
            make_disk_overlays(
                name=screen.name,
                center=screen.center,
                normal=screen.normal,
                radius=float(screen.radius),
                color=SCREEN_COLOR,
                in_plane_reference=screen.in_plane_reference,
                opacity=0.92,
            )
        )

    overlays.extend(
        make_cylindrical_surface_overlays(
            name=CYLINDRICAL_SCREEN_1.name,
            center=CYLINDRICAL_SCREEN_1.center,
            axis=CYLINDRICAL_SCREEN_1.axis,
            radius=float(CYLINDRICAL_SCREEN_1.radius),
            length=float(CYLINDRICAL_SCREEN_1.length),
            color=CYLINDRICAL_SCREEN_COLOR,
            opacity=0.0,
        )
    )

    for mirror in (
        TURNING_SQUARE_MIRROR_1,
        TURNING_SQUARE_MIRROR_2,
        TURNING_SQUARE_MIRROR_3,
        TURNING_SQUARE_MIRROR_4,
    ):
        overlays.append(
            make_rectangle_outline(
                name=mirror.name,
                center=mirror.center,
                normal=mirror.normal,
                width=float(mirror.width),
                height=float(mirror.height),
                color=PLANE_MIRROR_COLOR,
                in_plane_reference=mirror.in_plane_reference,
            )
        )

    for mirror in (
        SEMI_MIRROR_LEFT_1,
        SEMI_MIRROR_NEW,
    ):
        overlays.extend(
            make_rectangular_prism_overlays(
                name=mirror.name,
                center=mirror.center,
                normal=mirror.normal,
                width=float(mirror.width),
                height=float(mirror.height),
                thickness=float(mirror.thickness),
                color=SEMI_TRANSPARENT_MIRROR_COLOR,
                in_plane_reference=mirror.in_plane_reference,
            )
        )

    for prism in (
        PRISM_1,
        PRISM_2,
        PRISM_3,
        PRISM_4,
        PRISM_5,
    ):
        overlays.extend(
            make_triangular_prism_overlays(
                name=prism.name,
                center=prism.center,
                normal=prism.normal,
                vertices_2d=prism.vertices_2d,
                thickness=float(prism.thickness),
                color=TRIANGULAR_PRISM_COLOR,
                in_plane_reference=prism.in_plane_reference,
            )
        )

    overlays.extend(
        make_rectangular_prism_overlays(
            name=SEMI_MIRROR_3.name,
            center=SEMI_MIRROR_3.center,
            normal=SEMI_MIRROR_3.normal,
            width=float(SEMI_MIRROR_3.width),
            height=float(SEMI_MIRROR_3.height),
            thickness=float(SEMI_MIRROR_3.thickness),
            color=SEMI_TRANSPARENT_MIRROR_COLOR,
            in_plane_reference=SEMI_MIRROR_3.in_plane_reference,
        )
    )

    return overlays


def _print_scene_summary(
    *,
    rays: object,
    result: object,
    wrote_segments: bool,
    wrote_detector_hits: bool,
    segments_path: Path,
    detector_hits_path: Path,
) -> None:
    source_power = float(np.sum(to_numpy(rays.power)))
    power_summary = result.detector_power_summary()
    energy_summary = detector_energy_summary(
        result,
        INTEGRATION_TIME_S,
    )

    print("Initial scene: Gaussian source with scene scaffold")
    print(
        f"Integration time: "
        f"{INTEGRATION_TIME_S * 1e9:.3f} ns"
    )
    print(f"Source total power: {source_power:.6e} W")

    print("Scene elements:")
    for obj in BASE_SCENE_OBJECTS:
        print(f"  - {obj.name}")

    for sector in range(1, 5):
        for layer in range(1, 5):
            print(
                f"  - BUNDLE_{sector}_{layer} "
                "(7 disk mirrors)"
            )

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Initial 35 ns scene with the same Gaussian source as the demo."
        )
    )

    parser.add_argument(
        "--backend",
        default="numpy",
        choices=["numpy", "cupy"],
        help="Array backend",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=MAX_SECONDARY_RAY_GENERATIONS,
        help=(
            "Maximum number of ray interactions / "
            "secondary-ray generations"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="scene_gaussian_35ns_output",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--mirror-angle",
        action="append",
        default=[],
        help=(
            "Mirror angle override, for example "
            "MC1:x=0.01 or MS4:z=-0.02."
        ),
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip saving the Plotly trajectories plot",
    )
    parser.add_argument(
        "--no-open-plot",
        dest="open_plot",
        action="store_false",
        help="Do not open the saved plot in a browser",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(
        plot=True,
        open_plot=True,
    )

    args = parser.parse_args()

    print(
        "Loaded reconstructed mirror positions: "
        f"{RECONSTRUCTED_MICROASSEMBLIES_PATH}"
    )

    controlled_mirror_angles = _parse_controlled_mirror_angles(
        args.mirror_angle
    )
    _apply_controlled_mirror_angles(
        controlled_mirror_angles
    )

    source = build_initial_source(args.backend)
    rays = emit_initial_rays(source)

    scene = build_initial_scene()

    tracer = RayTracer(
        scene=scene,
        backend=args.backend,
        max_interactions=args.max_interactions,
        max_time_s=INTEGRATION_TIME_S,
        bundle_clip_inner_radius_m=BUNDLE_RAY_INNER_CYLINDER_RADIUS_M,
        bundle_clip_outer_radius_m=BUNDLE_RAY_OUTER_CYLINDER_RADIUS_M,
        bundle_clip_z_min_m=BUNDLE_RAY_Z_MIN_M,
        bundle_clip_z_max_m=BUNDLE_RAY_Z_MAX_M,
    )

    # The broad-phase acceleration itself belongs to the optimized raytrace.py.
    result = tracer.trace(rays)

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    segments_path = outdir / "segments.csv"
    detector_hits_path = outdir / "detector_hits.csv"

    wrote_segments = False
    wrote_detector_hits = False

    if not args.skip_csv:
        wrote_segments = write_segment_blocks_csv(
            segments_path,
            result.segments,
        )
        wrote_detector_hits = write_detector_hits_csv(
            detector_hits_path,
            result.detector_hits,
        )

    _print_scene_summary(
        rays=rays,
        result=result,
        wrote_segments=wrote_segments,
        wrote_detector_hits=wrote_detector_hits,
        segments_path=segments_path,
        detector_hits_path=detector_hits_path,
    )

    if args.plot:
        plot_path = outdir / "scene_gaussian_35ns.html"

        write_plotly_trajectories(
            plot_path,
            result,
            title="35 ns scene with reconstructed mirror positions",
            overlays=_build_plot_overlays(),
            detector_hit_exclude_prefixes=("Screen",),
            trim_end_surface_prefixes=("Screen",),
            trim_end_distance=5e-3,
            min_segment_power=20.0,
            max_segments_per_block=150,
            always_include_surface_prefixes=(
                "BUNDLE_",
                "Cylindrical Screen",
            ),
            always_include_child_segments_of_surface_prefixes=(
                "BUNDLE_",
            ),
        )

        _add_mirror_angle_controls(
            plot_path,
            controlled_mirror_angles,
        )

        print(f"Wrote: {plot_path}")

        if args.open_plot:
            _serve_interactive_plot(
                outdir=outdir,
                backend=args.backend,
                max_interactions=args.max_interactions,
            )


if __name__ == "__main__":
    main()
