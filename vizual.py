from __future__ import annotations
import csv
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from plotly.colors import sample_colorscale

from raytrace import to_numpy


_SURFACE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _parse_rgb_color(color: str) -> tuple[int, int, int] | None:
    if color.startswith("rgb(") and color.endswith(")"):
        parts = [part.strip() for part in color[4:-1].split(",")]
        if len(parts) == 3:
            return tuple(max(0, min(255, int(round(float(part))))) for part in parts)
    return None


def _darken_rgb_color(color: str, factor: float = 0.88) -> str:
    rgb = _parse_rgb_color(color)
    if rgb is None:
        return color
    r, g, b = (max(0, min(255, int(round(channel * factor)))) for channel in rgb)
    return f"rgb({r}, {g}, {b})"


def _increase_rgb_contrast(color: str, factor: float = 1.15, pivot: float = 128.0) -> str:
    rgb = _parse_rgb_color(color)
    if rgb is None:
        return color
    contrasted = tuple(
        max(0, min(255, int(round(pivot + factor * (channel - pivot)))))
        for channel in rgb
    )
    return f"rgb({contrasted[0]}, {contrasted[1]}, {contrasted[2]})"


def _blend_rgb_color(color: str, target: str, weight: float) -> str:
    rgb = _parse_rgb_color(color)
    target_rgb = _parse_rgb_color(target)
    if rgb is None or target_rgb is None:
        return color
    t = float(np.clip(weight, 0.0, 1.0))
    blended = tuple(
        max(0, min(255, int(round((1.0 - t) * src + t * dst))))
        for src, dst in zip(rgb, target_rgb)
    )
    return f"rgb({blended[0]}, {blended[1]}, {blended[2]})"


def _make_softened_colorscale(
    base: str,
    *,
    start: float = 0.12,
    end: float = 0.88,
    blend_to_mid: float = 0.16,
    darken_factor: float = 1.0,
    contrast_factor: float = 1.0,
    steps: int = 11,
) -> list[list[object]]:
    positions = np.linspace(float(start), float(end), int(steps), dtype=float)
    colors = sample_colorscale(base, positions.tolist())
    mid_color = sample_colorscale(base, [0.5])[0]
    scale: list[list[object]] = []
    normalized_positions = np.linspace(0.0, 1.0, int(steps), dtype=float)
    for pos, color in zip(normalized_positions, colors):
        softened = _blend_rgb_color(color, mid_color, weight=float(blend_to_mid))
        softened = _increase_rgb_contrast(softened, factor=float(contrast_factor))
        softened = _darken_rgb_color(softened, factor=float(darken_factor))
        scale.append([float(pos), softened])
    return scale


_LOW_INTENSITY_COLOR = "rgb(48, 112, 220)"
_HIGH_INTENSITY_COLOR = "rgb(225, 129, 215)"
_RAY_INTENSITY_COLORSCALE = [
    [0.0, _LOW_INTENSITY_COLOR],
    [1.0, _HIGH_INTENSITY_COLOR],
]

_DETECTOR_SCREEN_GRID_SIZE = 220
_SINGLE_DETECTOR_SCREEN_GRID_SIZE = 240
_DETECTOR_SCREEN_SMOOTH_PASSES = 4
_DETECTOR_HIT_MARKER_SIZE = 3.0
_DETECTOR_HIT_MARKER_LINE_WIDTH = 0.2
_DETECTOR_HIT_MARKER_OPACITY = 0.78
_DETECTOR_SUPPORT_SMOOTH_PASSES = 4
_DETECTOR_EDGE_FADE_PIXELS = 2.5


def _normalize(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 0.0:
        raise ValueError("Zero-length vector is not allowed")
    return arr / n


def _plane_basis(normal: Sequence[float], reference: Optional[Sequence[float]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = _normalize(normal)
    if reference is None:
        reference_vec = np.array((0.0, 0.0, 1.0), dtype=float)
        if abs(float(np.dot(reference_vec, w))) > 0.9:
            reference_vec = np.array((0.0, 1.0, 0.0), dtype=float)
    else:
        reference_vec = np.asarray(reference, dtype=float)
    u = reference_vec - np.dot(reference_vec, w) * w
    if float(np.linalg.norm(u)) <= 1e-12:
        fallback = np.array((1.0, 0.0, 0.0), dtype=float)
        if abs(float(np.dot(fallback, w))) > 0.9:
            fallback = np.array((0.0, 1.0, 0.0), dtype=float)
        u = fallback - np.dot(fallback, w) * w
    u = _normalize(u)
    v = _normalize(np.cross(w, u))
    return u, v, w


def make_rectangle_outline(
    *,
    name: str,
    center: Sequence[float],
    normal: Sequence[float],
    width: float,
    height: float,
    color: str,
    in_plane_reference: Optional[Sequence[float]] = None,
    line_width: float = 5,
) -> Dict[str, Any]:
    c = np.asarray(center, dtype=float)
    u, v, _ = _plane_basis(normal, in_plane_reference)
    corners = [
        c - 0.5 * width * u - 0.5 * height * v,
        c + 0.5 * width * u - 0.5 * height * v,
        c + 0.5 * width * u + 0.5 * height * v,
        c - 0.5 * width * u + 0.5 * height * v,
    ]
    corners.append(corners[0])
    return {
        "x": [float(p[0]) for p in corners],
        "y": [float(p[1]) for p in corners],
        "z": [float(p[2]) for p in corners],
        "mode": "lines",
        "name": name,
        "line": {"color": color, "width": line_width},
        "hoverinfo": "skip",
    }


def make_rectangular_prism_overlays(
    *,
    name: str,
    center: Sequence[float],
    normal: Sequence[float],
    width: float,
    height: float,
    thickness: float,
    color: str,
    in_plane_reference: Optional[Sequence[float]] = None,
    line_width: float = 5,
    opacity: float = 0.18,
) -> List[Dict[str, Any]]:
    c = np.asarray(center, dtype=float)
    u, v, w = _plane_basis(normal, in_plane_reference)
    half_w = 0.5 * float(width)
    half_h = 0.5 * float(height)
    half_t = 0.5 * float(thickness)

    front_center = c - half_t * w
    back_center = c + half_t * w
    front = [
        front_center - half_w * u - half_h * v,
        front_center + half_w * u - half_h * v,
        front_center + half_w * u + half_h * v,
        front_center - half_w * u + half_h * v,
    ]
    back = [
        back_center - half_w * u - half_h * v,
        back_center + half_w * u - half_h * v,
        back_center + half_w * u + half_h * v,
        back_center - half_w * u + half_h * v,
    ]
    vertices = front + back
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    edge_x: List[float | None] = []
    edge_y: List[float | None] = []
    edge_z: List[float | None] = []
    for i0, i1 in edges:
        p0 = vertices[i0]
        p1 = vertices[i1]
        edge_x.extend([float(p0[0]), float(p1[0]), None])
        edge_y.extend([float(p0[1]), float(p1[1]), None])
        edge_z.extend([float(p0[2]), float(p1[2]), None])

    mesh = {
        "type": "mesh3d",
        "x": [float(p[0]) for p in vertices],
        "y": [float(p[1]) for p in vertices],
        "z": [float(p[2]) for p in vertices],
        "i": [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
        "j": [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
        "k": [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
        "name": name,
        "color": color,
        "opacity": opacity,
        "flatshading": True,
        "hoverinfo": "skip",
        "showlegend": False,
    }
    outline = {
        "x": edge_x,
        "y": edge_y,
        "z": edge_z,
        "mode": "lines",
        "name": f"{name} outline",
        "line": {"color": color, "width": line_width},
        "hoverinfo": "skip",
        "showlegend": False,
    }
    return [mesh, outline]


def make_triangular_prism_overlays(
    *,
    name: str,
    center: Sequence[float],
    normal: Sequence[float],
    vertices_2d: Sequence[Sequence[float]],
    thickness: float,
    color: str,
    in_plane_reference: Sequence[float],
    line_width: float = 5,
    opacity: float = 0.18,
) -> List[Dict[str, Any]]:
    c = np.asarray(center, dtype=float)
    u, v, w = _plane_basis(normal, in_plane_reference)
    half_t = 0.5 * float(thickness)

    front_c = c - half_t * w
    back_c = c + half_t * w

    front = [front_c + p[0] * u + p[1] * v for p in vertices_2d]
    back = [back_c + p[0] * u + p[1] * v for p in vertices_2d]
    vertices = front + back

    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
    edge_x: List[float | None] = []
    edge_y: List[float | None] = []
    edge_z: List[float | None] = []
    for i0, i1 in edges:
        p0, p1 = vertices[i0], vertices[i1]
        edge_x.extend([float(p0[0]), float(p1[0]), None])
        edge_y.extend([float(p0[1]), float(p1[1]), None])
        edge_z.extend([float(p0[2]), float(p1[2]), None])

    mesh = {
        "type": "mesh3d",
        "x": [float(p[0]) for p in vertices],
        "y": [float(p[1]) for p in vertices],
        "z": [float(p[2]) for p in vertices],
        "i": [0, 3, 0, 0, 1, 1, 2, 2],
        "j": [1, 4, 1, 4, 2, 5, 0, 3],
        "k": [2, 5, 4, 3, 5, 4, 3, 5],
        "name": name,
        "color": color,
        "opacity": opacity,
        "flatshading": True,
        "hoverinfo": "skip",
        "showlegend": False,
    }
    outline = {
        "x": edge_x,
        "y": edge_y,
        "z": edge_z,
        "mode": "lines",
        "name": f"{name} outline",
        "line": {"color": color, "width": line_width},
        "hoverinfo": "skip",
        "showlegend": False,
    }
    return [mesh, outline]


def make_circle_outline(
    *,
    name: str,
    center: Sequence[float],
    normal: Sequence[float],
    radius: float,
    color: str,
    in_plane_reference: Optional[Sequence[float]] = None,
    line_width: float = 5,
    samples: int = 96,
) -> Dict[str, Any]:
    c = np.asarray(center, dtype=float)
    u, v, _ = _plane_basis(normal, in_plane_reference)
    theta = np.linspace(0.0, 2.0 * np.pi, samples + 1, dtype=float)
    pts = [c + radius * np.cos(t) * u + radius * np.sin(t) * v for t in theta]
    return {
        "x": [float(p[0]) for p in pts],
        "y": [float(p[1]) for p in pts],
        "z": [float(p[2]) for p in pts],
        "mode": "lines",
        "name": name,
        "line": {"color": color, "width": line_width},
        "hoverinfo": "skip",
    }


def make_disk_overlays(
    *,
    name: str,
    center: Sequence[float],
    normal: Sequence[float],
    radius: float,
    color: str,
    in_plane_reference: Optional[Sequence[float]] = None,
    line_width: float = 5,
    samples: int = 96,
    opacity: float = 0.22,
) -> List[Dict[str, Any]]:
    c = np.asarray(center, dtype=float)
    u, v, _ = _plane_basis(normal, in_plane_reference)
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False, dtype=float)
    rim = [c + radius * np.cos(t) * u + radius * np.sin(t) * v for t in theta]
    vertices = [c] + rim

    mesh = {
        "type": "mesh3d",
        "x": [float(p[0]) for p in vertices],
        "y": [float(p[1]) for p in vertices],
        "z": [float(p[2]) for p in vertices],
        "i": [0] * samples,
        "j": list(range(1, samples + 1)),
        "k": [idx + 1 for idx in range(1, samples)] + [1],
        "name": name,
        "color": color,
        "opacity": opacity,
        "flatshading": True,
        "hoverinfo": "skip",
        "showlegend": False,
    }
    outline = make_circle_outline(
        name=f"{name} outline",
        center=center,
        normal=normal,
        radius=radius,
        color=color,
        in_plane_reference=in_plane_reference,
        line_width=line_width,
        samples=samples,
    )
    outline["showlegend"] = False
    return [mesh, outline]


def make_cylindrical_surface_overlays(
    *,
    name: str,
    center: Sequence[float],
    axis: Sequence[float],
    radius: float,
    length: float,
    color: str,
    line_width: float = 4,
    samples: int = 96,
    opacity: float = 0.18,
) -> List[Dict[str, Any]]:
    c = np.asarray(center, dtype=float)
    u, v, w = _plane_basis(axis, None)
    half_l = 0.5 * float(length)
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False, dtype=float)

    bottom_center = c - half_l * w
    top_center = c + half_l * w
    bottom = [bottom_center + radius * np.cos(t) * u + radius * np.sin(t) * v for t in theta]
    top = [top_center + radius * np.cos(t) * u + radius * np.sin(t) * v for t in theta]
    vertices = bottom + top

    i_idx: List[int] = []
    j_idx: List[int] = []
    k_idx: List[int] = []
    for idx in range(samples):
        nxt = (idx + 1) % samples
        i_idx.extend([idx, idx])
        j_idx.extend([nxt, samples + nxt])
        k_idx.extend([samples + idx, samples + idx])

    mesh = {
        "type": "mesh3d",
        "x": [float(p[0]) for p in vertices],
        "y": [float(p[1]) for p in vertices],
        "z": [float(p[2]) for p in vertices],
        "i": i_idx,
        "j": j_idx,
        "k": k_idx,
        "name": name,
        "color": color,
        "opacity": opacity,
        "flatshading": True,
        "hoverinfo": "skip",
        "showlegend": False,
    }

    edge_x: List[float | None] = []
    edge_y: List[float | None] = []
    edge_z: List[float | None] = []
    for ring in (bottom, top):
        ring_closed = ring + [ring[0]]
        for p in ring_closed:
            edge_x.append(float(p[0]))
            edge_y.append(float(p[1]))
            edge_z.append(float(p[2]))
        edge_x.append(None)
        edge_y.append(None)
        edge_z.append(None)

    for idx in (0, samples // 4, samples // 2, (3 * samples) // 4):
        p0 = bottom[idx]
        p1 = top[idx]
        edge_x.extend([float(p0[0]), float(p1[0]), None])
        edge_y.extend([float(p0[1]), float(p1[1]), None])
        edge_z.extend([float(p0[2]), float(p1[2]), None])

    outline = {
        "x": edge_x,
        "y": edge_y,
        "z": edge_z,
        "mode": "lines",
        "name": f"{name} outline",
        "line": {"color": color, "width": line_width},
        "hoverinfo": "skip",
        "showlegend": False,
    }
    return [mesh, outline]


def make_marker_trace(
    *,
    name: str,
    points: Sequence[Sequence[float]],
    color: str,
    size: float = 5,
    text: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    trace: Dict[str, Any] = {
        "x": pts[:, 0].tolist(),
        "y": pts[:, 1].tolist(),
        "z": pts[:, 2].tolist(),
        "mode": "markers",
        "name": name,
        "marker": {"color": color, "size": size},
    }
    if text is not None:
        trace["text"] = list(text)
    return trace


def make_ray_bundle_preview_overlays(
    rays: Any,
    *,
    length: float,
    name: str = "Launched rays",
    color: str = "#4c78a8",
    line_width: float = 2,
    endpoint_color: str = "#111111",
    endpoint_size: float = 2,
    max_rays: int = 181,
) -> List[Dict[str, Any]]:
    positions = to_numpy(rays.position)
    directions = to_numpy(rays.direction)
    n = len(positions)
    if n == 0:
        return []

    stride = max(1, (n + max_rays - 1) // max_rays)
    xs: List[float | None] = []
    ys: List[float | None] = []
    zs: List[float | None] = []
    end_points: List[List[float]] = []
    for i in range(0, n, stride):
        start = np.asarray(positions[i], dtype=float)
        end = start + np.asarray(directions[i], dtype=float) * float(length)
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        zs.extend([float(start[2]), float(end[2]), None])
        end_points.append([float(end[0]), float(end[1]), float(end[2])])

    overlays: List[Dict[str, Any]] = [
        {
            "x": xs,
            "y": ys,
            "z": zs,
            "mode": "lines",
            "name": name,
            "line": {"color": color, "width": line_width},
            "hoverinfo": "skip",
        }
    ]
    if end_points:
        overlays.append(
            make_marker_trace(
                name=f"{name} ends",
                points=end_points,
                color=endpoint_color,
                size=endpoint_size,
            )
        )
        overlays[-1]["showlegend"] = False
        overlays[-1]["hoverinfo"] = "skip"
    return overlays


def _ellipticity_ratio(u: np.ndarray, v: np.ndarray, intensity: np.ndarray) -> float:
    if u.size < 2 or v.size < 2:
        return float("nan")
    weights = np.asarray(intensity, dtype=float).reshape(-1)
    if weights.size != u.size:
        raise ValueError("Intensity weights must have the same length as detector coordinates.")
    weights = np.clip(weights, 0.0, None)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return float("nan")

    pts = np.column_stack((u, v))
    center = np.average(pts, axis=0, weights=weights)
    rel = pts - center
    cov = (rel.T * weights) @ rel / total_weight
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    major = float(np.max(eigvals))
    minor = float(np.min(eigvals))
    if major <= 1e-30:
        return float("nan")
    return float(np.sqrt(minor / major))


def _select_detector_block(
    blocks: Sequence[Dict[str, Any]],
    *,
    primary_block_only: bool = False,
) -> List[Dict[str, Any]]:
    if not blocks:
        return []
    if not primary_block_only:
        return list(blocks)

    def block_key(block: Dict[str, Any]) -> Tuple[float, int]:
        intensity = np.asarray(to_numpy(block["intensity"]), dtype=float).reshape(-1)
        total_intensity = float(np.sum(np.clip(intensity, 0.0, None)))
        hits = int(intensity.size)
        return total_intensity, hits

    best = max(blocks, key=block_key)
    return [best]


def _apply_radial_quantile_filter(
    u: np.ndarray,
    v: np.ndarray,
    intensity: np.ndarray,
    *,
    radial_quantile: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radial_quantile is None or not (0.0 < float(radial_quantile) < 1.0):
        return u, v, intensity
    if u.size < 8:
        return u, v, intensity

    weights = np.clip(np.asarray(intensity, dtype=float).reshape(-1), 0.0, None)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return u, v, intensity

    pts = np.column_stack((u, v))
    center = np.average(pts, axis=0, weights=weights)
    radius = np.sqrt(np.sum((pts - center) ** 2, axis=1))
    threshold = float(np.quantile(radius, float(radial_quantile)))
    mask = radius <= threshold
    if int(np.count_nonzero(mask)) < 8:
        return u, v, intensity
    return u[mask], v[mask], intensity[mask]


def _empty_detector_hit_data() -> Dict[str, np.ndarray]:
    return {
        "u": np.zeros((0,), dtype=float),
        "v": np.zeros((0,), dtype=float),
        "intensity": np.zeros((0,), dtype=float),
        "power": np.zeros((0,), dtype=float),
        "position": np.zeros((0, 3), dtype=float),
    }


def _radial_quantile_mask(
    u: np.ndarray,
    v: np.ndarray,
    intensity: np.ndarray,
    *,
    radial_quantile: Optional[float] = None,
) -> np.ndarray:
    mask = np.ones(u.shape, dtype=bool)
    if radial_quantile is None or not (0.0 < float(radial_quantile) < 1.0):
        return mask
    if u.size < 8:
        return mask

    weights = np.clip(np.asarray(intensity, dtype=float).reshape(-1), 0.0, None)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return mask

    pts = np.column_stack((u, v))
    center = np.average(pts, axis=0, weights=weights)
    radius = np.sqrt(np.sum((pts - center) ** 2, axis=1))
    threshold = float(np.quantile(radius, float(radial_quantile)))
    candidate = radius <= threshold
    if int(np.count_nonzero(candidate)) < 8:
        return mask
    return candidate


def _collect_detector_hits(
    result: Any,
    name: str,
    *,
    primary_block_only: bool = False,
    radial_quantile: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _collect_detector_hit_data(
        result,
        name,
        primary_block_only=primary_block_only,
        radial_quantile=radial_quantile,
    )
    return data["u"], data["v"], data["intensity"]


def _collect_detector_hit_data(
    result: Any,
    name: str,
    *,
    primary_block_only: bool = False,
    radial_quantile: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    matching_blocks = [block for block in result.detector_hits if str(block["surface"]) == name]
    selected_blocks = _select_detector_block(matching_blocks, primary_block_only=primary_block_only)

    u_parts: List[np.ndarray] = []
    v_parts: List[np.ndarray] = []
    i_parts: List[np.ndarray] = []
    p_parts: List[np.ndarray] = []
    position_parts: List[np.ndarray] = []
    for block in selected_blocks:
        u = np.asarray(to_numpy(block["local_u"]), dtype=float).reshape(-1)
        v = np.asarray(to_numpy(block["local_v"]), dtype=float).reshape(-1)
        intensity = np.asarray(to_numpy(block["intensity"]), dtype=float).reshape(-1)
        power = np.asarray(to_numpy(block["power"]), dtype=float).reshape(-1)
        position = np.asarray(to_numpy(block["position"]), dtype=float).reshape(-1, 3)
        if u.size == 0:
            continue
        u_parts.append(u)
        v_parts.append(v)
        i_parts.append(intensity)
        p_parts.append(power)
        position_parts.append(position)

    if not u_parts:
        return _empty_detector_hit_data()

    u_all = np.concatenate(u_parts)
    v_all = np.concatenate(v_parts)
    i_all = np.concatenate(i_parts)
    p_all = np.concatenate(p_parts)
    position_all = np.concatenate(position_parts, axis=0)
    mask = _radial_quantile_mask(
        u_all, v_all, i_all,
        radial_quantile=radial_quantile,
    )
    return {
        "u": u_all[mask],
        "v": v_all[mask],
        "intensity": i_all[mask],
        "power": p_all[mask],
        "position": position_all[mask],
    }


def _finite_weights(intensity: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(intensity, dtype=float).reshape(-1), 0.0, None)
    if weights.size == 0 or float(np.sum(weights)) <= 0.0:
        return np.ones(weights.shape, dtype=float)
    return weights


def _spot_extent_metrics(u: np.ndarray, v: np.ndarray, intensity: np.ndarray) -> tuple[float, float]:
    if u.size < 2 or v.size < 2:
        return float("nan"), float("nan")

    weights = _finite_weights(intensity)
    pts = np.column_stack((u, v))
    center = np.average(pts, axis=0, weights=weights)
    rel = pts - center
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return float("nan"), float("nan")

    cov = (rel.T * weights) @ rel / total_weight
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    projected = rel @ axes
    diameters = np.ptp(projected, axis=0)
    diameters = np.asarray(diameters, dtype=float)
    if diameters.size < 2 or not np.all(np.isfinite(diameters)):
        return float("nan"), float("nan")

    mean_diameter = float(np.mean(diameters[:2]))
    area = float(np.pi * 0.5 * diameters[0] * 0.5 * diameters[1])
    if area <= 0.0:
        area = float("nan")
    return mean_diameter, area


def _screen_spot_metrics(data: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    u = data["u"]
    v = data["v"]
    intensity = data["intensity"]
    power = data["power"]
    position = data["position"]
    if u.size == 0:
        return {
            "integral_intensity": float("nan"),
            "ellipticity": float("nan"),
            "mean_diameter": float("nan"),
            "center": None,
        }

    weights = _finite_weights(intensity)
    center = np.average(position, axis=0, weights=weights)
    ellipticity = _ellipticity_ratio(u, v, intensity)
    mean_diameter, spot_area = _spot_extent_metrics(u, v, intensity)
    total_power = float(np.sum(np.clip(power, 0.0, None)))
    integral_intensity = total_power / spot_area if np.isfinite(spot_area) and spot_area > 0.0 else float("nan")
    return {
        "integral_intensity": integral_intensity,
        "ellipticity": ellipticity,
        "mean_diameter": mean_diameter,
        "center": center,
    }


def _format_table_number(value: Any, *, precision: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "н/д"
    if not np.isfinite(number):
        return "н/д"
    if number == 0.0:
        return "0"
    if abs(number) < 1e-4 or abs(number) >= 1e4:
        return f"{number:.{precision}e}"
    return f"{number:.{precision + 2}f}".rstrip("0").rstrip(".")


def _format_vector_m(value: Any) -> str:
    if value is None:
        return "н/д"
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return "н/д"
    return f"({_format_table_number(arr[0])}, {_format_table_number(arr[1])}, {_format_table_number(arr[2])})"


def _screen_number(screen: Mapping[str, Any]) -> int:
    label = str(screen.get("label", screen.get("name", "")))
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else 0


def _load_spot_reference_centers(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}

    centers: Dict[str, np.ndarray] = {}
    if isinstance(raw, dict):
        for name, value in raw.items():
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size == 3 and np.all(np.isfinite(arr)):
                centers[str(name)] = arr
    return centers


def _write_spot_reference_centers(path: Path, centers: Mapping[str, np.ndarray]) -> None:
    serializable = {
        str(name): [float(coord) for coord in np.asarray(center, dtype=float).reshape(3)]
        for name, center in centers.items()
    }
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _detector_summary_table_html(
    screens: Sequence[Dict[str, Any]],
    metrics_by_name: Mapping[str, Dict[str, Any]],
    reference_centers: Mapping[str, np.ndarray],
) -> str:
    ordered_screens = sorted(screens, key=_screen_number)

    def cells_for(metric: str) -> str:
        cells: List[str] = []
        for screen in ordered_screens:
            name = str(screen["name"])
            metrics = metrics_by_name[name]
            if metric == "integral_intensity":
                value = _format_table_number(metrics["integral_intensity"])
            elif metric == "ellipticity":
                value = _format_table_number(metrics["ellipticity"])
            elif metric == "mean_diameter":
                value = _format_table_number(metrics["mean_diameter"])
            elif metric == "center":
                value = _format_vector_m(metrics["center"])
            elif metric == "shift":
                center = metrics["center"]
                reference = reference_centers.get(name)
                value = _format_vector_m(np.asarray(center) - reference) if center is not None and reference is not None else "н/д"
            else:
                value = "н/д"
            cells.append(f"<td>{value}</td>")
        return "".join(cells)

    return f"""
<table class="detector-summary-table">
  <thead>
    <tr>
      <th>Характеристика</th>
      <th>Камера 1</th>
      <th>Камера 2</th>
      <th>Камера 3</th>
      <th>Камера 4</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><i>I</i>, Вт/м<sup>2</sup></td>{cells_for("integral_intensity")}</tr>
    <tr><td>Степень эллиптичности, <i>e</i></td>{cells_for("ellipticity")}</tr>
    <tr><td><i>d</i>, м</td>{cells_for("mean_diameter")}</tr>
    <tr><td>Координаты центра пятна, м</td>{cells_for("center")}</tr>
    <tr><td>Сдвиг центра пятна, м</td>{cells_for("shift")}</tr>
  </tbody>
</table>
"""


def _screen_b_indices(name: str) -> tuple[int, int]:
    parts = str(name).split("_")
    if len(parts) >= 4 and parts[0] == "screen" and parts[1] == "b":
        try:
            return int(parts[2]), int(parts[3])
        except ValueError:
            pass
    return 0, 0


def _format_bundle_center_m(row: Mapping[str, Any]) -> str:
    return f"({_format_table_number(row.get('s_center'))}, {_format_table_number(row.get('z_center'))})"


def _empty_bundle_metric_row(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "integral_intensity": float("nan"),
        "ray_count": 0,
        "mean_diameter": float("nan"),
        "total_power": float("nan"),
        "s_center": float("nan"),
        "z_center": float("nan"),
        "s_min": float("nan"),
        "s_max": float("nan"),
        "z_min": float("nan"),
        "z_max": float("nan"),
    }


def _screen_b_bundle_rows(
    result: Any,
    surface: Mapping[str, Any],
    *,
    cluster_count: int = 7,
) -> List[Dict[str, Any]]:
    name = str(surface["name"])
    rod, assembly = _screen_b_indices(name)
    prefix = f"b_{rod}_{assembly}" if rod and assembly else name
    data = _collect_detector_hit_blocks(result, name)
    positions = data["position"]
    intensity = data["intensity"]
    power = data["power"]

    rows_by_index: Dict[int, Dict[str, Any]] = {}
    if positions.size > 0:
        arc, axial = _cylindrical_unwrap_coordinates(
            positions,
            center=surface["center"],
            axis=surface["axis"],
            radius=float(surface["radius"]),
        )
        weights = _finite_weights(intensity)
        arc = arc - float(np.average(arc, weights=weights))
        axial = axial - float(np.average(axial, weights=weights))
        for row in _cylindrical_bundle_metrics(
            arc,
            axial,
            intensity,
            power,
            cluster_count=cluster_count,
            name_prefix=prefix,
        ):
            try:
                bundle_index = int(str(row["name"]).rsplit("_", 1)[-1])
            except (ValueError, TypeError):
                continue
            rows_by_index[bundle_index] = row

    return [
        rows_by_index.get(idx, _empty_bundle_metric_row(f"{prefix}_{idx}"))
        for idx in range(1, int(cluster_count) + 1)
    ]


def _gas_volume_bundle_table_html(
    result: Any,
    surfaces: Sequence[Mapping[str, Any]],
    *,
    cluster_count: int = 7,
) -> str:
    ordered_surfaces = sorted(surfaces, key=lambda surface: _screen_b_indices(str(surface["name"])))
    surface_by_index: Dict[tuple[int, int], Mapping[str, Any]] = {
        _screen_b_indices(str(surface["name"])): surface
        for surface in ordered_surfaces
    }

    rows: List[str] = [
        """
    <tr class="gas-volume-head-row">
      <th>Номер сборки</th>
      <th>Зеркало</th>
      <th>Координаты центра, м</th>
      <th>Диаметр <i>d</i>, м</th>
      <th>Количество лучей <i>N</i></th>
      <th>Интенсивность <i>I</i>, Вт/м<sup>2</sup></th>
      <th></th>
    </tr>"""
    ]

    for rod in range(1, 5):
        rows.append(f'    <tr class="rod-row"><td colspan="7">Стержень {rod}</td></tr>')
        for assembly in range(1, 5):
            surface = surface_by_index.get((rod, assembly))
            surface_name = f"screen_b_{rod}_{assembly}"
            if surface is None:
                bundle_rows = [_empty_bundle_metric_row(f"b_{rod}_{assembly}_{idx}") for idx in range(1, cluster_count + 1)]
            else:
                surface_name = str(surface["name"])
                bundle_rows = _screen_b_bundle_rows(result, surface, cluster_count=cluster_count)
            screen_label = f"Screen_b_{rod}_{assembly}"
            unwrap_href = f"{surface_name}_unwrap.html"

            for mirror_idx, bundle_row in enumerate(bundle_rows, start=1):
                cells: List[str] = []
                if mirror_idx == 1:
                    cells.append(f'<td class="assembly-cell" rowspan="{cluster_count}">{assembly}</td>')
                cells.extend(
                    [
                        f'<td class="mirror-cell">{mirror_idx}</td>',
                        f"<td>{_format_bundle_center_m(bundle_row)}</td>",
                        f"<td>{_format_table_number(bundle_row.get('mean_diameter'))}</td>",
                        f"<td>{int(bundle_row.get('ray_count', 0))}</td>",
                        f"<td>{_format_table_number(bundle_row.get('integral_intensity'))}</td>",
                    ]
                )
                if mirror_idx == 1:
                    cells.append(
                        f'<td class="screen-cell" rowspan="{cluster_count}">'
                        f'<iframe class="gas-volume-screen-frame" '
                        f'src="{escape(unwrap_href, quote=True)}" '
                        f'title="{escape(screen_label, quote=True)}"></iframe>'
                        "</td>"
                    )
                rows.append(f"    <tr>{''.join(cells)}</tr>")

    return f"""
  <div class="gas-volume-table-wrap">
    <table class="gas-volume-bundle-table">
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
  </div>
"""


def _smooth_grid(grid: np.ndarray, passes: int = 2) -> np.ndarray:
    arr = np.asarray(grid, dtype=float)
    if passes <= 0:
        return arr
    for _ in range(passes):
        padded = np.pad(arr, 1, mode="edge")
        arr = (
            padded[:-2, :-2]
            + 2.0 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2]
            + 4.0 * padded[1:-1, 1:-1]
            + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2]
            + 2.0 * padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 16.0
    return arr


def _smoothed_average_grid(
    u: np.ndarray,
    v: np.ndarray,
    intensity: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    smooth_passes: int = 2,
) -> np.ndarray:
    if u.size == 0:
        return np.full((x_edges.size - 1, y_edges.size - 1), np.nan, dtype=float)

    count_grid, _, _ = np.histogram2d(u, v, bins=(x_edges, y_edges))
    sum_grid, _, _ = np.histogram2d(u, v, bins=(x_edges, y_edges), weights=intensity)

    count_grid = _smooth_grid(count_grid, passes=smooth_passes)
    sum_grid = _smooth_grid(sum_grid, passes=smooth_passes)

    avg_grid = np.full(count_grid.shape, np.nan, dtype=float)
    mask = count_grid > 1e-12
    avg_grid[mask] = sum_grid[mask] / count_grid[mask]
    return avg_grid


def _detector_support_alpha(
    count_grid: np.ndarray,
    *,
    smooth_passes: int = _DETECTOR_SUPPORT_SMOOTH_PASSES,
    fade_pixels: float = _DETECTOR_EDGE_FADE_PIXELS,
) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    occupancy = _smooth_grid((np.asarray(count_grid, dtype=float) > 0.0).astype(float), passes=smooth_passes)
    support = occupancy >= 0.5
    if not np.any(support):
        return np.zeros_like(occupancy, dtype=float)

    inside_distance = distance_transform_edt(support)
    outside_distance = distance_transform_edt(~support)
    signed_distance = inside_distance - outside_distance
    alpha = np.clip(0.5 + signed_distance / max(float(fade_pixels), 1e-6), 0.0, 1.0)
    return _smooth_grid(alpha, passes=1)


def _detector_colorscale_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    cmap = getattr(_detector_colorscale_cmap, "_cached", None)
    if cmap is not None:
        return cmap

    color_points = []
    for position, color in _RAY_INTENSITY_COLORSCALE:
        rgb = _parse_rgb_color(str(color))
        if rgb is None:
            raise ValueError(f"Expected rgb(...) color, got {color!r}")
        rgba = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0)
        color_points.append((float(position), rgba))
    cmap = LinearSegmentedColormap.from_list("detector_screen_rgba", color_points)
    setattr(_detector_colorscale_cmap, "_cached", cmap)
    return cmap


def _detector_rgba_image(
    avg_grid: np.ndarray,
    alpha_grid: np.ndarray,
    *,
    zmin: float,
    zmax: float,
) -> np.ndarray:
    rgba = np.zeros(avg_grid.shape + (4,), dtype=np.uint8)
    valid = np.isfinite(avg_grid) & (alpha_grid > 1e-6)
    if not np.any(valid):
        return rgba

    denom = max(float(zmax) - float(zmin), 1e-12)
    normalized = np.clip((np.asarray(avg_grid, dtype=float) - float(zmin)) / denom, 0.0, 1.0)
    mapped = _detector_colorscale_cmap()(normalized)
    mapped[..., 3] = np.clip(alpha_grid, 0.0, 1.0)

    rgba = np.rint(np.clip(mapped, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba[~valid] = 0
    return rgba.transpose(1, 0, 2)


def _smoothed_detector_field(
    u: np.ndarray,
    v: np.ndarray,
    intensity: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    smooth_passes: int = 2,
    zmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    if u.size == 0:
        shape = (x_edges.size - 1, y_edges.size - 1)
        empty_avg = np.full(shape, np.nan, dtype=float)
        empty_rgba = np.zeros((y_edges.size - 1, x_edges.size - 1, 4), dtype=np.uint8)
        return empty_avg, empty_rgba

    count_grid, _, _ = np.histogram2d(u, v, bins=(x_edges, y_edges))
    avg_grid = _smoothed_average_grid(
        u,
        v,
        intensity,
        x_edges,
        y_edges,
        smooth_passes=smooth_passes,
    )
    alpha_grid = _detector_support_alpha(count_grid)
    rgba_image = _detector_rgba_image(avg_grid, alpha_grid, zmin=0.0, zmax=zmax)
    return avg_grid, rgba_image


def write_detector_screen_views(
    path: Path,
    result: Any,
    *,
    screens: Sequence[Dict[str, Any]],
    title: str,
    grid_size: int = _DETECTOR_SCREEN_GRID_SIZE,
    smooth_passes: int = _DETECTOR_SCREEN_SMOOTH_PASSES,
    remember_spot_centers: bool = False,
    gas_volume_unwrap_href: Optional[str] = None,
    gas_volume_unwrap_hrefs: Optional[Sequence[str]] = None,
    gas_volume_surfaces: Optional[Sequence[Mapping[str, Any]]] = None,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if len(screens) != 4:
        raise ValueError("Expected exactly 4 screens for a 2x2 detector view.")

    hit_map: Dict[str, Dict[str, np.ndarray]] = {}
    metrics_by_name: Dict[str, Dict[str, Any]] = {}
    max_intensity = 0.0
    for screen in screens:
        name = str(screen["name"])
        data = _collect_detector_hit_data(
            result,
            name,
            primary_block_only=bool(screen.get("primary_block_only", False)),
            radial_quantile=screen.get("radial_quantile"),
        )
        u_all = data["u"]
        i_all = data["intensity"]
        if u_all.size > 0:
            max_intensity = max(max_intensity, float(np.max(i_all)))
        hit_map[name] = data
        metrics_by_name[name] = _screen_spot_metrics(data)

    reference_centers_path = path.with_name(f"{path.stem}_spot_centers.json")
    reference_centers = _load_spot_reference_centers(reference_centers_path) if remember_spot_centers else {}
    if remember_spot_centers:
        updated_reference_centers = dict(reference_centers)
        for screen in screens:
            name = str(screen["name"])
            center = metrics_by_name[name]["center"]
            if name not in updated_reference_centers and center is not None:
                updated_reference_centers[name] = np.asarray(center, dtype=float)
        if updated_reference_centers:
            reference_centers_path.parent.mkdir(parents=True, exist_ok=True)
            _write_spot_reference_centers(reference_centers_path, updated_reference_centers)
            reference_centers = updated_reference_centers

    if max_intensity <= 0.0:
        max_intensity = 1.0

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[str(screen.get("label", screen["name"])) for screen in screens],
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 257, dtype=float)
    for idx, screen in enumerate(screens):
        row = idx // 2 + 1
        col = idx % 2 + 1
        name = str(screen["name"])
        label = str(screen.get("label", name))
        radius = float(screen["radius"])
        data = hit_map[name]
        u = data["u"]
        v = data["v"]
        intensity = data["intensity"]
        ellipticity = _ellipticity_ratio(u, v, intensity)
        screen_index = "".join(ch for ch in label if ch.isdigit()) or str(idx + 1)

        edges = np.linspace(-radius, radius, int(grid_size) + 1, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        avg_grid, rgba_image = _smoothed_detector_field(
            u,
            v,
            intensity,
            edges,
            edges,
            smooth_passes=smooth_passes,
            zmax=max_intensity,
        )

        fig.add_trace(
            go.Image(
                z=rgba_image,
                colormodel="rgba256",
                x0=centers[0],
                y0=centers[0],
                dx=centers[1] - centers[0],
                dy=centers[1] - centers[0],
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Heatmap(
                x=centers,
                y=centers,
                z=avg_grid.T,
                coloraxis="coloraxis",
                opacity=0.0,
                hovertemplate="u=%{x:.5f} m<br>v=%{y:.5f} m<br>I=%{z:.3e} W/m^2<extra></extra>",
                showscale=False,
                zmin=0.0,
                zmax=max_intensity,
            ),
            row=row,
            col=col,
        )

        circle_u = radius * np.cos(theta)
        circle_v = radius * np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=circle_u,
                y=circle_v,
                mode="lines",
                line={"color": "#444444", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=u.tolist(),
                y=v.tolist(),
                mode="markers",
                marker={
                    "size": _DETECTOR_HIT_MARKER_SIZE,
                    "color": intensity.tolist(),
                    "coloraxis": "coloraxis",
                    "line": {"color": "#111111", "width": _DETECTOR_HIT_MARKER_LINE_WIDTH},
                    "opacity": _DETECTOR_HIT_MARKER_OPACITY,
                },
                name=f"{label} hits",
                hovertemplate="u=%{x:.5f} m<br>v=%{y:.5f} m<br>I=%{marker.color:.3e} W/m^2<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(
            title_text="u [m]",
            range=[-radius, radius],
            constrain="domain",
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="v [m]",
            range=[-radius, radius],
            scaleanchor=f"x{idx + 1 if idx > 0 else ''}",
            scaleratio=1,
            constrain="domain",
            row=row,
            col=col,
        )

        axis_suffix = "" if idx == 0 else str(idx + 1)
        fig.add_annotation(
            x=1.06,
            y=0.98,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=f"e<sub>{screen_index}</sub> = {ellipticity:.4f}" if np.isfinite(ellipticity) else f"e<sub>{screen_index}</sub> = n/a",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="right",
            font={"size": 16, "color": "#111111"},
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor="rgba(0,0,0,0.18)",
            borderwidth=1,
        )

    fig.update_layout(
        title=None,
        width=1120,
        height=820,
        coloraxis={
            "colorscale": _RAY_INTENSITY_COLORSCALE,
            "cmin": 0.0,
            "cmax": max_intensity,
            "colorbar": {
                "title": "I [W/m^2]",
                "x": 1.02,
                "tickformat": ".2e",
            },
        },
        margin={"l": 60, "r": 135, "t": 45, "b": 70},
        showlegend=False,
    )
    plot_html = fig.to_html(include_plotlyjs=True, full_html=False, config={"responsive": True})
    summary_table_html = _detector_summary_table_html(screens, metrics_by_name, reference_centers)
    gas_volume_section_html = ""
    if gas_volume_surfaces:
        gas_volume_section_html = f"""
  <h2 class="section-heading">Попадание лучей в рабочий газовый объем</h2>
{_gas_volume_bundle_table_html(result, gas_volume_surfaces)}
"""
    unwrap_hrefs: List[str] = []
    if gas_volume_unwrap_href:
        unwrap_hrefs.append(str(gas_volume_unwrap_href))
    if gas_volume_unwrap_hrefs:
        unwrap_hrefs.extend(str(href) for href in gas_volume_unwrap_hrefs)
    if unwrap_hrefs and not gas_volume_section_html:
        iframe_html = "\n".join(
            f'    <iframe src="{escape(href, quote=True)}" title="Развертка {escape(Path(href).stem)}"></iframe>'
            for href in unwrap_hrefs
        )
        gas_volume_section_html = f"""
  <h2 class="section-heading">Попадание лучей в рабочий газовый объем</h2>
  <div class="plot-card gas-volume-card">
{iframe_html}
  </div>
"""

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Результаты моделирования</title>
  <style>
    :root {{
      color: #183052;
      background: #f4f7fb;
      font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif;
    }}
    body {{
      margin: 0;
      padding: 34px;
      background:
        radial-gradient(circle at 16% 0%, rgba(225, 129, 215, 0.18), transparent 30rem),
        radial-gradient(circle at 88% 12%, rgba(48, 112, 220, 0.14), transparent 28rem),
        #f4f7fb;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 31px;
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    h2 {{
      margin: 0 0 18px;
      font-size: 23px;
      font-weight: 650;
      color: #20385d;
    }}
    .section-heading {{
      margin-top: 32px;
    }}
    .plot-card {{
      width: min(1120px, 100%);
      padding: 14px 12px 4px;
      border: 1px solid rgba(60, 86, 125, 0.16);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.72);
      box-shadow: 0 18px 45px rgba(32, 58, 96, 0.11);
    }}
    .detector-summary-table {{
      width: min(1120px, 100%);
      margin-top: 26px;
      border-collapse: separate;
      border-spacing: 0;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(60, 86, 125, 0.18);
      border-radius: 18px;
      box-shadow: 0 18px 45px rgba(32, 58, 96, 0.12);
    }}
    .detector-summary-table th {{
      padding: 15px 14px;
      text-align: center;
      color: #182c4d;
      background: rgba(198, 211, 232, 0.92);
      border-bottom: 1px solid rgba(60, 86, 125, 0.18);
      font-size: 16px;
      font-weight: 700;
    }}
    .detector-summary-table th + th {{
      border-left: 1px solid rgba(60, 86, 125, 0.12);
    }}
    .detector-summary-table td {{
      padding: 15px 14px;
      text-align: center;
      vertical-align: middle;
      color: #14243d;
      border-bottom: 1px solid rgba(60, 86, 125, 0.12);
      font-size: 15px;
      line-height: 1.4;
    }}
    .detector-summary-table tr:last-child td {{
      border-bottom: 0;
    }}
    .detector-summary-table td + td {{
      border-left: 1px solid rgba(60, 86, 125, 0.10);
    }}
    .detector-summary-table td:first-child {{
      width: 24%;
      text-align: left;
      font-weight: 650;
      color: #20385d;
      background: rgba(230, 236, 246, 0.62);
    }}
    .gas-volume-card {{
      padding: 0;
      overflow: hidden;
    }}
    .gas-volume-card iframe {{
      display: block;
      width: 100%;
      height: 680px;
      border: 0;
      background: #f4f7fb;
    }}
    .gas-volume-card iframe + iframe {{
      border-top: 1px solid rgba(60, 86, 125, 0.16);
    }}
    .gas-volume-table-wrap {{
      width: min(1120px, 100%);
      overflow-x: auto;
      border: 1px solid rgba(60, 86, 125, 0.18);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 18px 45px rgba(32, 58, 96, 0.12);
    }}
    .gas-volume-bundle-table {{
      width: 100%;
      min-width: 2100px;
      border-collapse: separate;
      border-spacing: 0;
      color: #14243d;
      font-size: 14px;
    }}
    .gas-volume-bundle-table th,
    .gas-volume-bundle-table td {{
      padding: 10px 12px;
      text-align: center;
      vertical-align: middle;
      border-right: 1px solid rgba(60, 86, 125, 0.12);
      border-bottom: 1px solid rgba(60, 86, 125, 0.12);
      line-height: 1.35;
      white-space: nowrap;
    }}
    .gas-volume-bundle-table th:last-child,
    .gas-volume-bundle-table td:last-child {{
      border-right: 0;
    }}
    .gas-volume-bundle-table tr:last-child td {{
      border-bottom: 0;
    }}
    .gas-volume-head-row th {{
      background: rgba(198, 211, 232, 0.96);
      color: #182c4d;
      font-weight: 700;
      font-size: 15px;
    }}
    .gas-volume-bundle-table .rod-row td {{
      background: rgba(203, 216, 235, 0.92);
      color: #20385d;
      font-weight: 700;
      text-align: left;
      letter-spacing: 0.01em;
    }}
    .gas-volume-bundle-table .assembly-cell,
    .gas-volume-bundle-table .mirror-cell {{
      background: rgba(230, 236, 246, 0.70);
      color: #20385d;
      font-weight: 650;
    }}
    .gas-volume-bundle-table .screen-cell {{
      background: rgba(238, 243, 250, 0.82);
      color: #20385d;
      font-weight: 650;
      width: 1040px;
      min-width: 1040px;
      padding: 8px;
    }}
    .gas-volume-screen-frame {{
      display: block;
      width: 100%;
      height: 720px;
      border: 0;
      border-radius: 12px;
      background: #f4f7fb;
      box-shadow: inset 0 0 0 1px rgba(60, 86, 125, 0.12);
    }}
  </style>
</head>
<body>
  <h1>Результаты моделирования</h1>
  <h2>Попадание лучей на камеры</h2>
  <div class="plot-card">
    {plot_html}
  </div>
  {summary_table_html}
  {gas_volume_section_html}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_single_detector_screen_view(
    path: Path,
    result: Any,
    *,
    screen: Dict[str, Any],
    title: str,
    grid_size: int = _SINGLE_DETECTOR_SCREEN_GRID_SIZE,
    smooth_passes: int = _DETECTOR_SCREEN_SMOOTH_PASSES,
) -> None:
    import plotly.graph_objects as go

    name = str(screen["name"])
    label = str(screen.get("label", name))
    half_width = 0.5 * float(screen["width"])
    half_height = 0.5 * float(screen["height"])

    u, v, intensity = _collect_detector_hits(
        result,
        name,
        primary_block_only=bool(screen.get("primary_block_only", False)),
        radial_quantile=screen.get("radial_quantile"),
    )

    if u.size > 0:
        max_intensity = max(1.0, float(np.max(intensity)))
    else:
        max_intensity = 1.0

    ellipticity = _ellipticity_ratio(u, v, intensity)

    u_edges = np.linspace(-half_width, half_width, int(grid_size) + 1, dtype=float)
    v_edges = np.linspace(-half_height, half_height, int(grid_size) + 1, dtype=float)
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

    avg_grid, rgba_image = _smoothed_detector_field(
        u,
        v,
        intensity,
        u_edges,
        v_edges,
        smooth_passes=smooth_passes,
        zmax=max_intensity,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Image(
            z=rgba_image,
            colormodel="rgba256",
            x0=u_centers[0],
            y0=v_centers[0],
            dx=u_centers[1] - u_centers[0],
            dy=v_centers[1] - v_centers[0],
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Heatmap(
            x=u_centers,
            y=v_centers,
            z=avg_grid.T,
            colorscale=_RAY_INTENSITY_COLORSCALE,
            zmin=0.0,
            zmax=max_intensity,
            opacity=0.0,
            colorbar={"title": "I [W/m^2]", "x": 1.02, "tickformat": ".2e"},
            hovertemplate="u=%{x:.5f} m<br>v=%{y:.5f} m<br>I=%{z:.3e} W/m^2<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=u.tolist(),
            y=v.tolist(),
            mode="markers",
            marker={
                "size": _DETECTOR_HIT_MARKER_SIZE + 1.0,
                "color": intensity.tolist(),
                "colorscale": _RAY_INTENSITY_COLORSCALE,
                "cmin": 0.0,
                "cmax": max_intensity,
                "line": {"color": "#111111", "width": _DETECTOR_HIT_MARKER_LINE_WIDTH},
                "opacity": _DETECTOR_HIT_MARKER_OPACITY,
                "showscale": False,
            },
            name=f"{label} hits",
            hovertemplate="u=%{x:.5f} m<br>v=%{y:.5f} m<br>I=%{marker.color:.3e} W/m^2<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis={"title": "u [m]", "range": [-half_width, half_width], "constrain": "domain"},
        yaxis={"title": "v [m]", "range": [-half_height, half_height], "scaleanchor": "x", "scaleratio": 1, "constrain": "domain"},
        margin={"l": 60, "r": 95, "t": 60, "b": 50},
        showlegend=False,
    )
    fig.add_annotation(
        x=1.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"e = {ellipticity:.4f}" if np.isfinite(ellipticity) else "e = n/a",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"size": 16, "color": "#111111"},
        bgcolor="rgba(255,255,255,0.80)",
        bordercolor="rgba(0,0,0,0.18)",
        borderwidth=1,
    )
    fig.write_html(path, include_plotlyjs=True)


def _collect_detector_hit_blocks(result: Any, name: str) -> Dict[str, np.ndarray]:
    matching_blocks = [block for block in result.detector_hits if str(block["surface"]) == name]
    if not matching_blocks:
        return _empty_detector_hit_data()

    position_parts: List[np.ndarray] = []
    intensity_parts: List[np.ndarray] = []
    power_parts: List[np.ndarray] = []
    for block in matching_blocks:
        position = np.asarray(to_numpy(block["position"]), dtype=float).reshape(-1, 3)
        intensity = np.asarray(to_numpy(block["intensity"]), dtype=float).reshape(-1)
        power = np.asarray(to_numpy(block["power"]), dtype=float).reshape(-1)
        if position.size == 0:
            continue
        position_parts.append(position)
        intensity_parts.append(intensity)
        power_parts.append(power)

    if not position_parts:
        return _empty_detector_hit_data()

    return {
        "u": np.zeros((0,), dtype=float),
        "v": np.zeros((0,), dtype=float),
        "position": np.concatenate(position_parts, axis=0),
        "intensity": np.concatenate(intensity_parts),
        "power": np.concatenate(power_parts),
    }


def _cylindrical_unwrap_coordinates(
    positions: np.ndarray,
    *,
    center: Sequence[float],
    axis: Sequence[float],
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(center, dtype=float)
    u_axis, v_axis, w_axis = _plane_basis(axis, None)
    rel = np.asarray(positions, dtype=float).reshape(-1, 3) - c
    axial = rel @ w_axis
    radial = rel - axial[:, None] * w_axis
    x_local = radial @ u_axis
    y_local = radial @ v_axis
    angle = np.arctan2(y_local, x_local)
    arc = float(radius) * angle
    return arc, axial


def _cylindrical_bundle_metrics(
    arc: np.ndarray,
    axial: np.ndarray,
    intensity: np.ndarray,
    power: np.ndarray,
    *,
    cluster_count: int,
    name_prefix: str,
) -> List[Dict[str, Any]]:
    count = max(1, int(cluster_count))
    valid = (
        np.isfinite(arc)
        & np.isfinite(axial)
        & np.isfinite(intensity)
        & np.isfinite(power)
    )
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return []

    count = min(count, int(valid_indices.size))
    ordered_indices = valid_indices[np.argsort(arc[valid_indices])]
    if count == 1:
        groups = [ordered_indices]
    else:
        ordered_arc = arc[ordered_indices]
        gaps = np.diff(ordered_arc)
        split_points = np.sort(np.argsort(gaps)[-(count - 1):] + 1)
        groups = [group for group in np.split(ordered_indices, split_points) if group.size > 0]

    rows: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups, start=1):
        group_arc = arc[group]
        group_axial = axial[group]
        group_intensity = intensity[group]
        group_power = power[group]
        weights = _finite_weights(group_intensity)
        mean_diameter, spot_area = _spot_extent_metrics(group_arc, group_axial, group_intensity)
        total_power = float(np.sum(np.clip(group_power, 0.0, None)))
        integral_intensity = (
            total_power / spot_area
            if np.isfinite(spot_area) and float(spot_area) > 0.0
            else float("nan")
        )
        rows.append(
            {
                "name": f"{name_prefix}_{idx}",
                "integral_intensity": integral_intensity,
                "ray_count": int(group.size),
                "mean_diameter": mean_diameter,
                "total_power": total_power,
                "s_center": float(np.average(group_arc, weights=weights)),
                "z_center": float(np.average(group_axial, weights=weights)),
                "s_min": float(np.min(group_arc)),
                "s_max": float(np.max(group_arc)),
                "z_min": float(np.min(group_axial)),
                "z_max": float(np.max(group_axial)),
            }
        )
    return rows


def _write_cylindrical_bundle_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "integral_intensity_W_m2",
        "ray_count",
        "mean_diameter_m",
        "total_power_W",
        "s_center_m",
        "z_center_m",
        "s_min_m",
        "s_max_m",
        "z_min_m",
        "z_max_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "integral_intensity_W_m2": float(row["integral_intensity"]),
                    "ray_count": int(row["ray_count"]),
                    "mean_diameter_m": float(row["mean_diameter"]),
                    "total_power_W": float(row["total_power"]),
                    "s_center_m": float(row["s_center"]),
                    "z_center_m": float(row["z_center"]),
                    "s_min_m": float(row["s_min"]),
                    "s_max_m": float(row["s_max"]),
                    "z_min_m": float(row["z_min"]),
                    "z_max_m": float(row["z_max"]),
                }
            )


def _cylindrical_bundle_metrics_table_html(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    body = "\n".join(
        "      <tr>"
        f"<td>{escape(str(row['name']))}</td>"
        f"<td>{_format_table_number(row['integral_intensity'])}</td>"
        f"<td>{int(row['ray_count'])}</td>"
        f"<td>{_format_table_number(row['mean_diameter'])}</td>"
        "</tr>"
        for row in rows
    )
    return f"""
  <section class="bundle-section">
    <h2>Характеристики пучков на screen_b_1_1</h2>
    <table class="bundle-table">
      <thead>
        <tr>
          <th>Пучок</th>
          <th>Интегральная интенсивность, Вт/м<sup>2</sup></th>
          <th>Количество лучей</th>
          <th>Средний диаметр, м</th>
        </tr>
      </thead>
      <tbody>
{body}
      </tbody>
    </table>
  </section>
"""


def write_cylindrical_unwrap_view(
    path: Path,
    result: Any,
    *,
    surface: Dict[str, Any],
    title: str = "Развертка цилиндрической поверхности",
    grid_size: int = 180,
    smooth_passes: int = _DETECTOR_SCREEN_SMOOTH_PASSES,
    view_padding_m: float = 0.003,
    bundle_cluster_count: Optional[int] = None,
    bundle_name_prefix: Optional[str] = None,
    bundle_metrics_csv_path: Optional[Path] = None,
) -> None:
    import plotly.graph_objects as go

    name = str(surface["name"])
    center = surface["center"]
    axis = surface["axis"]
    radius = float(surface["radius"])
    length = float(surface["length"])

    data = _collect_detector_hit_blocks(result, name)
    positions = data["position"]
    intensity = data["intensity"]
    power = data["power"]
    if positions.size > 0:
        arc, axial = _cylindrical_unwrap_coordinates(
            positions,
            center=center,
            axis=axis,
            radius=radius,
        )
        weights = _finite_weights(intensity)
        arc_center = float(np.average(arc, weights=weights))
        axial_center = float(np.average(axial, weights=weights))
        arc = arc - arc_center
        axial = axial - axial_center
        max_intensity = max(1.0, float(np.max(intensity))) if intensity.size > 0 else 1.0
    else:
        arc = np.zeros((0,), dtype=float)
        axial = np.zeros((0,), dtype=float)
        max_intensity = 1.0

    bundle_rows: List[Dict[str, Any]] = []
    if bundle_cluster_count is not None and bundle_name_prefix:
        bundle_rows = _cylindrical_bundle_metrics(
            arc,
            axial,
            intensity,
            power,
            cluster_count=int(bundle_cluster_count),
            name_prefix=str(bundle_name_prefix),
        )
        if bundle_metrics_csv_path is not None:
            _write_cylindrical_bundle_metrics_csv(bundle_metrics_csv_path, bundle_rows)

    half_arc = np.pi * radius
    half_length = 0.5 * length
    if arc.size > 0:
        padding = max(float(view_padding_m), 0.0)
        x_min = max(-half_arc, float(np.min(arc)) - padding)
        x_max = min(half_arc, float(np.max(arc)) + padding)
        y_min = max(-half_length, float(np.min(axial)) - padding)
        y_max = min(half_length, float(np.max(axial)) + padding)
        min_span = max(2.0 * padding, 0.002)
        if x_max - x_min < min_span:
            center_x = 0.5 * (x_min + x_max)
            x_min = max(-half_arc, center_x - 0.5 * min_span)
            x_max = min(half_arc, center_x + 0.5 * min_span)
        if y_max - y_min < min_span:
            center_y = 0.5 * (y_min + y_max)
            y_min = max(-half_length, center_y - 0.5 * min_span)
            y_max = min(half_length, center_y + 0.5 * min_span)
    else:
        x_min, x_max = -half_arc, half_arc
        y_min, y_max = -half_length, half_length

    x_edges = np.linspace(x_min, x_max, int(grid_size) + 1, dtype=float)
    y_edges = np.linspace(y_min, y_max, int(max(40, grid_size // 2)) + 1, dtype=float)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    avg_grid, rgba_image = _smoothed_detector_field(
        arc,
        axial,
        intensity,
        x_edges,
        y_edges,
        smooth_passes=smooth_passes,
        zmax=max_intensity,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Image(
            z=rgba_image,
            colormodel="rgba256",
            x0=x_centers[0],
            y0=y_centers[0],
            dx=x_centers[1] - x_centers[0],
            dy=y_centers[1] - y_centers[0],
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=avg_grid.T,
            colorscale=_RAY_INTENSITY_COLORSCALE,
            zmin=0.0,
            zmax=max_intensity,
            opacity=0.0,
            colorbar={"title": "I [W/m^2]", "x": 1.02, "tickformat": ".2e"},
            hovertemplate="s-s_c=%{x:.5f} m<br>z-z_c=%{y:.5f} m<br>I=%{z:.3e} W/m^2<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=arc.tolist(),
            y=axial.tolist(),
            mode="markers",
            marker={
                "size": _DETECTOR_HIT_MARKER_SIZE,
                "color": intensity.tolist(),
                "colorscale": _RAY_INTENSITY_COLORSCALE,
                "cmin": 0.0,
                "cmax": max_intensity,
                "line": {"color": "#111111", "width": _DETECTOR_HIT_MARKER_LINE_WIDTH},
                "opacity": _DETECTOR_HIT_MARKER_OPACITY,
                "showscale": False,
            },
            name=f"{name} intersections",
            hovertemplate="s-s_c=%{x:.5f} m<br>z-z_c=%{y:.5f} m<br>I=%{marker.color:.3e} W/m^2<extra></extra>",
            showlegend=False,
        )
    )
    if arc.size == 0:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Пересечений с поверхностью нет",
            showarrow=False,
            font={"size": 18, "color": "#20385d"},
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(60,86,125,0.18)",
            borderwidth=1,
        )
    for row in bundle_rows:
        label_y = min(y_max, float(row["z_max"]) + 0.00022)
        fig.add_annotation(
            x=float(row["s_center"]),
            y=label_y,
            text=escape(str(row["name"])),
            showarrow=False,
            font={"size": 11, "color": "#20385d"},
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="rgba(60,86,125,0.18)",
            borderwidth=1,
        )
    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 22}},
        width=980,
        height=620,
        margin={"l": 70, "r": 110, "t": 80, "b": 70},
        plot_bgcolor="#edf2f8",
        paper_bgcolor="#f4f7fb",
        showlegend=False,
    )
    fig.update_xaxes(
        title_text="s - s_c [m]",
        range=[x_min, x_max],
        showgrid=True,
        gridcolor="rgba(255,255,255,0.85)",
    )
    fig.update_yaxes(
        title_text="z - z_c [m]",
        range=[y_min, y_max],
        showgrid=True,
        gridcolor="rgba(255,255,255,0.85)",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_html = fig.to_html(include_plotlyjs=True, full_html=False, config={"responsive": True})
    bundle_table_html = _cylindrical_bundle_metrics_table_html(bundle_rows)
    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color: #183052;
      background: #f4f7fb;
      font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif;
    }}
    body {{
      margin: 0;
      padding: 18px 22px 28px;
      background: #f4f7fb;
    }}
    .plot-wrap {{
      width: min(980px, 100%);
      margin: 0 auto;
    }}
    .bundle-section {{
      width: min(940px, calc(100% - 36px));
      margin: 12px auto 0;
      padding: 18px 18px 20px;
      border: 1px solid rgba(60, 86, 125, 0.18);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.9);
      box-shadow: 0 12px 34px rgba(32, 58, 96, 0.10);
    }}
    .bundle-section h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      color: #20385d;
    }}
    .bundle-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      overflow: hidden;
      border: 1px solid rgba(60, 86, 125, 0.18);
      border-radius: 12px;
      background: #ffffff;
      font-size: 14px;
    }}
    .bundle-table th,
    .bundle-table td {{
      padding: 10px 12px;
      border-right: 1px solid rgba(60, 86, 125, 0.12);
      border-bottom: 1px solid rgba(60, 86, 125, 0.12);
      text-align: center;
      white-space: nowrap;
    }}
    .bundle-table th {{
      background: rgba(190, 204, 224, 0.55);
      font-weight: 650;
    }}
    .bundle-table td:first-child,
    .bundle-table th:first-child {{
      text-align: left;
    }}
    .bundle-table tr:last-child td {{
      border-bottom: 0;
    }}
    .bundle-table th:last-child,
    .bundle-table td:last-child {{
      border-right: 0;
    }}
  </style>
</head>
<body>
  <div class="plot-wrap">
{plot_html}
  </div>
{bundle_table_html}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _format_scientific(value: float, *, precision: int = 4) -> str:
    return f"{float(value):.{precision}e}"


def _initial_beam_coordinates(source: Any, rays: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(to_numpy(rays.position), dtype=float)
    intensities = np.asarray(to_numpy(rays.intensity), dtype=float).reshape(-1)
    center = np.asarray(source.waist_position, dtype=float)
    u_axis, v_axis, _ = _plane_basis(source.axis, source.polarization_reference)
    rel = positions - center
    u = rel @ u_axis
    v = rel @ v_axis
    return u, v, intensities


def _beam_coordinates_at_z(source: Any, rays: Any, z_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(to_numpy(rays.position), dtype=float)
    directions = np.asarray(to_numpy(rays.direction), dtype=float)
    intensities = np.asarray(to_numpy(rays.intensity), dtype=float).reshape(-1)
    dz = directions[:, 2]
    valid = np.abs(dz) > 1e-15
    if not np.any(valid):
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    t = (float(z_m) - positions[:, 2]) / dz
    valid = valid & (t >= -1e-12)
    if not np.any(valid):
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    hit_points = positions[valid] + t[valid, None] * directions[valid]
    u_axis, v_axis, w_axis = _plane_basis(source.axis, source.polarization_reference)
    waist = np.asarray(source.waist_position, dtype=float)
    if abs(float(w_axis[2])) > 1e-15:
        center = waist + ((float(z_m) - waist[2]) / w_axis[2]) * w_axis
    else:
        center = np.mean(hit_points, axis=0)

    rel = hit_points - center
    u = rel @ u_axis
    v = rel @ v_axis
    return u, v, intensities[valid]


def _initial_beam_diameter(source: Any, rays: Any) -> float:
    u, v, _ = _initial_beam_coordinates(source, rays)
    if u.size == 0:
        return 2.0 * float(source.cutoff_ratio) * float(source.waist_radius)
    radius_from_rays = float(np.max(np.sqrt(u * u + v * v)))
    radius_from_source = float(source.cutoff_ratio) * float(source.waist_radius)
    return 2.0 * max(radius_from_rays, radius_from_source)


def _initial_integral_intensity(source: Any, rays: Any, diameter: float) -> float:
    powers = np.asarray(to_numpy(rays.power), dtype=float).reshape(-1)
    beam_radius = 0.5 * float(diameter)
    beam_area = np.pi * beam_radius * beam_radius
    if beam_area <= 0.0:
        return float("nan")
    return float(np.sum(powers) / beam_area)


def _beam_cross_section_html(
    source: Any,
    rays: Any,
    *,
    diameter: float,
    title: str,
    z_m: float | None = None,
) -> str:
    import plotly.graph_objects as go

    if z_m is None:
        u, v, intensity = _initial_beam_coordinates(source, rays)
    else:
        u, v, intensity = _beam_coordinates_at_z(source, rays, z_m)
    radius = 0.5 * float(diameter)
    if radius <= 0.0:
        radius = float(source.cutoff_ratio) * float(source.waist_radius)
    if radius <= 0.0:
        radius = 1.0

    grid_size = 180
    edges = np.linspace(-radius, radius, grid_size + 1, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    uu, vv = np.meshgrid(centers, centers, indexing="ij")
    rr = np.sqrt(uu * uu + vv * vv)
    waist_radius = max(float(source.waist_radius), 1e-12)
    max_intensity = float(getattr(source, "peak_intensity", np.max(intensity) if intensity.size > 0 else 1.0))
    avg_grid = max_intensity * np.exp(-2.0 * (rr / waist_radius) ** 2)
    avg_grid[rr > radius] = np.nan
    edge_width = max(2.5 * (2.0 * radius / grid_size), 1e-12)
    alpha_grid = np.clip(0.5 + (radius - rr) / edge_width, 0.0, 1.0)
    rgba_image = _detector_rgba_image(avg_grid, alpha_grid, zmin=0.0, zmax=max(max_intensity, 1.0))

    theta = np.linspace(0.0, 2.0 * np.pi, 241, dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Image(
            z=rgba_image,
            colormodel="rgba256",
            x0=centers[0],
            y0=centers[0],
            dx=centers[1] - centers[0],
            dy=centers[1] - centers[0],
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Heatmap(
            x=centers,
            y=centers,
            z=avg_grid.T,
            colorscale=_RAY_INTENSITY_COLORSCALE,
            zmin=0.0,
            zmax=max(max_intensity, 1.0),
            opacity=0.0,
            showscale=False,
            hovertemplate="u=%{x:.5f} m<br>v=%{y:.5f} m<br>I=%{z:.3e} W/m^2<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=(radius * np.cos(theta)).tolist(),
            y=(radius * np.sin(theta)).tolist(),
            mode="lines",
            line={"color": "rgba(30,45,70,0.70)", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 15}},
        width=440,
        height=360,
        margin={"l": 48, "r": 18, "t": 46, "b": 44},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#edf2f8",
        showlegend=False,
    )
    fig.update_xaxes(
        title_text="u [m]",
        range=[-radius, radius],
        constrain="domain",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.85)",
    )
    fig.update_yaxes(
        title_text="v [m]",
        range=[-radius, radius],
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.85)",
    )
    return fig.to_html(include_plotlyjs=True, full_html=False, config={"displayModeBar": False, "responsive": True})


def _format_angle_deg(value: Any, *, precision: int = 3) -> str:
    angle = float(value)
    if abs(angle) < 0.5 * 10 ** (-precision):
        angle = 0.0
    return f"{angle:.{precision}f}"


def _adjustable_mirror_angles_table_html(mirror_angles: Optional[Sequence[Dict[str, Any]]]) -> str:
    if not mirror_angles:
        return ""

    rows: list[str] = [
        "<table class=\"angle-table\">",
        "<thead><tr>"
        "<th>Зеркало</th>"
        "<th>Ось поворота</th>"
        "<th>Угол поворота, deg</th>"
        "</tr></thead>",
        "<tbody>",
    ]
    for item in mirror_angles:
        mirror_name = escape(str(item.get("mirror", item.get("name", ""))))
        x_angle = _format_angle_deg(item.get("x", 0.0))
        z_angle = _format_angle_deg(item.get("z", 0.0))
        rows.append(
            "<tr>"
            f"<td rowspan=\"2\" class=\"mirror-name\">{mirror_name}</td>"
            "<td class=\"axis-name\"><i>x</i></td>"
            f"<td>{x_angle}</td>"
            "</tr>"
        )
        rows.append(
            "<tr>"
            "<td class=\"axis-name\"><i>z</i></td>"
            f"<td>{z_angle}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def write_beam_characteristics_window(
    path: Path,
    source: Any,
    rays: Any,
    mirror_angles: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    ray_count = int(getattr(rays, "n_rays", len(to_numpy(rays.power))))
    diameter = _initial_beam_diameter(source, rays)
    integral_intensity = _initial_integral_intensity(source, rays, diameter)
    angle_table_html = _adjustable_mirror_angles_table_html(mirror_angles)
    cross_section_html = _beam_cross_section_html(
        source,
        rays,
        diameter=diameter,
        title="Сечение пучка при z = 3 м",
        z_m=3.0,
    )
    beam_type = "Гауссово распределение интенсивности"

    rows = [
        ("Начальная интенсивность, <i>I</i><sub>0</sub>", f"{_format_scientific(integral_intensity)} Вт/м<sup>2</sup>"),
        ("Количество лучей, <i>N</i>", f"{ray_count:d}"),
        ("Начальный диаметр, <i>D</i><sub>0</sub>", f"{_format_scientific(diameter)} м"),
    ]
    html_rows = "\n".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>"
        for label, value in rows
    )
    html_rows += (
        "\n<tr>"
        "<td>Тип пучка</td>"
        f"<td><div class=\"beam-type\">{escape(beam_type)}</div><div class=\"section-plot\">{cross_section_html}</div></td>"
        "</tr>"
    )

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Характеристики лазерного луча</title>
  <style>
    :root {{
      color: #183052;
      background: #f4f7fb;
      font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif;
    }}
    body {{
      margin: 0;
      padding: 36px;
      background:
        radial-gradient(circle at 20% 0%, rgba(225, 129, 215, 0.20), transparent 32rem),
        radial-gradient(circle at 90% 10%, rgba(48, 112, 220, 0.16), transparent 30rem),
        #f4f7fb;
    }}
    h1 {{
      margin: 0 0 24px;
      font-size: 30px;
      font-weight: 650;
      letter-spacing: 0.01em;
    }}
    h2 {{
      margin: 30px 0 16px;
      font-size: 24px;
      font-weight: 650;
      letter-spacing: 0.01em;
    }}
    table {{
      width: min(980px, 100%);
      border-collapse: separate;
      border-spacing: 0;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(60, 86, 125, 0.18);
      border-radius: 18px;
      box-shadow: 0 18px 45px rgba(32, 58, 96, 0.12);
    }}
    td {{
      padding: 18px 20px;
      vertical-align: top;
      border-bottom: 1px solid rgba(60, 86, 125, 0.14);
      font-size: 17px;
      line-height: 1.45;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    td:first-child {{
      width: 34%;
      font-weight: 650;
      color: #20385d;
      background: rgba(230, 236, 246, 0.65);
    }}
    td:nth-child(2) {{
      color: #14243d;
    }}
    .beam-type {{
      margin-bottom: 12px;
      font-weight: 650;
    }}
    .section-plot {{
      max-width: 460px;
    }}
    .angle-table {{
      margin-top: 0;
    }}
    .angle-table th {{
      padding: 15px 18px;
      text-align: center;
      color: #182c4d;
      background: rgba(198, 211, 232, 0.92);
      border-bottom: 1px solid rgba(60, 86, 125, 0.18);
      font-size: 16px;
      font-weight: 700;
    }}
    .angle-table th + th {{
      border-left: 1px solid rgba(60, 86, 125, 0.12);
    }}
    .angle-table td {{
      text-align: center;
      vertical-align: middle;
      background: rgba(255, 255, 255, 0.86);
    }}
    .angle-table td + td {{
      border-left: 1px solid rgba(60, 86, 125, 0.12);
    }}
    .angle-table td:first-child {{
      width: 22%;
      color: #20385d;
      background: rgba(230, 236, 246, 0.62);
    }}
    .angle-table .axis-name {{
      width: 22%;
      font-weight: 650;
    }}
    .angle-table .mirror-name {{
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <h1>Характеристики лазерного луча</h1>
  <table class="characteristics-table">
    {html_rows}
  </table>
  <h2>Угловые координаты регулируемых зеркал</h2>
  {angle_table_html}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_plotly_trajectories(
    path: Path,
    result: Any,
    *,
    title: str,
    overlays: Optional[List[Dict[str, Any]]] = None,
    max_segments_per_block: int = 400,
    intensity_bins: int = 16,
    detector_hit_exclude_prefixes: Optional[Sequence[str]] = None,
    trim_end_surface_prefixes: Optional[Sequence[str]] = None,
    trim_end_distance: float = 0.0,
    hide_end_surface_prefixes: Optional[Sequence[str]] = None,
    min_segment_power: float = 0.0,
    always_include_surface_prefixes: Optional[Sequence[str]] = None,
) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    sampled_segments: List[Dict[str, float]] = []
    for block in result.segments:
        x0 = to_numpy(block["x0"])
        y0 = to_numpy(block["y0"])
        z0 = to_numpy(block["z0"])
        x1 = to_numpy(block["x1"])
        y1 = to_numpy(block["y1"])
        z1 = to_numpy(block["z1"])
        powers = to_numpy(block["power"])
        intensities = to_numpy(block["intensity"])
        surfaces = np.asarray(block["surface"], dtype=object)
        n = len(x0)
        stride = max(1, (n + max_segments_per_block - 1) // max_segments_per_block)
        for i in range(n):
            surface = str(surfaces[i])
            include = (i % stride == 0)
            if always_include_surface_prefixes and any(surface.startswith(prefix) for prefix in always_include_surface_prefixes):
                include = True
            if not include:
                continue
            if float(powers[i]) < float(min_segment_power):
                continue
            end_x = float(x1[i])
            end_y = float(y1[i])
            end_z = float(z1[i])
            if hide_end_surface_prefixes and any(surface.startswith(prefix) for prefix in hide_end_surface_prefixes):
                continue
            if trim_end_surface_prefixes and trim_end_distance > 0.0:
                if any(surface.startswith(prefix) for prefix in trim_end_surface_prefixes):
                    start = np.array((float(x0[i]), float(y0[i]), float(z0[i])), dtype=float)
                    end = np.array((end_x, end_y, end_z), dtype=float)
                    delta = end - start
                    length = float(np.linalg.norm(delta))
                    if length > 1e-12:
                        trim = min(float(trim_end_distance), 0.25 * length)
                        end = end - (delta / length) * trim
                        end_x = float(end[0])
                        end_y = float(end[1])
                        end_z = float(end[2])
            sampled_segments.append(
                {
                    "x0": float(x0[i]),
                    "y0": float(y0[i]),
                    "z0": float(z0[i]),
                    "x1": end_x,
                    "y1": end_y,
                    "z1": end_z,
                    "intensity": float(intensities[i]),
                }
            )

    if sampled_segments:
        positive = np.asarray([seg["intensity"] for seg in sampled_segments if seg["intensity"] > 0.0], dtype=float)
        if positive.size:
            log_values = np.log10(positive)
            log_min = float(np.min(log_values))
            log_max = float(np.max(log_values))
            if not np.isfinite(log_min) or not np.isfinite(log_max):
                log_min, log_max = 0.0, 1.0
            if abs(log_max - log_min) < 1e-12:
                log_min -= 0.5
                log_max += 0.5
        else:
            log_min, log_max = 0.0, 1.0

        bins = max(1, int(intensity_bins))
        edges = np.linspace(log_min, log_max, bins + 1, dtype=float)
        colors = sample_colorscale(_RAY_INTENSITY_COLORSCALE, [i / max(1, bins - 1) for i in range(bins)])
        grouped_bins: List[Dict[str, List[float | None]]] = [{"x": [], "y": [], "z": []} for _ in range(bins)]
        for seg in sampled_segments:
            intensity = seg["intensity"]
            if intensity > 0.0 and np.isfinite(intensity):
                log_i = float(np.log10(intensity))
                idx = int(np.searchsorted(edges, log_i, side="right") - 1)
            else:
                idx = 0
            idx = max(0, min(bins - 1, idx))
            bucket = grouped_bins[idx]
            bucket["x"].extend([seg["x0"], seg["x1"], None])
            bucket["y"].extend([seg["y0"], seg["y1"], None])
            bucket["z"].extend([seg["z0"], seg["z1"], None])

        for idx, bucket in enumerate(grouped_bins):
            if not bucket["x"]:
                continue
            trace_name = f"log10(I) bin {idx + 1}"
            fig.add_scatter3d(
                x=bucket["x"],
                y=bucket["y"],
                z=bucket["z"],
                mode="lines",
                name=trace_name,
                line={"width": 3, "color": colors[idx]},
                hoverinfo="skip",
                showlegend=False,
            )

    if result.detector_hits:
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        labels: List[str] = []
        for block in result.detector_hits:
            surface = str(block["surface"])
            if detector_hit_exclude_prefixes and any(surface.startswith(prefix) for prefix in detector_hit_exclude_prefixes):
                continue
            points = to_numpy(block["position"])
            for point in points:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
                zs.append(float(point[2]))
                labels.append(surface)
        if xs:
            fig.add_scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker={"size": 2, "color": "#444444"},
                name="Detector hits",
                text=labels,
                hovertemplate="surface=%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
            )

    if sampled_segments:
        fig.add_scatter3d(
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            mode="markers",
            marker={
                "size": 0.1,
                "color": [log_min, log_max],
                "colorscale": _RAY_INTENSITY_COLORSCALE,
                "cmin": log_min,
                "cmax": log_max,
                "showscale": True,
                "colorbar": {"title": "log10(I [W/m^2])"},
            },
            hoverinfo="skip",
            showlegend=False,
            opacity=0.0,
        )

    for overlay in overlays or []:
        trace_type = str(overlay.get("type", "scatter3d")).lower()
        payload = {key: value for key, value in overlay.items() if key != "type"}
        if trace_type == "mesh3d":
            fig.add_trace(go.Mesh3d(**payload))
        else:
            fig.add_scatter3d(**payload)

    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 0.0, "y": 0.0, "z": 2.5},
                "up": {"x": 0.0, "y": 1.0, "z": 0.0},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"itemsizing": "constant"},
        showlegend=False,
    )
    fig.write_html(path, include_plotlyjs=True)
