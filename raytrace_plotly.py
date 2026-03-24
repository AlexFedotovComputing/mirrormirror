from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from plotly.colors import sample_colorscale

from comsol_like_raytrace import to_numpy


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


def write_plotly_trajectories(
    path: Path,
    result: Any,
    *,
    title: str,
    overlays: Optional[List[Dict[str, Any]]] = None,
    max_segments_per_block: int = 400,
    intensity_bins: int = 16,
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
        intensities = to_numpy(block["intensity"])
        n = len(x0)
        stride = max(1, (n + max_segments_per_block - 1) // max_segments_per_block)
        for i in range(0, n, stride):
            sampled_segments.append(
                {
                    "x0": float(x0[i]),
                    "y0": float(y0[i]),
                    "z0": float(z0[i]),
                    "x1": float(x1[i]),
                    "y1": float(y1[i]),
                    "z1": float(z1[i]),
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
        colors = sample_colorscale("Viridis", [i / max(1, bins - 1) for i in range(bins)])
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
            marker={"size": 4, "color": "#d62728"},
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
                "colorscale": "Viridis",
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
        fig.add_scatter3d(**overlay)

    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"itemsizing": "constant"},
        showlegend=False,
    )
    fig.write_html(path, include_plotlyjs=True)
