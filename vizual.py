from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


_RAY_INTENSITY_COLORSCALE = _make_softened_colorscale(
    "Dense",
    start=0.04,
    end=0.96,
    blend_to_mid=0.04,
    darken_factor=0.76,
    contrast_factor=1.20,
    steps=11,
)


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


def write_detector_screen_views(
    path: Path,
    result: Any,
    *,
    screens: Sequence[Dict[str, Any]],
    title: str,
    grid_size: int = 80,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if len(screens) != 4:
        raise ValueError("Expected exactly 4 screens for a 2x2 detector view.")

    hit_map: Dict[str, Dict[str, np.ndarray]] = {}
    max_intensity = 0.0
    for screen in screens:
        name = str(screen["name"])
        u_parts: List[np.ndarray] = []
        v_parts: List[np.ndarray] = []
        i_parts: List[np.ndarray] = []
        for block in result.detector_hits:
            if str(block["surface"]) != name:
                continue
            u = np.asarray(to_numpy(block["local_u"]), dtype=float).reshape(-1)
            v = np.asarray(to_numpy(block["local_v"]), dtype=float).reshape(-1)
            intensity = np.asarray(to_numpy(block["intensity"]), dtype=float).reshape(-1)
            if u.size == 0:
                continue
            u_parts.append(u)
            v_parts.append(v)
            i_parts.append(intensity)
        if u_parts:
            u_all = np.concatenate(u_parts)
            v_all = np.concatenate(v_parts)
            i_all = np.concatenate(i_parts)
            max_intensity = max(max_intensity, float(np.max(i_all)))
        else:
            u_all = np.zeros((0,), dtype=float)
            v_all = np.zeros((0,), dtype=float)
            i_all = np.zeros((0,), dtype=float)
        hit_map[name] = {"u": u_all, "v": v_all, "intensity": i_all}

    if max_intensity <= 0.0:
        max_intensity = 1.0

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[str(screen.get("label", screen["name"])) for screen in screens],
        horizontal_spacing=0.08,
        vertical_spacing=0.10,
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
        if u.size > 0:
            count_grid, _, _ = np.histogram2d(u, v, bins=(edges, edges))
            sum_grid, _, _ = np.histogram2d(u, v, bins=(edges, edges), weights=intensity)
            avg_grid = np.full(count_grid.shape, np.nan, dtype=float)
            mask = count_grid > 0
            avg_grid[mask] = sum_grid[mask] / count_grid[mask]
        else:
            avg_grid = np.full((grid_size, grid_size), np.nan, dtype=float)

        fig.add_trace(
            go.Heatmap(
                x=centers,
                y=centers,
                z=avg_grid.T,
                coloraxis="coloraxis",
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
                    "size": 4,
                    "color": intensity.tolist(),
                    "coloraxis": "coloraxis",
                    "line": {"color": "#111111", "width": 0.4},
                    "opacity": 0.9,
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
            font={"size": 13, "color": "#111111"},
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor="rgba(0,0,0,0.18)",
            borderwidth=1,
        )

    fig.update_layout(
        title=title,
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
        margin={"l": 50, "r": 120, "t": 60, "b": 40},
        showlegend=False,
    )
    fig.write_html(path, include_plotlyjs=True)


def write_single_detector_screen_view(
    path: Path,
    result: Any,
    *,
    screen: Dict[str, Any],
    title: str,
    grid_size: int = 80,
) -> None:
    import plotly.graph_objects as go

    name = str(screen["name"])
    label = str(screen.get("label", name))
    half_width = 0.5 * float(screen["width"])
    half_height = 0.5 * float(screen["height"])

    u_parts: List[np.ndarray] = []
    v_parts: List[np.ndarray] = []
    i_parts: List[np.ndarray] = []
    for block in result.detector_hits:
        if str(block["surface"]) != name:
            continue
        u = np.asarray(to_numpy(block["local_u"]), dtype=float).reshape(-1)
        v = np.asarray(to_numpy(block["local_v"]), dtype=float).reshape(-1)
        intensity = np.asarray(to_numpy(block["intensity"]), dtype=float).reshape(-1)
        if u.size == 0:
            continue
        u_parts.append(u)
        v_parts.append(v)
        i_parts.append(intensity)

    if u_parts:
        u = np.concatenate(u_parts)
        v = np.concatenate(v_parts)
        intensity = np.concatenate(i_parts)
        max_intensity = max(1.0, float(np.max(intensity)))
    else:
        u = np.zeros((0,), dtype=float)
        v = np.zeros((0,), dtype=float)
        intensity = np.zeros((0,), dtype=float)
        max_intensity = 1.0

    ellipticity = _ellipticity_ratio(u, v, intensity)

    u_edges = np.linspace(-half_width, half_width, int(grid_size) + 1, dtype=float)
    v_edges = np.linspace(-half_height, half_height, int(grid_size) + 1, dtype=float)
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

    if u.size > 0:
        count_grid, _, _ = np.histogram2d(u, v, bins=(u_edges, v_edges))
        sum_grid, _, _ = np.histogram2d(u, v, bins=(u_edges, v_edges), weights=intensity)
        avg_grid = np.full(count_grid.shape, np.nan, dtype=float)
        mask = count_grid > 0
        avg_grid[mask] = sum_grid[mask] / count_grid[mask]
    else:
        avg_grid = np.full((grid_size, grid_size), np.nan, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=u_centers,
            y=v_centers,
            z=avg_grid.T,
            colorscale=_RAY_INTENSITY_COLORSCALE,
            zmin=0.0,
            zmax=max_intensity,
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
                "size": 5,
                "color": intensity.tolist(),
                "colorscale": _RAY_INTENSITY_COLORSCALE,
                "cmin": 0.0,
                "cmax": max_intensity,
                "line": {"color": "#111111", "width": 0.4},
                "opacity": 0.92,
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
        font={"size": 13, "color": "#111111"},
        bgcolor="rgba(255,255,255,0.80)",
        bordercolor="rgba(0,0,0,0.18)",
        borderwidth=1,
    )
    fig.write_html(path, include_plotlyjs=True)


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
