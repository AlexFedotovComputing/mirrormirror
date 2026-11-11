"""Plotly figure data; display sampling never changes the optical calculation."""
from __future__ import annotations

import numpy as np

from .engine import CylinderSurface, PlaneSurface, to_numpy


def _line(points, name, color, **options):
    points = np.asarray(points, dtype=float)
    return dict(type="scatter3d", x=points[:, 0].tolist(), y=points[:, 1].tolist(),
                z=points[:, 2].tolist(), mode="lines", name=name,
                line=dict(color=color, width=2), **options)


def _surface_traces(surface):
    color = "#c58ad4" if surface.name.startswith("BUNDLE_") else "#72afbb"
    if surface.optics.detector:
        color = "#e2b767"
    if isinstance(surface, PlaneSurface):
        u, v = to_numpy(surface._u), to_numpy(surface._v)
        if surface.shape == "disk":
            a = np.linspace(0, 2 * np.pi, 25)
            local = surface.radius * np.column_stack((np.cos(a), np.sin(a)))
        elif surface.shape == "triangle":
            local = np.asarray(surface.triangle_vertices)
            local = np.vstack((local, local[0]))
        elif surface.shape == "rectangle":
            local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
            local = local * [surface.width / 2, surface.height / 2]
        else:
            return []  # An infinite physical plane has no finite outline.
        points = to_numpy(surface.center) + local[:, :1] * u + local[:, 1:] * v
        outline = _line(points, surface.name, color, showlegend=False,
                        hovertemplate=surface.name + "<br>%{x:.4f}, %{y:.4f}, %{z:.4f} м<extra></extra>")
        n = len(points) - 1
        mesh = dict(type="mesh3d", name=surface.name, x=points[:, 0].tolist(),
                    y=points[:, 1].tolist(), z=points[:, 2].tolist(),
                    i=[0] * (n - 2), j=list(range(1, n - 1)), k=list(range(2, n)),
                    color=color, opacity=0.35, showlegend=False, hoverinfo="skip")
        return [outline, mesh]
    if isinstance(surface, CylinderSurface):
        axis = to_numpy(surface.axis)
        helper = np.eye(3)[np.argmin(np.abs(axis))]
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        a = np.linspace(0, 2 * np.pi, 65)
        ring = surface.radius * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)
        center = to_numpy(surface.center)
        ends = [center + ring + s * surface.length / 2 * axis for s in (-1, 1)]
        return [_line(p, surface.name, color, showlegend=False, hoverinfo="name", opacity=0.25)
                for p in ends + [np.vstack((ends[0][i], ends[1][i])) for i in (0, 16, 32, 48)]]
    return []


class PlotlyRenderer:
    """Retains legacy log-intensity line grouping, in a reusable figure API."""

    def __init__(self, segment_budget=24000):
        self.segment_budget = segment_budget

    def render(self, configuration, result, frame=None):
        data = []
        for surface in configuration.scene.build_surfaces():
            data.extend(_surface_traces(surface))
        total = sum(len(block["x0"]) for block in result.segments)
        stride = max(1, int(np.ceil(total / self.segment_budget)))
        blocks, offset = [], 0
        for block in result.segments:
            n = len(block["x0"])
            index = np.arange((-offset) % stride, n, stride)
            offset += n
            if len(index):
                blocks.append({key: to_numpy(block[key])[index] for key in
                               ("x0", "y0", "z0", "x1", "y1", "z1", "intensity")})
        displayed = sum(len(b["x0"]) for b in blocks)
        if blocks:
            joined = {key: np.concatenate([b[key] for b in blocks]) for key in blocks[0]}
            log_i = np.log10(np.maximum(joined["intensity"], 1e-30))
            lower, upper = float(log_i.min()), float(log_i.max())
            bins = np.clip(((log_i - lower) / max(upper - lower, 1e-12) * 7).astype(int), 0, 7)
            colors = ["#393b83", "#465db1", "#368dc1", "#26b5b5", "#69cf8c", "#b5dd6e", "#f3da68", "#ff9d58"]
            for i, color in enumerate(colors):
                mask = bins == i
                if not mask.any():
                    continue
                coords = {}
                for axis in "xyz":
                    values = []
                    for start, end in zip(joined[axis + "0"][mask], joined[axis + "1"][mask]):
                        values.extend((float(start), float(end), None))
                    coords[axis] = values
                data.append(dict(type="scatter3d", mode="lines", name="Лучи",
                                 legendgroup="rays", showlegend=i == int(bins.min()),
                                 line=dict(width=2, color=color), hoverinfo="skip", **coords))
            data.append(dict(type="scatter3d", mode="markers", x=[None, None], y=[None, None], z=[None, None],
                             marker=dict(color=[lower, max(lower + 1e-9, upper)],
                                         colorscale=[[i / 7, color] for i, color in enumerate(colors)],
                                         autocolorscale=False, showscale=True,
                                         colorbar=dict(title=dict(text="log₁₀ I, Вт/м²"), thickness=12)),
                             showlegend=False, hoverinfo="skip"))
        for block in result.detector_hits:
            p = to_numpy(block["position"])
            data.append(dict(type="scatter3d", x=p[:, 0].tolist(), y=p[:, 1].tolist(), z=p[:, 2].tolist(),
                             mode="markers", marker=dict(size=2, color="#f4cc78"), name=str(block["surface"]),
                             showlegend=False, hovertemplate=str(block["surface"]) + "<br>%{x:.4f}, %{y:.4f}, %{z:.4f} м<extra></extra>"))
        p = to_numpy(configuration.source.waist_position)
        data.append(dict(type="scatter3d", x=[float(p[0])], y=[float(p[1])], z=[float(p[2])], mode="markers",
                         name="Лазер", marker=dict(size=5, color="#ffad66", symbol="diamond")))
        if frame is not None:
            frame.transform_traces(data)
            points, labels = [], []
            for row in frame.measurements:
                for screen, point in enumerate(row.screens, 1):
                    if point is not None and all(v is not None for v in point):
                        points.append(point)
                        labels.append(f"{row.id} · экран {screen}")
            p = np.asarray(points)
            data.append(dict(type="scatter3d", mode="markers", name="Геодезические точки",
                             x=p[:, 0].tolist(), y=p[:, 1].tolist(), z=p[:, 2].tolist(), text=labels,
                             marker=dict(size=4, color="#ffffff", symbol="circle-open"),
                             visible="legendonly", legendgroup="survey", meta=dict(kind="survey"),
                             hovertemplate="%{text}<br>%{x:.4f}, %{y:.4f}, %{z:.4f} м<extra></extra>"))
        data.append(dict(type="mesh3d", x=[-1.4, 1.4, 1.4, -1.4], y=[-1.4, -1.4, 1.4, 1.4], z=[0] * 4,
                         i=[0, 0], j=[1, 2], k=[2, 3], color="#98a7bc", opacity=0.08,
                         name="Граница EAST / WEST: z = 0", hoverinfo="name", showlegend=True))
        layout = dict(template="plotly_dark", paper_bgcolor="#101922", plot_bgcolor="#101922",
                      font=dict(family="Segoe UI, sans-serif", color="#c5d3df"),
                      margin=dict(l=0, r=0, t=10, b=0), uirevision="mirrorlab-static",
                      scene=dict(aspectmode="data", dragmode="orbit", camera=dict(projection=dict(type="orthographic")), xaxis=dict(title=dict(text="x, м")), yaxis=dict(title=dict(text="y, м")),
                                 zaxis=dict(title=dict(text="z, м · WEST + / EAST −" if frame else "z, м · локальная СК"))),
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(16,25,34,0.7)"))
        return dict(data=data, layout=layout), displayed


def measurement_figure(report):
    """Survey-frame comparison, deliberately separate from the optical scene."""
    traces = []
    points = [[], []]
    names = [[], []]
    lines = {axis: [] for axis in "xyz"}
    for ray in report["rays"]:
        complete = []
        for screen, point in enumerate(ray["screens_m"]):
            if point is None or any(value is None for value in point):
                continue
            points[screen].append(point)
            names[screen].append(ray["id"])
            complete.append(point)
        if len(complete) == 2:
            for i, axis in enumerate("xyz"):
                lines[axis].extend((complete[0][i], complete[1][i], None))
    traces.append(dict(type="scatter3d", mode="lines", name="Прямые по двум экранам",
                       line=dict(color="#6b8ca5", width=2), hoverinfo="skip", **lines))
    for i, (coordinates, labels) in enumerate(zip(points, names)):
        if coordinates:
            p = np.array(coordinates)
            traces.append(dict(type="scatter3d", x=p[:, 0].tolist(), y=p[:, 1].tolist(), z=p[:, 2].tolist(),
                               mode="markers", text=labels, name=f"Экран {i + 1}",
                               marker=dict(size=3, color=("#8ab2ff", "#e2b767")[i]),
                               hovertemplate="Луч %{text}<br>%{x:.4f}, %{y:.4f}, %{z:.4f} м<extra></extra>"))
    refined = report.get("lavsan", {}).get("refined_rays", [])
    if refined:
        p = np.array([ray["lavsan_point_m"] for ray in refined])
        traces.append(dict(type="scatter3d", x=p[:, 0].tolist(), y=p[:, 1].tolist(), z=p[:, 2].tolist(),
                           mode="markers", text=[ray["id"] for ray in refined], name="Лавсан после преобразования",
                           marker=dict(size=4, symbol="diamond", color="#69dac7"),
                           hovertemplate="Луч %{text}<br>%{x:.4f}, %{y:.4f}, %{z:.4f} м<extra></extra>"))
    return dict(data=traces, layout=dict(template="plotly_dark", paper_bgcolor="#101928",
                margin=dict(l=0, r=0, t=15, b=0), uirevision=report["direction"],
                scene=dict(aspectmode="data", dragmode="orbit", camera=dict(projection=dict(type="orthographic")), xaxis=dict(title=dict(text="x, м")), yaxis=dict(title=dict(text="y, м")), zaxis=dict(title=dict(text="z, м"))),
                legend=dict(orientation="h", y=1.08)))
