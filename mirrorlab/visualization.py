"""Bounded display data and full-sample diagnostics for the current tracer.

No legacy imports or retracing. Bundle identity follows ray ancestry, not spot
clustering. Extent diameter and P/ellipse-area retain the legacy definitions.
"""
from collections import defaultdict
import re

import numpy as np
from scipy.ndimage import gaussian_filter

from .engine import CylinderSurface, to_numpy
from .engine.backend import orthonormal_basis
from .endcap import endcap_projection

MIRROR = re.compile(r'BUNDLE_([1-4])_([1-4]) Mirror ([1-7])$')


def mirror_id(name):
    match = MIRROR.fullmatch(str(name))
    return '_'.join(match.groups()) if match else None


def spot_metrics(uv, positions, intensity, power):
    uv = np.asarray(uv, float).reshape(-1, 2)
    n = len(uv)
    result = dict(ray_count=n, center_m=None, center_uv_m=None,
                  diameter_m=None, ellipticity=None, mean_irradiance_w_m2=None,
                  power_w=float(np.sum(power)), peak_intensity_w_m2=None)
    if not n:
        return result
    weights = np.maximum(intensity, 0)
    if not np.sum(weights):
        weights = np.ones(n)
    center = np.average(uv, weights=weights, axis=0)
    result.update(center_m=np.average(positions, weights=weights, axis=0).tolist(),
                  center_uv_m=center.tolist(), peak_intensity_w_m2=float(np.max(intensity)))
    if n < 2:
        return result
    relative = uv - center
    values, axes = np.linalg.eigh((relative.T * weights) @ relative / np.sum(weights))
    diameters = np.ptp(relative @ axes, axis=0)
    result['diameter_m'] = float(np.mean(diameters))
    if values[-1] > 1e-30:
        result['ellipticity'] = float(np.sqrt(max(0., values[0]) / values[-1]))
    area = float(np.pi * diameters.prod() / 4)
    if area > 1e-24:
        result['mean_irradiance_w_m2'] = result['power_w'] / area
    return result


def field_data(uv, intensity, *, bounds=None, size=48):
    """Smoothed mean per-ray irradiance, with empty regions left transparent."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    if bounds is None:
        if len(uv):
            lo, hi = uv.min(axis=0), uv.max(axis=0)
            pad = np.maximum((hi - lo) * .08, .0001)
            bounds = np.array([lo - pad, hi + pad])
        else:
            bounds = np.array([[-.001, -.001], [.001, .001]])
    bounds = np.asarray(bounds)
    edges = [np.linspace(bounds[0, k], bounds[1, k], size + 1) for k in range(2)]
    counts, _, _ = np.histogram2d(*uv.T, bins=edges)
    weighted, _, _ = np.histogram2d(*uv.T, bins=edges, weights=intensity)
    support = gaussian_filter(counts, .85, mode='constant', truncate=2)
    total = gaussian_filter(weighted, .85, mode='constant', truncate=2)
    field = np.divide(total, support, out=np.zeros_like(total), where=support > 0)
    grid = [[float(v) if s > 1e-8 else None for v, s in zip(row, mask)]
            for row, mask in zip(field.T, support.T)]
    ids = np.linspace(0, len(uv)-1, min(len(uv), 2500), dtype=int)
    return dict(x_mm=(.5 * (edges[0][:-1] + edges[0][1:]) * 1000).tolist(),
                y_mm=(.5 * (edges[1][:-1] + edges[1][1:]) * 1000).tolist(), z=grid,
                points_mm=(uv[ids] * 1000).tolist(), intensity=np.asarray(intensity)[ids].tolist(),
                point_count=len(uv), displayed_points=len(ids))


def unwrap(positions, surface):
    u, v, axis = orthonormal_basis(surface.axis, xp=np)
    relative = np.asarray(positions).reshape(-1, 3) - np.asarray(surface.center)
    angle = np.arctan2(relative @ v, relative @ u)
    # Put the branch cut in the largest unoccupied angular gap. This avoids
    # splitting a compact spot at +/-pi and works for tilted cylinders too.
    if len(angle):
        ordered = np.sort(np.mod(angle, 2*np.pi))
        gap = np.argmax(np.diff(np.r_[ordered, ordered[0] + 2*np.pi]))
        start = ordered[(gap + 1) % len(ordered)]
        angle = np.mod(angle - start, 2*np.pi) + start
        angle -= 2*np.pi * np.floor(np.mean(angle) / (2*np.pi))
    return np.column_stack((angle * surface.radius, relative @ axis))


def ray_origins(result):
    """Labels for segments and hits; a hit on the cut labels only its children."""
    origins, ends = {}, {}
    for block in result.segments:
        for rid, parent, name, reflected in zip(block['ray_id'], block['parent_id'],
                                                block['surface'], block['bundle_reflected']):
            rid, parent = int(rid), int(parent)
            origin = (ends.get(parent) or origins.get(parent)) if reflected else None
            origins[rid] = origin
            cut = mirror_id(name)
            if cut:
                ends[rid] = cut
    return origins


def grouped_segments(result, origins, frame):
    """Keep identity and physical XYZ alongside the bounded display sampling.

    The customdata coordinates are in the display's physical coordinate frame,
    before any endcap rotation or projection. A projected view can retain Z and
    ray identity by copying these rows, updating XYZ if it clips the segment.
    """
    groups = defaultdict(list)
    counts = defaultdict(int)
    # Count first, then sample independently within every mirror. Rare bundles
    # remain visible even when the common optical path dominates the data.
    for block in result.segments:
        for rid in block['ray_id']:
            counts[origins.get(int(rid)) or 'common'] += 1
    offsets = defaultdict(int)
    for block in result.segments:
        starts = np.column_stack([block[a+'0'] for a in 'xyz'])
        ends = np.column_stack([block[a+'1'] for a in 'xyz'])
        for rid, start, end, intensity in zip(block['ray_id'], starts, ends, block['intensity']):
            key = origins.get(int(rid)) or 'common'
            budget = 2400 if key == 'common' else 100
            stride = max(1, int(np.ceil(counts[key] / budget)))
            if offsets[key] % stride == 0:
                groups[key].append((start, end, float(intensity), int(rid)))
            offsets[key] += 1
    data = []
    colors = ['#705f89', '#368dc1', '#278677', '#b07542']
    for key, rows in sorted(groups.items()):
        points = np.array([[row[0], row[1]] for row in rows]).reshape(-1, 3)
        if frame is not None:
            points = frame.points(points)
        coords = {}
        for i, axis in enumerate('xyz'):
            coords[axis] = [v for a, b in points[:, i].reshape(-1, 2) for v in (float(a), float(b), None)]
        rod, bundle, mirror = (None, None, None) if key == 'common' else map(int, key.split('_'))
        customdata = []
        for row, (start, end) in zip(rows, points.reshape(-1, 2, 3)):
            direction = end - start
            length = np.linalg.norm(direction)
            if length:
                direction /= length
            identity = [row[3], None if key == 'common' else key, rod, bundle, mirror]
            customdata.extend([start.tolist() + identity + direction.tolist(),
                               end.tolist() + identity + direction.tolist(), None])
        meta = dict(layer='traced_rays', mirror_id=None if key == 'common' else key,
                    rod=rod, bundle=bundle, mirror=mirror,
                    coordinate_frame='survey' if frame is not None else 'legacy',
                    customdata_fields=['x_m', 'y_m', 'z_m', 'ray_id', 'mirror_id',
                                       'rod', 'bundle', 'mirror', 'dx', 'dy', 'dz'],
                    segment_count=counts[key], displayed_segments=len(rows))
        identity_label = ('Общая оптика' if key == 'common' else
                          f'Луч {key} · стержень {rod}, бандл {bundle}, зеркало {mirror}')
        hovertemplate = (identity_label + '<br>ID трассировки: %{customdata[3]}'
                         '<br>X=%{customdata[0]:.7f} м'
                         '<br>Y=%{customdata[1]:.7f} м'
                         '<br>Z=%{customdata[2]:.7f} м'
                         '<br>Направление: (%{customdata[8]:.6f}; '
                         '%{customdata[9]:.6f}; %{customdata[10]:.6f})<extra></extra>')
        data.append(dict(type='scatter3d', mode='lines', **coords, meta=meta,
                         name='Общая оптика' if key == 'common' else f'Бандл {key[:3]} · зеркало {key[-1]}',
                         customdata=customdata, hovertemplate=hovertemplate,
                         legendgroup=key[:3], showlegend=False,
                         line=dict(width=2, color='#9995a0' if key == 'common' else colors[int(key[0])-1])))
    return data


def build_visualization(config, rays, result, frame):
    surfaces = {s.name: s for s in config.scene.build_surfaces()}
    origins = ray_origins(result)
    hit_map = defaultdict(list)
    for hit in result.detector_hits:
        hit_map[str(hit['surface'])].append(hit)

    def collect(name):
        blocks = hit_map[name]
        if not blocks:
            return dict(position=np.empty((0, 3)), local_u=np.empty(0), local_v=np.empty(0),
                        intensity=np.empty(0), power=np.empty(0), ray_id=np.empty(0, int))
        return {key: np.concatenate([b[key] for b in blocks]) for key in
                ('position', 'local_u', 'local_v', 'intensity', 'power', 'ray_id')}

    def display_points(p):
        return frame.points(p) if frame is not None else p

    cameras = []
    for i in range(1, 5):
        name = f'Screen {i}'
        hit = collect(name)
        uv = np.column_stack((hit['local_u'], hit['local_v']))
        surface = surfaces.get(name)
        radius = getattr(surface, 'radius', None)
        bounds = [[-radius, -radius], [radius, radius]] if radius else None
        cameras.append(dict(name=name, radius_mm=radius*1000 if radius else None,
            metrics=spot_metrics(uv, display_points(hit['position']), hit['intensity'], hit['power']),
            field=field_data(uv, hit['intensity'], bounds=bounds)))

    cylinders, gas_rows = [], []
    for rod in range(1, 5):
        for bundle in range(1, 5):
            name = f'screen_b_{rod}_{bundle}'
            surface = surfaces.get(name)
            hit = collect(name)
            uv = unwrap(hit['position'], surface) if isinstance(surface, CylinderSurface) else np.empty((0, 2))
            ids = np.array([origins.get(int(rid)) or '' for rid in hit['ray_id']], dtype=str)
            cylinder_rows, mirror_fields = [], {}
            for mirror in range(1, 8):
                key = f'{rod}_{bundle}_{mirror}'
                mask = ids == key
                row = dict(mirror_id=key, rod=rod, bundle=bundle, mirror=mirror, screen=name,
                           **spot_metrics(uv[mask], display_points(hit['position'][mask]),
                                          hit['intensity'][mask], hit['power'][mask]))
                cylinder_rows.append(row)
                gas_rows.append(row)
                if np.any(mask):
                    mirror_fields[key] = field_data(uv[mask], hit['intensity'][mask], size=32)
            cylinders.append(dict(name=name, rod=rod, bundle=bundle, available=surface is not None,
                                  field=field_data(uv, hit['intensity']), rows=cylinder_rows, mirror_fields=mirror_fields,
                                  unassigned_hits=int(np.sum(~np.isin(ids, [r['mirror_id'] for r in cylinder_rows])))))

    source = config.source
    u, v, _ = orthonormal_basis(source.axis, source.polarization_reference, xp=np)
    rel = to_numpy(rays.position) - np.asarray(source.waist_position)
    uv = np.column_stack((rel @ u, rel @ v))
    # Analytic Gaussian displayed together with the actual emitted sampling grid.
    field = field_data(uv, to_numpy(rays.intensity), bounds=[[-source.waist_radius*source.cutoff_ratio]*2,
                                                           [source.waist_radius*source.cutoff_ratio]*2])
    x, y = np.meshgrid(np.array(field['x_mm'])/1000, np.array(field['y_mm'])/1000)
    gaussian = source.peak_intensity * np.exp(-2*(x*x+y*y)/source.waist_radius**2)
    field['z'] = [[float(value) if radius2 <= (source.waist_radius*source.cutoff_ratio)**2 else None
                   for value, radius2 in zip(row, radii)] for row, radii in zip(gaussian, x*x+y*y)]
    angles = []
    for name, surface in surfaces.items():
        if re.fullmatch(r'M[PCS][1-4]', name) and hasattr(surface, 'normal'):
            normal = np.array(surface.normal, float, copy=True)
            normal /= np.linalg.norm(normal)
            angles.append(dict(name=name, normal=normal.tolist(),
                               normal_angles_deg=np.degrees(np.arccos(np.clip(normal, -1, 1))).tolist()))
    scene_rays = grouped_segments(result, origins, frame)
    return dict(schema_version=1, direction=config.direction, side=config.metadata['injection_side'],
                coordinate_frame='survey' if frame else 'legacy', cameras=cameras, cylinders=cylinders,
                gas_rows=gas_rows, scene_rays=scene_rays,
                endcap=endcap_projection(config, frame, scene_rays),
                source=dict(ray_count=rays.n_rays, waist_diameter_m=2*source.waist_radius,
                            sampled_diameter_m=2*source.waist_radius*source.cutoff_ratio,
                            peak_intensity_w_m2=source.peak_intensity,
                            power_w=float(np.sum(to_numpy(rays.power))), wavelength_m=source.wavelength_m,
                            field=field), mirror_angles=angles,
                reference_frame=dict(rotation=frame.rotation.tolist(), translation_m=frame.translation.tolist()) if frame else None)
