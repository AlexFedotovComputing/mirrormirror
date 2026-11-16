"""Orthographic end views aligned to mechanical landmarks in reference drawings.

This projects recorded optical paths, not photon intersections with the ROC
and not a simulation of electron drift or charge collection.
"""
import numpy as np
from .directions import SIDE_DESIGNATIONS

TDR_URL = 'https://mpd.jinr.ru/wp-content/uploads/2023/11/TpcTdr-v07.pdf'


def roc_geometry(sign):
    """Nominal TDR rev.07 chamber frames, not surveyed installation geometry.

    Chapter 4 p.17: bases 213/642 mm, height 800 mm, 12 at 30 degrees,
    3 mm frame gap. Radial placement is inferred from that gap and inner base;
    TDR rounded dimensions do not define a manufacturing/CAD registration.
    """
    half_angle = np.pi/12
    inner = (213/2*np.cos(half_angle)+3/2)/np.sin(half_angle)
    local = np.array([[inner,-213/2], [inner+800,-642/2],
                      [inner+800,642/2], [inner,213/2], [inner,-213/2]])
    outlines = []
    for i in range(12):
        angle = i*np.pi/6
        c, s = np.cos(angle), np.sin(angle)
        xy = local @ np.array([[c,s],[-s,c]])
        xy[:, 0] *= sign
        outlines.append(dict(type='scatter', mode='lines', x=xy[:,0].tolist(), y=xy[:,1].tolist(),
                             name=f'ROC · условный азимут {i*30}°', showlegend=False,
                             line=dict(color='#77717e', width=1.5), fill='toself',
                             fillcolor='rgba(112,95,137,0.035)', hoverinfo='name', meta=dict(layer='roc')))
    return dict(outlines=outlines, source_url=TDR_URL,
                geometry_note='Номинальные контуры рам ROC: 12 трапеций, основания 213/642 мм, высота 800 мм, зазор 3 мм. Шаблон накладывается после ориентации сцены по перископу: верхняя и нижняя камеры расположены вертикально, как на схемах. Это визуальная привязка, не измеренные контуры камер.')


def clip_half_segment(start, end, sign, half_length):
    """Clip a recorded segment to this side of the cathode, without extension."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    z0, z1 = sign*a[2], sign*b[2]
    delta = z1-z0
    if abs(delta) < 1e-14:
        return (a, b) if 0 <= z0 <= half_length else None
    t0, t1 = sorted((-z0/delta, (half_length-z0)/delta))
    lo, hi = max(0., t0), min(1., t1)
    if hi <= lo:
        return None
    return a+lo*(b-a), a+hi*(b-a)


def outer_xy(points, sign, axis):
    points = np.asarray(points, float)
    return (points[..., :2] - np.asarray(axis)[:2]) * [sign, 1]


def mechanical_basis(surfaces, frame, view_sign=None):
    """Outside the receiving endcap, looking inward, with periscope down.

    right x up points toward the camera: +Z for the positive survey half,
    -Z for the negative half. Branch numbering must not override handedness.
    Physical EAST/WEST labels are independent of the survey Z convention.
    """
    if view_sign is None:
        view_sign = frame.report.get('survey_z_signs', {}).get('injection')
    if view_sign not in (-1, 1):
        raise ValueError('Не определена внешняя сторона наблюдения торца.')
    points = {name: frame.points(surfaces[name].center) for name in ('MP1', 'MP2')}
    down = (points['MP1'] - points['MP2'])[:2]
    length = np.linalg.norm(down)
    if length < 1e-10:
        raise ValueError('Не определено поперечное направление трубы MP2 → MP1.')
    up = -down / length
    right = view_sign * np.array([up[1], -up[0]])
    return np.column_stack((right, up))


def drawing_xy(points, axis, basis):
    return (np.asarray(points)[..., :2] - np.asarray(axis)[:2]) @ basis


def orientation_landmarks(surfaces, frame, axis, basis):
    """Actual optical centres for checking orientation, without invented pipes."""
    traces = []
    groups = [('Перископ MP2 → MP1', ['MP2', 'MP1'], '#c05a54'),
              ('Края разрыва кольца', ['ML3', 'MR2'], '#a98319'),
              ('Поворотные зеркала', ['MS1', 'MS2', 'MS3', 'MS4'], '#575463')]
    for title, names, color in groups:
        centers = {name: surfaces.get(name, surfaces.get(f'{name}:front')) for name in names}
        names = [name for name in names if centers[name] is not None]
        if not names:
            continue
        xyz = frame.points([centers[name].center for name in names])
        uv = drawing_xy(xyz, axis, basis)*1000
        traces.append(dict(type='scatter', mode='lines+markers+text' if title.startswith('Перископ') else 'markers+text',
                           x=uv[:, 0].tolist(), y=uv[:, 1].tolist(), text=names, textposition='top center',
                           name=title, line=dict(color=color, width=5), marker=dict(color=color, size=8),
                           meta=dict(layer='orientation'), hoverinfo='skip', showlegend=False))
    return traces


def experimental_points(config, frame, axis, basis):
    """Project primary survey points, including unmatched and single-screen rows.

    Survey coordinates are already registered: do not transform them again.
    Model IDs are filter links only, never replacements for measured ray IDs.
    """
    dataset = getattr(config, 'reconstruction', {})
    records = dataset.get('observations')
    if isinstance(records, dict):
        records = records.get('records', [])
    if records is None:
        records = [m.to_dict() for m in getattr(frame, 'measurements', ())]
    links, groups = {}, {}
    by_id = {r.get('id', r.get('observation_id')): r for r in records}
    for mirror in dataset.get('mirrors', []):
        key = mirror.get('canonical_id', mirror.get('mirror_id'))
        if not key:
            continue
        for identifier in set(mirror.get('observation_ids', []) +
                              ([mirror['survey_observation_id']] if mirror.get('survey_observation_id') else [])):
            links.setdefault(identifier, []).append(key)
            row = by_id.get(identifier, {})
            groups.setdefault((row.get('rod'), row.get('window')), set()).add(int(key.split('_')[1]))
    traces = []
    for row in records:
        if row.get('source_type', 'primary') != 'primary' or row.get('exclusion_reason'):
            continue
        identifier = row.get('id', row.get('observation_id'))
        label = row.get('old_id', str(identifier).split(':')[-1])
        rod, window = row['rod'], row['window']
        bundles = groups.get((rod, window), set())
        bundle = next(iter(bundles)) if len(bundles) == 1 else frame.report.get(
            'survey_window_to_legacy_window', {}).get(str(window))
        xy, custom, labels, symbols, colors = [], [], [], [], []
        for screen, point in enumerate(row.get('screens_m', []), 1):
            if point is None or len(point) != 3 or any(v is None for v in point):
                continue
            if not np.isfinite(point).all():
                continue
            xy.append(drawing_xy(point, axis, basis)*1000)
            custom.append([*point, label.replace('_', ' '), screen])
            labels.append(f"{label.replace('_', ' ')} · э{screen}")
            symbols.append('circle-open' if screen == 1 else 'x')
            colors.append('#126caa' if screen == 1 else '#c44428')
        if xy:
            traces.append(dict(type='scatter', mode='markers',
                x=[float(p[0]) for p in xy], y=[float(p[1]) for p in xy],
                text=labels, textposition='top center', customdata=custom,
                name=f'Измерение {label}', showlegend=False,
                marker=dict(size=10, symbol=symbols, color=colors, line=dict(width=2)),
                meta=dict(layer='experimental_points', rod=rod, bundle=bundle,
                          observation_id=identifier, mirror_ids=links.get(identifier, [])),
                hovertemplate='Измерение %{customdata[3]} · экран %{customdata[4]}<br>'
                    'X=%{customdata[0]:.6f} м<br>Y=%{customdata[1]:.6f} м<br>Z=%{customdata[2]:.6f} м<extra></extra>'))
    return traces


def endcap_projection(config, frame, scene_rays):
    if frame is None:
        return dict(available=False, reason='Для вида ROC нужна привязка к общей системе координат TPC.')
    surfaces = {s.name: s for s in config.scene.build_surfaces()}
    if any(name not in surfaces for name in ('MP1', 'MP2')):
        return dict(available=False, reason='Для вертикальной ориентации нужны MP1 и MP2.')
    sign = frame.report.get('survey_z_signs', {}).get('injection')
    if sign not in (-1, 1):
        # Saved manual datasets can omit the registration report. Determine
        # the actual half from their microassembly positions, not a side label.
        centers = [s.center for s in config.scene.build_surfaces()
                   if s.name.startswith('BUNDLE_') and hasattr(s, 'center')]
        if centers:
            sign = int(np.sign(np.median(frame.points(centers)[:, 2])))
    if sign not in (-1, 1):
        return dict(available=False, reason='Не определено положение стороны ввода относительно катода.')
    try:
        basis = mechanical_basis(surfaces, frame, sign)
    except ValueError as error:
        return dict(available=False, reason=str(error))
    limits = [config.tracer_options[f'bundle_clip_z_{end}_m'] for end in ('min', 'max')]
    extent = frame.points([[0, 0, z] for z in limits])
    axis = extent.mean(axis=0)
    half_length = float(np.max(sign*extent[:, 2]))
    projected = []
    for trace in scene_rays:
        if not trace['meta']['mirror_id']:
            continue
        x, y, customdata = [], [], []
        for i in range(0, len(trace['x']), 3):
            a, b = [[trace[k][j] for k in 'xyz'] for j in (i, i+1)]
            clipped = clip_half_segment(a, b, sign, half_length)
            if clipped is None:
                continue
            uv = drawing_xy(clipped, axis, basis)*1000
            x.extend([float(uv[0, 0]), float(uv[1, 0]), None])
            y.extend([float(uv[0, 1]), float(uv[1, 1]), None])
            # Keep physical survey XYZ after clipping, not screen XY or ROC Z.
            source_rows = trace.get('customdata')
            for j, point in enumerate(clipped):
                row = list(source_rows[i+j]) if source_rows else [None]*11
                row[:3] = point.tolist()
                customdata.append(row)
            customdata.append(None)
        if x:
            projected.append(dict(type='scatter', mode='lines', x=x, y=y,
                                  name=trace['name'], line=trace['line'], meta=trace['meta'],
                                  customdata=customdata, hovertemplate=trace.get('hovertemplate'),
                                  hoverinfo='skip', showlegend=False))
    return dict(available=True, sign=sign, axis_m=axis.tolist(), plane_z_m=sign*half_length,
                horizontal_axis='u', vertical_axis='v', basis_survey=basis.tolist(),
                view_from_side=config.metadata['injection_side'],
                view_from_designation=SIDE_DESIGNATIONS[config.metadata['injection_side']],
                view_from='outside_receiving_endcap',
                camera_position_m=(axis + [0., 0., sign*(half_length+1.)]).tolist(),
                camera_target_m=axis.tolist(), viewing_direction_survey=[0., 0., -sign],
                orientation_note='Вид снаружи принимающего торца внутрь объёма: камера за плоскостью торца со стороны назначения лучей. Труба MP2 → MP1 направлена вниз. Координаты подсказок остаются геодезическими.',
                registration_note=('Кольцо WEST отражено, перископ направлен вниз. Стержни и MC сохранены по прежней юстировке; точное метрическое совпадение внешних коробов с CAD не задано.'
                                   if config.metadata.get('ring_alignment') else
                                   'Общая оптика WEST в исходной модели повторяет XY стороны EAST. При вертикальном перископе короба и разрыв кольца не совпадают со схемой WEST; точная механическая привязка не подтверждена.'
                                   if config.metadata['ring_side'] == 'WEST' else ''),
                landmarks=orientation_landmarks(surfaces, frame, axis, basis),
                experimental_points=experimental_points(config, frame, axis, basis),
                rays=projected, **roc_geometry(1))
