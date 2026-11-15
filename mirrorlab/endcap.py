"""Orthographic end views aligned to mechanical landmarks in reference drawings.

This projects recorded optical paths, not photon intersections with the ROC
and not a simulation of electron drift or charge collection.
"""
import numpy as np

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


def mechanical_basis(surfaces, frame):
    """Drawing orientation: periscope down; numbered branches 1 left, 4 right.

    The survey registration may reverse XY handedness. Drawing chirality is
    therefore fixed by named optical landmarks, not the survey Z sign.
    """
    points = {name: frame.points(surfaces[name].center) for name in ('MP1', 'MP2', 'MS1', 'MS4')}
    down = (points['MP1'] - points['MP2'])[:2]
    length = np.linalg.norm(down)
    if length < 1e-10:
        raise ValueError('Не определено поперечное направление трубы MP2 → MP1.')
    up = -down / length
    right = np.array([up[1], -up[0]])
    if np.dot((points['MS4'] - points['MS1'])[:2], right) < 0:
        right = -right
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
        names = [name for name in names if name in surfaces]
        if not names:
            continue
        xyz = frame.points([surfaces[name].center for name in names])
        uv = drawing_xy(xyz, axis, basis)*1000
        traces.append(dict(type='scatter', mode='lines+markers+text' if title.startswith('Перископ') else 'markers+text',
                           x=uv[:, 0].tolist(), y=uv[:, 1].tolist(), text=names, textposition='top center',
                           name=title, line=dict(color=color, width=5), marker=dict(color=color, size=8),
                           meta=dict(layer='orientation'), hoverinfo='skip', showlegend=False))
    return traces


def endcap_projection(config, frame, scene_rays):
    if frame is None:
        return dict(available=False, reason='Для вида ROC нужна привязка к общей системе координат TPC.')
    surfaces = {s.name: s for s in config.scene.build_surfaces()}
    if any(name not in surfaces for name in ('MP1', 'MP2', 'MS1', 'MS4')):
        return dict(available=False, reason='Для ориентации по схеме нужны MP1, MP2, MS1 и MS4.')
    try:
        basis = mechanical_basis(surfaces, frame)
    except ValueError as error:
        return dict(available=False, reason=str(error))
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
                view_from_side=config.metadata['ring_side'],
                orientation_note='Труба MP2 → MP1 направлена вниз; ветви MS1 и MS4 задают левую и правую стороны схемы. Координаты подсказок остаются геодезическими.',
                registration_note=('Общая оптика WEST в исходной модели повторяет XY стороны EAST. При вертикальном перископе короба и разрыв кольца не совпадают со схемой WEST; точная механическая привязка не подтверждена.'
                                   if config.metadata['ring_side'] == 'WEST' else ''),
                landmarks=orientation_landmarks(surfaces, frame, axis, basis),
                rays=projected, **roc_geometry(1))
