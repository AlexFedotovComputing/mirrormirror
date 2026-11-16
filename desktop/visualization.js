/* Views of the same calculation; no physical model or synthetic hits in the UI. */
window.mirrorVisualization = (() => {
  const el = id => document.getElementById(id);
  let result = null, view = 'scene', camera = null, revision = 0;
  let queue = Promise.resolve();
  const plots = new Set();
  const hoverCleanups = new Map();
  const referenceKey = 'mirrorlab.visualization.reference.v1';
  let references = {};
  try { references = JSON.parse(localStorage.getItem(referenceKey) || '{}'); } catch (_) { /* No stored reference. */ }
  const colorscale = [[0, '#3070dc'], [1, '#e181d7']];
  // Fixed shades, independent of selection: dark B1 through light B4.
  const rocBundleColors = [
    ['#48345f', '#624879', '#806397', '#a084b4'],
    ['#175079', '#226d9b', '#368dc1', '#62a5ce'],
    ['#145448', '#1c7060', '#278677', '#57a797'],
    ['#70421e', '#925a2b', '#b07542', '#c58d5c'],
  ];
  const number = (v, digits=4) => v == null || !Number.isFinite(v) ? '—' : v.toLocaleString('ru-RU', { maximumSignificantDigits: digits });
  const vector = v => v ? v.map(x => number(x, 6)).join('; ') : '—';
  const sides = () => (result?.visualization?.sides || []).filter(s => el('viz-side').value === 'both' || s.side === el('viz-side').value);
  const selected = (value, id) => el(id).value === 'all' || String(value) === el(id).value;
  const accepts = row => selected(row.rod, 'viz-rod') && selected(row.bundle, 'viz-bundle') && selected(row.mirror, 'viz-mirror');
  function node(tag, text, cls) {
    const n = document.createElement(tag);
    if (text != null) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  }
  function table(parent, headers, rows) {
    const t = node('table', null, 'viz-table');
    const head = node('tr'); headers.forEach(h => head.append(node('th', h)));
    t.append(node('thead')); t.firstChild.append(head);
    const body = node('tbody');
    for (const values of rows) {
      const row = node('tr'); values.forEach(v => row.append(node('td', v))); body.append(row);
    }
    t.append(body); parent.append(t);
  }
  function clear(id) {
    const parent = el(id);
    for (const plot of [...plots]) if (parent.contains(plot)) {
      hoverCleanups.get(plot)?.(); hoverCleanups.delete(plot);
      Plotly.purge(plot); plots.delete(plot);
    }
    parent.replaceChildren();
    return parent;
  }
  async function plot(parent, data, layout={}) {
    plots.add(parent);
    await Plotly.newPlot(parent, data, { paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      font: { family: 'Segoe UI, sans-serif', color: '#45464c', size: 12 },
      margin: { l: 62, r: 50, t: 40, b: 52 }, ...layout },
    { responsive: true, displaylogo: false, scrollZoom: true,
      toImageButtonOptions: { format: 'png', filename: 'mirrorlab-visualization', scale: 2 } });
  }
  async function fieldPlot(parent, field, title, axisNames=['u, мм', 'v, мм'], radius=null, maxIntensity=null, options={}) {
    const data = [];
    if (field && (options.heatmap ?? el('viz-smooth').checked)) data.push({ type: 'heatmap', x: field.x_mm, y: field.y_mm,
      z: field.z, colorscale, zmin: 0, ...(maxIntensity ? { zmax: maxIntensity } : {}),
      colorbar: { title: { text: 'I, Вт/м²' }, thickness: 10, tickformat: '.1e' },
      hoverongaps: false, hovertemplate: '%{x:.3f} мм; %{y:.3f} мм<br>I=%{z:.3e} Вт/м²<extra></extra>' });
    if (field && (options.points ?? el('viz-points').checked)) data.push({ type: 'scatter', mode: 'markers',
      x: field.points_mm.map(p => p[0]), y: field.points_mm.map(p => p[1]),
      marker: { size: options.source ? 5 : 3, color: field.intensity, colorscale, cmin: 0,
        ...(options.source ? {line:{color:'#303136',width:.7}} : {}),
        ...(maxIntensity ? { cmax: maxIntensity } : {}), opacity: .75 },
      customdata: field.intensity, hovertemplate: '%{x:.3f} мм; %{y:.3f} мм<br>I=%{customdata:.3e} Вт/м²<extra></extra>' });
    const shapes = radius ? [{ type: 'circle', x0: -radius, x1: radius, y0: -radius, y1: radius,
      line: { color: '#777', width: 1 } }] : [];
    await plot(parent, data, { title: { text: title, font: { size: 14 } }, showlegend: false,
      xaxis: { title: { text: axisNames[0] }, ...(radius ? { range: [-radius, radius] } : {}) },
      yaxis: { title: { text: axisNames[1] }, scaleanchor: 'x', scaleratio: 1,
        ...(radius ? { range: [-radius, radius] } : {}) }, shapes,
      annotations: !field?.point_count ? [{ text: 'Нет попаданий', x: .5, y: .5,
        xref: 'paper', yref: 'paper', showarrow: false }] : [] });
  }
  function refFor(side) {
    const ref = references[side.side];
    // Coordinates must have the same registration; never subtract unrelated frames.
    if (!ref) return null;
    let saved;
    try { saved = JSON.parse(ref.frame); } catch (_) { return null; }
    if (!saved || !side.reference_frame) return !saved && !side.reference_frame ? ref : null;
    const a = [...saved.rotation.flat(), ...saved.translation_m];
    const b = [...side.reference_frame.rotation.flat(), ...side.reference_frame.translation_m];
    return a.length === b.length && a.every((v,i) => Math.abs(v-b[i]) < 1e-10) ? ref : null;
  }
  function shift(side, key, center) {
    const reference = refFor(side)?.centers?.[key];
    return center && reference ? vector(center.map((v,i) => (v-reference[i])*1000)) : '—';
  }
  function updateReferenceNote() {
    el('viz-reference').disabled = result?.metadata?.static_configuration !== 'ideal';
    el('viz-reference-note').textContent = sides().map(side => {
      const ref = refFor(side);
      return `${side.side}: ${ref ? `опорный идеальный расчёт от ${ref.date}, ${ref.ray_count} лучей` : 'идеальная юстировка ещё не сохранена для этой системы координат'}`;
    }).join(' · ');
  }
  async function scene() {
    const data = [], chosen = sides();
    for (const side of chosen) {
      for (const trace of result.figure.data) {
        if (trace.meta?.side !== side.side) continue;
        const match = /BUNDLE_([1-4])_([1-4]) Mirror ([1-7])/.exec(trace.name || '');
        const key = trace.meta?.mirror_id || (match ? match.slice(1).join('_') : null);
        if (trace.meta?.layer === 'optics') {
          if (key) {
            const [rod,bundle,mirror] = key.split('_');
            if (!accepts({rod,bundle,mirror})) continue;
          } else if (!el('viz-optics').checked) continue;
          data.push(trace);
        } else if (trace.meta?.layer === 'sources' && el('viz-optics').checked) data.push(trace);
      }
      for (const trace of side.scene_rays) {
        const key = trace.meta.mirror_id;
        if (key) {
          const [rod,bundle,mirror] = key.split('_');
          if (!accepts({rod,bundle,mirror})) continue;
        } else if (!el('viz-common-rays').checked) continue;
        data.push({ ...trace, name: `${side.side} · ${trace.name}`, meta: { ...trace.meta, side: side.side } });
      }
    }
    const rayTraces = data.filter(t => t.meta?.layer === 'traced_rays');
    const total = rayTraces.reduce((n,t) => n+t.meta.segment_count, 0);
    const displayed = rayTraces.reduce((n,t) => n+t.meta.displayed_segments, 0);
    el('viz-scene-note').textContent = `Показано ${number(displayed,8)} из ${number(total,8)} отрезков выбранных групп. Цвет отражённых лучей обозначает стержень. ` +
      (chosen.every(s => s.coordinate_frame === 'survey') ? 'Геодезические координаты, м.' : 'Расчётные координаты, м.');
    const parent = el('viz-scene');
    await plot(parent, data, { margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false,
      scene: { aspectmode: 'data', camera: camera || { eye: { x:1.5,y:1.5,z:1.1 } },
        xaxis: { title: {text:'X, м'} }, yaxis: { title: {text:'Y, м'} }, zaxis: { title: {text:'Z, м'} } } });
    parent.on('plotly_relayout', event => { if (event['scene.camera']) camera = event['scene.camera']; });
  }
  const metricHeaders = ['Экран / зеркало', 'N', 'Мощность, Вт', 'Ī, Вт/м²', 'Диаметр, мм', 'Эллиптичность', 'Центр X; Y; Z, м', 'Сдвиг X; Y; Z, мм'];
  function metrics(side, key, label, m) {
    return [label, m.ray_count, number(m.power_w), number(m.mean_irradiance_w_m2), number(m.diameter_m == null ? null : m.diameter_m*1000),
      number(m.ellipticity), vector(m.center_m), shift(side, key, m.center_m)];
  }
  async function endcaps() {
    const parent = clear('viz-endcaps');
    const reports = sides();
    el('viz-roc-note').textContent = reports.find(s => s.endcap?.available)?.endcap.geometry_note || '';
    for (const side of reports) {
      const card = node('section', null, 'card viz-card'); parent.append(card);
      const e = side.endcap;
      card.append(node('h3', `${side.side} · вид снаружи со стороны ${e?.view_from_side || side.side}${e?.view_from_designation ? ` (${e.view_from_designation})` : ''}`));
      if (!e?.available) { card.append(node('p', e?.reason || 'Проекция недоступна.')); continue; }
      card.append(node('p', `${e.orientation_note} Плоскость модели Z=${number(e.plane_z_m)} м.`, 'field-note'));
      if (e.registration_note) card.append(node('p', e.registration_note, 'scene-stale'));
      const data = el('viz-roc-outlines').checked ? e.outlines : [];
      const rays = e.rays.filter(t => {
        const [rod,bundle,mirror] = t.meta.mirror_id.split('_');
        return accepts({rod,bundle,mirror});
      }).map(t => {
        const [rod, bundle] = t.meta.mirror_id.split('_').map(Number);
        return {...t, line: {...t.line, color: rocBundleColors[rod-1]?.[bundle-1] || t.line.color}};
      });
      const legend = node('div', null, 'viz-bundle-legend');
      legend.setAttribute('aria-label', 'Оттенки бандлов');
      for (let rod = 1; rod <= 4; rod++) {
        if (!selected(rod, 'viz-rod')) continue;
        const row = node('div', null, 'viz-bundle-legend-row');
        row.append(node('span', `Стержень ${rod}`));
        for (let bundle = 1; bundle <= 4; bundle++) {
          if (!selected(bundle, 'viz-bundle')) continue;
          const item = node('span', null, 'viz-bundle-legend-item');
          const swatch = node('i');
          swatch.style.backgroundColor = rocBundleColors[rod-1][bundle-1];
          item.append(swatch, node('span', `B${rod}-${bundle}`)); row.append(item);
        }
        legend.append(row);
      }
      card.append(legend);
      const p = node('div', null, 'viz-endcap-plot'); card.append(p);
      const measurements = el('viz-roc-measurements').checked ? (e.experimental_points || []).filter(t =>
        selected(t.meta.rod, 'viz-rod') && selected(t.meta.bundle, 'viz-bundle') &&
        (el('viz-mirror').value === 'all' || !t.meta.mirror_ids.length ||
          t.meta.mirror_ids.some(id => id.split('_')[2] === el('viz-mirror').value))) : [];
      await plot(p, [...data, ...rays, ...(el('viz-roc-landmarks').checked ? e.landmarks || [] : []),
        ...measurements.map(t => ({...t, mode: el('viz-roc-point-labels').checked ? 'markers+text' : 'markers'}))], {
        showlegend:false, hovermode:'closest', margin:{l:65,r:20,t:20,b:55},
        xaxis:{title:{text:'u, мм от оси TPC'},range:[-1450,1450],constrain:'domain'},
        yaxis:{title:{text:'v, мм · перископ вниз'},range:[-1450,1450],scaleanchor:'x',scaleratio:1,constrain:'domain'},
      });
      if (el('viz-roc-hover').checked) hoverCleanups.set(p, window.mirrorRayHover.attach(p));
      card.append(node('p', `Экспериментальные точки: ${measurements.reduce((n,t) => n+t.x.length, 0)}. Синий круг — экран 1; красный крест — экран 2. Номер в подсказке относится к измерению.`, 'field-note'));
      card.append(node('p', `${rays.reduce((n,t)=>n+t.x.length/3,0)} отрезков в проекции. Цвет обозначает стержень, оттенок — бандл: от тёмного B1 к светлому B4. Для отображения используется та же выборка лучей, что в 3D-сцене.`, 'field-note'));
    }
  }
  async function cameras() {
    const parent = clear('viz-cameras'), summary = clear('viz-camera-table'), rows=[];
    updateReferenceNote();
    for (const side of sides()) {
      const group = node('section', null, 'viz-camera-side'); group.append(node('h3', side.side));
      const grid = node('div', null, 'viz-camera-grid'); group.append(grid); parent.append(group);
      const peak = Math.max(...side.cameras.map(c => c.metrics.peak_intensity_w_m2 || 0));
      for (const c of side.cameras) {
        const p = node('div', null, 'viz-plot'); grid.append(p);
        await fieldPlot(p, c.field, `${c.name} · e=${number(c.metrics.ellipticity)} · d=${number(c.metrics.diameter_m == null ? null : c.metrics.diameter_m*1000)} мм`, undefined, c.radius_mm, peak);
        rows.push(metrics(side, c.name, `${side.side} · ${c.name}`, c.metrics));
      }
    }
    table(summary, metricHeaders, rows);
  }
  async function gas() {
    const parent = clear('viz-unwraps'), summary = clear('viz-gas-table');
    const rows=[];
    for (const side of sides()) {
      for (const row of side.gas_rows.filter(accepts)) rows.push(metrics(side, row.mirror_id, `${side.side} · ${row.mirror_id}`, row));
      for (const c of side.cylinders.filter(c => selected(c.rod,'viz-rod') && selected(c.bundle,'viz-bundle'))) {
        const card = node('section', null, 'card viz-card');
        card.append(node('h3', `${side.side} · стержень ${c.rod}, бандл ${c.bundle}`));
        const hits = c.rows.filter(accepts).reduce((n,r) => n+r.ray_count, 0);
        card.append(node('p', `${hits} попаданий выбранных зеркал${c.unassigned_hits ? ` · ${c.unassigned_hits} попаданий других или неустановленных зеркал` : ''}`, 'field-note'));
        const p = node('div', null, 'viz-plot'); card.append(p); parent.append(card);
        const key = `${c.rod}_${c.bundle}_${el('viz-mirror').value}`;
        const field = el('viz-mirror').value === 'all' ? c.field : c.mirror_fields[key];
        await fieldPlot(p, field, c.available ? c.name : `${c.name} отсутствует`, ['s, мм', 'h, мм']);
      }
    }
    table(summary, metricHeaders, rows);
  }
  async function source() {
    const summary = clear('viz-source-summary'), parent = clear('viz-source'), angles = clear('viz-angles');
    const rows=[], angleRows=[];
    for (const side of sides()) {
      const s = side.source;
      rows.push([side.side, s.ray_count, number(s.waist_diameter_m*1000), number(s.sampled_diameter_m*1000),
        number(s.peak_intensity_w_m2), number(s.power_w), number(s.wavelength_m*1e9)]);
      const p = node('div', null, 'viz-plot'); parent.append(p);
      await fieldPlot(p, s.field, `${side.side} · гауссово сечение в перетяжке`, ['u, мм','v, мм'], s.sampled_diameter_m*500, null,
        {source:true,heatmap:el('viz-source-map').checked,points:true});
      for (const a of side.mirror_angles) angleRows.push([side.side, a.name, ...a.normal_angles_deg.map(v => number(v,6))]);
    }
    table(summary, ['Сторона ввода','Лучей','Диаметр 2w₀, мм','Диаметр выборки, мм','Пиковая I₀, Вт/м²','Мощность выборки, Вт','λ, нм'], rows);
    table(angles, ['Сторона ввода','Зеркало','∠(n,X), °','∠(n,Y), °','∠(n,Z), °'], angleRows);
  }
  async function render(requested) {
    if (!result || el('distributions-panel').hidden) return;
    for (const panel of document.querySelectorAll('[data-viz-panel]')) panel.hidden = panel.dataset.vizPanel !== view;
    for (const button of document.querySelectorAll('[data-viz-view]')) button.setAttribute('aria-pressed', String(button.dataset.vizView === view));
    await ({scene, endcaps, cameras, gas, source})[view]();
    el('viz-content').dataset.rendered = String(requested);
  }
  function schedule() {
    const requested = ++revision;
    el('viz-content').dataset.pending = String(requested);
    queue = queue.then(() => requested === revision ? render(requested) : undefined)
      .catch(error => { el('notice').textContent = error.message; el('notice').hidden = false; });
    return queue;
  }
  document.addEventListener('DOMContentLoaded', () => {
    for (const id of ['viz-side','viz-rod','viz-bundle','viz-mirror','viz-optics','viz-common-rays','viz-smooth','viz-points','viz-roc-outlines','viz-roc-landmarks','viz-roc-hover','viz-roc-measurements','viz-roc-point-labels','viz-source-map']) el(id).addEventListener('change', schedule);
    for (const button of document.querySelectorAll('[data-viz-view]')) button.addEventListener('click', () => { view=button.dataset.vizView; schedule(); });
    el('viz-reset-camera').addEventListener('click', () => { camera=null; schedule(); });
    el('viz-reference').addEventListener('click', () => {
      if (result?.metadata?.static_configuration !== 'ideal') return;
      for (const side of sides()) {
        const centers = {};
        for (const c of side.cameras) centers[c.name] = c.metrics.center_m;
        for (const r of side.gas_rows) centers[r.mirror_id] = r.center_m;
        references[side.side] = { centers, frame: JSON.stringify(side.reference_frame),
          date: new Date().toLocaleString('ru-RU'), ray_count: side.source.ray_count };
      }
      try { localStorage.setItem(referenceKey, JSON.stringify(references)); }
      catch (_) { el('notice').textContent='Опорные центры сохранены только до закрытия окна.'; el('notice').hidden=false; }
      schedule();
    });
  });
  return {
    update(value) {
      result=value;
      const reports=value.visualization?.sides || [];
      el('viz-empty').hidden=reports.length > 0; el('viz-content').hidden=!reports.length;
      el('viz-stale').hidden=true;
      const previous=el('viz-side').value;
      el('viz-side').replaceChildren(...(reports.length > 1 ? [new Option('WEST и EAST','both')] : []),
        ...reports.map(s => new Option(s.side,s.side)));
      if ([...el('viz-side').options].some(o=>o.value===previous)) el('viz-side').value=previous;
      el('viz-context').textContent=`${value.request.kind === 'real' ? 'Реальная' : 'Идеальная'} · ${value.request.ray_count} лучей на лазер · ${value.request.max_interactions} взаимодействий. Результаты последней трассировки.`;
      schedule();
    },
    activate: schedule,
    stale() { if (result) el('viz-stale').hidden=false; },
  };
})();
