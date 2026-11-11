/* The renderer only receives structured data; all computation stays in Python. */
const $ = (id) => document.getElementById(id);
const bridge = window.mirrorlab;
const session = {};
let busy = false;
let camera = null;
let defaultCamera = null;
let overview = true;
const volumeRanges = { xaxis: [-1.5, 1.5], yaxis: [-1.5, 1.5], zaxis: [-1.95, 1.95] };
let elapsedTimer;
let capabilities = [];
let measurementReports = [];
let measurementContext = null;
const sideNames = {};
const routes = {};
let directions = [];
const directionLabel = (direction) => direction === 'both' ? 'WEST и EAST' : sideNames[direction];
const labels = {
  ray_count: 'Исходных лучей', initial_rays: 'Исходных лучей', rays: 'Исходных лучей',
  segments: 'Отрезков траекторий', segment_count: 'Отрезков траекторий',
  detector_hits: 'Попаданий на экраны', hits: 'Попаданий на экраны', hit_count: 'Попаданий на экраны',
  elapsed_s: 'Время расчёта, с', duration_s: 'Время расчёта, с', surfaces: 'Поверхностей',
  surface_count: 'Поверхностей', total_power_w: 'Мощность, Вт',
};
const format = (value) => typeof value === 'number'
  ? value.toLocaleString('ru-RU', { maximumFractionDigits: 4 }) : String(value ?? '—');

function notify(message, error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
  $('notice').hidden = !message;
}

function updateControls() {
  const real = $('kind').value === 'real';
  $('calibration-card').hidden = !real;
  $('calibrate').disabled = busy;
  showSideStatus();
}

function screensLabel(status, ideal = false) {
  if (!status) return 'Дополнительные экраны: проверка наличия данных…';
  if (!status.available) return 'Дополнительные экраны: нет данных';
  if (ideal) return 'Дополнительные экраны: доступны для реальной конфигурации; здесь не использованы';
  if (status.status === 'used_partial') return `Дополнительные экраны: использованы частично · ${status.refined_ray_count} лучей. Не приняты стержни: ${status.rejected_rods.join(', ')}.`;
  if (status.status === 'used') return `Дополнительные экраны: использованы · ${status.refined_ray_count} лучей`;
  if (status.status === 'rejected') return 'Дополнительные экраны: обработаны, уточнения не приняты';
  if (status.status === 'not_requested') return 'Дополнительные экраны: не использованы в этом расчёте';
  return 'Дополнительные экраны: доступны · будут использованы автоматически';
}

function showSideStatus() {
  const selected = $('direction').value;
  const kind = $('kind').value;
  const current = measurementContext?.direction === selected && measurementContext?.kind === kind;
  $('side-status').replaceChildren();
  for (const direction of directions) {
    const included = selected === 'both' || selected === direction;
    const report = current ? measurementReports.find(item => item.direction === direction) : null;
    const status = report?.additional_screens || capabilities.find(item => item.direction === direction);
    const card = document.createElement('div');
    card.className = 'side-status-item';
    card.dataset.direction = direction;
    card.dataset.included = String(included);
    const title = document.createElement('strong');
    title.textContent = `${sideNames[direction]} (${included ? 'лазер ' + routes[direction].laser_side : 'не выбрана'})`;
    card.append(title);
    if (included && status?.available) {
      const badge = document.createElement('span');
      badge.className = 'screen-status';
      badge.dataset.status = report ? status.status : 'pending';
      badge.textContent = status.refined_ray_count && kind === 'real'
        ? `Доп. экраны: использованы${status.status === 'used_partial' ? ' частично' : ''}, ${status.refined_ray_count} лучей`
        : kind === 'ideal' ? 'Доп. экраны: для реальной конфигурации' : 'Доп. экраны: доступны';
      badge.title = screensLabel(status, kind === 'ideal');
      card.append(badge);
    }
    $('side-status').append(card);
  }
}

function setBusy(active, label = 'Трассировка лучей') {
  busy = active;
  $('controls').disabled = active;
  $('cancel').hidden = !active;
  $('busy-overlay').hidden = !active;
  $('busy-label').textContent = label;
  clearInterval(elapsedTimer);
  if (active) {
    const start = Date.now();
    $('run-status').textContent = label + '…';
    elapsedTimer = setInterval(() => {
      $('run-status').textContent = `${label} · ${Math.floor((Date.now() - start) / 1000)} с`;
    }, 1000);
  }
  updateControls();
}

function paragraph(parent, text) {
  const p = document.createElement('p');
  p.textContent = typeof text === 'string' ? text : JSON.stringify(text);
  parent.append(p);
}

function metadata(result) {
  const { figure, ...rest } = result;
  return rest;
}

// Apply the same palette to traces and the intensity legend without changing the data.
const plotColors = {
  '#393b83': '#4c416f', '#465db1': '#63517f', '#368dc1': '#79618e',
  '#26b5b5': '#906f99', '#69cf8c': '#a97e9d', '#b5dd6e': '#bd8e9c',
  '#f3da68': '#cda193', '#ff9d58': '#d6b38f',
  '#c58ad4': '#9b80ad', '#72afbb': '#8c94a9', '#e2b767': '#bd9b78',
  '#f4cc78': '#a87c55', '#ffad66': '#b97a76', '#ffffff': '#705987',
  '#98a7bc': '#b3a6b5', '#6b8ca5': '#8c829e', '#8ab2ff': '#806e9e',
  '#69dac7': '#a8798c',
};
const themedPlotColor = (color) => typeof color === 'string' ? (plotColors[color.toLowerCase()] || color) : color;
function themedPlotData(data) {
  return data.map((trace) => ({
    ...trace,
    ...(trace.color ? { color: themedPlotColor(trace.color) } : {}),
    ...(trace.line ? { line: { ...trace.line, color: themedPlotColor(trace.line.color) } } : {}),
    ...(trace.marker ? { marker: { ...trace.marker,
      color: themedPlotColor(trace.marker.color),
      ...(Array.isArray(trace.marker.colorscale) ? {
        colorscale: trace.marker.colorscale.map(([value, color]) => [value, themedPlotColor(color)]),
      } : {}),
      ...(trace.marker.colorbar ? { colorbar: { ...trace.marker.colorbar,
        outlinewidth: 0, tickfont: { size: 12, color: '#55565b' },
        title: { ...trace.marker.colorbar.title, font: { size: 12, color: '#55565b' } },
      } } : {}),
    } } : {}),
  }));
}
function lightSceneAxis(axis = {}) {
  return { ...axis, gridcolor: '#e7e5e8', zerolinecolor: '#c8c4cb',
    backgroundcolor: '#ffffff', tickfont: { ...axis.tickfont, color: '#696965', size: 12 },
    title: { ...axis.title, font: { ...axis.title?.font, color: '#55565b', size: 12 } } };
}

async function equalAxisScale(plot) {
  // Use displayed ranges: data extents can differ from the cropped volume view.
  const spans = ['x', 'y', 'z'].map(axis => {
    const range = plot._fullLayout.scene[axis + 'axis'].range;
    return Math.abs(range[1] - range[0]);
  });
  if (!spans.every(value => Number.isFinite(value) && value > 0)) return;
  const largest = Math.max(...spans);
  const changes = { 'scene.aspectmode': 'manual',
    'scene.aspectratio': Object.fromEntries(['x', 'y', 'z'].map((axis, i) => [axis, spans[i] / largest])) };
  // Freeze the fitted full-scene bounds before setting proportions: otherwise
  // Plotly recalculates its padding and the physical axis scales diverge.
  for (const axis of ['xaxis', 'yaxis', 'zaxis']) {
    if (plot._fullLayout.scene[axis].autorange !== false) {
      changes[`scene.${axis}.range`] = [...plot._fullLayout.scene[axis].range];
      changes[`scene.${axis}.autorange`] = false;
    }
  }
  await Plotly.relayout(plot, changes);
  if (!plot.dataset.scaleListener) {
    plot.on('plotly_relayout', event => {
      if (Object.keys(event).some(key => /^scene\.[xyz]axis\.(range|autorange)/.test(key))) equalAxisScale(plot);
    });
    plot.dataset.scaleListener = 'true';
  }
}

async function drawScene(figure) {
  if (!window.Plotly) throw new Error('Plotly не загружен. Выполните npm install и перезапустите приложение.');
  if (!figure || !Array.isArray(figure.data)) throw new Error('Расчёт не вернул трёхмерную сцену.');
  defaultCamera = { eye: { x: 1.5, y: 1.5, z: 1.1 }, ...figure.layout?.scene?.camera,
    projection: { type: 'perspective' } };
  const sourceLayout = figure.layout || {};
  const axes = Object.fromEntries(Object.keys(volumeRanges).map(axis =>
    [axis, { ...lightSceneAxis(sourceLayout.scene?.[axis]), autorange: true }]));
  const layout = { ...sourceLayout, autosize: true, width: undefined, height: undefined,
    template: {}, paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
    legend: { ...sourceLayout.legend, bgcolor: 'rgba(255,255,255,0.92)', font: { size: 12, color: '#55565b' } },
    hoverlabel: { bgcolor: '#ffffff', bordercolor: '#c9c7c5', font: { family: 'Segoe UI, sans-serif', size: 12, color: '#303136' } },
    font: { ...sourceLayout.font, color: '#55565b', family: 'Segoe UI, sans-serif', size: 12 },
    margin: { l: 0, r: 0, t: 15, b: 0 }, uirevision: 'mirrorlab-camera',
    scene: { ...sourceLayout.scene, ...axes, dragmode: 'orbit', aspectmode: 'data',
      bgcolor: '#ffffff', camera: camera || defaultCamera },
  };
  await Plotly.react($('scene'), themedPlotData(figure.data), layout, {
    responsive: true, displaylogo: false, scrollZoom: true,
    toImageButtonOptions: { format: 'png', filename: 'mirrorlab-scene', scale: 2 },
  });
  await equalAxisScale($('scene'));
  if (!camera) await setSceneExtent();
  if (!$('scene').dataset.cameraListener) {
    $('scene').on('plotly_relayout', (event) => { if (event['scene.camera']) camera = event['scene.camera']; });
    $('scene').dataset.cameraListener = 'true';
  }
  $('empty-scene').hidden = true;
  $('reset-camera').disabled = false;
  $('scene-extent').disabled = false;
}

function showSummary(result, request) {
  $('metrics').replaceChildren();
  const summary = result.summary || {};
  for (const [key, value] of Object.entries(summary)) {
    if (!(key in labels) || typeof value !== 'number') continue;
    const metric = document.createElement('div');
    metric.className = 'metric';
    const strong = document.createElement('strong');
    strong.textContent = format(value);
    const caption = document.createElement('span');
    caption.textContent = labels[key];
    metric.append(strong, caption);
    $('metrics').append(metric);
  }
  $('result-notes').replaceChildren();
  const direction = directionLabel(request.direction);
  $('result-direction').textContent = direction;
  $('scene-label').textContent = `${direction} · ${request.kind === 'ideal' ? 'идеальная' : 'реальная'}`;
  paragraph($('result-notes'), `Поиск пересечений: ${request.collision_mode === 'exhaustive' ? 'полный перебор' : 'ожидаемые объекты'}. Лучей на лазер: ${request.ray_count}; максимум взаимодействий: ${request.max_interactions}.`);
  for (const warning of result.warnings || []) paragraph($('result-notes'), warning);
  $('result-json').textContent = JSON.stringify(metadata(result), null, 2);
  $('result-details').hidden = false;
}

async function showCalibration(result) {
  measurementReports = result.reports || [result];
  measurementContext = { direction: result.direction, kind: $('kind').value };
  $('measurement-sides').replaceChildren();
  for (const report of measurementReports) {
    const button = document.createElement('button');
    button.className = 'secondary-button';
    button.dataset.direction = report.direction;
    button.textContent = `${sideNames[report.direction]} · ${report.counts.total} лучей`;
    button.addEventListener('click', () => showMeasurementSide(report));
    $('measurement-sides').append(button);
  }
  await showMeasurementSide(measurementReports[0]);
  showSideStatus();
}

async function showMeasurementSide(result) {
  for (const button of $('measurement-sides').children) {
    button.setAttribute('aria-pressed', String(button.dataset.direction === result.direction));
  }
  const counts = result.counts || {};
  const summary = $('calibration-summary');
  summary.replaceChildren();
  const route = routes[result.direction];
  paragraph(summary, `Лазер: ${route.laser_side}; кольцо: ${route.ring_side}; ввод: ${route.injection_side}. Геодезия: ${route.survey_file}.`);
  paragraph(summary, screensLabel(result.additional_screens));
  paragraph(summary, 'Измерения и статическая сцена отображаются в общей геодезической системе координат. Привязка сцены по лучам не заменяет юстировку отдельных зеркал.');
  paragraph(summary, `Лучей: ${counts.total ?? '—'}; полных измерений: ${counts.complete ?? '—'}; с одним экраном: ${counts.single_screen ?? '—'}; без измерений: ${counts.missing ?? '—'}.`);
  const comparison = result.saved_equations_comparison;
  if (comparison) paragraph(summary, `Сравнение с сохранёнными уравнениями: совпало ${comparison.matched ?? '—'} из ${comparison.compared ?? '—'}.`);
  for (const warning of result.warnings || []) paragraph(summary, warning);
  if (result.lavsan && result.lavsan.status !== 'not_requested') {
    const status = { fitted: 'преобразование найдено', partial: 'частичное восстановление' }[result.lavsan.status] || result.lavsan.status;
    paragraph(summary, `Лавсановые экраны: ${status ?? '—'}. ${result.lavsan.assumption || ''}`);
    for (const rod of result.lavsan.rods || []) {
      paragraph(summary, `Стержень ${rod.rod}: ${rod.accepted ? 'преобразование принято' : 'преобразование не принято'}; RMS ${format(rod.rms_mm)} мм; максимум ${format(rod.max_mm)} мм; точек ${rod.point_count ?? '—'}.`);
    }
  }
  if (result.sources?.length) paragraph(summary, `Источники: ${result.sources.join('; ')}.`);
  const table = document.createElement('table');
  const head = document.createElement('thead');
  const row = document.createElement('tr');
  for (const title of ['Луч', 'Статус', 'Точек', 'RMS, мм', 'Начальная точка, м', 'Направление']) {
    const th = document.createElement('th'); th.textContent = title; row.append(th);
  }
  head.append(row); table.append(head);
  const body = document.createElement('tbody');
  const statuses = { complete: 'Полные данные', single_screen: 'Один экран', missing: 'Нет данных', fitted: 'Восстановлен',
    refined: 'Уточнён по лавсану', recovered_with_lavsan: 'Восстановлен с лавсаном' };
  const refined = new Map((result.lavsan?.refined_rays || []).map((ray) => [ray.id, ray]));
  for (const original of result.rays || []) {
    const ray = { ...original, ...refined.get(original.id) };
    const tr = document.createElement('tr');
    const vector = (v) => Array.isArray(v) ? v.map(format).join('; ') : '—';
    for (const value of [ray.id, statuses[ray.status] || ray.status, ray.point_count,
      format(ray.rms_mm), vector(ray.point_m), vector(ray.direction)]) {
      const td = document.createElement('td'); td.textContent = value ?? '—'; tr.append(td);
    }
    body.append(tr);
  }
  table.append(body);
  $('calibration-table').replaceChildren(table);
  $('calibration-json').textContent = JSON.stringify(metadata(result), null, 2);
  $('calibration-result').hidden = false;
  $('measurement-figure').hidden = !result.figure;
  if (result.figure) {
    const sourceLayout = result.figure.layout || {};
    await Plotly.react($('measurement-plot'), themedPlotData(result.figure.data), {
      ...sourceLayout, autosize: true, width: undefined, height: undefined,
      template: {}, paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
      legend: { ...sourceLayout.legend, bgcolor: 'rgba(255,255,255,0.92)', font: { size: 12, color: '#55565b' } },
      hoverlabel: { bgcolor: '#ffffff', bordercolor: '#c9c7c5', font: { family: 'Segoe UI, sans-serif', size: 12, color: '#303136' } },
      font: { color: '#55565b', family: 'Segoe UI, sans-serif', size: 12 },
      margin: { l: 0, r: 0, t: 15, b: 0 }, uirevision: 'survey-' + result.direction,
      scene: { ...sourceLayout.scene, dragmode: 'orbit', aspectmode: 'data', bgcolor: '#ffffff',
        camera: { ...sourceLayout.scene?.camera, projection: { type: 'perspective' } },
        ...Object.fromEntries(['xaxis', 'yaxis', 'zaxis'].map(axis => [axis, lightSceneAxis(sourceLayout.scene?.[axis])])) },
    }, { responsive: true, displaylogo: false, scrollZoom: true });
    await equalAxisScale($('measurement-plot'));
  }
}

async function run(method, payload) {
  if (busy) return;
  if (!bridge) { notify('Откройте приложение через npm start: требуется соединение Electron с Python.', true); return; }
  notify('');
  setBusy(true, method === 'trace' ? 'Трассировка лучей' : 'Обработка измерений');
  try {
    const result = await bridge.request(method, payload);
    session[method] = { parameters: payload, ...result };
    if (method === 'trace') {
      await drawScene(result.figure);
      showSummary(result, payload);
      if (result.measurements) {
        session.calibrate = result.measurements;
        await showCalibration(result.measurements);
      } else {
        measurementContext = null;
        measurementReports = [];
        delete session.calibrate;
        $('calibration-result').hidden = true;
      }
    } else await showCalibration(result);
    $('run-status').textContent = method === 'trace' ? 'Конфигурация построена' : 'Измерения обработаны';
    $('connection-label').textContent = 'Расчётный модуль подключён';
    if (method === 'calibrate') $('calibration-result').scrollIntoView({ block: 'start', behavior: 'smooth' });
  } catch (error) {
    notify(error.message || String(error), true);
    $('run-status').textContent = 'Расчёт не завершён';
  } finally { setBusy(false); }
}

$('trace-form').addEventListener('submit', (event) => {
  event.preventDefault();
  if (!$('trace-form').reportValidity()) return;
  run('trace', { direction: $('direction').value, kind: $('kind').value,
    collision_mode: $('collision-mode').value, ray_count: Number($('ray-count').value),
    max_interactions: Number($('max-interactions').value) });
});
$('calibrate').addEventListener('click', () => run('calibrate', {
  direction: $('direction').value,
}));
for (const id of ['direction', 'kind', 'collision-mode']) $(id).addEventListener('change', () => {
  if (id !== 'collision-mode') {
    measurementContext = null;
    $('calibration-result').hidden = true;
  }
  updateControls();
});
$('cancel').addEventListener('click', async () => {
  try { await bridge.request('cancel'); } catch (error) { notify(error.message, true); }
});
async function setSceneExtent() {
  // Keep every ray inside the axis bounds. A close-up changes the camera,
  // never the clipping box (which previously hid the laser-to-periscope path).
  const scene = $('scene')._fullLayout.scene;
  let view = structuredClone(defaultCamera);
  if (!overview) {
    const center = {};
    const spans = Object.keys(volumeRanges).map(axis => Math.abs(scene[axis].range[1] - scene[axis].range[0]));
    const zoom = Math.min(1, 3.9 / Math.max(...spans));
    for (const axis of ['x', 'y', 'z']) {
      const range = scene[axis + 'axis'].range;
      center[axis] = -(range[0] + range[1]) / (range[1] - range[0]) * scene.aspectratio[axis];
    }
    view.center = center;
    view.eye = Object.fromEntries(['x', 'y', 'z'].map(axis => [axis, center[axis] + defaultCamera.eye[axis] * zoom]));
  }
  $('scene-extent').textContent = overview ? 'К объёму' : 'Вся схема';
  await Plotly.relayout($('scene'), { 'scene.camera': view });
}
$('scene-extent').addEventListener('click', async () => {
  overview = !overview;
  await setSceneExtent();
});
$('reset-camera').addEventListener('click', async () => {
  camera = null;
  overview = true;
  await setSceneExtent();
  await Plotly.relayout($('scene'), { 'scene.camera': defaultCamera });
});
for (const button of document.querySelectorAll('[data-tab]')) button.addEventListener('click', () => {
  const selected = button.dataset.tab;
  for (const panel of document.querySelectorAll('.panel')) panel.hidden = panel.id !== selected + '-panel';
  for (const nav of document.querySelectorAll('.nav-button')) {
    const active = nav.dataset.tab === selected;
    nav.classList.toggle('active', active);
    if (active) nav.setAttribute('aria-current', 'page'); else nav.removeAttribute('aria-current');
  }
  $('breadcrumb').textContent = { static: 'Статика', dynamic: 'Динамика', distributions: 'Распределения' }[selected];
  if (selected === 'static') {
    if ($('scene').data) Plotly.Plots.resize($('scene'));
    if ($('measurement-plot').data) Plotly.Plots.resize($('measurement-plot'));
  }
});

updateControls();
(async () => {
  if (!bridge) { $('connection-label').textContent = 'Запустите через npm start'; return; }
  try {
    const catalog = await bridge.request('catalog');
    capabilities = catalog.additional_screens || [];
    directions = catalog.directions.filter(item => item.id !== 'both').map(item => item.id);
    for (const item of catalog.directions) {
      routes[item.id] = item;
      sideNames[item.id] = item.injection_side;
    }
    $('direction').replaceChildren(...catalog.directions.map(item => {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.id === 'both' ? 'Обе: WEST и EAST' : item.injection_side;
      return option;
    }));
    const defaults = catalog.defaults || {};
    for (const [key, id] of Object.entries({ direction: 'direction', kind: 'kind', collision_mode: 'collision-mode', ray_count: 'ray-count', max_interactions: 'max-interactions' })) {
      if (defaults[key] !== undefined) $(id).value = defaults[key];
    }
    $('connection-label').textContent = 'Расчётный модуль подключён';
    updateControls();
  } catch (error) {
    $('connection-label').textContent = 'Модуль расчёта недоступен';
    notify(error.message, true);
  }
})();
