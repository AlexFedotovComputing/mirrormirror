/* The renderer only receives structured data; all computation stays in Python. */
const $ = (id) => document.getElementById(id);
const bridge = window.mirrorlab;
const session = {};
let busy = false;
let ready = false;
let legendVisible = false;
let camera = null;
let defaultCamera = null;
let overview = true;
const volumeRanges = { xaxis: [-1.5, 1.5], yaxis: [-1.5, 1.5], zaxis: [-1.95, 1.95] };
let elapsedTimer;
let measurementReports = [];
let configurationVersion = 0;
let activeConfiguration = false;
let reconstructionResult = null;
let baseScene = null;
let reconstructionScene = null;
let sceneMeasurements = [];
let mirrorRecords = [];
let currentJob = null;
let busyText = '';
const sideNames = {};
const routes = {};
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
  const locked = busy || !ready;
  $('controls').disabled = locked;
  $('calibrate').disabled = locked;
  $('start-trace').disabled = $('start-load').disabled = locked;
  for (const id of ['optimize', 'save-configuration', 'load-configuration', 'load-manual', 'load-legacy', 'preview-candidate',
    'choose-legacy', 'accept-candidate', 'use-historical', 'save-source', 'legacy-frame']) $(id).disabled = locked;
  $('accept-candidate').disabled = locked || !!candidateBlockReason(reconstructionResult);
  const manual = hasManualReconstruction(reconstructionResult);
  const historicalIdeal = !activeConfiguration && !real && !manual;
  $('optimize').textContent = manual ? 'Уточнить ручную реконструкцию' : 'Восстановить зеркала';
  $('optimize').disabled = locked || historicalIdeal;
  $('optimize-reason').hidden = !historicalIdeal;
  $('kind').disabled = locked || activeConfiguration;
  $('historical-kind').hidden = activeConfiguration;
  $('calculation-source').hidden = !activeConfiguration;
  $('active-state').textContent = activeConfiguration ? `Принятая конфигурация · версия ${configurationVersion}`
    : $('kind').selectedOptions[0].textContent;
  $('save-source').options[0].textContent = activeConfiguration ? 'Принятая конфигурация' : 'Текущая историческая конфигурация';
  for (const button of document.querySelectorAll('[data-result-target]')) button.disabled = $(button.dataset.resultTarget).hidden;
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

function setBusy(active, label = 'Трассировка лучей') {
  busy = active;
  $('controls').disabled = active;
  $('cancel').hidden = !active;
  $('busy-overlay').hidden = !active;
  $('busy-label').textContent = label;
  busyText = label;
  clearInterval(elapsedTimer);
  if (active) {
    const start = Date.now();
    $('run-status').textContent = label + '…';
    elapsedTimer = setInterval(() => {
      $('run-status').textContent = `${busyText} · ${Math.floor((Date.now() - start) / 1000)} с`;
    }, 1000);
  }
  updateControls();
}

function paragraph(parent, text) {
  const p = document.createElement('p');
  p.textContent = typeof text === 'string' ? text : JSON.stringify(text);
  parent.append(p);
  return p;
}

function metadata(result) {
  const { figure, ...rest } = result;
  return rest;
}

function showCoordinateFrame(frame) {
  const survey = frame === 'survey';
  $('coordinate-title').hidden = false;
  $('coordinate-title').textContent = survey ? 'Геодезические XYZ · м' : 'Расчётная сцена · м';
  $('coordinate-title').title = survey ? 'Знак Z — по измерениям. Центральная плоскость объёма: z = 0.'
    : 'Координаты сохранённого источника. Привязка к геодезии не установлена.';
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
  if (!plot._fullLayout?.scene) return; // All layers may be switched off.
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
    [axis, { ...lightSceneAxis({ ...sourceLayout.scene?.[axis],
      title: sourceLayout.scene?.[axis]?.title ?? { text: `${axis[0]}, м` } }), autorange: true }]));
  const layout = { ...sourceLayout, autosize: true, width: undefined, height: undefined,
    showlegend: legendVisible,
    template: {}, paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
    legend: { ...sourceLayout.legend, bgcolor: 'rgba(255,255,255,0.92)', font: { size: 12, color: '#55565b' } },
    hoverlabel: { bgcolor: '#ffffff', bordercolor: '#c9c7c5', font: { family: 'Segoe UI, sans-serif', size: 12, color: '#303136' } },
    font: { ...sourceLayout.font, color: '#55565b', family: 'Segoe UI, sans-serif', size: 12 },
    ...(sourceLayout.title?.text ? { title: { ...sourceLayout.title, x: .02, y: .98, xanchor: 'left',
      font: { size: 13, color: '#55565b' } } } : {}),
    margin: { l: 0, r: 0, t: sourceLayout.title?.text ? 45 : 15, b: 0 }, uirevision: 'mirrorlab-camera',
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
  if (!$('scene').dataset.mirrorListener) {
    $('scene').on('plotly_click', event => {
      const point = event.points?.[0];
      const info = point?.customdata && typeof point.customdata === 'object' && !Array.isArray(point.customdata)
        ? point.customdata : point?.data?.meta;
      const id = info?.mirror_id ?? info?.canonical_id;
      if (!id) return;
      let selected = mirrorRecords.findIndex(item => (item.mirror.canonical_id === id || item.mirror.mirror_id === id)
        && (!info.direction || item.direction === info.direction));
      if (selected < 0) selected = mirrorRecords.findIndex(item => item.mirror.old_id === id &&
        (!info.direction || item.direction === info.direction));
      if (selected >= 0) { $('mirror-select').value = String(selected); showMirror(); }
    });
    $('scene').dataset.mirrorListener = 'true';
  }
  $('empty-scene').hidden = true;
  $('reset-camera').disabled = !$('scene')._fullLayout?.scene;
  $('scene-extent').disabled = !$('scene')._fullLayout?.scene;
  $('toggle-legend').disabled = !figure.data.length;
}

async function drawSceneLayers() {
  const result = composeScene(baseScene, reconstructionScene, sceneMeasurements, {
    base: $('show-base').checked, reconstruction: $('show-reconstruction').checked,
    measurements: $('show-measurements').checked,
    sides: Object.fromEntries([...document.querySelectorAll('[data-scene-side]')].map(input => [input.dataset.sceneSide, input.checked])),
    layers: Object.fromEntries([...document.querySelectorAll('[data-scene-layer]')].map(input => [input.dataset.sceneLayer, input.checked])),
  });
  if (!result) return;
  await drawScene(result.figure);
  $('filtered-scene').hidden = result.figure.data.length > 0;
  const hiddenLayers = [...document.querySelectorAll('.scene-layers input')].filter(input => !input.checked).length;
  $('layers-summary').textContent = hiddenLayers ? `Скрыто групп: ${hiddenLayers}` : '';
  $('layers-summary').hidden = !hiddenLayers;
  showCoordinateFrame(result.frame);
  $('scene-label').textContent = result.label;
  $('scene-comparison-note').textContent = result.note;
  $('scene-comparison-note').hidden = !result.note;
}

function candidateBlockReason(result) {
  if (!result?.document) return 'Нет конфигурации для принятия.';
  if (result.active) return 'Эта конфигурация уже принята.';
  if (hasManualReconstruction(result) && result.status !== 'validated') return 'Ручная реконструкция не прошла все проверки.';
  if (result.status === 'infeasible' || reconstructionReports(result).some(report => report.status === 'infeasible')) {
    return 'Физические проверки не пройдены. Причины — в отчёте реконструкции.';
  }
  if (result.can_activate === false) return 'Не установлена общая оптика или конфигурация неполна.';
  return '';
}

function hasManualReconstruction(result) {
  return Object.values(result?.document?.datasets ?? {}).some(dataset =>
    dataset?.model_kind === 'manual_reconstruction' || !!dataset?.manual_references);
}

function showSummary(result, request) {
  showCoordinateFrame(result.metadata?.coordinate_frame ?? 'survey');
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
  $('scene-label').textContent = `${direction} · ${result.active ? 'принятая реконструкция' : request.kind === 'ideal' ? 'историческая идеальная' : 'историческая реальная'}`;
  paragraph($('result-notes'), `Поиск пересечений: ${request.collision_mode === 'exhaustive' ? 'полный перебор' : 'ожидаемые объекты'}. Лучей на лазер: ${request.ray_count}; максимум взаимодействий: ${request.max_interactions}.`);
  for (const warning of result.warnings || []) paragraph($('result-notes'), warning);
  $('result-json').textContent = JSON.stringify(metadata(result), null, 2);
  $('result-details').hidden = false;
}

async function showCalibration(result) {
  measurementReports = result.reports || [result];
  sceneMeasurements = measurementReports.map(report => ({ ...report, side: sideNames[report.direction] }));
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
  await drawSceneLayers();
}

async function showMeasurementSide(result) {
  for (const button of $('measurement-sides').children) {
    button.setAttribute('aria-pressed', String(button.dataset.direction === result.direction));
  }
  const counts = result.counts || {};
  const summary = $('calibration-summary');
  summary.replaceChildren();
  const route = routes[result.direction];
  const sources = comparisonDetails(summary, 'Источники и привязка');
  paragraph(sources, `Лазер: ${route.laser_side}; кольцо: ${route.ring_side}; ввод: ${route.injection_side}. Геодезия: ${route.survey_file}.`);
  paragraph(summary, screensLabel(result.additional_screens));
  paragraph(sources, 'Измерения и статическая сцена отображаются в общей геодезической системе координат. Привязка сцены по лучам не заменяет юстировку отдельных зеркал.');
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
  if (result.sources?.length) paragraph(sources, `Источники: ${result.sources.join('; ')}.`);
  summary.append(sources);
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

const reconstructionStatuses = { validated: 'Проверено', partial: 'Частичный результат',
  infeasible: 'Недопустимо', legacy_unvalidated: 'Историческая геометрия · не проверена',
  needs_context: 'Нужны сведения об источнике' };
const basisStatuses = { data_supported: 'По измерениям', data_and_mechanics: 'Измерения и механика',
  prior_dominated: 'Определено предположениями', ambiguous: 'Неоднозначно', infeasible: 'Недопустимо',
  measured: 'Измерено', partial: 'Частичные измерения', predicted: 'Предсказано', failed: 'Не восстановлено' };
const precise = value => typeof value === 'number'
  ? value !== 0 && Math.abs(value) < 1e-7 ? value.toExponential(4) : value.toLocaleString('ru-RU', { maximumFractionDigits: 9 })
  : String(value);
const describe = value => value === undefined || value === null ? 'Не определено'
  : Array.isArray(value) ? value.map(item => typeof item === 'number' ? precise(item) : describe(item)).join('; ')
    : typeof value === 'object' ? JSON.stringify(value, null, 2) : precise(value);

function reconstructionReports(result) {
  if (Array.isArray(result.reports)) return result.reports;
  const datasets = result.document?.datasets ?? result.document?.directions;
  if (Array.isArray(datasets)) return datasets.filter(item => item && typeof item === 'object');
  if (datasets && typeof datasets === 'object') return Object.entries(datasets).map(([direction, item]) => ({ direction, ...item }));
  return [];
}

function comparisonTable(parent, headings, rows, captionText) {
  const wrap = document.createElement('div'); wrap.className = 'table-wrap';
  const table = document.createElement('table');
  if (captionText) { const caption = document.createElement('caption'); caption.className = 'sr-only'; caption.textContent = captionText; table.append(caption); }
  const head = document.createElement('thead'); const headRow = document.createElement('tr');
  for (const label of headings) { const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = label; headRow.append(cell); }
  head.append(headRow); table.append(head);
  const body = document.createElement('tbody');
  for (const row of rows) {
    const tr = document.createElement('tr');
    if (row.outlier) tr.dataset.outlier = 'true';
    for (const value of Array.isArray(row) ? row : row.values) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
    body.append(tr);
  }
  table.append(body); wrap.append(table); parent.append(wrap);
  return table;
}

function comparisonDetails(parent, label) {
  const details = document.createElement('details'); const summary = document.createElement('summary');
  summary.textContent = label; details.append(summary); parent.append(details); return details;
}

function comparisonScore(parent, report, className) {
  const score = document.createElement('p'); score.className = `comparison-score ${className}`;
  const total = report.observation_count ?? 0;
  score.textContent = `${report.matched_count ?? 0} из ${total}`;
  const percent = document.createElement('small');
  percent.textContent = total ? ` · ${((report.matched_count ?? 0) * 100 / total).toLocaleString('ru-RU', { maximumFractionDigits: 1 })}%` : ' · нет полных измерений';
  score.append(percent); parent.append(score);
}

const trajectoryReasonLabels = {
  matched: 'Измерение воспроизведено',
  geometric_mismatch: 'Подогнанная линия вне допуска',
  no_fitted_assignment: 'Не назначена подогнанная линия',
  geometric_fit_unavailable: 'Нет данных геометрической подгонки',
  one_to_one_conflict: 'Подходящий путь назначен другому измерению',
  intercepted_before_mirror: 'До нужного зеркала встретилась другая поверхность',
  incoming_branch_missing: 'Входящая ветвь не достигнута',
  outside_illumination: 'Центр среза вне входящего пучка',
  incoming_ambiguous: 'Неоднозначная входящая ветвь',
  incoming_verification_failed: 'Не подтверждено попадание входящего луча',
  incoming_unavailable: 'Нет доступной входящей ветви',
  wrong_reflecting_side: 'Неверная сторона отражения или скользящее падение',
  outside_reflecting_ellipse: 'Попадание вне отражающего эллипса',
  no_reflected_child: 'Нужный отражённый луч не возник',
  outgoing_mirror_interception: 'Выходной путь перехвачен зеркалом до измеренных точек',
  trace_time_limit: 'Время трассировки закончилось до измеренных точек',
  path_turn_or_discontinuity: 'Поворот или разрыв пути до измеренных точек',
  trace_ended_before_points: 'Сохранённый участок закончился до измеренных точек',
  physical_path_mismatch: 'Прослеженный путь вне допуска',
  points_behind_reflection: 'Измеренная точка позади начала отражённого пути',
  reflected_path_missing: 'Нет сохранённых отражённых участков',
  optical_evidence_missing: 'В сохранении нет оптической диагностики зеркала',
  physical_comparison_unavailable: 'Физическое сравнение недоступно',
  two_complete_points_required: 'Нужны две полностью измеренные точки',
  invalid_or_coincident_points: 'Точки некорректны или совпадают',
};

function renderTrajectoryDiagnostics(parent, report) {
  if (report?.status !== 'evaluated') return;
  const rows = report.observations ?? [];
  const section = comparisonDetails(parent, `Причины по траекториям · ${rows.length}`);
  section.className = 'trajectory-diagnostics';
  paragraph(section, `Линия попадает в допуск, но физический путь не воспроизведён: ${report.fit_only_count}.`);
  paragraph(section, 'Зеркало подгонки — гипотеза геометрического сопоставления. Ближайший физический путь может принадлежать другому зеркалу. Механические конфликты показаны отдельно.');
  const controls = document.createElement('div'); controls.className = 'trajectory-diagnostic-controls';
  const filter = document.createElement('select'); filter.setAttribute('aria-label', 'Фильтр траекторий');
  for (const [value, label] of [['fit_only', 'Подгонка есть, пути нет'], ['unmatched', 'Все невоспроизведённые'],
    ['matched', 'Воспроизведённые'], ['excluded', 'Неполные измерения'], ['all', 'Все траектории']]) {
    const option = document.createElement('option'); option.value = value; option.textContent = label; filter.append(option);
  }
  filter.value = report.fit_only_count ? 'fit_only' : 'all';
  const search = document.createElement('input'); search.type = 'search'; search.placeholder = 'Измерение, зеркало, поверхность или причина';
  search.setAttribute('aria-label', 'Поиск траектории'); controls.append(filter, search); section.append(controls);
  const count = paragraph(section, ''); count.setAttribute('aria-live', 'polite');
  const list = document.createElement('div'); list.className = 'trajectory-diagnostic-list'; section.append(list);
  const reasonLabel = code => trajectoryReasonLabels[code] ?? code;
  const mm = value => value == null ? '—' : format(value * 1000);
  const point = value => value ? value.map(v => Number(v).toFixed(6)).join('; ') : 'Нет данных';
  const renderRow = row => {
    const card = comparisonDetails(list, `${row.observation_id} · ${row.reason_codes.map(reasonLabel).join('; ')}`);
    card.className = 'trajectory-diagnostic-row'; card.dataset.observationId = row.observation_id;
    card.dataset.status = row.status;
    // Render detailed tables on first expansion; a side may contain 112 rows.
    card.addEventListener('toggle', () => {
      if (!card.open || card.dataset.loaded) return;
      card.dataset.loaded = 'true';
      const fit = row.fitted; const actual = row.physical; const mirror = row.fitted_mirror; const path = row.path;
      comparisonTable(card, ['Проверка', 'Зеркало', 'Ошибка по двум точкам, мм', 'Результат'], [
        ['Подгонка', fit?.mirror_id ?? '—', mm(fit?.maximum_distance_m), fit ? (fit.within_tolerance ? 'В допуске' : 'Вне допуска') : 'Нет данных'],
        [actual?.matched ? 'Совпавший физический путь' : 'Ближайший физический путь', actual?.mirror_id ?? '—',
          mm(actual?.maximum_distance_m), actual ? (actual.matched ? 'Воспроизведено' : 'Не засчитан') : 'Нет пути'],
      ], 'Подгонка и физическая траектория');
      if (actual?.assigned_observation_id && actual.assigned_observation_id !== row.observation_id)
        paragraph(card, `Этот физический путь назначен измерению ${actual.assigned_observation_id}.`);
      if (mirror) {
        paragraph(card, `Для зеркала подгонки ${mirror.mirror_id}: ${mirror.optical_reason_codes.length
          ? mirror.optical_reason_codes.map(reasonLabel).join('; ') : 'нет зарегистрированной ошибки входа и отражения'}.`);
        if (mirror.first_surface) paragraph(card, `Первая достигнутая поверхность вместо нужного среза: ${mirror.first_surface}.`);
        if (mirror.incident_segment) paragraph(card, `Входящий участок: ${point(mirror.incident_segment.start_m)} → ${point(mirror.incident_segment.end_m)} м; поверхность ${mirror.incident_segment.surface}.`);
        if (mirror.outgoing_blockers?.length) paragraph(card, `В дальнейших ветвях встретились тела: ${mirror.outgoing_blockers.join(', ')}. Это само по себе не означает остановку до измеренных точек.`);
        const dimensions = mirror.dimensions;
        if (dimensions) paragraph(card, `Размеры в сохранённой модели: цилиндр Ø ${format(dimensions.cylinder_diameter_mm)} мм; оси эллипса ${format(dimensions.ellipse_minor_axis_mm)} × ${format(dimensions.ellipse_major_axis_mm)} мм.`);
        if (mirror.axial_shift_m != null) paragraph(card, `Подстройка центра вдоль выходной линии: ${mm(mirror.axial_shift_m)} мм. Это результат поиска, не измеренный допуск крепления.`);
        if (mirror.body_conflicts?.length) comparisonTable(card, ['Пара тел', 'Состояние'],
          mirror.body_conflicts.map(c => [c.pair.join(' / '), c.status ?? 'Зазор не подтверждён']), 'Механические конфликты зеркала');
        if (mirror.violations?.length) paragraph(card, `Ограничения конфигурации: ${mirror.violations.join('; ')}.`);
      }
      if (path) {
        paragraph(card, `Показан ${row.path_basis === 'matched_physical_path' ? 'совпавший физический путь' : 'лучший сохранённый путь зеркала подгонки'}: ${path.trajectory_id}.`);
        paragraph(card, `Длина: ${mm(path.length_m)} мм. Конец: ${point(path.end_m)} м. Последняя поверхность: ${path.terminal_surface ?? 'не сохранена'}.`);
        comparisonTable(card, ['Точка', 'До подгонки, мм', 'До участка, мм', 'До продолжения, мм', 'За концом, мм', 'Позади начала, мм'],
          path.point_checks.map((p, i) => [p.point_index, mm(fit?.point_distances_m?.[i]), mm(p.distance_m),
            mm(p.extension_distance_m), mm(p.beyond_end_m), mm(p.behind_origin_m)]), 'Ошибки и положение измеренных точек');
        paragraph(card, 'Продолжение используется только для объяснения причины; за пределами прослеженного участка оно не засчитывается. Координаты — геодезические, м.');
        comparisonTable(card, ['Поверхность', 'Начало XYZ, м', 'Конец XYZ, м'],
          path.segments.map(s => [s.surface ?? 'Не сохранена', point(s.start_m), point(s.end_m)]), 'Последовательность прослеженных участков');
      } else {
        if (row.screens_m) comparisonTable(card, ['Точка', 'XYZ, м', 'До подгонки, мм'],
          row.screens_m.map((p, i) => [i + 1, point(p), mm(fit?.point_distances_m?.[i])]), 'Измеренные точки');
        paragraph(card, 'Нет связанного прослеженного участка для проверки положения точек относительно его конца.');
      }
    });
  };
  const refresh = () => {
    const query = search.value.trim().toLocaleLowerCase();
    const selected = rows.filter(row => (filter.value === 'all' || (filter.value === 'fit_only' ? row.fit_only : row.status === filter.value)) &&
      (!query || `${JSON.stringify(row)} ${row.reason_codes.map(reasonLabel).join(' ')}`.toLocaleLowerCase().includes(query)));
    list.replaceChildren(); count.textContent = `Показано: ${selected.length} из ${rows.length}`;
    selected.forEach(renderRow);
  };
  filter.addEventListener('change', refresh); search.addEventListener('input', refresh); refresh();
}

function renderTrajectoryComparison(reports, context) {
  const container = $('trajectory-comparison-sides'); container.replaceChildren();
  const preview = $('trajectory-comparison-preview');
  $('trajectory-comparison-result').hidden = !reports.length;
  preview.hidden = !reports.length;
  $('trajectory-comparison-context').textContent = context;
  let fitMatched = 0, fitTotal = 0, physicalMatched = 0, physicalTotal = 0;
  for (const report of reports) {
    const fit = report.diagnostics?.geometric_fit;
    const physical = report.diagnostics?.trajectory_matching;
    const side = document.createElement('article'); side.className = 'comparison-side trajectory-comparison-side'; side.dataset.direction = report.direction;
    const heading = document.createElement('h3'); heading.textContent = sideNames[report.direction] ?? report.direction; side.append(heading);
    const panels = document.createElement('div'); panels.className = 'comparison-panels'; side.append(panels);
    const fitPanel = document.createElement('section'); fitPanel.className = 'comparison-panel geometric-fit-panel';
    const fitHeading = document.createElement('h4'); fitHeading.textContent = 'Точность подгонки'; fitPanel.append(fitHeading);
    if (fit?.status === 'evaluated') {
      fitMatched += fit.matched_count; fitTotal += fit.observation_count;
      comparisonScore(fitPanel, fit, 'geometric-fit-summary');
      paragraph(fitPanel, `Порог: ${format(fit.distance_tolerance_m * 1000)} мм · сопоставлено: ${fit.assigned_count}/${fit.observation_count}`);
      const statistics = comparisonTable(fitPanel, ['Ошибка', 'Медиана', '90%', '95%', 'Максимум'], [
        ['Положение, мм', ...['median', 'p90', 'p95', 'maximum'].map(k => format(fit.position_error_mm?.[k]))],
        ['Угол, мрад', ...['median', 'p90', 'p95', 'maximum'].map(k => format(fit.angular_error_mrad?.[k]))],
      ], 'Распределение ошибок геометрической подгонки'); statistics.className = 'comparison-statistics';
      const pairs = [...(fit.matches ?? [])].sort((a, b) => b.maximum_distance_m - a.maximum_distance_m);
      const details = comparisonDetails(fitPanel, `Пары и наибольшие отклонения · ${pairs.length}`);
      paragraph(details, `В пределах 1 мм: ${fit.counts_within_1mm}/${fit.observation_count}; в пределах 3 мм: ${fit.counts_within_3mm}/${fit.observation_count}.`);
      paragraph(details, 'Сначала показаны наибольшие ошибки. Строки выше порога выделены. Номера используются только для поиска записей.');
      comparisonTable(details, ['Измерение', 'Расчётная траектория', 'Ошибка, мм', 'Угол, мрад'], pairs.map(pair => ({
        outlier: !pair.within_tolerance, values: [pair.observation_id, pair.trajectory_id,
          format(pair.maximum_distance_m * 1000), format(pair.angular_error_rad * 1000)],
      })), 'Соответствия измеренных и подогнанных траекторий');
      for (const item of fit.unmatched_observations ?? []) paragraph(details, `Без расчётной пары: ${item.observation_id ?? item}.`);
      if (fit.excluded_observations?.length) {
        const excluded = comparisonDetails(fitPanel, `Не входят в оценку · ${fit.excluded_observations.length}`);
        paragraph(excluded, 'Для проверки траектории нужны две различные полностью измеренные точки.');
        for (const item of fit.excluded_observations) paragraph(excluded, item.observation_id);
      }
    } else {
      paragraph(fitPanel, 'Нет реконструкции с измерениями. Загрузите её или восстановите зеркала.').className = 'comparison-empty';
    }
    panels.append(fitPanel);
    const physicalPanel = document.createElement('section'); physicalPanel.className = 'comparison-panel physical-path-panel';
    const physicalHeading = document.createElement('h4'); physicalHeading.textContent = 'Физическая проходимость'; physicalPanel.append(physicalHeading);
    if (physical?.status === 'evaluated') {
      physicalMatched += physical.matched_count; physicalTotal += physical.observation_count;
      comparisonScore(physicalPanel, physical, 'physical-path-summary');
      paragraph(physicalPanel, `Порог: ${format(physical.distance_tolerance_m * 1000)} мм · не воспроизведено: ${physical.unmatched_observation_count}`);
      paragraph(physicalPanel, `Расчётных путей без пары измерений: ${physical.unmatched_trajectory_count}`);
      const details = comparisonDetails(physicalPanel, 'Прошедшие пути и несовпавшие измерения');
      comparisonTable(details, ['Измерение', 'Прослеженный путь', 'Ошибка, мм', 'Угол, мрад'], (physical.matches ?? []).map(pair => [
        pair.observation_id, pair.trajectory_id, format(pair.maximum_distance_m * 1000), format(pair.angular_error_rad * 1000),
      ]), 'Геометрически совпавшие физические пути');
      for (const item of physical.unmatched_observations ?? []) paragraph(details,
        `Не воспроизведено: ${item.observation_id}${item.nearest ? `; ближайший путь ${item.nearest.trajectory_id}, ${format(item.nearest.maximum_distance_m * 1000)} мм${item.reason === 'one_to_one_conflict' ? ' — уже сопоставлен другому измерению' : ''}` : '; расчётного пути нет'}.`);
      for (const id of physical.unmatched_trajectory_ids ?? []) paragraph(details, `Путь без пары измерений: ${id}.`);
    } else {
      paragraph(physicalPanel, 'Нет измерений, сохранённых физических путей или их общей привязки.').className = 'comparison-empty';
    }
    panels.append(physicalPanel);
    renderTrajectoryDiagnostics(side, report.diagnostics?.trajectory_diagnostics);
    container.append(side);
  }
  const toleranceLabels = [...new Set(reports.flatMap(report => [report.diagnostics?.geometric_fit, report.diagnostics?.trajectory_matching])
    .filter(value => value?.status === 'evaluated').map(value => format(value.distance_tolerance_m * 1000)))];
  preview.hidden = !fitTotal && !physicalTotal;
  $('trajectory-comparison-summary').textContent = fitTotal || physicalTotal
    ? `Подгонка: ${fitTotal ? `${fitMatched}/${fitTotal}` : 'недоступна'} · Проходимость: ${physicalTotal ? `${physicalMatched}/${physicalTotal}` : 'недоступна'} · ${toleranceLabels.length === 1 ? `порог ${toleranceLabels[0]} мм` : 'пороги в отчёте'}`
    : 'Нет реконструкции с измерениями для сравнения.';
}

function showMirror() {
  const selected = mirrorRecords[Number($('mirror-select').value)];
  $('mirror-details').replaceChildren();
  $('mirror-technical').textContent = '';
  if (!selected) return;
  const mirror = selected.mirror;
  const body = mirror.body ?? mirror;
  const frame = selected.displayFrame;
  const hasTransform = (frame?.coordinate_frame === 'survey' || frame?.coordinate_operation === 'scene_to_survey') &&
    Array.isArray(frame.rotation) && Array.isArray(frame.translation_m);
  const transform = (vector, point = false) => !hasTransform || !Array.isArray(vector) ? vector
    : frame.rotation.map((row, i) => row.reduce((sum, v, j) => sum + v * vector[j], 0) + (point ? frame.translation_m[i] : 0));
  const uncertainty = mirror.uncertainty ?? {};
  const constraint = value => ({ not_evaluated: 'Не проверено', passed: 'Проверено', violated: 'Нарушение' }[value?.status ?? value] ?? value?.status ?? value);
  const alternatives = mirror.assignment_alternatives ?? mirror.alternatives ?? selected.assignment?.alternatives ?? [];
  const fields = [
    ['Сторона', sideNames[selected.direction] ?? selected.direction],
    ['Исходный номер', mirror.old_id ?? mirror.mirror_id], ['Каноническое место', mirror.canonical_id],
    ['Ручной источник', mirror.source_manual_id], ['Геодезическое наблюдение', mirror.survey_observation_id],
    ['Физический идентификатор', mirror.physical_native_id ?? mirror.physical_id ?? mirror.native_id], ['Угловое место', mirror.canonical_slot],
    ['Происхождение назначения', mirror.mapping_origin],
    ['Статус', basisStatuses[mirror.status] ?? mirror.status],
    ['Наблюдение', selected.datasetStatus === 'legacy_unvalidated' && !mirror.manual_reference_survey
      ? 'Исторический источник; первичные измерения не привязаны' : basisStatuses[mirror.observation_status] ?? mirror.observation_status],
    ['Идентификаторы измерений', mirror.observation_old_ids ?? mirror.observation_ids],
    ['Основание', basisStatuses[mirror.support_status ?? mirror.basis ?? mirror.confidence_status ?? mirror.status] ?? mirror.support_status ?? mirror.basis ?? mirror.confidence_status],
    ['Система координат', hasTransform ? 'Геодезическая · сохранённое преобразование'
      : selected.calculationFrame === 'survey' ? 'Геодезическая · исходные координаты' : 'Расчётная сцена'],
    ['Центр отражающего среза, м', transform(body.center_m ?? mirror.center_m, true)],
    ['Физическая нормаль', transform(body.normal ?? body.normal_world)], ['Ось цилиндра', transform(body.axis ?? body.axis_world)],
    ['Сдвиг от шаблона, мм', mirror.template_deviation?.center_m?.map(value => value * 1000)],
    ['Поворот от шаблона, мрад', mirror.template_deviation?.angular_rad == null ? null : mirror.template_deviation.angular_rad * 1000],
    ['Остаток по измерениям, м', mirror.measurement_residual_m ?? mirror.residual_m ?? mirror.residuals],
    ['Зазор, м', mirror.clearance_m ?? mirror.minimum_clearance_m],
    ['Неопределённость центра, мм', uncertainty.center_sigma_m == null ? null : uncertainty.center_sigma_m * 1000],
    ['Предиктивный угловой разброс, мрад', uncertainty.predictive_angle_sigma_rad == null ? null : uncertainty.predictive_angle_sigma_rad * 1000],
    ['Ошибка среднего шаблона, мрад', uncertainty.mean_template_sigma_rad == null ? null : uncertainty.mean_template_sigma_rad * 1000],
    ['Размещение на стержне', constraint(mirror.rod_containment)], ['Сторона перетяжки', constraint(mirror.waist_side)],
    ['Допустимые гипотезы соответствия', alternatives.length || null],
  ];
  for (const [label, value] of fields) {
    const dt = document.createElement('dt'); dt.textContent = label;
    const dd = document.createElement('dd'); dd.textContent = describe(value);
    $('mirror-details').append(dt, dd);
  }
  $('mirror-technical').textContent = JSON.stringify({ mirror, assignment: selected.assignment,
    coordinate_frame: { displayed: hasTransform ? 'survey' : selected.calculationFrame, stored: selected.calculationFrame,
      note: 'Компоненты неопределённости и параметры тела сохранены в расчётной системе координат. Стоимости гипотез не являются вероятностями.' } }, null, 2);
}

async function showReconstruction(result) {
  reconstructionResult = result;
  configurationVersion = result.configuration_version ?? configurationVersion;
  if (result.active) {
    activeConfiguration = true;
    if (baseScene) markSceneStale('Источник расчёта изменён. На сцене сохранена предыдущая трассировка для сравнения. Нажмите «Рассчитать лучи» для нового расчёта.');
  }
  const reports = reconstructionReports(result);
  renderTrajectoryComparison(reports, result.active ? 'Источник: принятая реконструкция.'
    : 'Источник: кандидат, не принятый для расчётов.');
  if (result.active && reports.length === 1) $('direction').value = reports[0].direction;
  reconstructionScene = null;
  if (result.figure) {
    reconstructionScene = { figure: result.figure, frame: result.coordinate_frame ??
      (reports.every(report => report.display_frame?.coordinate_frame === 'survey') ? 'survey' : 'legacy'),
      direction: result.scene_direction ?? (reports.length > 1 ? 'both' : reports[0]?.direction),
      label: `${reports.map(report => sideNames[report.direction]).join(' и ')} · ${result.active ? 'принятая реконструкция' : 'диагностический кандидат'}` };
  }
  const status = reconstructionStatuses[result.status] ?? result.status ?? 'Диагностический результат';
  $('reconstruction-status').textContent = status;
  $('candidate-state').hidden = false;
  $('candidate-state').textContent = `${result.active ? 'Принято для расчётов' : 'Кандидат'} · ${status}`;
  $('candidate-report').hidden = false;
  $('preview-candidate').hidden = result.active || !result.figure;
  $('preview-candidate').textContent = 'Показать кандидата в сцене';
  // Manual candidates require an explicit comparison preview; the accepted base stays intact.
  if (hasManualReconstruction(result) && !result.active) $('show-reconstruction').checked = false;
  $('candidate-note').textContent = result.active ? '' : 'Кандидат не принят для следующих расчётов.';
  $('candidate-note').hidden = result.active;
  $('accept-candidate').hidden = !result.document;
  $('accept-candidate').textContent = result.active ? 'Конфигурация принята' : 'Принять для расчётов';
  $('accept-candidate').disabled = busy || !!candidateBlockReason(result);
  $('accept-reason').textContent = candidateBlockReason(result);
  $('accept-reason').hidden = result.active || !candidateBlockReason(result);
  $('save-source').hidden = $('save-source-label').hidden = result.active || !result.document;
  $('save-source').value = result.active ? 'active' : 'candidate';
  $('active-state').textContent = activeConfiguration ? `Принятая конфигурация · версия ${configurationVersion}` : 'Историческая конфигурация';
  $('use-historical').hidden = !activeConfiguration;
  await drawSceneLayers();
  $('reconstruction-sides').replaceChildren();
  mirrorRecords = [];
  for (const report of reports) {
    const card = document.createElement('div'); card.className = 'reconstruction-side'; card.dataset.status = report.status;
    const title = document.createElement('strong');
    title.textContent = `${sideNames[report.direction] ?? report.direction}: ${reconstructionStatuses[report.status] ?? report.status ?? '—'}`;
    card.append(title);
    const mirrors = report.mirrors ?? [];
    const count = state => mirrors.filter(mirror => (mirror.observation_status ?? mirror.measurement_status ?? mirror.trajectory_status ?? mirror.status) === state).length;
    const counts = report.counts ?? {};
    paragraph(card, report.status === 'legacy_unvalidated' && !report.manual_references
      ? `Зеркал: ${mirrors.length}. Геометрия из исторического источника; независимые первичные измерения к этому импорту не привязаны.`
      : `Зеркал: ${mirrors.length || counts.total || '—'} · полные: ${counts.complete ?? counts.measured ?? count('measured')} · частичные: ${counts.partial ?? counts.single_screen ?? count('partial')} · предсказанные: ${counts.predicted ?? count('predicted')}`);
    if (report.manual_references) {
      paragraph(card, `Источник: ${report.manual_references.source_file ?? '—'}`).style.overflowWrap = 'anywhere';
      const sourceDetails = comparisonDetails(card, 'Происхождение данных');
      paragraph(sourceDetails, `Система: ${report.manual_references.coordinate_frame ?? '—'} · единицы: ${report.manual_references.units ?? '—'}.`);
      paragraph(sourceDetails, 'Ручные ориентиры сохранены отдельно от первичных измерений и не увеличивают число независимых наблюдений.');
      paragraph(sourceDetails, `SHA-256: ${report.manual_references.sha256 ?? '—'}`).style.overflowWrap = 'anywhere';
    }
    const ambiguous = mirrors.filter(mirror => mirror.status === 'ambiguous' || mirror.support_status === 'ambiguous').length;
    const failed = mirrors.filter(mirror => ['failed', 'infeasible'].includes(mirror.status)).length;
    paragraph(card, `Неоднозначные: ${counts.ambiguous ?? ambiguous} · недопустимые: ${counts.failed ?? failed}`);
    const violations = new Map();
    for (const mirror of mirrors) for (const violation of new Set(mirror.violations ?? [])) {
      violations.set(violation, (violations.get(violation) ?? 0) + 1);
    }
    const violationNames = { body_clearance_not_certified: 'Зазор между телами не подтверждён',
      'incoming_field:outside_illumination': 'Вне области освещения',
      'reflection:unavailable': 'Отражённая траектория не получена',
      'reflection:infeasible': 'Проверка отражения не пройдена' };
    for (const [reason, count] of violations) paragraph(card, `${violationNames[reason] ?? reason}: ${count} зеркал.`);
    const heldout = report.diagnostics?.withheld;
    if (heldout?.status === 'evaluated') {
      const heldoutDetails = comparisonDetails(card, 'Контроль со скрытыми измерениями');
      if (heldout.geometric_matching?.status === 'evaluated' && heldout.geometric_matching.angle_tolerance_rad == null) {
        const geometric = heldout.geometric_matching;
        paragraph(heldoutDetails, `Среди предсказаний геометрически совпало ${geometric.matched_count}/${geometric.observation_count} скрытых траекторий в пределах ${format(geometric.distance_tolerance_m * 1000)} мм, без углового отсечения. Это проверка предсказанных прямых; физическое прохождение здесь не проверяется.`);
      } else {
        paragraph(heldoutDetails, 'Сохранённый контроль использовал выбор номера или угловое отсечение. Он не является текущей оценкой совпадения траекторий без учёта номеров. Подробные прежние метрики сохранены в технической диагностике.');
      }
    }
    for (const warning of report.warnings ?? report.limitations ?? report.diagnostics?.limitations ?? []) paragraph(card, warning);
    $('reconstruction-sides').append(card);
    for (const mirror of mirrors) mirrorRecords.push({ direction: report.direction, mirror, displayFrame: report.display_frame, datasetStatus: report.status,
      calculationFrame: report.display_frame?.calculation_frame ?? report.coordinate_frame ?? 'legacy',
      assignment: (Array.isArray(report.assignments) ? report.assignments : []).find(item =>
        item.canonical_id === mirror.canonical_id || item.old_id === mirror.old_id ||
        item.physical_group === mirror.canonical_id?.split('_').slice(0, 2).join('_')) });
  }
  $('mirror-select').replaceChildren(...mirrorRecords.map(({ direction, mirror }, index) => {
    const option = document.createElement('option'); option.value = String(index);
    option.textContent = `${sideNames[direction] ?? direction} · ${mirror.old_id ?? mirror.mirror_id ?? '—'} → ${mirror.canonical_id ?? '—'}`;
    return option;
  }));
  showMirror();
  $('reconstruction-json').textContent = JSON.stringify(metadata(result), null, 2);
  $('reconstruction-result').hidden = false;
  for (const warning of result.warnings ?? []) paragraph($('reconstruction-sides'), warning);
}

const stageLabels = { data: 'Проверка данных', checking_data: 'Проверка данных', observations: 'Проверка данных',
  template: 'Построение шаблона', matching: 'Сопоставление', correspondence: 'Сопоставление',
  fitting: 'Подгонка сборок', assembly_fit: 'Подгонка сборок', individual: 'Индивидуальная подстройка',
  geometry: 'Проверка геометрии', validation: 'Оценка качества', quality: 'Оценка качества' };
bridge?.onProgress(message => {
  if (!busy || message.configuration_version !== configurationVersion) return;
  if (currentJob && currentJob !== message.job_id) return;
  currentJob = message.job_id;
  const progress = message.progress ?? message;
  const stage = progress.label ?? stageLabels[progress.stage] ?? progress.stage ?? 'Вычисление';
  busyText = progress.direction ? `${sideNames[progress.direction] ?? progress.direction} · ${stage}` : stage;
  $('busy-label').textContent = busyText;
  $('progress-stage').textContent = busyText;
  $('reconstruction-progress').hidden = false;
  const total = progress.total;
  const completed = progress.completed ?? progress.current;
  if (Number.isFinite(total) && total > 0 && Number.isFinite(completed)) {
    $('progress-bar').max = total; $('progress-bar').value = completed;
  } else $('progress-bar').removeAttribute('value');
});

async function run(method, payload) {
  if (busy || !ready) return;
  if (!bridge) { notify('Откройте приложение через npm start: требуется соединение Electron с Python.', true); return; }
  notify('');
  currentJob = null;
  const actionLabels = { trace: 'Трассировка лучей', calibrate: 'Обработка измерений', optimize: 'Проверка данных',
    save_configuration: 'Сохранение конфигурации', load_configuration: 'Проверка файла конфигурации',
    import_legacy: 'Проверка исторических данных', import_manual: 'Проверка ручной реконструкции',
    refine_manual: 'Уточнение ручной реконструкции', accept_candidate: 'Применение результата', use_historical: 'Историческая конфигурация' };
  setBusy(true, actionLabels[method]);
  try {
    const result = await bridge.request(method, payload);
    if (result.canceled || result.saved === false) { $('run-status').textContent = 'Действие отменено'; return; }
    configurationVersion = result.configuration_version ?? configurationVersion;
    session[method] = { parameters: payload, ...result };
    if (method === 'trace') {
      $('scene-stale').hidden = true;
      showSummary(result, payload);
      const comparisons = result.comparison_reports ?? [];
      if (comparisons.some(report => report.diagnostics?.geometric_fit?.status === 'evaluated') || !reconstructionResult) {
        renderTrajectoryComparison(comparisons, result.active
          ? 'Источник: принятая реконструкция.'
          : 'Историческая геометрия. Нет реконструкции для оценки подгонки.');
      } else {
        $('trajectory-comparison-context').textContent = 'Сравнение кандидата. Последняя трассировка использует другую конфигурацию.';
      }
      baseScene = { figure: result.figure, frame: result.metadata?.coordinate_frame ?? 'survey',
        direction: payload.direction, label: $('scene-label').textContent };
      if (reconstructionResult?.active) reconstructionScene = null;
      await drawSceneLayers();
      if (result.measurements) {
        session.calibrate = result.measurements;
        await showCalibration(result.measurements);
      } else {
        measurementReports = [];
        delete session.calibrate;
        $('calibration-result').hidden = true;
      }
    } else if (method === 'calibrate') await showCalibration(result);
    else if (method === 'save_configuration') notify(`Конфигурация сохранена${result.path ? ': ' + result.path : ''}`);
    else if (method === 'use_historical') {
      activeConfiguration = false;
      reconstructionResult = null;
      reconstructionScene = null;
      await drawSceneLayers();
      $('reconstruction-result').hidden = true;
      renderTrajectoryComparison([], '');
      $('accept-candidate').hidden = $('accept-reason').hidden = true;
      $('candidate-state').hidden = $('candidate-report').hidden = true;
      $('preview-candidate').hidden = true;
      $('active-state').textContent = 'Историческая конфигурация';
      $('use-historical').hidden = true;
      $('save-source').hidden = $('save-source-label').hidden = true;
      $('save-source').value = 'active';
      markSceneStale('Выбрана историческая конфигурация. На сцене предыдущий результат. Нажмите «Рассчитать лучи» для обновления.');
    } else await showReconstruction(result);
    $('run-status').textContent = method === 'trace' ? 'Конфигурация построена' : method === 'calibrate'
      ? 'Измерения обработаны' : method === 'save_configuration' ? 'Конфигурация сохранена' : 'Реконструкция обработана';
    if (method === 'calibrate') $('calibration-result').scrollIntoView({ block: 'start', behavior: 'smooth' });
  } catch (error) {
    notify(error.message || String(error), true);
    $('run-status').textContent = 'Расчёт не завершён';
  } finally { setBusy(false); $('reconstruction-progress').hidden = true; currentJob = null; }
}

$('trace-form').addEventListener('submit', (event) => {
  event.preventDefault();
  if (!$('trace-form').reportValidity()) return;
  run('trace', { direction: $('direction').value, kind: $('kind').value,
    collision_mode: $('collision-mode').value, ray_count: Number($('ray-count').value),
    max_interactions: Number($('max-interactions').value) });
});
$('trace-form').addEventListener('invalid', event => {
  const details = event.target.closest('details');
  if (details) details.open = true;
}, true);
$('start-trace').addEventListener('click', () => $('trace-form').requestSubmit());
$('start-load').addEventListener('click', () => $('load-configuration').click());
$('calibrate').addEventListener('click', () => run('calibrate', {
  direction: $('direction').value,
}));
const configurationSelection = () => ({ direction: $('direction').value, kind: $('kind').value });
$('optimize').addEventListener('click', () => run(hasManualReconstruction(reconstructionResult) ? 'refine_manual' : 'optimize', {
  ...configurationSelection(),
  ...(hasManualReconstruction(reconstructionResult) ? { direction: Object.keys(reconstructionResult.document.datasets).length > 1
    ? 'both' : Object.keys(reconstructionResult.document.datasets)[0] } : {}),
}));
$('load-manual').addEventListener('click', () => run('import_manual', configurationSelection()));
$('preview-candidate').addEventListener('click', async () => {
  $('show-reconstruction').checked = !$('show-reconstruction').checked;
  $('preview-candidate').textContent = $('show-reconstruction').checked ? 'Скрыть кандидата в сцене' : 'Показать кандидата в сцене';
  await drawSceneLayers();
});
$('save-configuration').addEventListener('click', () => run('save_configuration', {
  ...configurationSelection(), source: $('save-source').value,
}));
$('load-configuration').addEventListener('click', () => run('load_configuration', configurationSelection()));
$('load-legacy').addEventListener('click', () => {
  $('legacy-context').hidden = !$('legacy-context').hidden;
  $('load-legacy').setAttribute('aria-expanded', String(!$('legacy-context').hidden));
});
$('choose-legacy').addEventListener('click', () => run('import_legacy', {
  ...configurationSelection(), coordinate_frame: $('legacy-frame').value,
}));
$('accept-candidate').addEventListener('click', () => run('accept_candidate', { configuration_version: configurationVersion }));
$('use-historical').addEventListener('click', () => run('use_historical', configurationSelection()));
$('mirror-select').addEventListener('change', showMirror);
for (const input of document.querySelectorAll('.scene-layers input')) {
  input.addEventListener('change', () => drawSceneLayers().catch(error => notify(error.message, true)));
}
$('restore-layers').addEventListener('click', () => {
  for (const input of document.querySelectorAll('.scene-layers input')) input.checked = true;
  drawSceneLayers().catch(error => notify(error.message, true));
});
$('toggle-legend').addEventListener('click', async () => {
  legendVisible = !legendVisible;
  $('toggle-legend').setAttribute('aria-pressed', String(legendVisible));
  try { await Plotly.relayout($('scene'), { showlegend: legendVisible }); }
  catch (error) { notify(error.message, true); }
});
for (const button of document.querySelectorAll('[data-result-target]')) button.addEventListener('click', () => {
  const target = $(button.dataset.resultTarget);
  if (target.hidden) return;
  target.focus({ preventScroll: true });
  target.scrollIntoView({ block: 'start', behavior: 'smooth' });
});
function markSceneStale(message = 'Параметры изменены. На сцене предыдущий результат. Нажмите «Рассчитать лучи» для обновления.') {
  if (!baseScene && !reconstructionScene) return;
  $('scene-stale').textContent = message;
  $('scene-stale').hidden = false;
}
for (const id of ['direction', 'kind', 'collision-mode', 'ray-count', 'max-interactions']) $(id).addEventListener('change', () => {
  if (id === 'direction' || id === 'kind') {
    $('calibration-result').hidden = true;
  }
  markSceneStale();
  updateControls();
});
$('cancel').addEventListener('click', async () => {
  try { await bridge.request('cancel'); } catch (error) { notify(error.message, true); }
});
async function setSceneExtent() {
  // Keep every ray inside the axis bounds. A close-up changes the camera,
  // never the clipping box (which previously hid the laser-to-periscope path).
  const scene = $('scene')._fullLayout.scene;
  if (!scene) return;
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
  $('breadcrumb').textContent = { static: 'Статическая конфигурация', dynamic: 'Динамический анализ', distributions: 'Распределения' }[selected];
  if (selected === 'static') {
    if ($('scene').data) Plotly.Plots.resize($('scene'));
    if ($('measurement-plot').data) Plotly.Plots.resize($('measurement-plot'));
  }
});

$('open-help').addEventListener('click', () => {
  $('help-dialog').showModal();
  $('help-dialog').scrollTop = 0;
});

updateControls();
(async () => {
  if (!bridge) {
    $('run-status').textContent = 'Расчёт недоступен';
    notify('Запустите приложение через run.py или npm start.', true);
    return;
  }
  try {
    const catalog = await bridge.request('catalog');
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
    ready = true;
    $('run-status').textContent = '';
    updateControls();
  } catch (error) {
    $('run-status').textContent = 'Нет связи с расчётным модулем';
    notify(error.message, true);
  }
})();
