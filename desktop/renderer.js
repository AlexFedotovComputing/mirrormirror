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
let configurationVersion = 0;
let activeConfiguration = false;
let reconstructionResult = null;
let mirrorRecords = [];
let currentJob = null;
let busyText = '';
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
  for (const id of ['optimize', 'save-configuration', 'load-configuration', 'load-legacy',
    'choose-legacy', 'accept-candidate', 'use-historical', 'save-source', 'legacy-frame']) $(id).disabled = busy;
  $('kind').disabled = busy || activeConfiguration;
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
}

function metadata(result) {
  const { figure, ...rest } = result;
  return rest;
}

function showCoordinateFrame(frame) {
  const survey = frame === 'survey';
  $('coordinate-title').textContent = survey ? 'Геодезические XYZ · метры' : 'Расчётная сцена · метры';
  $('coordinate-note').textContent = survey ? 'Знак Z — по измерениям. Центральная плоскость объёма: z = 0.'
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
  $('reset-camera').disabled = false;
  $('scene-extent').disabled = false;
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
    ['Статус', basisStatuses[mirror.status] ?? mirror.status],
    ['Наблюдение', selected.datasetStatus === 'legacy_unvalidated'
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
  $('metrics').replaceChildren();
  $('result-details').hidden = true;
  $('calibration-result').hidden = true;
  measurementReports = [];
  measurementContext = null;
  configurationVersion = result.configuration_version ?? configurationVersion;
  if (result.active) activeConfiguration = true;
  const reports = reconstructionReports(result);
  if (result.active && reports.length === 1) $('direction').value = reports[0].direction;
  if (result.figure) await drawScene(result.figure);
  if (result.figure) showCoordinateFrame(result.coordinate_frame ??
    (reports.every(report => report.display_frame?.coordinate_frame === 'survey') ? 'survey' : 'legacy'));
  const status = reconstructionStatuses[result.status] ?? result.status ?? 'Диагностический результат';
  $('reconstruction-status').textContent = status;
  const missingOptics = reports.some(report => report.optics_snapshot === null);
  $('candidate-note').textContent = result.active
    ? 'Эта конфигурация принята. Построение сцены и сохранение используют её геометрию и оптику.'
    : result.can_activate === false && result.status !== 'infeasible'
      ? missingOptics ? 'Кандидат неполон: общая оптика не установлена. Доступны просмотр геометрии и сохранение.'
        : 'В конфигурации есть недопустимые части. Причины показаны отдельно для каждой стороны; доступны просмотр и сохранение кандидата.'
    : result.figure ? 'Показан диагностический кандидат. Принятая конфигурация сохранена; непроверенные условия перечислены в отчёте.'
      : 'Импорт доступен как данные для просмотра и сохранения. Полная оптическая сцена не определена; текущая сцена сохранена.';
  $('accept-candidate').hidden = result.active || !result.document || result.status === 'infeasible' || result.can_activate === false;
  $('accept-candidate').textContent = result.status === 'legacy_unvalidated' ? 'Использовать исторический импорт' : 'Использовать частичный результат';
  $('save-source').hidden = $('save-source-label').hidden = result.active || !result.document;
  $('save-source').value = result.active ? 'active' : 'candidate';
  $('active-state').textContent = activeConfiguration ? `Принятая конфигурация · версия ${configurationVersion}` : 'Историческая конфигурация';
  $('use-historical').hidden = !activeConfiguration;
  if (result.figure) $('scene-label').textContent = `${(result.scene_direction
    ? [sideNames[result.scene_direction] ?? result.scene_direction]
    : reports.map(report => sideNames[report.direction] ?? report.direction)).join(' и ')} · ${result.active ? 'принятая реконструкция' : 'диагностический кандидат'}`;
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
    paragraph(card, report.status === 'legacy_unvalidated'
      ? `Зеркал: ${mirrors.length}. Геометрия из исторического источника; независимые первичные измерения к этому импорту не привязаны.`
      : `Зеркал: ${mirrors.length || counts.total || '—'} · полные: ${counts.complete ?? counts.measured ?? count('measured')} · частичные: ${counts.partial ?? counts.single_screen ?? count('partial')} · предсказанные: ${counts.predicted ?? count('predicted')}`);
    const ambiguous = mirrors.filter(mirror => mirror.status === 'ambiguous' || mirror.support_status === 'ambiguous').length;
    const failed = mirrors.filter(mirror => ['failed', 'infeasible'].includes(mirror.status)).length;
    paragraph(card, `Неоднозначные: ${counts.ambiguous ?? ambiguous} · недопустимые: ${counts.failed ?? failed}`);
    const heldout = report.diagnostics?.withheld;
    if (heldout?.status === 'evaluated') {
      paragraph(card, `Скрытые лучи: ${heldout.predicted}/${heldout.planned}; медианная угловая ошибка ${format(heldout.angular_error_mrad?.median)} мрад, 90-й процентиль ${format(heldout.angular_error_mrad?.p90)} мрад. Жёсткий эмпирический шаблон: ${format(heldout.rigid_empirical_error_mrad?.median)} мрад.`);
      if (heldout.conditional_interval_coverage != null) paragraph(card,
        `Покрытие условных 95% интервалов: ${format(heldout.conditional_interval_coverage * 100)}%. Оценка по малой выборке; метрологическая точность не подтверждена.`);
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
  showSideStatus();
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
  if (busy) return;
  if (!bridge) { notify('Откройте приложение через npm start: требуется соединение Electron с Python.', true); return; }
  notify('');
  currentJob = null;
  const actionLabels = { trace: 'Трассировка лучей', calibrate: 'Обработка измерений', optimize: 'Проверка данных',
    save_configuration: 'Сохранение конфигурации', load_configuration: 'Проверка файла конфигурации',
    import_legacy: 'Проверка исторических данных', accept_candidate: 'Применение результата', use_historical: 'Историческая конфигурация' };
  setBusy(true, actionLabels[method]);
  try {
    const result = await bridge.request(method, payload);
    if (result.canceled || result.saved === false) { $('run-status').textContent = 'Действие отменено'; return; }
    configurationVersion = result.configuration_version ?? configurationVersion;
    session[method] = { parameters: payload, ...result };
    if (method === 'trace') {
      await drawScene(result.figure);
      showSummary(result, payload);
      if (reconstructionResult && !reconstructionResult.active) {
        $('candidate-note').textContent = 'Отчёт ниже относится к диагностическому кандидату. Сцена показывает текущую принятую конфигурацию.';
      }
      if (result.measurements) {
        session.calibrate = result.measurements;
        await showCalibration(result.measurements);
      } else {
        measurementContext = null;
        measurementReports = [];
        delete session.calibrate;
        $('calibration-result').hidden = true;
      }
    } else if (method === 'calibrate') await showCalibration(result);
    else if (method === 'save_configuration') notify(`Конфигурация сохранена${result.path ? ': ' + result.path : ''}`);
    else if (method === 'use_historical') {
      activeConfiguration = false;
      reconstructionResult = null;
      $('reconstruction-result').hidden = true;
      $('active-state').textContent = 'Историческая конфигурация';
      $('use-historical').hidden = true;
      $('save-source').hidden = $('save-source-label').hidden = true;
      $('save-source').value = 'active';
      notify('Выбрана историческая конфигурация. Нажмите «Построить конфигурацию» для обновления сцены.');
    } else await showReconstruction(result);
    $('run-status').textContent = method === 'trace' ? 'Конфигурация построена' : method === 'calibrate'
      ? 'Измерения обработаны' : method === 'save_configuration' ? 'Конфигурация сохранена' : 'Реконструкция обработана';
    $('connection-label').textContent = 'Расчётный модуль подключён';
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
$('calibrate').addEventListener('click', () => run('calibrate', {
  direction: $('direction').value,
}));
const configurationSelection = () => ({ direction: $('direction').value, kind: $('kind').value });
$('optimize').addEventListener('click', () => run('optimize', configurationSelection()));
$('save-configuration').addEventListener('click', () => run('save_configuration', {
  ...configurationSelection(), source: $('save-source').value,
}));
$('load-configuration').addEventListener('click', () => run('load_configuration', configurationSelection()));
$('load-legacy').addEventListener('click', () => { $('legacy-context').hidden = !$('legacy-context').hidden; });
$('choose-legacy').addEventListener('click', () => run('import_legacy', {
  ...configurationSelection(), coordinate_frame: $('legacy-frame').value,
}));
$('accept-candidate').addEventListener('click', () => run('accept_candidate', { configuration_version: configurationVersion }));
$('use-historical').addEventListener('click', () => run('use_historical', configurationSelection()));
$('mirror-select').addEventListener('change', showMirror);
for (const id of ['direction', 'kind', 'collision-mode']) $(id).addEventListener('change', () => {
  if (id !== 'collision-mode') {
    measurementContext = null;
    $('calibration-result').hidden = true;
    if ($('scene').data) notify('Параметры изменены. Сцена показывает последний результат; нажмите «Построить конфигурацию», чтобы применить выбранные параметры.');
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
