/* Compose fresh layers, never append to Plotly's already composed data. */
function composeScene(base, reconstruction, measurements = [], visible = {}) {
  const compare = base && reconstruction && base.frame === 'survey' && reconstruction.frame === 'survey';
  const primary = reconstruction && !compare && (visible.reconstruction !== false || !base) ? reconstruction : base || reconstruction;
  if (!primary) return null;
  const data = [];
  const add = (traces, source, label = '') => traces.forEach((trace, index) => {
    // Side means the injection side, consistently with the direction selector.
    const side = trace.meta?.side;
    const layer = trace.meta?.layer;
    if (visible.sides?.[side] === false || visible.layers?.[layer] === false) return;
    data.push({
      ...trace, uid: `${source.replace(/[^a-zA-Z0-9_-]/g, '-')}-${index}`,
      name: label + (trace.name || ''),
      legendgroup: `${source}:${trace.legendgroup || index}`,
      meta: { ...trace.meta, scene_source: source },
      ...(trace.meta?.kind === 'survey' && visible.layers?.measurement_points === true ? { visible: true } : {}),
    });
  });
  if (primary === base && visible.base !== false) add(base.figure.data, 'base');
  if (reconstruction && visible.reconstruction !== false) {
    const traces = compare ? reconstruction.figure.data.filter(trace => trace.meta?.mirror_id) : reconstruction.figure.data;
    add(traces, 'reconstruction', 'Реконструкция · ');
  }
  if (primary.frame === 'survey' && visible.measurements !== false) {
    for (const report of measurements) {
      const included = primary.direction === 'both' || primary.direction === report.direction ||
        (compare && (reconstruction.direction === 'both' || reconstruction.direction === report.direction));
      if (included) add(report.figure?.data || [], `measurements:${report.direction}`, `${report.side} · Измерения · `);
    }
  }
  return { figure: { ...primary.figure, data }, frame: primary.frame,
    label: compare && visible.base !== false && visible.reconstruction !== false
      ? `${base.label} + реконструкция` : visible.base === false && reconstruction ? reconstruction.label : primary.label,
    note: base && reconstruction && !compare && visible.base !== false && visible.reconstruction !== false
      ? 'Совместный просмотр недоступен: общая геодезическая система координат не установлена. Отключите слой реконструкции, чтобы увидеть сохранённую построенную сцену.'
      : '',
  };
}

if (typeof module !== 'undefined') module.exports = { composeScene };
