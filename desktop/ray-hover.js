/* Physical coordinates under the pointer on projected ray segments. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.mirrorRayHover = api;
})(typeof window === 'undefined' ? globalThis : window, function () {
  'use strict';

  function closestPoint(point, start, end) {
    const dx = end[0] - start[0], dy = end[1] - start[1];
    const lengthSquared = dx * dx + dy * dy;
    const t = lengthSquared ? Math.max(0, Math.min(1,
      ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared)) : 0;
    const x = start[0] + t * dx, y = start[1] + t * dy;
    return { t, x, y, distanceSquared: (point[0] - x) ** 2 + (point[1] - y) ** 2 };
  }

  function attach(plot) {
    const doc = plot.ownerDocument, win = doc.defaultView;
    const tooltip = doc.createElement('div');
    tooltip.className = 'mirror-ray-tooltip';
    tooltip.setAttribute('role', 'tooltip');
    Object.assign(tooltip.style, { position: 'fixed', pointerEvents: 'none', zIndex: '10000',
      whiteSpace: 'pre-line', background: '#fff', color: '#303136', border: '1px solid #77717e',
      borderRadius: '6px', padding: '8px 10px', boxShadow: '0 2px 10px #0002',
      font: '12px/1.5 "Segoe UI", sans-serif', maxWidth: '350px' });
    tooltip.hidden = true;
    doc.body.appendChild(tooltip);
    let segments = [], pointer = null, pending = null, disposed = false;
    const numeric = value => typeof value === 'number' && Number.isFinite(value);
    const physicalPoint = row => Array.isArray(row) && row.slice(0, 3).length === 3 && row.slice(0, 3).every(numeric);
    const axisKey = (value, name) => !value || value === name ? `${name}axis` : `${name}axis${value.slice(1)}`;

    function rebuild() {
      if (disposed) return;
      segments = [];
      const layout = plot._fullLayout;
      if (!layout) return;
      for (const [index, trace] of (plot.data || []).entries()) {
        const full = plot._fullData?.[index] || trace;
        if (trace.visible === false || trace.visible === 'legendonly' || full.visible === false ||
            full.visible === 'legendonly' || !trace.meta?.mirror_id || !Array.isArray(trace.customdata)) continue;
        const xa = layout[axisKey(full.xaxis, 'x')], ya = layout[axisKey(full.yaxis, 'y')];
        if (!xa?.l2p || !ya?.l2p) continue;
        for (let i = 0; i < trace.x.length - 1; i++) {
          const a = trace.customdata[i], b = trace.customdata[i + 1];
          if (!physicalPoint(a) || !physicalPoint(b) ||
              ![trace.x[i], trace.y[i], trace.x[i + 1], trace.y[i + 1]].every(numeric)) continue;
          const start = [xa._offset + xa.l2p(trace.x[i]), ya._offset + ya.l2p(trace.y[i])];
          const end = [xa._offset + xa.l2p(trace.x[i + 1]), ya._offset + ya.l2p(trace.y[i + 1])];
          if (![...start, ...end].every(numeric)) continue;
          segments.push({ start, end, a, b, meta: trace.meta, name: trace.name, xa, ya });
        }
      }
      schedule();
    }

    function render() {
      pending = null;
      if (disposed || !pointer) return;
      tooltip.hidden = true;
      const rect = plot.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      // Plotly's axis transforms use layout pixels; pointer coordinates use CSS pixels.
      const scaleX = rect.width / (plot._fullLayout?.width || rect.width);
      const scaleY = rect.height / (plot._fullLayout?.height || rect.height);
      const local = [pointer.x - rect.left, pointer.y - rect.top];
      let nearest = null, distanceSquared = 8 * 8;
      for (const segment of segments) {
        const { xa, ya } = segment;
        if (local[0] < xa._offset * scaleX || local[0] > (xa._offset + xa._length) * scaleX ||
            local[1] < ya._offset * scaleY || local[1] > (ya._offset + ya._length) * scaleY) continue;
        const hit = closestPoint(local, [segment.start[0] * scaleX, segment.start[1] * scaleY],
          [segment.end[0] * scaleX, segment.end[1] * scaleY]);
        if (hit.distanceSquared <= distanceSquared) {
          distanceSquared = hit.distanceSquared;
          nearest = { segment, t: hit.t };
        }
      }
      if (!nearest) return;
      const { segment, t } = nearest, { a, b, meta } = segment;
      const xyz = a.slice(0, 3).map((value, index) => value + t * (b[index] - value));
      const number = value => value.toLocaleString('ru-RU', { minimumFractionDigits: 7, maximumFractionDigits: 7 });
      tooltip.textContent = `Луч ${a[4] ?? meta.mirror_id}\n` +
        `Стержень ${a[5] ?? meta.rod} · бандл ${a[6] ?? meta.bundle} · зеркало ${a[7] ?? meta.mirror}\n` +
        `ID трассировки: ${a[3]}\n` +
        `X = ${number(xyz[0])} м\nY = ${number(xyz[1])} м\nZ = ${number(xyz[2])} м`;
      tooltip.hidden = false;
      tooltip.style.left = `${Math.max(4, Math.min(pointer.x + 14, win.innerWidth - tooltip.offsetWidth - 6))}px`;
      tooltip.style.top = `${Math.max(4, Math.min(pointer.y + 14, win.innerHeight - tooltip.offsetHeight - 6))}px`;
    }

    function schedule() {
      if (!disposed && pointer && pending === null) pending = win.requestAnimationFrame(render);
    }
    function leave() {
      pointer = null;
      tooltip.hidden = true;
      if (pending !== null) win.cancelAnimationFrame(pending);
      pending = null;
    }
    function move(event) {
      if (event.buttons) { leave(); return; }
      pointer = { x: event.clientX, y: event.clientY };
      schedule();
    }
    plot.addEventListener('pointermove', move);
    plot.addEventListener('pointerleave', leave);
    plot.addEventListener('pointerdown', leave);
    plot.on('plotly_afterplot', rebuild);
    plot.on('plotly_relayout', rebuild);
    rebuild();
    return function cleanup() {
      if (disposed) return;
      disposed = true;
      leave();
      plot.removeEventListener('pointermove', move);
      plot.removeEventListener('pointerleave', leave);
      plot.removeEventListener('pointerdown', leave);
      plot.removeListener('plotly_afterplot', rebuild);
      plot.removeListener('plotly_relayout', rebuild);
      tooltip.remove();
      segments = [];
    };
  }
  return { closestPoint, attach };
});
