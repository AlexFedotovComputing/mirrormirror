const { test } = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { closestPoint, attach } = require('../desktop/ray-hover.js');

test('nearest point interpolates the interior and clamps outside endpoints', () => {
  assert.deepEqual(closestPoint([30, 14], [10, 10], [50, 10]), { t: 0.5, x: 30, y: 10, distanceSquared: 16 });
  assert.equal(closestPoint([0, 10], [10, 10], [50, 10]).t, 0);
  assert.equal(closestPoint([80, 10], [10, 10], [50, 10]).t, 1);
  assert.deepEqual(closestPoint([13, 14], [10, 10], [10, 10]), { t: 0, x: 10, y: 10, distanceSquared: 25 });
});

function fixture() {
  const frames = new Map();
  let nextFrame = 0;
  const tooltip = { style: {}, hidden: true, offsetWidth: 100, offsetHeight: 100,
    setAttribute() {}, remove() { this.removed = true; } };
  const win = { innerWidth: 1000, innerHeight: 1000,
    requestAnimationFrame(callback) { frames.set(++nextFrame, callback); return nextFrame; },
    cancelAnimationFrame(id) { frames.delete(id); } };
  const doc = { defaultView: win, body: { appendChild() {} }, createElement() { return tooltip; } };
  const plot = new EventEmitter();
  plot.ownerDocument = doc;
  plot.addEventListener = plot.on.bind(plot);
  plot.removeEventListener = plot.removeListener.bind(plot);
  plot.getBoundingClientRect = () => ({ left: 100, top: 200, width: 400, height: 300 });
  plot._fullLayout = { width: 400, height: 300,
    xaxis: { _offset: 50, _length: 300, l2p: value => value * 100 },
    yaxis: { _offset: 30, _length: 200, l2p: value => 200 - value * 100 } };
  plot.data = [{ x: [0, 2, null], y: [1, 1, null], name: 'Бандл 2_3',
    meta: { mirror_id: '2_3_4', rod: 2, bundle: 3, mirror: 4 },
    customdata: [[10, 20, 1, 42, '2_3_4', 2, 3, 4], [12, 24, 3, 42, '2_3_4', 2, 3, 4], null] }];
  const flush = () => { const callbacks = [...frames.values()]; frames.clear(); callbacks.forEach(callback => callback()); };
  return { plot, tooltip, frames, flush };
}

test('pointer hover reports interpolated physical XYZ and rebuilds after axis changes', () => {
  const { plot, tooltip, flush } = fixture();
  const cleanup = attach(plot);
  plot.emit('pointermove', { clientX: 250, clientY: 333, buttons: 0 }); flush();
  assert.equal(tooltip.hidden, false);
  assert.match(tooltip.textContent, /Луч 2_3_4/);
  assert.match(tooltip.textContent, /бандл 3/);
  assert.match(tooltip.textContent, /ID трассировки: 42/);
  assert.match(tooltip.textContent, /X = 11,0000000 м/);
  assert.match(tooltip.textContent, /Y = 22,0000000 м/);
  assert.match(tooltip.textContent, /Z = 2,0000000 м/);
  plot._fullLayout.xaxis.l2p = value => value * 200;
  plot.emit('plotly_relayout'); flush();
  assert.match(tooltip.textContent, /Z = 1,5000000 м/);
  cleanup();
});

test('threshold, hidden traces and cleanup prevent stale tooltips', () => {
  const { plot, tooltip, frames, flush } = fixture();
  const cleanup = attach(plot);
  plot.emit('pointermove', { clientX: 250, clientY: 339, buttons: 0 }); flush();
  assert.equal(tooltip.hidden, true);
  plot.data[0].visible = 'legendonly'; plot.emit('plotly_afterplot');
  plot.emit('pointermove', { clientX: 250, clientY: 330, buttons: 0 }); flush();
  assert.equal(tooltip.hidden, true);
  plot.data[0].visible = true; plot.emit('plotly_afterplot'); flush();
  assert.equal(tooltip.hidden, false);
  plot.emit('pointermove', { clientX: 251, clientY: 330, buttons: 0 });
  assert.equal(frames.size, 1);
  cleanup(); cleanup();
  assert.equal(frames.size, 0);
  assert.equal(tooltip.removed, true);
  assert.deepEqual(plot.eventNames(), []);
});
