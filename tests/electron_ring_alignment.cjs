/* Focused UI QA for the temporary WEST lock and physical EAST ring correction.
 * Launches Electron; run only with the normal GUI execution approval.
 */
const path = require('node:path');
const fs = require('node:fs/promises');
const assert = require('node:assert/strict');
const { _electron } = require('playwright-core');

(async () => {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'artifacts', 'ui', 'ring-alignment');
  await fs.mkdir(output, { recursive: true });
  const env = { ...process.env, MIRRORLAB_HEADLESS: '1', MIRRORLAB_USER_DATA: path.join(output, 'profile') };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({
    executablePath: path.join(root, 'node_modules/electron/dist/electron.exe'), args: [root], cwd: root, env,
  });
  try {
    const page = await app.firstWindow();
    page.setDefaultTimeout(120000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false));
    await page.waitForFunction(() => !document.querySelector('#controls').disabled &&
      document.querySelector('#run-status').textContent !== 'Запуск…');

    const catalog = await page.evaluate(() => window.mirrorlab.request('catalog'));
    assert.deepEqual(catalog.directions.map(d => d.id), ['west_to_east']);
    assert.equal(catalog.defaults.direction, 'west_to_east');
    assert.deepEqual(await page.locator('#direction option').evaluateAll(options => options.map(o => o.value)), ['west_to_east']);
    assert.equal(await page.locator('#direction').inputValue(), 'west_to_east');
    const rejected = await page.evaluate(async () => {
      try { await window.mirrorlab.request('trace', { direction: 'east_to_west', kind: 'ideal',
        collision_mode: 'expected', ray_count: 1, cpu_workers: 1, max_interactions: 48 }); return ''; }
      catch (error) { return error.message; }
    });
    assert.match(rejected, /временно отключён/);

    await page.selectOption('#kind', 'ideal');
    await page.fill('#ray-count', '217');
    await page.fill('#cpu-workers', '1');
    await page.click('#run');
    await page.waitForFunction(() => !document.querySelector('#controls').disabled &&
      ['Конфигурация построена', 'Расчёт не завершён'].includes(document.querySelector('#run-status').textContent));
    assert.equal(await page.locator('#run-status').innerText(), 'Конфигурация построена', await page.locator('#notice').innerText());

    const report = await page.evaluate(() => {
      const result = JSON.parse(document.querySelector('#result-json').textContent);
      const traces = document.querySelector('#optical-scene').data;
      const center = name => {
        const trace = traces.find(t => t.type === 'mesh3d' && (t.name === name || t.name?.endsWith(' · ' + name)));
        if (!trace) throw new Error('Missing optical mesh: ' + name);
        // The circular mesh includes its repeated closing vertex; omit it.
        return ['x', 'y', 'z'].map(axis => trace[axis].slice(0, -1).reduce((a, b) => a + b, 0) / (trace[axis].length - 1));
      };
      const mp1 = center('MP1'), mp2 = center('MP2');
      const laser = traces.find(t => t.name === 'Лазер' || t.name?.startsWith('Лазер WEST'));
      if (!laser) throw new Error('Missing source marker');
      const source = ['x', 'y', 'z'].map(axis => laser[axis][0]);
      const rotation = result.metadata.frame_registration.rotation;
      const nativeVector = vector => [0, 1, 2].map(i => vector.reduce((sum, value, j) => sum + rotation[j][i] * value, 0));
      return {
        request: result.request,
        summary: result.summary,
        sides: result.sides.map(s => s.direction),
        directions: [...new Set(traces.map(t => t.meta?.direction).filter(Boolean))],
        injectionSides: [...new Set(traces.map(t => t.meta?.side).filter(Boolean))],
        mp1, mp2, source, rotation,
        periscopeNative: nativeVector(mp1.map((v, i) => v - mp2[i])),
        sourceOffsetNative: nativeVector(source.map((v, i) => v - mp1[i])),
      };
    });
    assert.equal(report.request.direction, 'west_to_east');
    assert.equal(report.summary.ray_count, 217);
    assert.deepEqual(report.sides, ['west_to_east']);
    assert.deepEqual(report.directions, ['west_to_east']);
    assert.deepEqual(report.injectionSides, ['EAST']);
    assert.ok(report.summary.segment_count > 217, 'Trace must pass the source entrance');
    assert.ok(Math.abs(report.periscopeNative[0]) < 1e-10, JSON.stringify(report.periscopeNative));
    assert.ok(report.periscopeNative[1] < -.01, 'Actual MP2 → MP1 must point down in the native frame');
    assert.ok(Math.hypot(...report.sourceOffsetNative.slice(0, 2)) < 1e-5,
      'Laser entrance must retain its original micrometre alignment with MP1');
    await fs.writeFile(path.join(output, 'checks.json'), JSON.stringify(report, null, 2) + '\n');

    await page.locator('#scene-card').scrollIntoViewIfNeeded();
    const capture = async name => {
      // Plotly resolves relayout before the next WebGL paint in a hidden window.
      await page.waitForTimeout(250);
      const png = await app.evaluate(async ({ BrowserWindow }) =>
        (await BrowserWindow.getAllWindows()[0].webContents.capturePage(undefined, { stayHidden: true })).toPNG().toString('base64'));
      await fs.writeFile(path.join(output, name), Buffer.from(png, 'base64'));
    };
    await capture('east-ring-3d.png');
    await page.evaluate(async rotation => {
      const transform = vector => rotation.map(row => row.reduce((sum, value, i) => sum + value * vector[i], 0));
      const xyz = v => ({ x: v[0], y: v[1], z: v[2] });
      await Plotly.relayout(document.querySelector('#optical-scene'), {
        'scene.camera': { eye: xyz(transform([0, 0, -2.5])), up: xyz(transform([0, 1, 0])),
          center: { x: 0, y: 0, z: 0 }, projection: { type: 'perspective' } },
      });
    }, report.rotation);
    await capture('east-ring-end-view.png');
    await page.selectOption('#kind', 'real');
    await page.click('#run');
    await page.waitForFunction(() => !document.querySelector('#controls').disabled &&
      ['Конфигурация построена', 'Расчёт не завершён'].includes(document.querySelector('#run-status').textContent));
    assert.equal(await page.locator('#run-status').innerText(), 'Конфигурация построена', await page.locator('#notice').innerText());
    const real = await page.evaluate(() => JSON.parse(document.querySelector('#result-json').textContent));
    assert.equal(real.metadata.ring_alignment.illuminated_mirrors_after, 112);
    assert.equal(real.metadata.ring_alignment.footprint_recentering.length, 2);
    assert.ok(real.warnings.some(w => w.includes('MS/MC')));
    await capture('east-ring-real.png');
    const endcap = await page.evaluate(() => session.trace.visualization.sides[0].endcap);
    assert.equal(endcap.view_from_side, 'EAST');
    assert.equal(endcap.view_from_designation, '+');
    assert.ok(endcap.sign * endcap.camera_position_m[2] > endcap.sign * endcap.plane_z_m);
    await page.click('.nav-button[data-tab="distributions"]');
    await page.click('[data-viz-view="endcaps"]');
    await page.waitForFunction(() => document.querySelector('#viz-endcaps .js-plotly-plot')?.data?.length > 0);
    assert.equal(await page.locator('#viz-endcaps h3').innerText(), 'EAST · вид снаружи со стороны EAST (+)');
    await page.selectOption('#viz-rod', '2');
    await page.selectOption('#viz-bundle', '3');
    await page.waitForFunction(() => {
      const traces = document.querySelector('#viz-endcaps .js-plotly-plot')?.data || [];
      const measured = traces.filter(t => t.meta?.layer === 'experimental_points');
      return measured.length > 0 && measured.every(t => t.meta.rod === 2 && t.meta.bundle === 3);
    });
    const measured = await page.locator('#viz-endcaps .js-plotly-plot').evaluate(p =>
      p.data.filter(t => t.meta?.layer === 'experimental_points').flatMap(t => t.customdata));
    for (const id of ['2 2 4', '2 2 5']) assert.ok(measured.some(p => p[3] === id && p[4] === 2));
    await page.uncheck('#viz-roc-measurements');
    await page.waitForFunction(() => document.querySelector('#viz-endcaps .js-plotly-plot')?.data.every(t => t.meta?.layer !== 'experimental_points'));
    await page.check('#viz-roc-measurements');
    await page.check('#viz-roc-point-labels');
    await page.waitForFunction(() => document.querySelector('#viz-endcaps .js-plotly-plot')?.data.some(t => t.meta?.layer === 'experimental_points' && t.mode === 'markers+text'));
    const endcapImage = await page.locator('#viz-endcaps .js-plotly-plot').evaluate(
      p => Plotly.toImage(p, {format:'png', width:850, height:850}));
    await fs.writeFile(path.join(output, 'east-outside-endcap.png'), Buffer.from(endcapImage.split(',')[1], 'base64'));
    assert.deepEqual(errors, []);
    console.log('EAST-only UI, blocked WEST requests, actual downward periscope and source alignment passed.');
    console.log('Artifacts: ' + output);
  } finally { await app.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
