/* Run with node tests/electron_smoke.cjs; requires playwright-core for UI QA. */
const path = require('node:path');
const fs = require('node:fs/promises');
const assert = require('node:assert/strict');
const { _electron } = require('playwright-core');

(async () => {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'artifacts', 'ui');
  await fs.mkdir(output, { recursive: true });
  const errors = [];
  const env = { ...process.env, MIRRORLAB_HEADLESS: '1', MIRRORLAB_USER_DATA: path.join(output, 'profile') };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({
    executablePath: path.join(root, 'node_modules', 'electron', 'dist', 'electron.exe'),
    args: [root], cwd: root,
    env,
  });
  try {
    const page = await app.firstWindow();
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false));
    page.on('pageerror', error => errors.push(error.message));
    await page.waitForFunction(() => document.querySelector('#connection-label')?.textContent.includes('подключён'));
    assert.equal(await page.evaluate(() => typeof window.require), 'undefined');
    assert.equal(await page.locator('#direction').inputValue(), 'both');
    assert.equal(await page.locator('#use-lavsan').count(), 0);
    await page.locator('#run').click();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent === 'Конфигурация построена');
    assert.ok(await page.locator('#scene .gl-container').count());
    const plotted = await page.evaluate(() => {
      const plot = document.querySelector('#scene');
      return { axis: plot._fullLayout.scene.xaxis.title.text,
        scale: plot._fullData.find(trace => trace.marker?.showscale)?.marker.colorscale,
        dragmode: plot._fullLayout.scene.dragmode, aspectmode: plot._fullLayout.scene.aspectmode,
        projection: plot._fullLayout.scene.camera.projection.type };
    });
    assert.equal(plotted.axis, 'x, м');
    assert.equal(plotted.scale[0][1], '#4c416f');
    assert.equal(plotted.dragmode, 'orbit');
    assert.equal(plotted.aspectmode, 'manual');
    const axisScales = await page.evaluate(() => {
      const scene = document.querySelector('#scene')._fullLayout.scene;
      return ['x', 'y', 'z'].map(axis => scene.aspectratio[axis] /
        Math.abs(scene[axis + 'axis'].range[1] - scene[axis + 'axis'].range[0]));
    });
    assert.ok(Math.max(...axisScales) - Math.min(...axisScales) < 1e-12);
    assert.equal(plotted.projection, 'perspective');
    const sources = await page.evaluate(() => document.querySelector('#scene').data
      .filter(trace => trace.name?.includes('Лазер')).map(trace => trace.name));
    assert.deepEqual(sources.sort(), ['Лазер EAST → ввод WEST', 'Лазер WEST → ввод EAST']);
    const inwardSegments = await page.evaluate(() => {
      const counts = { WEST: 0, EAST: 0 };
      for (const trace of document.querySelector('#scene').data) {
        if (!trace.name?.endsWith('· Лучи')) continue;
        for (let i = 0; i < trace.x.length; i += 3) {
          if (Math.hypot(trace.x[i], trace.y[i]) > 1.2 &&
              Math.hypot(trace.x[i + 1], trace.y[i + 1]) < 1.0) {
            counts[trace.z[i] > 0 ? 'WEST' : 'EAST']++;
          }
        }
      }
      return counts;
    });
    assert.ok(inwardSegments.WEST > 0 && inwardSegments.EAST > 0, JSON.stringify(inwardSegments));
    const coordinates = await page.evaluate(() => {
      const data = document.querySelector('#scene').data;
      return {
        lasers: data.filter(t => t.name?.includes('Лазер')).map(t => ({ name: t.name, z: t.z[0] })),
        westSurvey: data.find(t => t.name === 'WEST · Геодезические точки').z,
        eastSurvey: data.find(t => t.name === 'EAST · Геодезические точки').z,
        prismZ: data.filter(t => t.name?.endsWith('PR3:top')).map(t => t.z[0]),
      };
    });
    assert.ok(coordinates.lasers.find(t => t.name.includes('Лазер EAST')).z < 0);
    assert.ok(coordinates.lasers.find(t => t.name.includes('Лазер WEST')).z > 0);
    assert.ok(coordinates.westSurvey.every(z => z > 0) && coordinates.eastSurvey.every(z => z < 0));
    assert.ok(Math.min(...coordinates.prismZ) < -1.7 && Math.max(...coordinates.prismZ) > 1.7);
    assert.equal(await page.locator('#export, .local-badge, #kind-note, #collision-note, #coverage-label').count(), 0);
    const layout = await page.evaluate(() => ({
      top: document.querySelector('#scene').getBoundingClientRect().top,
      height: document.querySelector('#scene').clientHeight,
      viewport: window.innerHeight,
      labels: [...document.querySelector('#direction').options].map(option => option.textContent),
      range: document.querySelector('#scene')._fullLayout.scene.zaxis.range,
    }));
    assert.ok(layout.top < 190 && layout.height > layout.viewport * .7, JSON.stringify(layout));
    assert.ok(layout.labels.every(label => !/[.·]/.test(label)));
    assert.ok(layout.range[0] < -5.08537 && layout.range[1] > 5.08537);
    const laserPathsVisible = await page.evaluate(() => {
      const plot = document.querySelector('#scene');
      const lines = plot.data.filter(t => t.name?.endsWith('· Лучи'));
      return plot.data.filter(t => t.name?.includes('Лазер')).every(laser =>
        ['x', 'y', 'z'].every(a => laser[a][0] >= plot._fullLayout.scene[a + 'axis'].range[0] &&
          laser[a][0] <= plot._fullLayout.scene[a + 'axis'].range[1]) &&
        lines.some(t => t.z.some((z, i) => i % 3 === 0 && Math.hypot(
          t.x[i] - laser.x[0], t.y[i] - laser.y[0], z - laser.z[0]) < .02)));
    });
    assert.ok(laserPathsVisible, 'Laser-to-periscope paths must be present inside the visible bounds');
    await page.locator('#scene-extent').click();
    await page.waitForFunction(() => document.querySelector('#scene-extent').textContent === 'Вся схема' &&
      document.querySelector('#scene')._fullLayout.scene.zaxis.range[1] > 5);
    assert.deepEqual(await page.evaluate(() => document.querySelector('#scene')._fullLayout.scene.zaxis.range), layout.range);
    assert.equal(await page.evaluate(() => document.querySelector('#scene')._fullLayout.scene.aspectmode), 'manual');
    await page.locator('#reset-camera').click();
    await page.waitForFunction(() => document.querySelector('#scene-extent').textContent === 'К объёму');
    assert.deepEqual(await page.evaluate(() => document.querySelector('#scene')._fullLayout.scene.zaxis.range), layout.range);
    assert.equal(await page.evaluate(() => document.querySelector('#scene')._fullLayout.scene.aspectmode), 'manual');
    await page.screenshot({ path: path.join(output, 'static.png'), fullPage: true });
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1100, 820));
    await page.waitForFunction(() => window.innerWidth < 1200);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    await page.screenshot({ path: path.join(output, 'static-compact.png'), fullPage: true });
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1480, 980));
    await page.waitForFunction(() => window.innerWidth > 1200);
    // Inspect the actual scene's prism faces at a scale where closure is visible.
    const prismImage = await page.evaluate(async () => {
      const faces = document.querySelector('#scene').data.filter(trace => trace.name?.startsWith('WEST · PR3:'));
      const preview = document.createElement('div');
      preview.id = 'prism-preview';
      preview.style.cssText = 'position:fixed;left:0;top:0;width:640px;height:640px;z-index:10000';
      document.body.append(preview);
      await Plotly.newPlot(preview, faces, {
        template: 'plotly_dark', paper_bgcolor: '#101922', showlegend: false,
        title: { text: 'PR3' }, margin: { l: 0, r: 0, t: 40, b: 0 },
        scene: { aspectmode: 'data', camera: { eye: { x: 1.4, y: 1.8, z: 1.1 } } },
      });
      return Plotly.toImage(preview, { format: 'png', width: 640, height: 640 });
    });
    await fs.writeFile(path.join(output, 'prism.png'), Buffer.from(prismImage.split(',')[1], 'base64'));
    await page.evaluate(() => {
      const preview = document.querySelector('#prism-preview');
      Plotly.purge(preview);
      preview.remove();
    });
    await page.locator('#kind').selectOption('real');
    await page.locator('#run').click();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent === 'Конфигурация построена');
    assert.match(await page.locator('#side-status [data-direction="east_to_west"]').innerText(), /использованы частично.*75/s);
    assert.equal(await page.locator('#side-status [data-direction="west_to_east"] .screen-status').count(), 0);
    assert.equal(await page.locator('#measurement-sides button').count(), 2);
    const measurementScene = await page.evaluate(() => {
      const scene = document.querySelector('#measurement-plot')._fullLayout.scene;
      return { dragmode: scene.dragmode, aspectmode: scene.aspectmode, projection: scene.camera.projection.type };
    });
    assert.deepEqual(measurementScene, { dragmode: 'orbit', aspectmode: 'manual', projection: 'perspective' });
    for (const direction of ['east_to_west', 'west_to_east']) {
      await page.locator(`#measurement-sides [data-direction="${direction}"]`).click();
      await page.waitForFunction(dir => document.querySelector(`#measurement-sides [data-direction="${dir}"]`).getAttribute('aria-pressed') === 'true', direction);
      assert.equal(await page.locator('#calibration-table tbody tr').count(), 112);
    }
    await page.screenshot({ path: path.join(output, 'both-real.png'), fullPage: true });
    for (const direction of ['west_to_east', 'east_to_west']) {
      await page.locator('#direction').selectOption(direction);
      await page.locator('#kind').selectOption('real');
      await page.locator('#collision-mode').selectOption('exhaustive');
      await page.locator('#run').click();
      await page.waitForFunction(() => document.querySelector('#run-status').textContent === 'Конфигурация построена');
      assert.match(await page.locator('#side-status').innerText(), /не выбрана/);
      await page.locator('#calibrate').click();
      await page.waitForFunction(() => document.querySelector('#run-status').textContent === 'Измерения обработаны', null, { timeout: 60000 });
      assert.equal(await page.locator('#calibration-table tbody tr').count(), 112);
      await page.screenshot({ path: path.join(output, `calibration-${direction}.png`), fullPage: true });
    }
    for (const tab of ['dynamic', 'distributions']) {
      await page.locator(`.nav-button[data-tab="${tab}"]`).click();
      assert.ok(await page.locator(`#${tab}-panel`).isVisible());
    }
    // Validation and cancellation exercise the actual isolated preload bridge.
    const rejected = await page.evaluate(async () => {
      try { await window.mirrorlab.request('exec', {}); return false; } catch { return true; }
    });
    assert.ok(rejected);
    // The file dialog is stubbed only inside this test, still exercising save IPC.
    const exportPath = path.join(output, 'export.json');
    await app.evaluate(({ dialog }, filePath) => {
      dialog.showSaveDialog = async () => ({ canceled: false, filePath });
    }, exportPath);
    await page.locator('.nav-button[data-tab="static"]').click();
    await page.evaluate(() => window.mirrorlab.saveResult({ application: 'MirrorLab', format_version: 1 }));
    const exported = JSON.parse(await fs.readFile(exportPath, 'utf8'));
    assert.deepEqual(exported, { application: 'MirrorLab', format_version: 1 });
    await page.evaluate(() => window.mirrorlab.request('cancel'));
    assert.equal((await page.evaluate(() => window.mirrorlab.request('catalog'))).modules.static, 'ready');
    assert.deepEqual(errors, []);
    console.log('Electron smoke passed: native UI, Plotly, real traces, both surveys, lavsan, stubs, IPC, worker restart.');
  } finally { await app.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
