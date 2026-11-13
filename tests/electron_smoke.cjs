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
    const capture = async filename => {
      const png = await app.evaluate(async ({ BrowserWindow }) =>
        (await BrowserWindow.getAllWindows()[0].webContents.capturePage(undefined, { stayHidden: true })).toPNG().toString('base64'));
      await fs.writeFile(path.join(output, filename), Buffer.from(png, 'base64'));
    };
    page.setDefaultTimeout(120000);
    const waitForStatus = async (expected, timeout = 120000) => {
      await page.waitForFunction(text => [text, 'Расчёт не завершён'].includes(document.querySelector('#run-status').textContent), expected, { timeout });
      assert.equal(await page.locator('#run-status').innerText(), expected, await page.locator('#notice').innerText());
    };
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false));
    page.on('pageerror', error => errors.push(error.message));
    await page.waitForFunction(() => document.querySelector('#connection-label')?.textContent.includes('подключён'));
    assert.equal(await page.evaluate(() => typeof window.require), 'undefined');
    assert.equal(await page.locator('#direction').inputValue(), 'both');
    assert.equal(await page.locator('#use-lavsan').count(), 0);
    assert.equal(await page.locator('#optimize').innerText(), 'Восстановить зеркала');
    await page.locator('#kind').selectOption('ideal');
    assert.equal(await page.locator('#optimize').isDisabled(), true);
    assert.equal(await page.locator('#optimize-reason').isVisible(), true);
    await page.locator('#kind').selectOption('real');
    assert.equal(await page.locator('#optimize').isEnabled(), true);
    assert.equal(await page.locator('#optimize-reason').isVisible(), false);
    await page.locator('#kind').selectOption('ideal');
    assert.equal(await page.locator('#optimize').isDisabled(), true);
    for (const [id, label] of [['save-configuration', 'Сохранить'], ['load-configuration', 'Открыть конфигурацию'], ['load-legacy', 'Импорт старого формата…']]) {
      assert.equal(await page.locator('#' + id).innerText(), label);
    }
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
    assert.ok(layout.top < 300 && layout.height > layout.viewport * .5, JSON.stringify(layout));
    assert.equal(await page.evaluate(() => document.querySelector('#scene')._fullLayout.showlegend), false);
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
    await capture('static.png');
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1100, 820));
    await page.waitForFunction(() => window.innerWidth < 1200);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    await capture('static-compact.png');
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
    assert.ok(await page.evaluate(() => document.querySelector('#scene').data.some(trace => trace.meta?.kind === 'measurement_ray')));
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
    await capture('both-real.png');
    await page.locator('#layers-panel > summary').click();
    const beforeFilters = await page.evaluate(() => ({
      count: document.querySelector('#scene').data.length,
      camera: document.querySelector('#scene')._fullLayout.scene.camera,
    }));
    await page.locator('#show-east').uncheck();
    await page.waitForFunction(() => document.querySelector('#scene').data.every(t => t.meta?.side !== 'EAST'));
    assert.ok(await page.evaluate(() => document.querySelector('#scene').data.some(t => t.meta?.side === 'WEST')));
    await page.locator('#show-traced-rays').uncheck();
    await page.waitForFunction(() => document.querySelector('#scene').data.every(t => t.meta?.layer !== 'traced_rays'));
    assert.ok(await page.evaluate(() => document.querySelector('#scene').data.every(t => !t.marker?.showscale)));
    await page.locator('#show-optics').uncheck();
    await page.waitForFunction(() => document.querySelector('#scene').data.every(t => t.meta?.layer !== 'optics'));
    await page.locator('#show-east').check();
    await page.locator('#show-traced-rays').check();
    await page.locator('#show-optics').check();
    await page.waitForFunction(count => document.querySelector('#scene').data.length === count, beforeFilters.count);
    assert.deepEqual(await page.evaluate(() => document.querySelector('#scene')._fullLayout.scene.camera), beforeFilters.camera);
    await page.locator('#show-east').uncheck();
    await page.locator('#show-west').uncheck();
    await page.locator('#show-boundary').uncheck();
    await page.waitForFunction(() => document.querySelector('#scene').data.length === 0);
    await page.locator('#show-east').check();
    await page.locator('#show-west').check();
    await page.locator('#show-boundary').check();
    await page.waitForFunction(count => document.querySelector('#scene').data.length === count, beforeFilters.count);
    assert.equal(await page.locator('#notice').innerText(), '');
    await page.locator('#trace-settings > summary').click();
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
      await capture(`calibration-${direction}.png`);
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
    // Physical configuration persistence uses the actual buttons and Python schema validation.
    const configurationPath = path.join(output, 'configuration.json');
    await app.evaluate(({ dialog }, filePath) => {
      dialog.showSaveDialog = async () => ({ canceled: false, filePath });
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] });
    }, configurationPath);
    await page.locator('#direction').selectOption('west_to_east');
    await page.locator('#save-configuration').click();
    await waitForStatus('Конфигурация сохранена');
    const savedConfiguration = JSON.parse(await fs.readFile(configurationPath, 'utf8'));
    assert.ok(savedConfiguration.schema_version);
    assert.ok(savedConfiguration.datasets.west_to_east);
    await page.locator('#load-configuration').click();
    await waitForStatus('Реконструкция обработана');
    assert.ok(await page.evaluate(() => {
      const data = document.querySelector('#scene').data;
      return data.some(trace => trace.meta?.scene_source === 'base' && trace.name?.includes('Лучи')) &&
        data.some(trace => trace.meta?.scene_source === 'reconstruction');
    }), 'Loading a reconstruction preserves the previous traced rays');
    // Loading an unvalidated historical document previews it; acceptance is explicit.
    if (await page.locator('#accept-candidate').isEnabled()) {
      await page.locator('#accept-candidate').click();
      await waitForStatus('Реконструкция обработана');
    }
    assert.match(await page.locator('#active-state').innerText(), /Принятая конфигурация/);
    assert.equal(await page.locator('#mirror-select option').count(), 112);
    const selectedGeometry = await page.evaluate(() => {
      const result = JSON.parse(document.querySelector('#reconstruction-json').textContent);
      const dataset = result.document.datasets.west_to_east;
      const mirror = dataset.mirrors[0];
      const c = mirror.center_m;
      const expected = dataset.display_frame.rotation.map((row, i) => row.reduce((sum, x, j) => sum + x * c[j], 0) + dataset.display_frame.translation_m[i]);
      const entry = [...document.querySelectorAll('#mirror-details dt')].find(node => node.textContent === 'Центр отражающего среза, м');
      return { expected, displayed: entry.nextElementSibling.textContent.split(';').map(x => Number(x.trim().replace(/\s/g, '').replace(',', '.'))) };
    });
    assert.ok(selectedGeometry.expected.every((value, i) => Math.abs(value - selectedGeometry.displayed[i]) < 1e-9));
    const mirrorClicked = await page.evaluate(() => {
      const plot = document.querySelector('#scene');
      const target = plot.data.find(trace => trace.meta?.mirror_id === '1_1_2' && trace.meta?.direction === 'west_to_east');
      if (!target) return null;
      plot.emit('plotly_click', { points: [{ data: target }] });
      return document.querySelector('#mirror-select').selectedOptions[0].textContent;
    });
    assert.match(mirrorClicked, /1_1_2/);
    const activeBeforeFailure = await page.locator('#active-state').innerText();
    const badPath = path.join(output, 'invalid-configuration.json');
    await fs.writeFile(badPath, JSON.stringify({ ...savedConfiguration, schema_version: 999999 }));
    await app.evaluate(({ dialog }, filePath) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] }); }, badPath);
    await page.locator('#load-configuration').click();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent === 'Расчёт не завершён');
    assert.equal(await page.locator('#active-state').innerText(), activeBeforeFailure);
    // Killing and restarting Python must retain the accepted document in the app.
    await page.evaluate(() => window.mirrorlab.request('cancel'));
    await page.locator('#run').click();
    await waitForStatus('Конфигурация построена');
    assert.match(await page.locator('#scene-label').innerText(), /принятая реконструкция/);
    assert.equal(await page.locator('#active-state').innerText(), activeBeforeFailure);
    await capture('loaded-configuration.png');
    // An infeasible candidate is inspectable alongside the accepted trace, but cannot replace it.
    const diagnosticPath = path.join(output, 'diagnostic-configuration.json');
    const diagnostic = structuredClone(savedConfiguration);
    diagnostic.status = diagnostic.datasets.west_to_east.status = 'infeasible';
    const diagnosticMirror = diagnostic.datasets.west_to_east.mirrors[0];
    diagnosticMirror.trajectory = { point_m: diagnosticMirror.center_m, direction: [0, 0, 1],
      points_m: [diagnosticMirror.center_m, diagnosticMirror.center_m.map((x, i) => x + (i === 2 ? .1 : 0))] };
    await fs.writeFile(diagnosticPath, JSON.stringify(diagnostic));
    await app.evaluate(({ dialog }, filePath) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] });
    }, diagnosticPath);
    await page.locator('#load-configuration').click();
    await waitForStatus('Реконструкция обработана');
    assert.equal(await page.locator('#active-state').innerText(), activeBeforeFailure);
    assert.ok(await page.locator('#accept-candidate').isVisible());
    assert.ok(await page.locator('#accept-candidate').isDisabled());
    assert.match(await page.locator('#accept-reason').innerText(), /infeasible/);
    const layerState = () => page.evaluate(() => {
      const plot = document.querySelector('#scene');
      return { camera: plot._fullLayout.scene.camera, count: plot.data.length,
        base: plot.data.filter(t => t.meta?.scene_source === 'base').length,
        reconstructed: plot.data.filter(t => t.meta?.kind === 'reconstructed_ray').length };
    });
    const combined = await layerState();
    assert.ok(combined.base > 0 && combined.reconstructed === 1);
    await page.locator('#show-reconstructed-rays').uncheck();
    await page.waitForFunction(() => !document.querySelector('#scene').data.some(t => t.meta?.kind === 'reconstructed_ray'));
    assert.ok(await page.evaluate(() => document.querySelector('#scene').data.some(t => t.meta?.scene_source === 'reconstruction' && t.meta?.layer === 'optics')));
    await page.locator('#show-reconstructed-rays').check();
    await page.waitForFunction(() => document.querySelector('#scene').data.some(t => t.meta?.kind === 'reconstructed_ray'));
    await page.locator('#show-reconstruction').uncheck();
    await page.waitForFunction(() => !document.querySelector('#scene').data.some(t => t.meta?.scene_source === 'reconstruction'));
    assert.equal((await layerState()).base, combined.base);
    await page.locator('#show-reconstruction').check();
    await page.waitForFunction(() => document.querySelector('#scene').data.some(t => t.meta?.kind === 'reconstructed_ray'));
    assert.deepEqual(await layerState(), combined);
    await page.locator('#run').click();
    await waitForStatus('Конфигурация построена');
    assert.deepEqual(await layerState(), combined);
    await capture('candidate-comparison.png');
    // Opt-in long integration run: full reconstruction, candidate save, reload, exact dataset round trip.
    if (process.env.MIRRORLAB_EXTENDED_UI === '1') {
      await page.locator('#optimize').click();
      await page.waitForFunction(() => !document.querySelector('#reconstruction-progress').hidden ||
        document.querySelector('#run-status').textContent === 'Расчёт не завершён');
      assert.equal(await page.locator('#reconstruction-progress').isVisible(), true, await page.locator('#notice').innerText());
      await page.locator('#cancel').click();
      await waitForStatus('Расчёт не завершён');
      assert.equal(await page.locator('#active-state').innerText(), activeBeforeFailure);
      await page.locator('#run').click();
      await waitForStatus('Конфигурация построена');
      assert.match(await page.locator('#scene-label').innerText(), /принятая реконструкция/);
      await page.locator('#optimize').click();
      await page.waitForFunction(() => !document.querySelector('#reconstruction-progress').hidden ||
        document.querySelector('#run-status').textContent === 'Расчёт не завершён');
      assert.equal(await page.locator('#reconstruction-progress').isVisible(), true, await page.locator('#notice').innerText());
      await waitForStatus('Реконструкция обработана', 600000);
      assert.equal(await page.locator('#mirror-select option').count(), 112);
      const candidate = await page.evaluate(() => JSON.parse(document.querySelector('#reconstruction-json').textContent));
      assert.ok(candidate.document.datasets.west_to_east.template);
      if (!candidate.active) assert.equal(await page.locator('#active-state').innerText(), activeBeforeFailure);
      const candidatePath = path.join(output, 'reconstruction.json');
      await app.evaluate(({ dialog }, filePath) => {
        dialog.showSaveDialog = async () => ({ canceled: false, filePath });
        dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] });
      }, candidatePath);
      await page.locator('#save-configuration').click();
      await waitForStatus('Конфигурация сохранена');
      const reconstructed = JSON.parse(await fs.readFile(candidatePath, 'utf8'));
      assert.deepEqual(reconstructed.datasets, candidate.document.datasets);
      await page.locator('#load-configuration').click();
      await waitForStatus('Реконструкция обработана');
      const loaded = await page.evaluate(() => JSON.parse(document.querySelector('#reconstruction-json').textContent));
      assert.deepEqual(loaded.document.datasets, reconstructed.datasets);
      const empiricalFrameError = await page.evaluate(() => {
        const dataset = JSON.parse(document.querySelector('#reconstruction-json').textContent).document.datasets.west_to_east;
        const point = dataset.mirrors[0].center_m;
        const expected = dataset.display_frame.rotation.map((row, i) => row.reduce((sum, x, j) => sum + x * point[j], 0) + dataset.display_frame.translation_m[i]);
        const label = [...document.querySelectorAll('#mirror-details dt')].find(node => node.textContent === 'Центр отражающего среза, м');
        const displayed = label.nextElementSibling.textContent.split(';').map(value => Number(value.trim().replace(/\s/g, '').replace(',', '.')));
        return Math.max(...expected.map((value, i) => Math.abs(value - displayed[i])));
      });
      assert.ok(empiricalFrameError < 1e-9, 'Empirical scene_to_survey operation is also applied to mirror details');
      await capture('reconstruction.png');
    }
    // Legacy JSON is an explicit diagnostic import; accepting it is a separate action.
    const activeBeforeImport = await page.locator('#active-state').innerText();
    await app.evaluate(({ dialog }, filePath) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] });
    }, path.join(root, 'data', 'static', 'west_to_east_real.json'));
    await page.locator('#load-legacy').click();
    await page.locator('#choose-legacy').click();
    await waitForStatus('Реконструкция обработана');
    assert.equal(await page.locator('#active-state').innerText(), activeBeforeImport);
    assert.match(await page.locator('#reconstruction-status').innerText(), /не проверена/);
    assert.ok(await page.locator('#accept-candidate').isVisible());
    await page.locator('#accept-candidate').click();
    await waitForStatus('Реконструкция обработана');
    const activeLegacy = await page.locator('#active-state').innerText();
    assert.notEqual(activeLegacy, activeBeforeImport);
    assert.match(await page.locator('#coordinate-title').innerText(), /Расчётная сцена/);
    // A geometry-only table cannot silently replace an accepted full optical configuration.
    await app.evaluate(({ dialog }, filePath) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] });
    }, path.join(root, 'Восстановленные_микросборки.txt'));
    await page.locator('#legacy-frame').selectOption('legacy');
    await page.locator('#choose-legacy').click();
    await waitForStatus('Реконструкция обработана');
    assert.equal(await page.locator('#active-state').innerText(), activeLegacy);
    assert.equal(await page.locator('#accept-candidate').isVisible(), true);
    assert.equal(await page.locator('#accept-candidate').isDisabled(), true);
    assert.match(await page.locator('#accept-reason').innerText(), /оптика/);
    const tableImport = await page.evaluate(() => JSON.parse(document.querySelector('#reconstruction-json').textContent));
    assert.equal(tableImport.can_activate, false);
    assert.equal(tableImport.document.datasets.west_to_east.optics_snapshot, null);
    assert.ok(await page.locator('#mirror-select option').count() > 0);
    await capture('legacy-table.png');
    const permutedSelection = await page.evaluate(async () => {
      const original = JSON.parse(document.querySelector('#reconstruction-json').textContent);
      const candidate = structuredClone(original);
      const mirrors = candidate.document.datasets.west_to_east.mirrors;
      [mirrors[0].old_id, mirrors[1].old_id] = [mirrors[1].canonical_id, mirrors[0].canonical_id];
      candidate.reports = [candidate.document.datasets.west_to_east];
      await showReconstruction(candidate);
      document.querySelector('#scene').emit('plotly_click', { points: [{ data: { meta: {
        direction: 'west_to_east', mirror_id: mirrors[1].canonical_id,
      } } }] });
      const selected = JSON.parse(document.querySelector('#mirror-technical').textContent).mirror.canonical_id;
      await showReconstruction(original);
      return { selected, expected: mirrors[1].canonical_id };
    });
    assert.equal(permutedSelection.selected, permutedSelection.expected, 'Plot selection resolves canonical identity before conflicting old numbering');
    assert.deepEqual(errors, []);
    console.log('Electron smoke passed: native UI, Plotly, real traces, both surveys, lavsan, configuration save/load, malformed load, worker restart.');
  } finally { await app.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
