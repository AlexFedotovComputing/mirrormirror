/* Simplified static presets → actual Python tracing → unchanged Plotly controls. */
const path = require('node:path');
const fs = require('node:fs/promises');
const assert = require('node:assert/strict');
const { _electron } = require('playwright-core');

(async () => {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'artifacts', 'ui', 'static-presets');
  await fs.mkdir(output, { recursive: true });
  const env = { ...process.env, MIRRORLAB_HEADLESS: '1', MIRRORLAB_USER_DATA: path.join(output, 'profile') };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({ executablePath: path.join(root, 'node_modules/electron/dist/electron.exe'), args: [root], cwd: root, env });
  try {
    const page = await app.firstWindow();
    page.setDefaultTimeout(120000);
    const errors = [];
    const dragScene = async (label) => {
      await page.locator('#optical-scene').first().scrollIntoViewIfNeeded();
      const before = await page.evaluate(() => JSON.stringify(document.querySelector('#optical-scene')._fullLayout.scene._scene.getCamera()));
      const box = await page.locator('#optical-scene canvas').first().boundingBox();
      assert.ok(box, label + ': canvas exists');
      await page.mouse.move(box.x + box.width * .6, box.y + box.height * .5);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width * .7, box.y + box.height * .6, { steps: 3 });
      await page.mouse.up();
      await page.waitForFunction(previous => JSON.stringify(document.querySelector('#optical-scene')._fullLayout.scene._scene.getCamera()) !== previous, before, { timeout: 5000 });
      await page.mouse.click(box.x + box.width * .6, box.y + box.height * .5, { clickCount: 2 });
      await page.mouse.click(box.x + box.width * .55, box.y + box.height * .45);
      const beforeWheel = await page.evaluate(() => JSON.stringify(document.querySelector('#optical-scene')._fullLayout.scene._scene.getCamera()));
      await page.mouse.wheel(0, -120);
      await page.waitForFunction(previous => JSON.stringify(document.querySelector('#optical-scene')._fullLayout.scene._scene.getCamera()) !== previous, beforeWheel, { timeout: 5000 });
      console.log('Mouse drag: ' + label);
    };
    page.on('pageerror', e => { errors.push(e.message); console.error(e.stack); });
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false));
    await page.waitForFunction(() => !document.querySelector('#controls').disabled && document.querySelector('#run-status').textContent !== 'Запуск…');
    for (const id of ['load-configuration', 'load-manual', 'save-configuration', 'calibrate', 'add-manual-rays', 'optimize', 'trace-source']) {
      assert.equal(await page.locator('#' + id).isVisible(), false, id + ' should not be in the static panel');
    }
    assert.deepEqual(await page.locator('#kind option').allTextContents(), ['Идеальная', 'Реальная']);
    assert.equal(await page.locator('#run').innerText(), 'Выполнить трассировку');
    assert.equal(await page.locator('#ray-count').inputValue(), '1000');
    assert.equal(await page.locator('#ray-count').isVisible(), true);
    assert.equal(await page.locator('#ray-count').getAttribute('max'), null);
    assert.equal(await page.locator('#cpu-workers').inputValue(), '6');
    assert.equal(await page.locator('#trace-settings').evaluate(el => el.open), false);
    await page.locator('#trace-settings summary').click();
    await page.selectOption('#direction', 'east_to_west');
    await page.fill('#ray-count', '37');
    const trace = async (kind, expected) => {
      await page.selectOption('#kind', kind);
      await page.click('#run');
      await page.waitForFunction(() => !document.querySelector('#controls').disabled && ['Конфигурация построена', 'Расчёт не завершён'].includes(document.querySelector('#run-status').textContent));
      assert.equal(await page.locator('#run-status').innerText(), 'Конфигурация построена', await page.locator('#notice').innerText());
      const result = JSON.parse(await page.locator('#result-json').textContent());
      assert.equal(result.trace_source, 'preset');
      assert.equal(result.metadata.static_configuration, expected);
      assert.equal(result.summary.ray_count, 37);
      assert.equal(result.request.cpu_workers, 6);
      assert.ok(await page.evaluate(() => document.querySelector('#optical-scene').data.length > 0));
    };
    await trace('ideal', 'ideal');
    await dragScene('initial');
    assert.match(await page.locator('#scene-label').innerText(), /Идеальная/);
    await trace('real', 'manual');
    await dragScene('manual');
    assert.match(await page.locator('#scene-label').innerText(), /ручные зеркала/);
    // Switching back must select the ideal preset, not reuse the last manual scene.
    await trace('ideal', 'ideal');
    await trace('real', 'manual');
    await page.locator('#layers-panel').evaluate(el => { el.open = true; });
    await page.uncheck('#show-optics');
    await page.waitForFunction(() => !document.querySelector('#optical-scene').data.some(t => t.meta?.layer === 'optics'));
    await page.check('#show-optics');
    await page.waitForFunction(() => document.querySelector('#optical-scene').data.some(t => t.meta?.layer === 'optics'));
    await dragScene('layers');
    await page.uncheck('#show-base');
    await page.waitForFunction(() => !document.querySelector('#filtered-scene').hidden);
    await page.check('#show-base');
    await page.waitForFunction(() => document.querySelector('#filtered-scene').hidden);
    await dragScene('restore empty scene');
    await page.click('#toggle-legend');
    assert.equal(await page.locator('#toggle-legend').getAttribute('aria-pressed'), 'true');
    await page.click('#reset-camera');
    await page.locator('#trace-settings').evaluate(el => { el.open = false; });
    await page.locator('#layers-panel').evaluate(el => { el.open = false; });
    await page.locator('#scene-card').scrollIntoViewIfNeeded();
    const screenshot = await app.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage(undefined, { stayHidden: true })).toPNG().toString('base64'));
    await fs.writeFile(path.join(output, 'static-presets.png'), Buffer.from(screenshot, 'base64'));
    assert.deepEqual(errors, []);
    console.log('Static ideal/manual presets, 37-ray tracing, switching and Plotly controls passed.');
  } finally { await app.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
