/* Focused regression for scene filtering, including the empty Plotly scene. */
const path = require('node:path');
const assert = require('node:assert/strict');
const { _electron } = require('playwright-core');

(async () => {
  const root = path.resolve(__dirname, '..');
  const env = { ...process.env, MIRRORLAB_HEADLESS: '1', MIRRORLAB_USER_DATA: path.join(root, 'artifacts/ui/layers-profile') };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({ executablePath: path.join(root, 'node_modules/electron/dist/electron.exe'), args: [root], cwd: root, env });
  try {
    const page = await app.firstWindow();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false));
    await page.waitForFunction(() => document.querySelector('#connection-label').textContent.includes('подключён'));
    await page.locator('#ray-count').fill('19');
    await page.locator('#run').click();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent === 'Конфигурация построена');
    const before = await page.evaluate(() => ({ count: document.querySelector('#scene').data.length,
      camera: document.querySelector('#scene')._fullLayout.scene.camera }));
    await page.locator('#layers-panel > summary').click();
    // Await the same renderer used by change events to ensure WebGL updates finish.
    for (const id of ['show-east', 'show-west', 'show-boundary']) {
      await page.locator('#' + id).uncheck();
      await page.evaluate(() => drawSceneLayers());
    }
    assert.equal(await page.evaluate(() => document.querySelector('#scene').data.length), 0);
    assert.ok(await page.locator('#filtered-scene').isVisible());
    await page.locator('#restore-layers').click();
    await page.evaluate(() => drawSceneLayers());
    assert.equal(await page.locator('#filtered-scene').isVisible(), false);
    assert.deepEqual(await page.evaluate(() => ({ count: document.querySelector('#scene').data.length,
      camera: document.querySelector('#scene')._fullLayout.scene.camera })), before);
    await page.locator('#show-west').uncheck();
    await page.locator('#show-traced-rays').uncheck();
    await page.evaluate(() => drawSceneLayers());
    await page.locator('#run').click();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent === 'Конфигурация построена');
    assert.ok(await page.evaluate(() => document.querySelector('#scene').data.every(t => t.meta?.side !== 'WEST' && t.meta?.layer !== 'traced_rays')));
    assert.equal(await page.locator('#notice').innerText(), '');
    assert.deepEqual(errors, []);
    console.log('Layer UI passed: empty scene, restored camera, combined filters and persistence after tracing.');
  } finally { await app.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
