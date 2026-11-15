/* Real button → full Python trace → scene layers → persisted candidate. */
const path = require('node:path');
const fs = require('node:fs/promises');
const assert = require('node:assert/strict');
const { _electron } = require('playwright-core');

(async () => {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'artifacts', 'ui', 'manual-rays');
  await fs.mkdir(output, { recursive: true });
  const env = { ...process.env, MIRRORLAB_HEADLESS: '1', MIRRORLAB_USER_DATA: path.join(output, 'profile') };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({ executablePath: path.join(root, 'node_modules/electron/dist/electron.exe'), args: [root], cwd: root, env });
  try {
    const page = await app.firstWindow();
    page.setDefaultTimeout(120000);
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false));
    await page.waitForFunction(() => document.querySelector('#add-manual-rays')?.disabled === false, null, { timeout: 30000 });
    console.log('Worker connected');
    await page.selectOption('#direction', 'east_to_west');
    await page.selectOption('#kind', 'real');
    const finish = async text => {
      await page.waitForFunction(() => !document.querySelector('#add-manual-rays').disabled);
      assert.equal(await page.locator('#run-status').innerText(), text, await page.locator('#notice').innerText());
    };
    await page.click('#add-manual-rays');
    await finish('Лучи из ручного анализа добавлены');
    console.log('Manual rays computed');
    assert.match(await page.locator('.manual-rays-summary').innerText(), /\/83/);
    assert.equal(await page.locator('#trace-source').inputValue(), 'candidate');
    const count = () => page.evaluate(() => document.querySelector('#optical-scene').data.filter(t => t.meta?.kind === 'manual_aperture_ray').length);
    const initialCount = await count();
    assert.ok(await page.locator('#accept-candidate').isDisabled());
    // Open the layer details without relying on its current expanded state.
    await page.locator('#show-manual-rays').evaluate(el => { const d = el.closest('details'); if (d) d.open = true; });
    await page.uncheck('#show-manual-rays');
    await page.waitForFunction(() => !document.querySelector('#optical-scene').data.some(t => t.meta?.kind === 'manual_aperture_ray'));
    await page.check('#show-manual-rays');
    await page.waitForFunction(() => document.querySelector('#optical-scene').data.some(t => t.meta?.kind === 'manual_aperture_ray'));
    await page.click('#add-manual-rays');
    await finish('Лучи из ручного анализа добавлены');
    assert.equal(await count(), initialCount);
    const saved = path.join(output, 'manual-rays.json');
    await app.evaluate(({ dialog }, filePath) => { dialog.showSaveDialog = async () => ({ canceled: false, filePath }); }, saved);
    await page.click('#save-configuration');
    await finish('Конфигурация сохранена');
    const document = JSON.parse(await fs.readFile(saved, 'utf8'));
    assert.ok(document.datasets.east_to_west.manual_rays);
    await page.fill('#ray-count', '73');
    await page.click('#run');
    await finish('Конфигурация построена');
    assert.match(await page.locator('#scene-label').innerText(), /трассировка кандидата/);
    await page.evaluate(() => window.mirrorlab.request('cancel'));
    await app.evaluate(({ dialog }, filePath) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] }); }, saved);
    await page.click('#load-configuration');
    await finish('Реконструкция обработана');
    assert.equal(await count(), initialCount);
    assert.match(await page.locator('.manual-rays-summary').innerText(), /\/83/);
    await page.locator('#scene-card').scrollIntoViewIfNeeded();
    const screenshot = await app.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage(undefined, { stayHidden: true })).toPNG().toString('base64'));
    await fs.writeFile(path.join(output, 'manual-rays.png'), Buffer.from(screenshot, 'base64'));
    assert.deepEqual(errors, []);
    console.log('Manual mirrors, candidate tracing, layers, rerun, save and reload passed.');
  } finally { await app.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
