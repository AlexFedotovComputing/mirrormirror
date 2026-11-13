/* Real worker + isolated renderer: manual import, explicit preview, refine and persistence. */
const path = require('node:path');
const fs = require('node:fs/promises');
const assert = require('node:assert/strict');
const { _electron } = require('playwright-core');

(async () => {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'artifacts', 'ui', 'manual');
  await fs.mkdir(output, { recursive: true });
  const env = { ...process.env, MIRRORLAB_HEADLESS: '1', MIRRORLAB_USER_DATA: path.join(output, 'profile') };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({ executablePath: path.join(root, 'node_modules', 'electron', 'dist', 'electron.exe'), args: [root], cwd: root, env });
  try {
    const page = await app.firstWindow();
    page.setDefaultTimeout(120000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false));
    await page.waitForFunction(() => document.querySelector('#connection-label').textContent.includes('подключён'));
    const finish = async expected => {
      await page.waitForFunction(() => !document.querySelector('#load-manual').disabled, null, { timeout: 600000 });
      assert.equal(await page.locator('#run-status').innerText(), expected, await page.locator('#notice').innerText());
    };
    await page.locator('#run').click();
    await finish('Конфигурация построена');
    const base = await page.evaluate(() => JSON.stringify(document.querySelector('#scene').data));
    for (const [filename, direction] of [['west_manual_corrected_survey.json', 'east_to_west'], ['east_recovered_centers.json', 'west_to_east']]) {
      const input = path.join(root, 'data', 'manual_refinement', filename);
      const saved = path.join(output, direction + '.json');
      await app.evaluate(({ dialog }, filePath) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] }); }, input);
      await page.locator('#load-manual').click();
      await finish('Реконструкция обработана');
      assert.equal(await page.locator('#mirror-select option').count(), 112);
      assert.equal(await page.locator('#optimize').innerText(), 'Уточнить ручную реконструкцию');
      assert.ok(await page.locator('#optimize').isEnabled());
      assert.ok(await page.locator('#accept-candidate').isDisabled());
      assert.equal(await page.evaluate(() => JSON.stringify(document.querySelector('#scene').data)), base);
      await page.locator('#preview-candidate').click();
      await page.waitForFunction(() => document.querySelector('#scene').data.some(trace => trace.meta?.scene_source === 'reconstruction'));
      await page.locator('#optimize').click();
      await finish('Реконструкция обработана');
      assert.ok(await page.locator('#accept-candidate').isDisabled());
      await app.evaluate(({ dialog }, filePath) => { dialog.showSaveDialog = async () => ({ canceled: false, filePath }); }, saved);
      await page.locator('#save-configuration').click();
      await finish('Конфигурация сохранена');
      const document = JSON.parse(await fs.readFile(saved, 'utf8'));
      assert.ok(document.datasets[direction].manual_references);
      assert.ok(document.datasets[direction].diagnostics.manual_refinement);
      await page.evaluate(() => window.mirrorlab.request('cancel'));
      await app.evaluate(({ dialog }, filePath) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] }); }, saved);
      await page.locator('#load-configuration').click();
      await finish('Реконструкция обработана');
      assert.equal(await page.locator('#mirror-select option').count(), 112);
      assert.ok(await page.locator('#accept-candidate').isDisabled());
      assert.equal(await page.evaluate(() => JSON.stringify(document.querySelector('#scene').data)), base);
      const screenshot = await app.evaluate(async ({ BrowserWindow }) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage(undefined, { stayHidden: true })).toPNG().toString('base64'));
      await fs.writeFile(path.join(output, direction + '.png'), Buffer.from(screenshot, 'base64'));
    }
    assert.deepEqual(errors, []);
    console.log('Manual UI import → preview → refine → save → worker restart → load passed for both sides.');
  } finally { await app.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
