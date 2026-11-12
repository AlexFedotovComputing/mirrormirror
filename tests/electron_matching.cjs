/* Load an old reconstruction and inspect the numbering-independent report. */
const path = require('node:path');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const { _electron } = require('playwright-core');

(async () => {
  const root = path.resolve(__dirname, '..');
  const fixture = path.resolve(root, process.argv[2] || 'data/examples/reconstruction-both.json');
  const env = { ...process.env, MIRRORLAB_HEADLESS: '1',
    MIRRORLAB_USER_DATA: path.join(root, 'artifacts/ui/matching-profile') };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({ executablePath: path.join(root, 'node_modules/electron/dist/electron.exe'),
    args: [root], cwd: root, env });
  try {
    const page = await app.firstWindow();
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false));
    await page.emulateMedia({ reducedMotion: 'reduce' });
    page.setDefaultTimeout(60000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.waitForFunction(() => document.querySelector('#connection-label')?.textContent.includes('подключён'));
    await app.evaluate(({ dialog }, filePath) => {
      dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [filePath] });
    }, fixture);
    await page.locator('#load-configuration').click();
    await page.waitForFunction(() => ['Реконструкция обработана', 'Расчёт не завершён'].includes(document.querySelector('#run-status').textContent));
    assert.equal(await page.locator('#run-status').textContent(), 'Реконструкция обработана', await page.locator('#notice').textContent());
    console.log('Loaded reconstruction.');
    await page.locator('.result-navigation [data-result-target="trajectory-comparison-result"]').click();
    assert.ok(await page.locator('#trajectory-comparison-result').isVisible());
    const reports = await page.evaluate(() => JSON.parse(document.querySelector('#reconstruction-json').textContent).reports);
    assert.equal(reports.length, 2);
    const cards = page.locator('#trajectory-comparison-result .trajectory-comparison-side');
    assert.equal(await cards.count(), 2);
    for (let i = 0; i < reports.length; i++) {
      const matching = reports[i].diagnostics.trajectory_matching;
      const fit = reports[i].diagnostics.geometric_fit;
      assert.equal(matching.status, 'evaluated');
      assert.equal(fit.status, 'evaluated');
      assert.equal(fit.angle_tolerance_rad, null);
      assert.equal(matching.angle_tolerance_rad, null);
      assert.equal(new Set(matching.matches.map(p => p.trajectory_id)).size, matching.matched_count);
      const card = page.locator(`.trajectory-comparison-side[data-direction="${reports[i].direction}"]`);
      assert.match(await card.textContent(), /Точность подгонки/);
      assert.match(await card.textContent(), /Физическая проходимость/);
      assert.match(await card.textContent(), /3 мм/);
      assert.match((await card.locator('.geometric-fit-summary').textContent()).replace(/\s/g, '').replace(/из/g, '/'), new RegExp(`${fit.matched_count}/${fit.observation_count}`));
      const details = card.locator('details').first();
      await details.locator('summary').click();
      assert.ok(await details.evaluate(node => node.open));
      await details.locator('summary').click();
      if (fixture.endsWith('mirrorlab-configuration-both1.json')) {
        assert.equal(fit.counts_within_3mm, reports[i].direction === 'east_to_west' ? 82 : 81);
      }
    }
    const fitCounts = reports.map(r => r.diagnostics.geometric_fit.matched_count);
    console.log('Verified both comparison cards.');
    if (!(await page.locator('#trace-settings').evaluate(node => node.open))) await page.locator('#trace-settings > summary').click();
    await page.locator('#ray-count').fill('19');
    await page.locator('#run').click();
    await page.waitForFunction(() => ['Конфигурация построена', 'Расчёт не завершён'].includes(document.querySelector('#run-status').textContent));
    assert.equal(await page.locator('#run-status').textContent(), 'Конфигурация построена', await page.locator('#notice').textContent());
    console.log('Historical trace completed.');
    await page.locator('.result-navigation [data-result-target="trajectory-comparison-result"]').click();
    assert.match(await page.locator('#trajectory-comparison-result').textContent(), /кандидат/i);
    for (let i = 0; i < reports.length; i++) {
      const card = page.locator(`.trajectory-comparison-side[data-direction="${reports[i].direction}"]`);
      assert.match((await card.locator('.geometric-fit-summary').textContent()).replace(/\s/g, '').replace(/из/g, '/'), new RegExp(`${fitCounts[i]}/${reports[i].diagnostics.geometric_fit.observation_count}`));
    }
    await fs.mkdir(path.join(root, 'artifacts/ui'), { recursive: true });
    await page.screenshot({ path: path.join(root, 'artifacts/ui/trajectory-comparison.png'), fullPage: true, animations: 'disabled' });
    for (const width of [1480, 1100]) {
      await page.setViewportSize({ width, height: 980 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    }
    assert.deepEqual(errors, []);
    console.log('Trajectory comparison UI passed: fit vs physical paths, old JSON, report navigation, preserved candidate after trace, responsive layout.');
  } finally { await app.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
