/* Focused renderer QA using freshly recomputed diagnostics of a saved document. */
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs/promises');
const { spawnSync } = require('node:child_process');
const { _electron } = require('playwright-core');

(async () => {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'artifacts', 'trajectory-diagnostics');
  await fs.mkdir(output, { recursive: true });
  const result = spawnSync('python', ['-B', '-c', [
    'import json,sys',
    'from pathlib import Path',
    'from mirrorlab.calibration.trajectory_matching import dataset_comparison_report',
    'doc=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))',
    'print(json.dumps([dataset_comparison_report(d) for d in doc["datasets"].values()]))',
  ].join('\n'), path.join(root, 'mirrorlab', 'saved restored', 'mirrorlab-configuration-both1.json')],
  { cwd: root, encoding: 'utf8', windowsHide: true });
  assert.equal(result.status, 0, result.stderr);
  const reports = JSON.parse(result.stdout);
  const report = reports.find(r => r.diagnostics.trajectory_diagnostics.fit_only_count > 0);
  assert.ok(report, 'Saved fixture must contain a fit/physical discrepancy');
  const diagnostic = report.diagnostics.trajectory_diagnostics;
  const env = { ...process.env, MIRRORLAB_HEADLESS: '1', MIRRORLAB_USER_DATA: path.join(output, 'profile') };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({ executablePath: path.join(root, 'node_modules', 'electron', 'dist', 'electron.exe'),
    args: [root], cwd: root, env });
  try {
    const page = await app.firstWindow();
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await app.evaluate(({ BrowserWindow }) => {
      const win = BrowserWindow.getAllWindows()[0];
      win.webContents.setBackgroundThrottling(false); win.setSize(1440, 1100);
    });
    await page.waitForFunction(() => typeof renderTrajectoryComparison === 'function');
    await page.evaluate(data => renderTrajectoryComparison([data], 'Сохранённая реконструкция'), report);
    const panel = page.locator('.trajectory-diagnostics');
    await panel.locator(':scope > summary').click();
    const rows = panel.locator('.trajectory-diagnostic-row');
    assert.equal(await rows.count(), diagnostic.fit_only_count);
    const blocked = diagnostic.observations.find(r => r.fit_only && r.reason_codes.includes('outgoing_mirror_interception'));
    assert.ok(blocked);
    await panel.getByRole('searchbox').fill(blocked.observation_id);
    assert.equal(await rows.count(), 1);
    await rows.first().locator('summary').click();
    await rows.first().getByText('Последняя поверхность:', { exact: false }).waitFor();
    const text = await rows.first().innerText();
    assert.ok(text.includes(blocked.path.terminal_surface));
    assert.ok(text.includes('Оси') || text.includes('оси эллипса'));
    assert.ok(text.includes('За концом, мм'));
    assert.ok(text.includes('Подгонка'));
    assert.equal(await rows.first().locator('table').count() >= 3, true);
    await rows.first().evaluate(el => el.scrollIntoView({ block: 'start' }));
    await page.evaluate(() => window.scrollBy(0, -70));
    const png = await app.evaluate(async ({ BrowserWindow }) =>
      (await BrowserWindow.getAllWindows()[0].webContents.capturePage(undefined, { stayHidden: true })).toPNG().toString('base64'));
    await fs.writeFile(path.join(output, 'trajectory-detail.png'), Buffer.from(png, 'base64'));
    await panel.getByRole('searchbox').fill('нет-такой-траектории');
    assert.equal(await rows.count(), 0);
    await panel.getByRole('searchbox').fill('');
    await panel.getByRole('combobox').selectOption('matched');
    assert.equal(await rows.count(), diagnostic.observations.filter(r => r.status === 'matched').length);
    await panel.getByRole('combobox').selectOption('excluded');
    assert.equal(await rows.count(), diagnostic.observations.filter(r => r.status === 'excluded').length);
    await panel.getByRole('combobox').selectOption('all');
    assert.equal(await rows.count(), diagnostic.observations.length);
    assert.deepEqual(errors, []);
    console.log('Trajectory diagnostics UI passed: filters, search, detailed evidence, saved-data compatibility.');
  } finally { await app.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
