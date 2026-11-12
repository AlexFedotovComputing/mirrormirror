/* Verify startup discoverability and the revised navigation with real Electron. */
const path = require('node:path');
const assert = require('node:assert/strict');
const { _electron } = require('playwright-core');

(async () => {
  const root = path.resolve(__dirname, '..');
  const env = { ...process.env, MIRRORLAB_HEADLESS: '1', MIRRORLAB_USER_DATA: path.join(root, 'artifacts/ui/ergonomics-test-profile') };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await _electron.launch({ executablePath: path.join(root, 'node_modules/electron/dist/electron.exe'), args: [root], cwd: root, env });
  try {
    const page = await app.firstWindow();
    page.setDefaultTimeout(30000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.setBackgroundThrottling(false));
    await page.waitForFunction(() => document.querySelector('#connection-label').textContent.includes('подключён'));
    for (const size of [[1480, 980], [1100, 820], [1000, 700]]) {
      await app.evaluate(({ BrowserWindow }, [width, height]) => BrowserWindow.getAllWindows()[0].setSize(width, height), size);
      const layout = await page.evaluate(() => ({
        width: window.innerWidth, documentWidth: document.documentElement.scrollWidth, height: window.innerHeight,
        actions: ['load-configuration', 'save-configuration', 'start-trace', 'layers-panel'].map(id => {
          const node = id === 'layers-panel' ? document.querySelector('#layers-panel > summary') : document.getElementById(id);
          const box = node.getBoundingClientRect(); return { id, top: box.top, bottom: box.bottom };
        }),
      }));
      assert.ok(layout.documentWidth <= layout.width, JSON.stringify(layout));
      assert.ok(layout.actions.every(box => box.top >= 0 && box.bottom <= layout.height), JSON.stringify(layout));
    }
    assert.ok(await page.locator('#run-status').isVisible());
    assert.ok(await page.locator('#calibrate').isVisible());
    assert.ok(await page.locator('.result-navigation [data-result-target="result-details"]').isDisabled());
    await page.locator('#ray-count').fill('19');
    await page.locator('#start-trace').click();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent === 'Конфигурация построена');
    assert.equal(await page.evaluate(() => document.querySelector('#scene')._fullLayout.showlegend), false);
    const camera = await page.evaluate(() => document.querySelector('#scene')._fullLayout.scene.camera);
    await page.locator('#toggle-legend').click();
    await page.waitForFunction(() => document.querySelector('#scene')._fullLayout.showlegend);
    await page.locator('#toggle-legend').click();
    await page.waitForFunction(() => !document.querySelector('#scene')._fullLayout.showlegend);
    assert.deepEqual(await page.evaluate(() => document.querySelector('#scene')._fullLayout.scene.camera), camera);
    await page.locator('.result-navigation [data-result-target="result-details"]').click();
    assert.equal(await page.evaluate(() => document.activeElement.id), 'result-details');
    await page.locator('.result-navigation [data-result-target="scene-card"]').click();
    await page.locator('#ray-count').fill('20');
    await page.locator('#ray-count').press('Tab');
    assert.ok(await page.locator('#scene-stale').isVisible());
    await page.locator('#trace-settings > summary').click();
    await page.locator('#max-interactions').fill('0');
    await page.locator('#trace-settings > summary').click();
    await page.locator('#run').click();
    assert.ok(await page.evaluate(() => document.querySelector('#trace-settings').open));
    await page.locator('#max-interactions').fill('48');
    await page.locator('#run').click();
    await page.waitForFunction(() => document.querySelector('#run-status').textContent === 'Конфигурация построена' && document.querySelector('#scene-stale').hidden);
    assert.deepEqual(errors, []);
    console.log('Ergonomics UI passed: startup at three sizes, visible actions, legend, report navigation, stale-state notice and hidden-field validation.');
  } finally { await app.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
