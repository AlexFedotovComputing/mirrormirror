/* Real Windows process-tree cancellation and tracing above the former ray cap. */
const assert = require('node:assert/strict');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { WorkerClient } = require('../desktop/worker-client.cjs');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const processes = () => {
  const text = execFileSync('powershell.exe', ['-NoProfile', '-Command',
    'Get-CimInstance Win32_Process -Filter "Name = \'python.exe\'" | Select-Object ProcessId,ParentProcessId | ConvertTo-Json -Compress'],
    { windowsHide: true, encoding: 'utf8' }).trim();
  return [].concat(text ? JSON.parse(text) : []);
};
(async () => {
  const client = new WorkerClient(path.resolve(__dirname, '..'));
  try {
    await client.request('catalog', {});
    const root = client.child.pid;
    const pending = client.request('trace_static', { direction: 'east_to_west', kind: 'real',
      ray_count: 200000, cpu_workers: 6 }).then(() => 'completed', () => 'cancelled');
    let descendants = [];
    for (let i = 0; i < 20; i++) {
      descendants = processes().filter(p => p.ParentProcessId === root).map(p => p.ProcessId);
      if (descendants.length >= 6) break;
      await pause(250);
    }
    assert.ok(descendants.length >= 6, 'CPU pool actually started');
    client.stop();
    assert.equal(await pending, 'cancelled');
    const remaining = new Set(processes().map(p => p.ProcessId));
    assert.ok(!remaining.has(root));
    for (const pid of descendants) assert.ok(!remaining.has(pid), `orphan CPU worker ${pid}`);
    const result = await client.request('trace_static', { direction: 'east_to_west', kind: 'real',
      ray_count: 50001, cpu_workers: 6 });
    assert.equal(result.summary.ray_count, 50001);
    assert.equal(result.summary.cpu_workers, 6);
    assert.ok(result.summary.segment_count > 0);
    console.log('Six CPU workers cancelled without orphans; restart and 50001-ray manual trace passed.');
  } finally { client.stop(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
