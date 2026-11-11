const { test } = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { PassThrough } = require('node:stream');
const { WorkerClient, validateRequest } = require('../desktop/worker-client.cjs');

function fixture(options = {}) {
  const children = [];
  const client = new WorkerClient('/project', { ...options, spawnProcess(command, args, config) {
    const child = new EventEmitter();
    child.stdin = new PassThrough(); child.stdout = new PassThrough(); child.stderr = new PassThrough();
    child.kill = () => { child.killed = true; child.emit('exit', null, 'SIGTERM'); };
    Object.assign(child, { command, args, config });
    children.push(child);
    return child;
  } });
  return { client, children };
}

test('protocol transports UTF-8 data and correlates out-of-order responses', async () => {
  const { client, children } = fixture();
  const first = client.request('catalog');
  const second = client.request('calibrate', { direction: 'west_to_east' });
  const child = children[0];
  assert.equal(child.config.windowsHide, true);
  assert.deepEqual(child.args, ['-u', '-m', 'mirrorlab.worker']);
  const requests = child.stdin.read().toString().trim().split('\n').map(JSON.parse);
  assert.equal(requests[1].payload.direction, 'west_to_east');
  child.stdout.write(JSON.stringify({ id: 2, result: { name: 'Измерения' } }) + '\n');
  child.stdout.write(JSON.stringify({ id: 1, result: { directions: [] } }) + '\n');
  assert.deepEqual(await second, { name: 'Измерения' });
  assert.deepEqual(await first, { directions: [] });
  client.stop();
});

test('cancellation rejects pending requests and next request starts a clean worker', async () => {
  const { client, children } = fixture();
  const pending = client.request('trace');
  const rejected = assert.rejects(pending, /отменён/);
  client.stop();
  await rejected;
  assert.equal(children[0].killed, true);
  const next = client.request('catalog');
  assert.equal(children.length, 2);
  children[0].stdout.write('{"id":2,"result":"stale"}\n');
  children[1].stdout.write('{"id":2,"result":"fresh"}\n');
  assert.equal(await next, 'fresh');
  client.stop();
});

test('worker failures and malformed output surface as errors', async () => {
  const { client, children } = fixture();
  const first = client.request('catalog');
  const rejected = assert.rejects(first, /некорректный ответ/);
  children[0].stdout.write('debug text on protocol channel\n');
  await rejected;
  const next = client.request('trace');
  children[1].stdout.write('{"id":2,"error":{"message":"Missing input file"}}\n');
  await assert.rejects(next, /Missing input file/);
  client.stop();
});

test('timeout stops calculation instead of leaving a busy orphan process', async () => {
  const { client, children } = fixture({ timeout: 10 });
  await assert.rejects(client.request('trace'), /Превышено время/);
  assert.equal(children[0].killed, true);
  assert.equal(client.pending.size, 0);
});

test('IPC allowlist and numeric bounds reject arbitrary methods and parameters', () => {
  const payload = { direction: 'west_to_east', kind: 'ideal', collision_mode: 'exhaustive', ray_count: 217, max_interactions: 48 };
  assert.deepEqual(validateRequest('trace', payload), payload);
  assert.equal(validateRequest('trace', { ...payload, direction: 'both' }).direction, 'both');
  assert.deepEqual(validateRequest('calibrate', { direction: 'both' }), { direction: 'both' });
  for (const ray_count of [0, 50001, NaN, 1.5, '217']) assert.throws(() => validateRequest('trace', { ...payload, ray_count }));
  assert.throws(() => validateRequest('exec', {}));
  assert.throws(() => validateRequest('trace', { ...payload, collision_mode: 'unexpected' }));
  assert.throws(() => validateRequest('calibrate', { direction: 'west_to_east', use_lavsan: 'false' }));
  assert.deepEqual(validateRequest('catalog', { ignored: true }), {});
});
