const { test } = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { PassThrough } = require('node:stream');
const { WorkerClient, validateRequest } = require('../desktop/worker-client.cjs');
const { ConfigurationState } = require('../desktop/configuration-state.cjs');

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

test('progress keeps the request pending and forwards only the matching job and state', async () => {
  const { client, children } = fixture();
  const progress = [];
  let completed = false;
  const pending = client.request('optimize', { configuration_version: 7 }, { onProgress: message => progress.push(message) });
  pending.then(() => { completed = true; });
  const request = JSON.parse(children[0].stdin.read().toString());
  const send = message => children[0].stdout.write(JSON.stringify(message) + '\n');
  send({ id: request.id, type: 'progress', job_id: 'old-job', configuration_version: 7, progress: { stage: 'template' } });
  send({ id: request.id, type: 'progress', job_id: request.job_id, configuration_version: 6, progress: { stage: 'template' } });
  send({ id: request.id, type: 'progress', job_id: request.job_id, configuration_version: 7, progress: { stage: 'fitting' } });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(completed, false);
  assert.equal(client.pending.size, 1);
  assert.equal(progress.length, 1);
  assert.equal(progress[0].progress.stage, 'fitting');
  await assert.rejects(client.request('trace'), /Дождитесь/);
  send({ id: request.id, type: 'result', job_id: request.job_id, configuration_version: 7, result: { status: 'partial' } });
  assert.deepEqual(await pending, { status: 'partial' });
  client.stop();
});

test('terminal results for the wrong configuration cannot be applied', async () => {
  const { client, children } = fixture();
  const pending = client.request('optimize', { configuration_version: 3 });
  children[0].stdout.write('{"id":1,"type":"result","job_id":"job-1","configuration_version":2,"result":{"status":"validated"}}\n');
  await assert.rejects(pending, /другой версии/);
  client.stop();
});

test('optimizer timeout describes optimization and preserves application state', async () => {
  const { client } = fixture({ timeout: 10 });
  await assert.rejects(client.request('optimize'), error => /оптимизации/.test(error.message) && !/фотон|число лучей/.test(error.message));
});

test('partial preview, cancellation and process restart do not replace accepted state', async () => {
  const state = new ConfigurationState();
  const first = { document: { datasets: { east_to_west: { marker: 1 } } }, status: 'validated' };
  state.offer(first, 0);
  assert.equal(state.version, 1);
  state.offer({ document: { datasets: { east_to_west: { marker: 2 } } }, status: 'partial' }, 1);
  assert.equal(state.active.document.datasets.east_to_west.marker, 1);
  const { client, children } = fixture();
  const pending = client.request('optimize', state.payload());
  const rejected = assert.rejects(pending, /отменён/);
  client.stop();
  await rejected;
  const next = client.request('trace', state.payload());
  const request = JSON.parse(children[1].stdin.read().toString());
  assert.equal(request.payload.active_configuration.datasets.east_to_west.marker, 1);
  assert.equal(request.configuration_version, 1);
  children[1].stdout.write('{"id":2,"type":"result","job_id":"job-2","configuration_version":1,"result":{}}\n');
  await next;
  state.accept(1);
  assert.equal(state.active.status, 'partial');
  assert.equal(state.active.document.datasets.east_to_west.marker, 2);
  client.stop();
});

test('load preserves partial status and stale candidates cannot replace it', () => {
  const state = new ConfigurationState();
  state.offer({ document: { id: 'a' }, status: 'partial' }, 0, { load: true });
  assert.equal(state.active.status, 'partial');
  assert.throws(() => state.offer({ document: { id: 'old' }, status: 'validated' }, 0), /Запоздалый/);
  assert.equal(state.active.document.id, 'a');
  const loadedInvalid = state.offer({ document: { id: 'invalid-load' }, status: 'infeasible' }, 1, { load: true });
  assert.equal(loadedInvalid.active, false);
  assert.equal(state.active.document.id, 'a');
  state.offer({ document: { id: 'bad' }, status: 'infeasible' }, 1);
  assert.throws(() => state.accept(1), /Недопустимая/);
  assert.equal(state.active.document.id, 'a');
  state.historical();
  assert.equal(state.active, null);
  assert.throws(() => state.accept(1), /устарел/);
});

test('file operations allow documented contexts, not renderer-provided paths or state', () => {
  assert.deepEqual(validateRequest('load_configuration', { direction: 'both', kind: 'real', path: 'arbitrary', active_configuration: {} }), { direction: 'both', kind: 'real' });
  assert.deepEqual(validateRequest('import_legacy', { direction: 'east_to_west', kind: 'real', coordinate_frame: 'survey' }), {
    direction: 'east_to_west', kind: 'real', coordinate_frame: 'survey',
  });
  assert.throws(() => validateRequest('import_legacy', { direction: 'east_to_west', kind: 'real', coordinate_frame: 'guess' }));
  assert.throws(() => validateRequest('accept_candidate', { configuration_version: -1 }));
});
