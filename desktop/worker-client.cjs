const { spawn } = require('node:child_process');
const { createInterface } = require('node:readline');
const { existsSync } = require('node:fs');
const path = require('node:path');

/** One Python process; stdout is a JSON-lines channel, stderr is diagnostics. */
class WorkerClient {
  constructor(root, { spawnProcess = spawn, timeout = 600_000, python } = {}) {
    this.root = root;
    this.spawnProcess = spawnProcess;
    this.timeout = timeout;
    const venv = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    this.python = python || process.env.MIRRORLAB_PYTHON || (existsSync(venv) ? venv : 'python');
    this.pending = new Map();
    this.sequence = 0;
    this.child = null;
    this.stderr = '';
  }

  start() {
    if (this.child) return;
    this.stderr = '';
    const child = this.spawnProcess(this.python, ['-u', '-m', 'mirrorlab.worker'], {
      cwd: this.root,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
      stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true,
    });
    this.child = child;
    const lines = createInterface({ input: child.stdout });
    lines.on('line', (line) => {
      if (this.child !== child || !line.trim()) return;
      let message;
      try { message = JSON.parse(line); }
      catch { this.stop('Python вернул некорректный ответ. ' + line.slice(0, 200)); return; }
      const pending = this.pending.get(message.id);
      if (!pending) return;
      if ((message.job_id !== undefined && message.job_id !== pending.jobId) ||
          (message.configuration_version !== undefined && message.configuration_version !== pending.version)) {
        if (message.type !== 'progress') {
          clearTimeout(pending.timer);
          this.pending.delete(message.id);
          pending.reject(new Error('Ответ относится к другой версии конфигурации. Состояние не изменено.'));
        }
        return;
      }
      if (message.type === 'progress') {
        pending.onProgress?.(message);
        return;
      }
      if (message.type && !['result', 'error'].includes(message.type)) return;
      clearTimeout(pending.timer);
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(message.error.message || String(message.error)));
      else pending.resolve(message.result);
    });
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk) => { this.stderr = (this.stderr + chunk).slice(-8000); });
    child.stdin.on('error', (error) => {
      if (this.child === child) this.stop('Не удалось передать запрос Python: ' + error.message);
    });
    child.on('error', (error) => {
      if (this.child === child) this.stop(`Не удалось запустить Python (${this.python}). Установите зависимости и проверьте MIRRORLAB_PYTHON. ${error.message}`);
    });
    child.on('exit', (code, signal) => {
      lines.close();
      if (this.child === child) this.stop(`Расчётный процесс завершился (${code ?? signal}). ${this.stderr.trim()}`);
    });
  }

  request(method, payload = {}, { onProgress } = {}) {
    if (this.pending.size >= 2 || (method !== 'catalog' && [...this.pending.values()].some(item => item.method !== 'catalog'))) {
      return Promise.reject(new Error('Дождитесь завершения текущего расчёта.'));
    }
    this.start();
    const id = ++this.sequence;
    const jobId = `job-${id}`;
    const version = payload.configuration_version ?? 0;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.stop(method === 'optimize'
        ? 'Превышено время оптимизации. Предыдущая конфигурация сохранена; проверьте диагностику данных и ограничений.'
        : 'Превышено время расчёта. Предыдущая конфигурация сохранена.'), this.timeout);
      this.pending.set(id, { resolve, reject, timer, method, jobId, version, onProgress });
      this.child.stdin.write(JSON.stringify({ id, job_id: jobId, configuration_version: version, method, payload }) + '\n', 'utf8');
    });
  }

  stop(reason = 'Расчёт отменён.') {
    const child = this.child;
    this.child = null;
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer);
      reject(new Error(reason));
    }
    this.pending.clear();
    if (child && !child.killed) child.kill();
  }
}

function validateRequest(method, payload) {
  if (!['catalog', 'trace', 'calibrate', 'cancel', 'optimize', 'save_configuration',
    'load_configuration', 'import_legacy', 'accept_candidate', 'use_historical'].includes(method)) throw new Error('Неизвестная команда.');
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('Некорректные параметры.');
  if (method === 'catalog' || method === 'cancel') return {};
  if (method === 'accept_candidate') {
    if (!Number.isInteger(payload.configuration_version) || payload.configuration_version < 0) throw new Error('Некорректная версия конфигурации.');
    return { configuration_version: payload.configuration_version };
  }
  if (!['both', 'west_to_east', 'east_to_west'].includes(payload.direction)) throw new Error('Выберите направление лучей.');
  if (['save_configuration', 'load_configuration', 'import_legacy', 'optimize', 'use_historical'].includes(method)) {
    if (!['ideal', 'real'].includes(payload.kind)) throw new Error('Выберите конфигурацию.');
    const clean = { direction: payload.direction, kind: payload.kind };
    if (method === 'save_configuration') {
      if (!['active', 'candidate'].includes(payload.source ?? 'active')) throw new Error('Выберите сохраняемую конфигурацию.');
      clean.source = payload.source ?? 'active';
    }
    if (method === 'import_legacy') {
      if (!['unknown', 'legacy', 'survey'].includes(payload.coordinate_frame ?? 'unknown')) throw new Error('Неизвестная система координат.');
      clean.coordinate_frame = payload.coordinate_frame ?? 'unknown';
    }
    return clean;
  }
  if (method === 'calibrate') {
    if (payload.use_lavsan !== undefined && typeof payload.use_lavsan !== 'boolean') throw new Error('Некорректный флаг использования лавсана.');
    return payload.use_lavsan === undefined ? { direction: payload.direction }
      : { direction: payload.direction, use_lavsan: payload.use_lavsan };
  }
  if (!['ideal', 'real'].includes(payload.kind)) throw new Error('Выберите конфигурацию.');
  if (!['exhaustive', 'expected'].includes(payload.collision_mode)) throw new Error('Выберите режим пересечений.');
  for (const [name, max] of [['ray_count', 50000], ['max_interactions', 100]]) {
    if (!Number.isInteger(payload[name]) || payload[name] < 1 || payload[name] > max) throw new Error(`${name}: требуется целое число от 1 до ${max}.`);
  }
  return { direction: payload.direction, kind: payload.kind, collision_mode: payload.collision_mode,
    ray_count: payload.ray_count, max_interactions: payload.max_interactions };
}

module.exports = { WorkerClient, validateRequest };
