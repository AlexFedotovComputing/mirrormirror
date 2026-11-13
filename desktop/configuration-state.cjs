/** Accepted state belongs to the app, not the disposable computation process. */
class ConfigurationState {
  constructor() {
    this.version = 0;
    this.active = null;
    this.candidate = null;
  }

  payload() {
    return { active_configuration: this.active?.document ?? null, configuration_version: this.version };
  }

  offer(result, expectedVersion, { load = false } = {}) {
    if (expectedVersion !== this.version) throw new Error('Конфигурация уже изменена. Запоздалый результат не применён.');
    if (!result?.document) return { ...result, configuration_version: this.version, active: false };
    this.candidate = { ...result, configuration_version: this.version };
    if (result.status === 'validated' && result.can_activate !== false) {
      return this.accept(this.version, { load });
    }
    return { ...this.candidate, active: false };
  }

  accept(expectedVersion, { load = false } = {}) {
    if (!this.candidate || expectedVersion !== this.version || this.candidate.configuration_version !== this.version) {
      throw new Error('Кандидат устарел. Запустите расчёт или загрузите файл заново.');
    }
    if (this.candidate.status === 'infeasible') throw new Error('Недопустимая конфигурация доступна только для диагностики.');
    if (isManualConfiguration(this.candidate.document) && this.candidate.status !== 'validated') throw new Error('Ручная реконструкция не прошла все проверки. Доступны уточнение, просмотр и сохранение кандидата.');
    if (this.candidate.can_activate === false) throw new Error('Для этой геометрии не установлена общая оптика или конфигурация неполна. Результат можно сохранить как частичный импорт.');
    this.version++;
    this.active = { ...this.candidate, configuration_version: this.version, active: true };
    this.candidate = null;
    return this.active;
  }

  historical() {
    this.version++;
    this.active = this.candidate = null;
    return { active: false, configuration_version: this.version };
  }
}

function isManualConfiguration(document) {
  return Object.values(document?.datasets ?? {}).some(dataset =>
    dataset?.model_kind === 'manual_reconstruction' || !!dataset?.manual_references);
}

module.exports = { ConfigurationState, isManualConfiguration };
