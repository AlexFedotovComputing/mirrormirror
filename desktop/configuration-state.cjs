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
    if ((load || result.status === 'validated') && result.status !== 'infeasible' && result.can_activate !== false) {
      return this.accept(this.version, { load });
    }
    return { ...this.candidate, active: false };
  }

  accept(expectedVersion, { load = false } = {}) {
    if (!this.candidate || expectedVersion !== this.version || this.candidate.configuration_version !== this.version) {
      throw new Error('Кандидат устарел. Запустите расчёт или загрузите файл заново.');
    }
    if (!load && this.candidate.status === 'infeasible') throw new Error('Недопустимая конфигурация доступна только для диагностики.');
    if (!load && this.candidate.can_activate === false) throw new Error('Для этой геометрии не установлена общая оптика. Результат можно сохранить как частичный импорт.');
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

module.exports = { ConfigurationState };
