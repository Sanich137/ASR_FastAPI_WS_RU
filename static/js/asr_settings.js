/**
 * ASRSettings — модуль для хранения эксперт-настроек ASR в localStorage.
 * Ключ: asr_expert_settings_v1
 * Поддерживает миграцию схемы и сброс к дефолтам.
 */
(function() {
  'use strict';

  const STORAGE_KEY = 'asr_expert_settings_v1';
  const SCHEMA_VERSION = 1;

  // Дефолты по Pydantic-моделям (SyncASRRequest, PostFileRequest, WSConfigMessage)
  const DEFAULTS = {
    url: {
      keep_raw: true,
      do_echo_clearing: true,
      do_dialogue: true,
      do_punctuation: true,
      do_diarization: false,
      make_mono: true,
      diar_vad_sensity: 3,
      do_auto_speech_speed_correction: false,
      speech_speed_correction_multiplier: 1.0,
      use_batch: true,
      batch_size: 8,
      expert: false,
      fast_speech: false,
      split_phrases: true
    },
    file: {
      keep_raw: true,
      do_echo_clearing: false,
      do_dialogue: true,
      do_punctuation: true,
      do_diarization: false,
      make_mono: true,
      diar_vad_sensity: 3,
      do_auto_speech_speed_correction: false,
      speech_speed_correction_multiplier: 1.0,
      use_batch: true,
      batch_size: 8,
      expert: false,
      fast_speech: false,
      split_phrases: true
    },
    ws: {
      sample_rate: 16000,
      audio_format: 'pcm16',
      audio_transport: 'json_base64',
      wait_null_answers: true,
      do_dialogue: true,
      do_punctuation: true,
      channel_name: null,
      use_base64: true,
      expert: false,
      fast_speech: false,
      split_phrases: true
    }
  };

  function _loadRaw() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      console.warn('ASRSettings: failed to parse localStorage, resetting');
      return null;
    }
  }

  function _saveRaw(data) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  }

  function _ensureStorage() {
    let data = _loadRaw();
    if (!data || data._schema !== SCHEMA_VERSION) {
      data = { _schema: SCHEMA_VERSION };
      Object.keys(DEFAULTS).forEach(mode => {
        data[mode] = { ...DEFAULTS[mode] };
      });
      _saveRaw(data);
    }
    // Гарантируем наличие всех полей (если схема менялась частично)
    Object.keys(DEFAULTS).forEach(mode => {
      data[mode] = data[mode] || {};
      Object.keys(DEFAULTS[mode]).forEach(key => {
        if (!(key in data[mode])) {
          data[mode][key] = DEFAULTS[mode][key];
        }
      });
    });
    return data;
  }

  const ASRSettings = {
    /**
     * Вернуть настройки для режима (url | file | ws).
     * Всегда возвращает полный объект с дефолтами для отсутствующих ключей.
     */
    getFor(mode) {
      const data = _ensureStorage();
      return { ...DEFAULTS[mode], ...(data[mode] || {}) };
    },

    /**
     * Сохранить настройки для режима.
     * @param {string} mode — 'url', 'file', 'ws'
     * @param {object} values — объект с обновлёнными значениями
     */
    setFor(mode, values) {
      const data = _ensureStorage();
      data[mode] = { ...(data[mode] || {}), ...values };
      _saveRaw(data);
    },

    /**
     * Сбросить настройки режима к дефолтам.
     */
    reset(mode) {
      const data = _ensureStorage();
      data[mode] = { ...DEFAULTS[mode] };
      _saveRaw(data);
    },

    /**
     * Полный сброс всех настроек.
     */
    resetAll() {
      const data = { _schema: SCHEMA_VERSION };
      Object.keys(DEFAULTS).forEach(mode => {
        data[mode] = { ...DEFAULTS[mode] };
      });
      _saveRaw(data);
    },

    /**
     * Проверить и при необходимости мигрировать схему.
     * При несовпадении версии — сброс с уведомлением через callback.
     */
    migrate(onReset) {
      const data = _loadRaw();
      if (!data || data._schema !== SCHEMA_VERSION) {
        this.resetAll();
        if (typeof onReset === 'function') onReset();
        return false;
      }
      return true;
    },

    /**
     * Проверить, отличаются ли текущие настройки режима от дефолтов.
     */
    isDirty(mode) {
      const current = this.getFor(mode);
      const defs = DEFAULTS[mode];
      return Object.keys(defs).some(key => current[key] !== defs[key]);
    },

    /**
     * Вернуть дефолтные настройки для режима (копия).
     */
    getDefaults(mode) {
      return { ...DEFAULTS[mode] };
    }
  };

  // Экспорт в глобальную область
  window.ASRSettings = ASRSettings;
})();
