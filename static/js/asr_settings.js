(function() {
  'use strict';

  const STORAGE_KEY = 'asr_expert_settings_v1';
  const DEFAULTS = {
    url: {
      expert: false,
      keep_raw: true,
      do_echo_clearing: false,
      do_dialogue: true,
      do_punctuation: true,
      do_diarization: false,
      make_mono: true,
      diar_vad_sensity: 3,
      do_auto_speech_speed_correction: false,
      speech_speed_correction_multiplier: 1.0,
      use_batch: false,
      batch_size: 8,
      fast_speech: false,
      split_phrases: true,
    },
    file: {
      expert: false,
      keep_raw: true,
      do_echo_clearing: false,
      do_dialogue: true,
      do_punctuation: true,
      do_diarization: false,
      make_mono: true,
      diar_vad_sensity: 3,
      do_auto_speech_speed_correction: false,
      speech_speed_correction_multiplier: 1.0,
      use_batch: false,
      batch_size: 8,
      fast_speech: false,
      split_phrases: true,
    },
    ws: {
      expert: false,
      do_dialogue: true,
      do_punctuation: true,
      use_base64: true,
      wait_null_answers: true,
      audio_transport: 'json_base64',
      sample_rate: 16000,
      audio_format: 'pcm16',
      channel_name: '',
      fast_speech: false,
      split_phrases: true,
    }
  };

  function _load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (e) {
      return {};
    }
  }

  function _save(data) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  }

  function getFor(mode) {
    const stored = _load();
    const defs = DEFAULTS[mode] || {};
    return { ...defs, ...(stored[mode] || {}) };
  }

  function setFor(mode, values) {
    const stored = _load();
    stored[mode] = { ...(stored[mode] || {}), ...values };
    _save(stored);
  }

  function reset(mode) {
    const stored = _load();
    if (mode) {
      delete stored[mode];
    } else {
      Object.keys(stored).forEach(k => delete stored[k]);
    }
    _save(stored);
  }

  function getDefaults(mode) {
    return DEFAULTS[mode] || {};
  }

  function isDirty(mode) {
    const current = getFor(mode);
    const defs = getDefaults(mode);
    return Object.keys(defs).some(key => current[key] !== defs[key]);
  }

  window.ASRSettings = { getFor, setFor, reset, getDefaults, isDirty };
})();
