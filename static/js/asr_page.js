(function() {
  'use strict';

  function renderAsrResult(payload, mode) {
    const settings = ASRSettings.getFor(mode);
    const sentenced = payload.sentenced_data || payload.data?.sentenced_data;
    if (settings.split_phrases && sentenced?.list_of_sentenced_recognitions) {
      return sentenced.list_of_sentenced_recognitions.map(item => {
        const start = item.start != null ? item.start : (item.start_time || '');
        return `${start} - ${item.text || ''}`;
      }).join('\n');
    }
    if (!settings.split_phrases && sentenced?.full_text_only) {
      const ft = sentenced.full_text_only;
      return Array.isArray(ft) ? ft.join('\n') : String(ft);
    }
    return sentenced?.raw_text_sentenced_recognition || payload.data?.text || payload.data?.raw_data?.channel_1?.map(x => x.data?.text).join('\n') || JSON.stringify(payload, null, 2);
  }

  // --- Табы ---
  function switchTab(name) {
    document.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('panel-' + name).style.display = 'block';
    document.getElementById('tab-' + name).classList.add('active');
  }
  window.switchTab = switchTab;

  // --- Утилиты ---
  function copyText(elId) {
    const text = document.getElementById(elId).textContent;
    navigator.clipboard.writeText(text).then(() => UI.toast('Скопировано', 'success'));
  }
  window.copyText = copyText;

  function downloadText(elId, filename) {
    const text = document.getElementById(elId).textContent;
    const blob = new Blob([text], {type:'text/plain'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
  }
  window.downloadText = downloadText;

  // --- URL ---
  function validateExpert(mode) {
    let ok = true;
    const showErr = (id, msg) => {
      const el = document.getElementById(mode + '_err_' + id);
      if (el) { el.textContent = msg; el.style.display = msg ? 'block' : 'none'; }
    };
    if (mode !== 'ws') {
      const sensity = parseInt(document.getElementById(mode + '_diar_vad_sensity').value);
      if (isNaN(sensity) || sensity < 1 || sensity > 5) {
        showErr('diar_vad_sensity', 'Допустимые значения: 1–5'); ok = false;
      } else { showErr('diar_vad_sensity', ''); }

      const speed = parseFloat(document.getElementById(mode + '_speech_speed_correction_multiplier').value);
      if (isNaN(speed) || speed <= 0) {
        showErr('speech_speed_correction_multiplier', 'Должно быть > 0'); ok = false;
      } else { showErr('speech_speed_correction_multiplier', ''); }

      const batch = parseInt(document.getElementById(mode + '_batch_size').value);
      if (isNaN(batch) || batch < 1) {
        showErr('batch_size', 'Минимум 1'); ok = false;
      } else { showErr('batch_size', ''); }
    } else {
      const sampleRate = parseInt(document.getElementById('ws_sample_rate').value);
      if (isNaN(sampleRate) || sampleRate < 8000 || sampleRate > 48000) {
        showErr('sample_rate', 'Допустимый диапазон: 8000–48000'); ok = false;
      } else { showErr('sample_rate', ''); }
    }
    return ok;
  }
  window.validateExpert = validateExpert;

  function getUrlParams() {
    const expert = document.getElementById('url_expert').checked;
    const defaults = ASRSettings.getDefaults('url');
    const domBool = (id) => document.getElementById(id).checked;
    const domInt = (id, def) => {
      const v = parseInt(document.getElementById(id).value);
      return isNaN(v) ? def : v;
    };
    const domFloat = (id, def) => {
      const v = parseFloat(document.getElementById(id).value);
      return isNaN(v) ? def : v;
    };
    const fastSpeech = domBool('url_fast_speech');
    const splitPhrases = domBool('url_split_phrases');
    return {
      AudioFileUrl: document.getElementById('urlInput').value.trim(),
      keep_raw: expert ? domBool('url_keep_raw') : defaults.keep_raw,
      do_echo_clearing: expert ? domBool('url_do_echo_clearing') : defaults.do_echo_clearing,
      do_dialogue: splitPhrases ? true : (expert ? domBool('url_do_dialogue') : defaults.do_dialogue),
      do_punctuation: splitPhrases ? true : (expert ? domBool('url_do_punctuation') : defaults.do_punctuation),
      do_diarization: expert ? domBool('url_do_diarization') : defaults.do_diarization,
      make_mono: expert ? domBool('url_make_mono') : defaults.make_mono,
      diar_vad_sensity: expert ? domInt('url_diar_vad_sensity', defaults.diar_vad_sensity) : defaults.diar_vad_sensity,
      do_auto_speech_speed_correction: fastSpeech ? true : (expert ? domBool('url_do_auto_speech_speed_correction') : defaults.do_auto_speech_speed_correction),
      speech_speed_correction_multiplier: expert ? domFloat('url_speech_speed_correction_multiplier', defaults.speech_speed_correction_multiplier) : defaults.speech_speed_correction_multiplier,
      use_batch: fastSpeech ? false : (expert ? domBool('url_use_batch') : defaults.use_batch),
      batch_size: expert ? domInt('url_batch_size', defaults.batch_size) : defaults.batch_size,
    };
  }

  async function sendUrl() {
    const btn = document.getElementById('btnUrl');
    UI.setLoading(btn, true);
    const payload = getUrlParams();
    if (!payload.AudioFileUrl) { UI.toast('Введите ссылку', 'warning'); UI.setLoading(btn, false); return; }
    if (document.getElementById('url_expert').checked && !validateExpert('url')) {
      UI.setLoading(btn, false); return;
    }
    try {
      const resp = await Auth.apiFetch('/api/v1/asr/url', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      const el = document.getElementById('urlResult');
      const pre = document.getElementById('urlResultText');
      el.style.display = 'block';
      pre.textContent = renderAsrResult(data, 'url');
    } catch (e) {
      UI.toast('Ошибка: ' + e.message, 'error');
    } finally {
      UI.setLoading(btn, false);
    }
  }
  window.sendUrl = sendUrl;

  // --- Файл (drag & drop) ---
  let selectedFile = null;

  function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.style.borderColor = 'var(--color-border)';
    if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
  }
  window.handleDrop = handleDrop;

  function handleFileSelect(file) {
    selectedFile = file;
    document.getElementById('btnFile').disabled = false;
    UI.toast('Файл выбран: ' + file.name, 'info');
  }
  window.handleFileSelect = handleFileSelect;

  function getFileParams() {
    const expert = document.getElementById('file_expert').checked;
    const defaults = ASRSettings.getDefaults('file');
    const domBool = (id) => document.getElementById(id).checked;
    const domInt = (id, def) => {
      const v = parseInt(document.getElementById(id).value);
      return isNaN(v) ? def : v;
    };
    const domFloat = (id, def) => {
      const v = parseFloat(document.getElementById(id).value);
      return isNaN(v) ? def : v;
    };
    const fastSpeech = domBool('file_fast_speech');
    const splitPhrases = domBool('file_split_phrases');
    const form = new FormData();
    form.append('keep_raw', expert ? domBool('file_keep_raw') : defaults.keep_raw);
    form.append('do_echo_clearing', expert ? domBool('file_do_echo_clearing') : defaults.do_echo_clearing);
    form.append('do_dialogue', splitPhrases ? true : (expert ? domBool('file_do_dialogue') : defaults.do_dialogue));
    form.append('do_diarization', expert ? domBool('file_do_diarization') : defaults.do_diarization);
    form.append('do_punctuation', splitPhrases ? true : (expert ? domBool('file_do_punctuation') : defaults.do_punctuation));
    form.append('make_mono', expert ? domBool('file_make_mono') : defaults.make_mono);
    form.append('diar_vad_sensity', expert ? domInt('file_diar_vad_sensity', defaults.diar_vad_sensity) : defaults.diar_vad_sensity);
    form.append('do_auto_speech_speed_correction', fastSpeech ? true : (expert ? domBool('file_do_auto_speech_speed_correction') : defaults.do_auto_speech_speed_correction));
    form.append('speech_speed_correction_multiplier', expert ? domFloat('file_speech_speed_correction_multiplier', defaults.speech_speed_correction_multiplier) : defaults.speech_speed_correction_multiplier);
    form.append('use_batch', fastSpeech ? false : (expert ? domBool('file_use_batch') : defaults.use_batch));
    form.append('batch_size', expert ? domInt('file_batch_size', defaults.batch_size) : defaults.batch_size);
    return form;
  }

  async function sendFile() {
    if (!selectedFile) return;
    const btn = document.getElementById('btnFile');
    UI.setLoading(btn, true);
    document.getElementById('fileProgress').style.display = 'block';
    if (document.getElementById('file_expert').checked && !validateExpert('file')) {
      UI.setLoading(btn, false); document.getElementById('fileProgress').style.display = 'none'; return;
    }
    const form = getFileParams();
    form.append('file', selectedFile);
    try {
      const resp = await Auth.apiFetch('/api/v1/asr/file', {method:'POST', body:form});
      const data = await resp.json();
      document.getElementById('fileResult').style.display = 'block';
      document.getElementById('fileResultText').textContent = renderAsrResult(data, 'file');
    } catch (e) {
      UI.toast('Ошибка: ' + e.message, 'error');
    } finally {
      UI.setLoading(btn, false);
      document.getElementById('fileProgress').style.display = 'none';
    }
  }
  window.sendFile = sendFile;

  document.getElementById('dropZone').addEventListener('click', () => document.getElementById('fileInput').click());

  // --- WebSocket (legacy logic from index.html, adapted for new protocol) ---
  let wsSockets = [];

  function onWsFileSelected() {
    const f = document.getElementById('wsFileInput').files[0];
    document.getElementById('btnWs').disabled = !f;
  }
  window.onWsFileSelected = onWsFileSelected;

  function setWsStatus(status) {
    const dot = document.getElementById('wsStatusDot');
    const txt = document.getElementById('wsStatusText');
    if (status === 'connected') { dot.style.background = 'var(--color-success)'; txt.textContent = 'Подключено'; }
    else if (status === 'connecting') { dot.style.background = 'var(--color-warning)'; txt.textContent = 'Подключение...'; }
    else { dot.style.background = 'var(--color-danger)'; txt.textContent = 'Отключено'; }
  }

  function readWavHeader(arrayBuffer) {
    const dataView = new DataView(arrayBuffer);
    if (String.fromCharCode(...new Uint8Array(arrayBuffer, 0, 4)) !== 'RIFF') {
      throw new Error('Файл не является WAV файлом');
    }
    const sampleRate = dataView.getUint32(24, true);
    const numChannels = dataView.getUint16(22, true);
    const bitsPerSample = dataView.getUint16(34, true);
    return { sampleRate, numChannels, bitsPerSample };
  }

  function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  }

  function getWsParams() {
    const expert = document.getElementById('ws_expert').checked;
    const defaults = ASRSettings.getDefaults('ws');
    const domBool = (id) => document.getElementById(id).checked;
    const domInt = (id, def) => {
      const v = parseInt(document.getElementById(id).value);
      return isNaN(v) ? def : v;
    };
    const domString = (id, def) => {
      const el = document.getElementById(id);
      return el ? (el.value || def) : def;
    };
    const fastSpeech = domBool('ws_fast_speech');
    const splitPhrases = domBool('ws_split_phrases');
    return {
      sample_rate: defaults.sample_rate,
      audio_format: expert ? domString('ws_audio_format', defaults.audio_format) : defaults.audio_format,
      audio_transport: expert ? domString('ws_audio_transport', defaults.audio_transport) : defaults.audio_transport,
      wait_null_answers: expert ? domBool('ws_wait_null_answers') : defaults.wait_null_answers,
      do_dialogue: splitPhrases ? true : (expert ? domBool('ws_do_dialogue') : defaults.do_dialogue),
      do_punctuation: splitPhrases ? true : (expert ? domBool('ws_do_punctuation') : defaults.do_punctuation),
      channel_name: expert ? (domString('ws_channel_name', defaults.channel_name) || null) : defaults.channel_name,
    };
  }

  async function sendChannel(arrayBuffer, channel, numChannels, chunkSize, sampleRate, wsParams) {
    return new Promise((resolve, reject) => {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/asr/ws`;
      const socket = new WebSocket(wsUrl);
      wsSockets.push(socket);

      const token = Auth.getAccessToken();

      socket.onopen = async function() {
        const useBase64 = wsParams.audio_transport === 'json_base64';
        if (token) {
          socket.send(JSON.stringify({ type: 'auth', access_token: token }));
        }
        socket.send(JSON.stringify({
          type: 'config',
          sample_rate: sampleRate,
          audio_format: wsParams.audio_format,
          audio_transport: wsParams.audio_transport,
          wait_null_answers: wsParams.wait_null_answers,
          do_dialogue: wsParams.do_dialogue,
          do_punctuation: wsParams.do_punctuation,
          channel_name: wsParams.channel_name || ('channel_' + (channel + 1))
        }));

        const dataView = new Uint8Array(arrayBuffer);
        const bytesPerSample = 2;
        let offset = 44 + channel * bytesPerSample;
        let seqNum = 0;

        while (offset < dataView.length) {
          const endOffset = Math.min(offset + chunkSize * numChannels, dataView.length);
          const chunk = new Uint8Array((endOffset - offset) / numChannels);

          for (let i = 0; i < chunk.length; i += bytesPerSample) {
            const sampleOffset = offset + i * numChannels;
            chunk[i] = dataView[sampleOffset];
            chunk[i + 1] = dataView[sampleOffset + 1];
          }

          if (useBase64) {
            const base64Chunk = arrayBufferToBase64(chunk);
            socket.send(JSON.stringify({
              type: 'audio_chunk',
              audio_base64: base64Chunk,
              seq_num: seqNum++
            }));
          } else {
            socket.send(chunk);
          }
          offset += chunkSize * numChannels;
          await new Promise((resolve) => setTimeout(resolve, 100));
        }

        socket.send(JSON.stringify({ type: 'eos' }));
      };

      socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'partial_result') {
          const text = data.data?.text || '';
          document.getElementById('wsPartial').textContent = `[Канал ${channel+1}]: ${text} ...`;
        }
        if (data.type === 'final_result' || data.last_message) {
          const text = renderAsrResult(data, 'ws');
          const current = document.getElementById('wsResultText').textContent;
          document.getElementById('wsResultText').textContent = current + (current ? '\n\n' : '') + `=== Канал ${channel+1} ===\n${text}`;
          document.getElementById('wsPartial').textContent = '';
        }
        if (data.last_message) {
          socket.close();
          resolve();
        }
      };

      socket.onerror = function(error) {
        UI.toast('Ошибка WebSocket канала ' + (channel+1), 'error');
        reject(error);
      };

      socket.onclose = function() {
        console.log(`Сокет для канала ${channel+1} закрыт`);
      };
    });
  }

  async function sendAllChannels(arrayBuffer, numChannels, chunkSize, sampleRate, wsParams) {
    for (let channel = 0; channel < numChannels; channel++) {
      await sendChannel(arrayBuffer, channel, numChannels, chunkSize, sampleRate, wsParams);
      console.log("конец канала");
    }
  }

  async function sendWs() {
    const file = document.getElementById('wsFileInput').files[0];
    if (!file) return;
    if (document.getElementById('ws_expert').checked && !validateExpert('ws')) {
      return;
    }
    UI.setLoading(document.getElementById('btnWs'), true);
    document.getElementById('wsResult').style.display = 'block';
    document.getElementById('wsResultText').textContent = '';
    document.getElementById('wsPartial').textContent = '';
    document.getElementById('btnWsStop').style.display = 'inline-flex';
    setWsStatus('connecting');
    wsSockets = [];
    const wsParams = getWsParams();

    try {
      const arrayBuffer = await file.arrayBuffer();
      const { sampleRate, numChannels } = readWavHeader(arrayBuffer);
      const chunkSize = 65536;
      await sendAllChannels(arrayBuffer, numChannels, chunkSize, sampleRate, wsParams);
      setWsStatus('disconnected');
      UI.toast('Распознавание завершено', 'success');
    } catch (e) {
      UI.toast('Ошибка: ' + e.message, 'error');
      setWsStatus('disconnected');
    } finally {
      UI.setLoading(document.getElementById('btnWs'), false);
      document.getElementById('btnWsStop').style.display = 'none';
    }
  }
  window.sendWs = sendWs;

  function stopWs() {
    wsSockets.forEach(s => { try { s.close(); } catch(e) {} });
    wsSockets = [];
    setWsStatus('disconnected');
    UI.setLoading(document.getElementById('btnWs'), false);
    document.getElementById('btnWsStop').style.display = 'none';
  }
  window.stopWs = stopWs;

  // --- История ---
  async function loadHistory() {
    const el = document.getElementById('asrHistory');
    try {
      const resp = await Auth.apiFetch('/api/v1/user/sessions?limit=5');
      const data = await resp.json();
      if (!data || !data.length) { el.innerHTML = '<div class="text-muted">Нет сессий</div>'; return; }
      el.innerHTML = '<table style="width:100%;border-collapse:collapse;">' +
        data.map(s => `<tr style="border-bottom:1px solid var(--color-border);">
          <td style="padding:8px 0;" class="text-sm">${UI.formatDate(s.created_at)}</td>
          <td style="padding:8px 0;" class="text-sm"><span class="badge badge-${s.status==='completed'?'success':s.status==='failed'?'danger':'warning'}">${s.status}</span></td>
          <td style="padding:8px 0;" class="text-sm">${s.session_type}</td>
          <td style="padding:8px 0;" class="text-sm"><a href="/history/${s.id}" style="color:var(--color-primary);">Открыть</a></td>
        </tr>`).join('') + '</table>';
    } catch (e) {
      el.innerHTML = '<div class="text-muted">Не удалось загрузить</div>';
    }
  }

  // Live validation for expert number inputs
  ['url','file','ws'].forEach(mode => {
    const panel = document.getElementById(mode + '_expert_panel');
    if (!panel) return;
    panel.querySelectorAll('input[type="number"]').forEach(input => {
      input.addEventListener('input', () => {
        if (document.getElementById(mode + '_expert').checked) {
          validateExpert(mode);
        }
      });
    });
  });

  // Инициализация
  loadHistory();
})();
