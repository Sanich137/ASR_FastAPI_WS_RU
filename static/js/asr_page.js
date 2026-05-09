(function() {
  'use strict';

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
  async function sendUrl() {
    const btn = document.getElementById('btnUrl');
    UI.setLoading(btn, true);
    const url = document.getElementById('urlInput').value.trim();
    if (!url) { UI.toast('Введите ссылку', 'warning'); UI.setLoading(btn, false); return; }
    try {
      const resp = await Auth.apiFetch('/api/v1/asr/url', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          AudioFileUrl: url,
          keep_raw: document.getElementById('url_keep_raw').checked,
          do_echo_clearing: document.getElementById('url_do_echo').checked,
          do_dialogue: document.getElementById('url_do_dialogue').checked,
          do_punctuation: document.getElementById('url_do_punct').checked
        })
      });
      const data = await resp.json();
      const el = document.getElementById('urlResult');
      const pre = document.getElementById('urlResultText');
      el.style.display = 'block';
      pre.textContent = data.data?.sentenced_data?.raw_text_sentenced_recognition || data.data?.raw_data?.channel_1?.map(x => x.data?.text).join('\n') || JSON.stringify(data, null, 2);
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

  async function sendFile() {
    if (!selectedFile) return;
    const btn = document.getElementById('btnFile');
    UI.setLoading(btn, true);
    document.getElementById('fileProgress').style.display = 'block';
    const form = new FormData();
    form.append('file', selectedFile);
    form.append('keep_raw', document.getElementById('file_keep_raw').checked);
    form.append('do_dialogue', document.getElementById('file_do_dialogue').checked);
    form.append('do_diarization', document.getElementById('file_do_diar').checked);
    form.append('do_punctuation', document.getElementById('file_do_punct').checked);
    try {
      const resp = await Auth.apiFetch('/api/v1/asr/file', {method:'POST', body:form});
      const data = await resp.json();
      document.getElementById('fileResult').style.display = 'block';
      document.getElementById('fileResultText').textContent = data.data?.sentenced_data?.raw_text_sentenced_recognition || data.data?.raw_data?.channel_1?.map(x => x.data?.text).join('\n') || JSON.stringify(data, null, 2);
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

  async function sendChannel(arrayBuffer, channel, numChannels, chunkSize, sampleRate) {
    return new Promise((resolve, reject) => {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/asr/ws`;
      const socket = new WebSocket(wsUrl);
      wsSockets.push(socket);

      const doDialogue = document.getElementById('ws_do_dialogue').checked;
      const doPunctuation = document.getElementById('ws_do_punct').checked;
      const token = Auth.getAccessToken();

      socket.onopen = async function() {
        const useBase64 = document.getElementById('ws_use_base64').checked;
        if (token) {
          socket.send(JSON.stringify({ type: 'auth', access_token: token }));
        }
        socket.send(JSON.stringify({
          type: 'config',
          sample_rate: sampleRate,
          audio_format: 'pcm16',
          audio_transport: useBase64 ? 'json_base64' : 'binary',
          wait_null_answers: false,
          do_dialogue: doDialogue,
          do_punctuation: doPunctuation,
          channel_name: 'channel_' + (channel + 1)
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
          const text = data.sentenced_data?.raw_text_sentenced_recognition || data.data?.text || JSON.stringify(data, null, 2);
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

  async function sendAllChannels(arrayBuffer, numChannels, chunkSize, sampleRate) {
    for (let channel = 0; channel < numChannels; channel++) {
      await sendChannel(arrayBuffer, channel, numChannels, chunkSize, sampleRate);
      console.log("конец канала");
    }
  }

  async function sendWs() {
    const file = document.getElementById('wsFileInput').files[0];
    if (!file) return;
    UI.setLoading(document.getElementById('btnWs'), true);
    document.getElementById('wsResult').style.display = 'block';
    document.getElementById('wsResultText').textContent = '';
    document.getElementById('wsPartial').textContent = '';
    document.getElementById('btnWsStop').style.display = 'inline-flex';
    setWsStatus('connecting');
    wsSockets = [];

    try {
      const arrayBuffer = await file.arrayBuffer();
      const { sampleRate, numChannels } = readWavHeader(arrayBuffer);
      const chunkSize = 65536;
      await sendAllChannels(arrayBuffer, numChannels, chunkSize, sampleRate);
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

  // Инициализация
  loadHistory();
})();
