(function() {
  class ASRClient {
    constructor() {
      this.ws = null;
      this.reconnectAttempts = 0;
      this.maxReconnectAttempts = 5;
      this.baseDelay = 1000;
      this.maxDelay = 30000;
      this._onPartial = null;
      this._onFinal = null;
      this._onError = null;
      this._pendingMessages = [];
      this._isConnected = false;
      this._authToken = null;
      this._config = null;
    }

    onPartial(cb) { this._onPartial = cb; }
    onFinal(cb) { this._onFinal = cb; }
    onError(cb) { this._onError = cb; }

    connect(authToken, config) {
      this._authToken = authToken;
      this._config = config;
      this._connect();
    }

    _connect() {
      if (this.ws) {
        try { this.ws.close(); } catch(e) {}
      }

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = protocol + '//' + window.location.host + '/api/v1/asr/ws';

      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this._isConnected = true;
        if (this._authToken) {
          this.ws.send(JSON.stringify({ type: 'auth', access_token: this._authToken }));
        }
        this.ws.send(JSON.stringify({ type: 'config', ...this._config }));
        while (this._pendingMessages.length > 0) {
          const msg = this._pendingMessages.shift();
          this.ws.send(msg);
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'partial_result' && this._onPartial) {
            this._onPartial(msg);
          } else if (msg.type === 'final_result' && this._onFinal) {
            this._onFinal(msg);
          } else if (msg.type === 'error' && this._onError) {
            this._onError(msg);
          }
        } catch (e) {
          console.error('WS parse error', e);
        }
      };

      this.ws.onerror = () => {
        if (this._onError) this._onError({ code: 'ws_error', message: 'WebSocket error' });
      };

      this.ws.onclose = (event) => {
        this._isConnected = false;
        if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
          this._doReconnect();
        }
      };
    }

    _doReconnect() {
      const delay = Math.min(this.baseDelay * Math.pow(2, this.reconnectAttempts), this.maxDelay);
      this.reconnectAttempts++;
      setTimeout(() => this._connect(), delay);
    }

    sendAudioChunk(base64Chunk, seqNum) {
      const payload = JSON.stringify({ type: 'audio_chunk', audio_base64: base64Chunk, seq_num: seqNum });
      if (this._isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(payload);
      } else {
        this._pendingMessages.push(payload);
      }
    }

    sendEOS() {
      const payload = JSON.stringify({ type: 'eos' });
      if (this._isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(payload);
      } else {
        this._pendingMessages.push(payload);
      }
    }

    disconnect() {
      this.reconnectAttempts = this.maxReconnectAttempts;
      if (this.ws) {
        try { this.ws.close(1000, 'Client disconnect'); } catch(e) {}
      }
      this._isConnected = false;
    }
  }

  window.ASRClient = ASRClient;
})();
