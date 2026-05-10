(function() {
  'use strict';

  class AdminDashboardWS {
    constructor() {
      this.ws = null;
      this.reconnectAttempts = 0;
      this.maxReconnectAttempts = 10;
      this.baseDelay = 1000;
      this.maxDelay = 30000;
      this.heartbeatInterval = null;
      this._isConnected = false;
    }

    connect(token) {
      this._token = token;
      this._connect();
    }

    _connect() {
      if (this.ws) {
        try { this.ws.close(); } catch(e) {}
      }

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${protocol}//${window.location.host}/api/v1/admin/ws`;

      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this._isConnected = true;
        if (this._token) {
          this.ws.send(JSON.stringify({ type: 'auth', access_token: this._token }));
        }
        this._startHeartbeat();
        this._setWidgetsStatus('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'metrics') {
            this._updateWidgets(msg.data);
          } else if (msg.type === 'alert') {
            this._showAlert(msg.data);
          }
        } catch (e) {
          console.error('WS parse error', e);
        }
      };

      this.ws.onerror = () => {
        this._setWidgetsStatus('error');
      };

      this.ws.onclose = (event) => {
        this._isConnected = false;
        this._stopHeartbeat();
        // Если закрытие из-за истёкшего токена — пробуем refresh и сразу переподключаемся
        if (event.code === 1008 || event.code === 1011) {
          Auth.refreshToken().then((ok) => {
            if (ok) {
              this.reconnectAttempts = 0;
              this._connect();
            } else {
              Auth.clearAuth();
              window.location.href = '/admin/login';
            }
          });
          return;
        }
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          const delay = Math.min(this.baseDelay * Math.pow(2, this.reconnectAttempts), this.maxDelay);
          this.reconnectAttempts++;
          setTimeout(() => this._connect(), delay);
          this._setWidgetsStatus('reconnecting');
        } else {
          this._setWidgetsStatus('disconnected');
        }
      };
    }

    _startHeartbeat() {
      this.heartbeatInterval = setInterval(() => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000);
    }

    _stopHeartbeat() {
      if (this.heartbeatInterval) {
        clearInterval(this.heartbeatInterval);
        this.heartbeatInterval = null;
      }
    }

    _updateWidgets(data) {
      const setText = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = `<div class="text-muted text-sm">${id.split('-')[1].toUpperCase()}</div><div style="font-size:24px; font-weight:700; margin-top:4px;">${text}</div>`;
      };

      setText('widget-cpu', data.cpu_utilization_percent != null ? data.cpu_utilization_percent.toFixed(1) + '%' : '—');
      setText('widget-gpu', data.gpu_utilization_percent != null ? data.gpu_utilization_percent.toFixed(1) + '%' : '—');
      setText('widget-tasks', data.active_tasks != null ? data.active_tasks : '—');
      setText('widget-queue', data.queue_depth != null ? data.queue_depth : '—');
      setText('widget-uptime', data.uptime_formatted || '—');
      setText('widget-adapter-status', data.adapter_status || '—');
    }

    _showAlert(data) {
      const banner = document.getElementById('alertBanner');
      const text = document.getElementById('alertText');
      if (!banner || !text) return;
      if (data.cpu_utilization_percent > 90 || data.queue_depth > 50) {
        text.textContent = `CPU: ${data.cpu_utilization_percent?.toFixed(1)}%, Очередь: ${data.queue_depth}`;
        banner.style.display = 'block';
      }
    }

    _setWidgetsStatus(status) {
      const color = status === 'connected' ? 'var(--color-success)' : status === 'reconnecting' ? 'var(--color-warning)' : 'var(--color-danger)';
      const label = status === 'connected' ? 'Live' : status === 'reconnecting' ? 'Reconnect...' : 'Offline';
      // Можно добавить индикатор статуса где-то на странице
    }

    disconnect() {
      this.reconnectAttempts = this.maxReconnectAttempts;
      this._stopHeartbeat();
      if (this.ws) {
        try { this.ws.close(1000, 'Client disconnect'); } catch(e) {}
      }
    }
  }

  window.AdminDashboardWS = AdminDashboardWS;
})();
