(function() {
  let _token = null;

  function setAccessToken(t) { _token = t; }
  function getAccessToken() { return _token; }
  function clearAuth() { _token = null; }

  let _isRefreshing = false;
  let _refreshPromise = null;

  async function apiFetch(url, opts = {}) {
    opts.headers = opts.headers || {};
    if (!opts.credentials) {
      opts.credentials = 'include';
    }

    // Если токена нет, пробуем восстановить сессию через refresh cookie
    if (!_token) {
      await refreshToken();
    }

    if (_token) {
      opts.headers['Authorization'] = 'Bearer ' + _token;
    }

    let resp = await fetch(url, opts);
    if (resp.status === 401) {
      const refreshed = await refreshToken();
      if (refreshed) {
        opts.headers['Authorization'] = 'Bearer ' + _token;
        resp = await fetch(url, opts);
      } else {
        clearAuth();
        window.location.href = '/login';
        return Promise.reject(new Error('Session expired'));
      }
    }
    if (resp.status === 403) {
      clearAuth();
      window.location.href = '/login';
      return Promise.reject(new Error('Forbidden'));
    }
    return resp;
  }

  async function refreshToken() {
    if (_isRefreshing) {
      return await _refreshPromise;
    }
    _isRefreshing = true;
    _refreshPromise = _doRefresh();
    const result = await _refreshPromise;
    _isRefreshing = false;
    _refreshPromise = null;
    return result;
  }

  async function _doRefresh() {
    try {
      const resp = await fetch('/api/v1/auth/refresh', {
        method: 'POST',
        credentials: 'include'
      });
      if (!resp.ok) return false;
      const data = await resp.json();
      if (data.access_token) {
        setAccessToken(data.access_token);
        return true;
      }
      return false;
    } catch (e) {
      return false;
    }
  }

  async function initAuth() {
    const publicPaths = ['/login', '/register', '/tg'];
    if (publicPaths.some(p => window.location.pathname.startsWith(p))) return;
    if (!_token) {
      // Пытаемся восстановить сессию через refresh cookie
      const refreshed = await refreshToken();
      if (!refreshed) {
        window.location.href = '/login';
      }
    }
  }

  window.Auth = { setAccessToken, getAccessToken, clearAuth, apiFetch, initAuth };
})();
