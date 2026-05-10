(function() {
  'use strict';

  function renderBadge(status) {
    const map = {
      completed: 'badge-success',
      active: 'badge-success',
      failed: 'badge-danger',
      cancelled: 'badge-danger',
      processing: 'badge-warning',
      pending: 'badge-warning',
      expired: 'badge-warning',
    };
    const cls = map[status] || 'badge-info';
    return `<span class="badge ${cls}">${status}</span>`;
  }

  function renderDate(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    const dd = String(d.getDate()).padStart(2, '0');
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const yyyy = d.getFullYear();
    const hh = String(d.getHours()).padStart(2, '0');
    const min = String(d.getMinutes()).padStart(2, '0');
    return `${dd}.${mm}.${yyyy} ${hh}:${min}`;
  }

  function renderDuration(sec) {
    if (!sec) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  async function initAdmin() {
    if (!Auth.getAccessToken()) {
      const refreshed = await Auth.refreshToken();
      if (!refreshed) {
        window.location.href = '/admin/login';
        return false;
      }
    }
    return true;
  }

  window.AdminUI = { renderBadge, renderDate, renderDuration, initAdmin };
})();
