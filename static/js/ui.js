(function() {
  const container = document.createElement('div');
  container.className = 'toast-container';
  document.body.appendChild(container);

  function toast(msg, type = 'info', duration = 5000) {
    const el = document.createElement('div');
    el.className = 'toast ' + type;
    el.textContent = msg;

    const progress = document.createElement('div');
    progress.className = 'progress';
    progress.style.animationDuration = duration + 'ms';
    el.appendChild(progress);

    container.appendChild(el);
    setTimeout(() => {
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 300);
    }, duration);
  }

  function confirmDialog(title, text, onConfirm) {
    const dialog = document.createElement('dialog');
    dialog.innerHTML = `
      <div style="padding:16px;max-width:360px;">
        <h3 style="margin:0 0 8px;">${escapeHtml(title)}</h3>
        <p style="margin:0 0 16px;color:var(--color-muted);">${escapeHtml(text)}</p>
        <div style="display:flex;gap:8px;justify-content:flex-end;">
          <button class="btn btn-ghost" id="dlg-cancel">Отмена</button>
          <button class="btn btn-danger" id="dlg-confirm">Подтвердить</button>
        </div>
      </div>
    `;
    document.body.appendChild(dialog);
    dialog.showModal();

    dialog.querySelector('#dlg-cancel').onclick = () => { dialog.close(); dialog.remove(); };
    dialog.querySelector('#dlg-confirm').onclick = () => { dialog.close(); dialog.remove(); onConfirm(); };
    dialog.addEventListener('close', () => dialog.remove());
  }

  function setLoading(element, isLoading) {
    if (!element) return;
    if (isLoading) {
      element.disabled = true;
      element.dataset.originalText = element.innerHTML;
      element.innerHTML = '<span class="spinner"></span>';
    } else {
      element.disabled = false;
      element.innerHTML = element.dataset.originalText || element.innerHTML;
    }
  }

  function formatDate(iso) {
    if (!iso) return '-';
    const d = new Date(iso);
    return d.toLocaleString('ru-RU');
  }

  function formatBytes(bytes) {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B','KB','MB','GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function formatDuration(sec) {
    if (!sec) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  window.UI = { toast, confirmDialog, setLoading, formatDate, formatBytes, formatDuration };
})();
