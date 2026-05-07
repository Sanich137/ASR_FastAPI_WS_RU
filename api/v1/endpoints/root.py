from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from models.fast_api_models import V1BaseResponse
from services.ws_metrics import SystemMetricsCollector
from config import settings
from static import mo
router = APIRouter(prefix="", tags=["System"])


@router.get("/", response_model=V1BaseResponse)
async def root():
    """
    Корневой эндпоинт API v1.

    Returns:
        V1BaseResponse: базовый ответ с приветственным сообщением.
    """
    return V1BaseResponse(
        success=True,
        error_description=None,
        data={
            "message": "No_service_selected",
            "available_endpoints": {
                "POST /api/v1/asr/url": "ASR by URL",
                "POST /api/v1/asr/file": "ASR by file upload",
                "GET /api/v1/health/is_alive": "Service health check",
                "WS /api/v1/asr/ws": "WebSocket streaming ASR",
                "GET /docs": "API documentation",
                "/demo": "DEMO UI page"
            },
            "try_addr": f"http://{settings.HOST}:{settings.PORT}/docs"
        }
    )


@router.get("/status", response_model=V1BaseResponse)
async def health_status(request: Request):
    """
    REST endpoint для получения системных метрик (Задача 6.7).
    Возвращает тот же JSON, что и WSStatusResponse.
    """
    metrics: SystemMetricsCollector = request.app.state.metrics_collector
    status = metrics.collect(
        active_connections=request.app.state.ws_manager.active_connections_count,
        max_connections=request.app.state.ws_manager.max_connections,
    )
    return V1BaseResponse(
        success=True,
        error_description=None,
        data=status.model_dump()
    )


@router.get("/monitor", response_class=HTMLResponse)
async def monitor_page():
    """
    Возвращает HTML-страницу мониторинга (Задача 6.7).
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ASR Monitor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
            .card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .card h3 { margin: 0 0 10px; font-size: 14px; color: #666; text-transform: uppercase; }
            .value { font-size: 28px; font-weight: bold; color: #333; }
            .badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; color: white; }
            .badge.idle { background: #28a745; }
            .badge.busy { background: #ffc107; color: #333; }
            .badge.overloaded { background: #dc3545; }
            .progress { width: 100%; height: 8px; background: #e9ecef; border-radius: 4px; margin-top: 8px; overflow: hidden; }
            .progress-bar { height: 100%; background: #007bff; transition: width 0.3s; }
            .log { background: #212529; color: #fff; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; }
            .log-entry { margin-bottom: 4px; }
            .disconnected { color: #dc3545; }
            .connected { color: #28a745; }
        </style>
    </head>
    <body>
        <h1>ASR Monitor</h1>
        <div class="grid">
            <div class="card">
                <h3>GPU Memory</h3>
                <div class="value" id="gpu-mem">—</div>
                <div class="progress"><div class="progress-bar" id="gpu-mem-bar" style="width:0%"></div></div>
            </div>
            <div class="card">
                <h3>GPU Utilization</h3>
                <div class="value" id="gpu-util">—</div>
                <div class="progress"><div class="progress-bar" id="gpu-util-bar" style="width:0%"></div></div>
            </div>
            <div class="card">
                <h3>CPU Memory</h3>
                <div class="value" id="cpu-mem">—</div>
                <div class="progress"><div class="progress-bar" id="cpu-mem-bar" style="width:0%"></div></div>
            </div>
            <div class="card">
                <h3>CPU Utilization</h3>
                <div class="value" id="cpu-util">—</div>
                <div class="progress"><div class="progress-bar" id="cpu-util-bar" style="width:0%"></div></div>
            </div>
            <div class="card">
                <h3>Active Tasks</h3>
                <div class="value" id="active-tasks">—</div>
            </div>
            <div class="card">
                <h3>Connections</h3>
                <div class="value" id="connections">—</div>
            </div>
            <div class="card">
                <h3>Adapter Status</h3>
                <div class="badge idle" id="status-badge">idle</div>
            </div>
            <div class="card">
                <h3>Uptime</h3>
                <div class="value" id="uptime">—</div>
            </div>
        </div>
        <h2>Event Log</h2>
        <div class="log" id="log"></div>

        <script>
            const wsUrl = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/api/v1/asr/ws';
            let ws;
            let reconnectTimer;
            let seq = 0;

            function log(msg, type='info') {
                const el = document.getElementById('log');
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                const time = new Date().toLocaleTimeString();
                entry.textContent = `[${time}] ${msg}`;
                if (type === 'error') entry.style.color = '#dc3545';
                if (type === 'success') entry.style.color = '#28a745';
                el.prepend(entry);
                while (el.children.length > 20) el.lastChild.remove();
            }

            function connect() {
                log('Connecting to ' + wsUrl, 'info');
                ws = new WebSocket(wsUrl);

                ws.onopen = () => {
                    log('Connected', 'success');
                    // Send config
                    ws.send(JSON.stringify({
                        type: 'config',
                        sample_rate: 16000,
                        wait_null_answers: true,
                        do_dialogue: false,
                        do_punctuation: false,
                        channel_name: 'monitor'
                    }));
                    // Request status and subscribe to periodic updates
                    requestStatus();
                };

                ws.onmessage = (event) => {
                    try {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'status_response') {
                            updateUI(msg);
                        }
                    } catch (e) {
                        log('Parse error: ' + e, 'error');
                    }
                };

                ws.onclose = () => {
                    log('Disconnected', 'error');
                    document.getElementById('status-badge').textContent = 'disconnected';
                    document.getElementById('status-badge').className = 'badge overloaded';
                    reconnectTimer = setTimeout(connect, 3000);
                };

                ws.onerror = (err) => {
                    log('WebSocket error', 'error');
                };
            }

            function requestStatus() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'status_request', command: 'get_status'}));
                }
            }

            function updateUI(data) {
                const gpuTotal = data.gpu_memory_total_mb || 1;
                const cpuTotal = data.cpu_memory_total_mb || 1;

                document.getElementById('gpu-mem').textContent = (data.gpu_memory_free_mb !== null ? `${data.gpu_memory_free_mb} / ${data.gpu_memory_total_mb} MB` : 'N/A');
                document.getElementById('gpu-mem-bar').style.width = ((data.gpu_memory_total_mb - data.gpu_memory_free_mb) / gpuTotal * 100) + '%';

                document.getElementById('gpu-util').textContent = (data.gpu_utilization_percent !== null ? data.gpu_utilization_percent.toFixed(1) + '%' : 'N/A');
                document.getElementById('gpu-util-bar').style.width = (data.gpu_utilization_percent || 0) + '%';

                document.getElementById('cpu-mem').textContent = (data.cpu_memory_free_mb !== null ? `${data.cpu_memory_free_mb} / ${data.cpu_memory_total_mb} MB` : 'N/A');
                document.getElementById('cpu-mem-bar').style.width = ((data.cpu_memory_total_mb - data.cpu_memory_free_mb) / cpuTotal * 100) + '%';

                document.getElementById('cpu-util').textContent = (data.cpu_utilization_percent !== null ? data.cpu_utilization_percent.toFixed(1) + '%' : 'N/A');
                document.getElementById('cpu-util-bar').style.width = (data.cpu_utilization_percent || 0) + '%';

                document.getElementById('active-tasks').textContent = data.active_tasks_count;
                document.getElementById('connections').textContent = data.active_connections_count;
                document.getElementById('uptime').textContent = data.uptime_sec.toFixed(0) + 's';

                const badge = document.getElementById('status-badge');
                badge.textContent = data.adapter_status;
                badge.className = 'badge ' + data.adapter_status;
            }

            connect();
            // Fallback: request status every 5s if push fails
            setInterval(requestStatus, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
