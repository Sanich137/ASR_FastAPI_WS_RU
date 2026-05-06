"""
Модуль services/ws_metrics.py
Содержит класс SystemMetricsCollector для сбора системных метрик
(GPU, CPU, активные задачи, статус адаптера) и формирования
WSStatusResponse для передачи через WebSocket.
"""

import time
import logging
from typing import Optional, Tuple

from models.ws_models import WSStatusResponse
from config import settings

logger = logging.getLogger(__name__)


class SystemMetricsCollector:
    """
    Собирает системные метрики и вычисляет статус адаптера ASR.

    Attributes:
        start_time: Unix-timestamp запуска коллектора (для uptime).
        _active_tasks: Счётчик активных задач распознавания.
        _nvml_handle: Опциональный handle pynvml (из app.state).
    """

    def __init__(
        self,
        nvml_handle: Optional[object] = None,
        start_time: Optional[float] = None,
    ) -> None:
        """
        Инициализирует коллектор метрик.

        Args:
            nvml_handle: Handle pynvml (например, из app.state.nvml_handle).
            start_time: Unix-timestamp старта приложения. По умолчанию — текущее время.
        """
        self.start_time: float = start_time if start_time is not None else time.time()
        self._nvml_handle: Optional[object] = nvml_handle
        self._active_tasks: int = 0

    def get_gpu_stats(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Возвращает свободную и общую память GPU в мегабайтах.

        Returns:
            Кортеж (free_mb, total_mb). Если pynvml недоступен или не инициализирован —
            возвращает (None, None).
        """
        if self._nvml_handle is None:
            return None, None
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            free_mb = int(info.free / 1024 / 1024)
            total_mb = int(info.total / 1024 / 1024)
            return free_mb, total_mb
        except Exception as exc:
            logger.warning("Failed to get GPU stats: %s", exc)
            return None, None

    def get_cpu_stats(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Возвращает свободную и общую память RAM в мегабайтах.

        Returns:
            Кортеж (free_mb, total_mb). Если psutil недоступен — (None, None).
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            free_mb = int(mem.available / 1024 / 1024)
            total_mb = int(mem.total / 1024 / 1024)
            return free_mb, total_mb
        except Exception:
            return None, None

    def increment_tasks(self) -> None:
        """Увеличивает счётчик активных задач распознавания на 1."""
        self._active_tasks += 1

    def decrement_tasks(self) -> None:
        """Уменьшает счётчик активных задач на 1 (не ниже 0)."""
        self._active_tasks = max(0, self._active_tasks - 1)

    def get_active_tasks_count(self) -> int:
        """Возвращает текущее количество активных задач распознавания."""
        return self._active_tasks

    def get_queue_depth(self) -> int:
        """
        Возвращает глубину очереди задач на обработку.

        Returns:
            Заглушка (0) до реализации очереди в последующих этапах.
        """
        return 0

    def get_adapter_status(
        self,
        active_connections: int,
        max_connections: int,
    ) -> str:
        """
        Вычисляет статус адаптера: idle / busy / overloaded.

        Пороги берутся из конфигурации:
        - GPU memory usage > WS_STATUS_GPU_OVERLOAD_THRESHOLD_PCT -> overloaded.
        - active_connections / max_connections > WS_STATUS_BUSY_CONNECTIONS_THRESHOLD_PCT -> busy.

        Args:
            active_connections: Текущее число активных WebSocket-соединений.
            max_connections: Максимально допустимое число соединений.

        Returns:
            Строка-статус: "idle", "busy" или "overloaded".
        """
        gpu_free, gpu_total = self.get_gpu_stats()
        if gpu_total and gpu_total > 0:
            gpu_used_pct = (gpu_total - gpu_free) / gpu_total * 100
            threshold = getattr(settings, "WS_STATUS_GPU_OVERLOAD_THRESHOLD_PCT", 90.0)
            if gpu_used_pct >= threshold:
                return "overloaded"

        if max_connections > 0:
            conn_ratio = active_connections / max_connections
            busy_threshold = getattr(settings, "WS_STATUS_BUSY_CONNECTIONS_THRESHOLD_PCT", 80.0) / 100
            if conn_ratio >= busy_threshold:
                return "busy"

        if self._active_tasks > 0:
            return "busy"

        return "idle"

    def collect(
        self,
        active_connections: int = 0,
        max_connections: int = 100,
    ) -> WSStatusResponse:
        """
        Собирает полный набор метрик и возвращает WSStatusResponse.

        Args:
            active_connections: Текущее количество активных WS-соединений.
            max_connections: Максимально допустимое количество соединений.

        Returns:
            WSStatusResponse с актуальными метриками.
        """
        gpu_free, gpu_total = self.get_gpu_stats()
        cpu_free, cpu_total = self.get_cpu_stats()
        uptime = time.time() - self.start_time

        status = self.get_adapter_status(active_connections, max_connections)

        return WSStatusResponse(
            adapter_status=status,
            gpu_memory_free_mb=gpu_free,
            gpu_memory_total_mb=gpu_total,
            cpu_memory_free_mb=cpu_free,
            cpu_memory_total_mb=cpu_total,
            active_tasks_count=self._active_tasks,
            active_connections_count=active_connections,
            queue_depth=self.get_queue_depth(),
            uptime_sec=round(uptime, 2),
        )
