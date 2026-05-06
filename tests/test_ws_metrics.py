"""
Тесты для services/ws_metrics.py (SystemMetricsCollector).
"""

import time

import pytest

from services.ws_metrics import SystemMetricsCollector
from models.ws_models import WSStatusResponse


class TestSystemMetricsCollectorLifecycle:
    def test_init_default_start_time(self):
        """Проверяет, что start_time устанавливается при инициализации."""
        collector = SystemMetricsCollector()
        assert collector.start_time <= time.time()

    def test_init_custom_start_time(self):
        """Проверяет передачу кастомного start_time."""
        now = time.time() - 100
        collector = SystemMetricsCollector(start_time=now)
        assert collector.start_time == now

    def test_get_gpu_stats_without_handle(self):
        """Без NVML-handle GPU-метрики возвращают None."""
        collector = SystemMetricsCollector(nvml_handle=None)
        free, total = collector.get_gpu_stats()
        assert free is None
        assert total is None


class TestSystemMetricsCollectorTasks:
    def test_active_tasks_counter(self):
        """Проверка инкремента/декремента счётчика задач."""
        collector = SystemMetricsCollector()
        assert collector.get_active_tasks_count() == 0
        collector.increment_tasks()
        assert collector.get_active_tasks_count() == 1
        collector.increment_tasks()
        assert collector.get_active_tasks_count() == 2
        collector.decrement_tasks()
        assert collector.get_active_tasks_count() == 1
        collector.decrement_tasks()
        collector.decrement_tasks()  # не уходит ниже 0
        assert collector.get_active_tasks_count() == 0

    def test_get_queue_depth_default(self):
        """По умолчанию глубина очереди равна 0."""
        collector = SystemMetricsCollector()
        assert collector.get_queue_depth() == 0


class TestSystemMetricsCollectorStatus:
    def test_get_adapter_status_idle(self):
        """Статус idle при отсутствии соединений и задач."""
        collector = SystemMetricsCollector()
        assert collector.get_adapter_status(0, 100) == "idle"

    def test_get_adapter_status_busy_by_tasks(self):
        """Статус busy при наличии активных задач."""
        collector = SystemMetricsCollector()
        collector.increment_tasks()
        assert collector.get_adapter_status(0, 100) == "busy"

    def test_get_adapter_status_busy_by_connections(self):
        """Статус busy при высокой загрузке соединений (порог 80%)."""
        collector = SystemMetricsCollector()
        assert collector.get_adapter_status(85, 100) == "busy"

    def test_get_adapter_status_idle_low_connections(self):
        """Статус idle при низкой загрузке соединений."""
        collector = SystemMetricsCollector()
        assert collector.get_adapter_status(50, 100) == "idle"


class TestSystemMetricsCollectorCollect:
    def test_collect_returns_ws_status_response(self):
        """collect() возвращает валидный WSStatusResponse."""
        collector = SystemMetricsCollector(start_time=time.time() - 10)
        collector.increment_tasks()
        response = collector.collect(active_connections=5, max_connections=100)
        assert isinstance(response, WSStatusResponse)
        assert response.adapter_status == "busy"
        assert response.active_tasks_count == 1
        assert response.active_connections_count == 5
        assert response.uptime_sec >= 10
        assert response.queue_depth == 0

    def test_collect_zero_connections_idle(self):
        """collect() при нулевых соединениях и задачах возвращает idle."""
        collector = SystemMetricsCollector()
        response = collector.collect(active_connections=0, max_connections=100)
        assert response.adapter_status == "idle"
        assert response.active_tasks_count == 0
        assert response.active_connections_count == 0
        assert response.uptime_sec >= 0

    def test_collect_cpu_stats_optional(self):
        """CPU-метрики либо числа, либо None — не вызывают ошибок."""
        collector = SystemMetricsCollector()
        response = collector.collect()
        if response.cpu_memory_free_mb is not None:
            assert isinstance(response.cpu_memory_free_mb, int)
        if response.cpu_memory_total_mb is not None:
            assert isinstance(response.cpu_memory_total_mb, int)
