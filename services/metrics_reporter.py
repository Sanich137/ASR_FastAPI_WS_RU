"""Фоновая задача записи системных метрик в БД."""

import asyncio
import logging

from db.enums import SystemLogLevel
from db.models import SystemLog
from db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def metrics_reporter_loop(app_state, interval_sec: float = 30.0):
    """Каждые interval_sec секунд собирает метрики и пишет в SystemLog."""
    while True:
        try:
            await asyncio.sleep(interval_sec)

            metrics_collector = getattr(app_state, "metrics_collector", None)
            ws_manager = getattr(app_state, "ws_manager", None)

            if not metrics_collector:
                continue

            active_conn = ws_manager.active_connections_count if ws_manager else 0
            max_conn = getattr(ws_manager, "max_connections", 100)

            metrics = metrics_collector.collect(
                active_connections=active_conn,
                max_connections=max_conn,
            )

            # Преобразуем Pydantic-модель WSStatusResponse в dict для JSONB
            metrics_dict = metrics.model_dump() if hasattr(metrics, "model_dump") else metrics.dict()

            async with AsyncSessionLocal() as session:
                log = SystemLog(
                    level=SystemLogLevel.info,
                    component="SystemMetricsCollector",
                    message="Periodic metrics snapshot",
                    meta=metrics_dict,
                )
                session.add(log)
                await session.commit()
                logger.debug("Метрики записаны в SystemLog")

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Ошибка в metrics_reporter_loop: {exc}")
