import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from logging.config import dictConfig

from config import settings

# ContextVar для проброса request_id из middleware в лог-записи
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    """
    Форматтер, выводящий лог в виде JSON.
    Поля: timestamp, level, logger, message, request_id, module, function, line.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


class RequestIDFilter(logging.Filter):
    """Фильтр, пробрасывающий request_id из ContextVar в атрибут записи."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging():
    """
    Централизованная настройка логирования.
    """
    handlers = {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "json",
            "filters": ["request_id"],
        },
    }

    if settings.IS_PROD:
        handlers["file"] = {
            "()": "logging.handlers.TimedRotatingFileHandler",
            "filename": settings.FILENAME,
            "when": "midnight",
            "interval": 1,
            "backupCount": settings.LOG_BACKUP_COUNT,
            "encoding": "UTF-8",
            "formatter": "json",
            "filters": ["request_id"],
        }

    handler_names = list(handlers.keys())

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "core.logging_config.JsonFormatter",
            },
        },
        "filters": {
            "request_id": {
                "()": "core.logging_config.RequestIDFilter",
            },
        },
        "handlers": handlers,
        "root": {
            "level": settings.LOGGING_LEVEL,
            "handlers": handler_names,
        },
        "loggers": {
            "uvicorn": {"level": "WARNING", "handlers": handler_names, "propagate": False},
            "uvicorn.access": {"level": "WARNING", "handlers": handler_names, "propagate": False},
            "httpx": {"level": "WARNING", "handlers": handler_names, "propagate": False},
            "httpcore": {"level": "WARNING", "handlers": handler_names, "propagate": False},
        },
    }

    dictConfig(config)
