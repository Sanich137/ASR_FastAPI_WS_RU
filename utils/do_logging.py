# -*- coding: utf-8 -*-
from config import settings

import logging
from logging.handlers import TimedRotatingFileHandler
from fastapi.logger import logger as fastapi_logger

logger = logging.getLogger(__name__)

if settings.IS_PROD:
    # Создаем обработчик, который ротирует логи каждый день в полночь
    file_handler = TimedRotatingFileHandler(
        filename=settings.FILENAME,  # Базовое имя файла
        when='midnight',  # Ротация каждый день в полночь
        interval=1,  # Интервал - каждый день
        backupCount=settings.LOG_BACKUP_COUNT if hasattr(settings, 'LOG_BACKUP_COUNT') else 7,
        # Хранить 7 дней логов по умолчанию
        encoding='UTF-8'
    )

    file_handler.setLevel(settings.LOGGING_LEVEL)
    file_handler.setFormatter(logging.Formatter(settings.LOGGING_FORMAT))

    # Настраиваем логгер
    logger.addHandler(file_handler)
    fastapi_logger.addHandler(file_handler)

    # Убираем basicConfig, так как мы используем кастомный обработчик
    logging.basicConfig(level=settings.LOGGING_LEVEL)
else:
    logging.basicConfig(
        level=settings.LOGGING_LEVEL,
        format=settings.LOGGING_FORMAT,
        encoding="UTF-8"
    )