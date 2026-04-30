# -*- coding: utf-8 -*-
import logging

# Логгер модуля. Полная конфигурация производится в lifespan через core.logging_config.setup_logging()
# Этот модуль не должен содержать side-effects при импорте.
logger = logging.getLogger(__name__)
