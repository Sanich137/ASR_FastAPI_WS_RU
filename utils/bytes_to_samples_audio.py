import numpy as np
from  time import perf_counter
import logging

import config


def get_np_array_samples_float32(audio_bytes: bytes, sample_width: int = 2) -> np.ndarray:
    """
    Преобразует аудио в байтах в массив float32.
    :param audio_bytes: Аудиоданные в байтах.
    :param sample_width: Размер одного сэмпла в байтах (обычно 2 для 16-битного аудио).
    :return: Массив numpy с данными в формате float32.
    """
    logger = logging.getLogger(__name__)
    timer_start = perf_counter()
    # Проверка входных данных
    if not audio_bytes:
        logger.error("Ошибка: audio_bytes пустой")
        raise ValueError("audio_bytes не может быть пустым")

    if sample_width not in [2, 4]:
        logger.error(f"Неподдерживаемый sample_width: {sample_width}")
        raise ValueError("sample_width должен быть 2 или 4")

    # Определяем тип данных на основе sample_width
    dtype = np.int16 if sample_width == 2 else np.int32

    try:
        # Преобразуем байты в массив numpy
        samples = np.frombuffer(audio_bytes, dtype=dtype)
        logger.debug(f"Длина samples: {len(samples)}, dtype: {samples.dtype}")

        if len(samples) == 0:
            logger.error("Ошибка: samples пустой после np.frombuffer")
            raise ValueError("Не удалось преобразовать audio_bytes в массив")

        # Преобразуем в float32
        samples_float32 = samples.astype(np.float32)

        # Нормализация в зависимости от битности
        if sample_width == 2:
            samples_float32 /= 32768.0  # Для 16-bit
        else:
            samples_float32 /= 2147483648.0  # Для 32-bit

        # Проверка на NaN и бесконечные значения
        if np.any(np.isnan(samples_float32)) or np.any(np.isinf(samples_float32)):
            logger.error("Обнаружены NaN или бесконечные значения в samples_float32")
            raise ValueError("Некорректные значения в samples_float32")

        # Проверка диапазона значений
        if np.max(np.abs(samples_float32)) > 1.0:
            logger.warning(f"Значения samples_float32 вне диапазона [-1, 1]: max={np.max(np.abs(samples_float32))}")
            samples_float32 = np.clip(samples_float32, -1.0, 1.0)

        logger.debug(
            f"Длина samples_float32: {len(samples_float32)}, min={np.min(samples_float32)}, max={np.max(samples_float32)}")

        end_timer = perf_counter()
        logger.debug(f"Время на конвертацию аудио длинной {len(samples)/config.BASE_SAMPLE_RATE} сек. затрачено {(end_timer-timer_start)} сек.")
        return samples_float32

    except Exception as e:
        logger.error(f"Ошибка в get_np_array_samples_float32: {e}")
        raise





async def get_np_array_samples_int16(audio_bytes: bytes, sample_width: int = 2) -> np.ndarray:
    """
    Преобразует аудио в байтах в массив float32.
    :param audio_bytes: Аудиоданные в байтах.
    :param sample_width: Размер одного сэмпла в байтах (обычно 2 для 16-битного аудио).
    :return: Массив numpy с данными в формате int16.
    """

    # Определяем тип данных на основе sample_width
    dtype = np.int16 if sample_width == 2 else np.int32

    # Преобразуем байты в массив numpy
    np_samples = np.frombuffer(audio_bytes, dtype=dtype)

   # Преобразование в int16
    if np_samples.dtype != np.int16:
        np_samples = (np_samples * 32767).astype(np.int16)

    return np_samples