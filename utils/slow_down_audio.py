import numpy as np
from pydub import AudioSegment
from utils.bytes_to_samples_audio import get_np_array_samples_float32
from utils.do_logging import logger
from scipy.interpolate import CubicSpline
from pathlib import Path

async def do_slow_down_audio(audio_segment: AudioSegment, slowdown_rate: float) -> AudioSegment:
    """
    Замедляет аудио без сохранения тональности с использованием кубической интерполяции.
    :param audio_segment: Исходный моно AudioSegment
    :param slowdown_rate: Коэффициент замедления (0.5 = 2x медленнее)
    :return: Замедленный AudioSegment
    """

    # Проверка коэффициента
    if slowdown_rate <= 0 or slowdown_rate > 1:
        raise ValueError("Коэффициент замедления должен быть в диапазоне (0, 1]")
    if slowdown_rate == 1:
        logger.info("slowdown_rate = 1. Изменение длины аудио не производится")
        return audio_segment  # Без изменений

    # Получение параметров аудио
    sample_width = audio_segment.sample_width
    frame_rate = audio_segment.frame_rate
    channels = audio_segment.channels
    logger.debug(f"Текущая длина аудиосегмента: {audio_segment.duration_seconds}")

    # Преобразование в сырые байты
    raw_data = audio_segment.raw_data

    # Конвертация в float32 массив
    samples_float32 = get_np_array_samples_float32(raw_data, sample_width)

    # Расчет новых размеров
    original_length = len(samples_float32)
    new_length = int(original_length / slowdown_rate)

    # Создание кубического сплайна
    x_original = np.arange(original_length)
    spline = CubicSpline(x_original, samples_float32)

    # Генерация новых точек
    x_new = np.linspace(0, original_length - 1, new_length)
    new_samples = spline(x_new)

    # Обеспечение нормализации и клиппинг
    new_samples = np.clip(new_samples, -1.0, 1.0)

    # Преобразование обратно в int16/int32
    if sample_width == 2:
        new_samples_int = (new_samples * 32767).astype(np.int16)
    elif sample_width == 4:
        new_samples_int = (new_samples * 2147483647).astype(np.int32)
    else:
        raise ValueError(f"Неподдерживаемый sample width: {sample_width}")

    slowered_audiosegment = AudioSegment(
        new_samples_int.tobytes(),
        sample_width=sample_width,
        frame_rate=frame_rate,
        channels=channels
        )

    logger.debug(f"Новая длина аудиосегмента: {slowered_audiosegment.duration_seconds}")
    logger.debug(f"Соотношение длин: {audio_segment.duration_seconds / slowered_audiosegment.duration_seconds}")

    return slowered_audiosegment