import logging
import numpy as np
import librosa
from pydub import AudioSegment
import asyncio



def sync_resample_audiosegment(audio_data: AudioSegment, target_sample_rate: int) -> AudioSegment:

    """
    Выполняет ресемплинг аудиоданных из AudioSegment в заданную частоту дискретизации, сохраняя каналы.
    Блокирующие операции выполняются в отдельном потоке, чтобы не блокировать цикл событий.

    :param audio_data: Входные аудиоданные в формате AudioSegment (pydub).
    :param target_sample_rate: Целевая частота дискретизации (Hz).
    :return: AudioSegment с новой частотой дискретизации.
    """
    logger = logging.getLogger(__name__)

    # П проверка входных данных
    if not audio_data:
        logger.error("Ошибка: audio_data пустой")
        raise ValueError("audio_data не может быть пустым")

    if audio_data.sample_width not in [2, 4]:
        logger.error(f"Неподдерживаемый sample_width: {audio_data.sample_width}")
        raise ValueError("sample_width должен быть 2 или 4")

    if audio_data.frame_rate <= 0 or target_sample_rate <= 0:
        logger.error(
            f"Некорректные частоты: frame_rate={audio_data.frame_rate}, target_sample_rate={target_sample_rate}")
        raise ValueError("Частота дискретизации должна быть положительной")

    if audio_data.frame_rate == target_sample_rate:
        return audio_data

    try:
        # Получаем количество каналов и битность
        channels = audio_data.channels
        sample_width = audio_data.sample_width

        # Преобразуем AudioSegment в numpy массив
        samples = np.array(audio_data.get_array_of_samples(), dtype=np.float32)
        logger.debug(f"Исходные samples: длина={len(samples)}, channels={channels}, sample_width={sample_width}")

        # Нормализация в зависимости от битности
        if sample_width == 2:
            samples /= 32768.0  # Для 16-bit
        elif sample_width == 4:
            samples /= 2147483648.0  # Для 32-bit

        # Проверка на NaN и бесконечные значения
        if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
            logger.error("Обнаружены NaN или бесконечные значения в samples")
            raise ValueError("Некорректные значения в samples")

        # Разделяем каналы
        samples = samples.reshape(-1, channels)

        # Ресемплинг каждого канала отдельно
        resampled_channels = []
        for channel in range(channels):
            channel_samples = samples[:, channel]
            resampled = librosa.resample(
                channel_samples,
                orig_sr=audio_data.frame_rate,
                target_sr=target_sample_rate,
                res_type='soxr_hq'  # Высококачественный ресемплинг
            )
            resampled_channels.append(resampled)

        # Объединяем каналы обратно
        resampled_samples = np.stack(resampled_channels, axis=1).ravel()

        # Проверка на NaN и бесконечные значения после ресемплинга
        if np.any(np.isnan(resampled_samples)) or np.any(np.isinf(resampled_samples)):
            logger.error("Обнаружены NaN или бесконечные значения в resampled_samples")
            raise ValueError("Некорректные значения в resampled_samples")

        # Обратное преобразование в int16 или int32
        if sample_width == 2:
            resampled_samples = (resampled_samples * 32768.0).astype(np.int16)
        elif sample_width == 4:
            resampled_samples = (resampled_samples * 2147483648.0).astype(np.int32)

        # Создаем новый AudioSegment
        resampled_audio = AudioSegment(
            resampled_samples.tobytes(),
            frame_rate=target_sample_rate,
            sample_width=sample_width,
            channels=channels
        )

        logger.debug(f"Ресемплированное аудио: длина={len(resampled_samples)}, frame_rate={target_sample_rate}")
        return resampled_audio

    except Exception as e:
        logger.error(f"Ошибка в resample_in_thread: {e}")
        raise


async def async_resample_audiosegment(audio_data: AudioSegment, target_sample_rate: int) -> AudioSegment:
    """ асинхронная обёртка для sync_resample_audiosegment"""
    return await asyncio.to_thread(sync_resample_audiosegment, audio_data, target_sample_rate)