from config import settings
import numpy as np
from pydub import AudioSegment
from utils.bytes_to_samples_audio import get_np_array_samples_float32

from utils.pre_start_init import (audio_overlap,
                                  audio_buffer,
                                  audio_to_asr,
                                  )

from VoiceActivityDetector import vad
from utils.resamppling import async_resample_audiosegment
import logging
logger = logging.getLogger(__name__)


async def find_last_speech_position(socket_id, is_last_chunk):
    """
    1. Берём собранное аудио, добавляем в начало overlap
    2. Конвертируем его в np.float32
    3. Находим позицию последнего сегмента тишины перед речью в аудио.
    4. Всё до этой позиции отправляем на распознавание
    5. Остаток, хвост, складываем отдельно как overlap
    6. Если не находит ни одного сегмента без речи, помечаем его как полностью речь и отдаём на распознавание
    """

    if is_last_chunk:
        last_audio =  audio_overlap[socket_id] + audio_buffer[socket_id]
        # for i in last_audio[::settings.MAX_OVERLAP_DURATION*1000]:
        #     audio_to_asr[socket_id].append(last_audio[:i])

        for i in range(0, len(last_audio), settings.MAX_OVERLAP_DURATION*1000):
            audio_to_asr[socket_id].append(last_audio[i:min(i + settings.MAX_OVERLAP_DURATION*1000, len(last_audio))])



    else:
        # дописали к буферу хвост предыдущего прохода
        audio_buffer[socket_id] = audio_overlap[socket_id] + audio_buffer[socket_id]

        frame_rate = audio_buffer[socket_id].frame_rate
        # 16000 - битрейт, требуемый Silero VAD
        silero_bitrate = 16000

        # Проверка входного аудио
        if not audio_buffer[socket_id]:
            logger.error("Ошибка: audio_buffer пустой")
            raise ValueError("audio_buffer не может быть пустым")

        if audio_buffer[socket_id].frame_rate != silero_bitrate:
            audio_for_vad = await async_resample_audiosegment(audio_buffer[socket_id], silero_bitrate)
        else:
            audio_for_vad = audio_buffer[socket_id]

        logger.debug(f"Получено из буфера на обработку аудио продолжительностью {audio_buffer[socket_id].duration_seconds}")

        # Приставляем буфер к полученному аудио
        audio_for_vad = audio_overlap[socket_id]+audio_for_vad
        # Переводим в float32 для VAD
        try:
            audio = get_np_array_samples_float32(audio_for_vad.raw_data, audio_for_vad.sample_width)
            logger.debug(f"Аудио для VAD: длина={len(audio)}, min={np.min(audio)}, max={np.max(audio)}")
        except Exception as e:
            logger.error(f"Ошибка в get_np_array_samples_float32: {e}")
            raise

        # Проверка на корректность данных
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            logger.error("Обнаружены NaN или бесконечные значения в audio")
            raise ValueError("Некорректные значения в audio")

        # Входные данные для деления фреймов
        duration_seconds = 0.5
        # Длина фрейма для Silero VAD: 256 семплов для 8 кГц, 512 семплов для 16 кГц
        frame_length = 512 if audio_for_vad.frame_rate == 16000 else 256

        if frame_length is None:
            raise ValueError("для VAD Поддерживаются только фреймрейты 8000 или 16000 Гц")

        # Длительность одного фрейма в секундах
        frame_duration = frame_length / frame_rate

        # Вычисляем количество фреймов
        min_silence_frames = int(duration_seconds / frame_duration)

        # Устанавливаем стартовые значения.
        max_audio_length = len(audio) if len(audio) < settings.MAX_OVERLAP_DURATION*silero_bitrate else settings.MAX_OVERLAP_DURATION*silero_bitrate

        partial_frame_length = 0

        # Разделение на фрагменты
        frames = [audio[i:i + frame_length] for i in range(int(len(audio)//3), max_audio_length, frame_length)]
        logger.debug(f"Создано фреймов: {len(frames)}, frame_length={frame_length}")

        # Проверка каждого фрагмента на наличие голоса
        silence_frames = 0
        await vad.reset_state()
        vad_state = vad.state

        no_silent = False
        for i, frame in enumerate(reversed(frames)):
            vad.state = vad_state
            try:
                if len(frame) < frame_length:
                    partial_frame_length = len(frame)
                    logger.debug(f"Пропущен неполный фрейм: длина={partial_frame_length}")
                    continue
                else:
                    logger.debug(f"Обработка фрейма {i}: длина={len(frame)}, min={np.min(frame)}, max={np.max(frame)}")
                    speech_prob, vad_state = await vad.is_speech(frame, audio_for_vad.frame_rate)
                    if speech_prob < vad.prob_level:
                        logger.debug(f"Найден не голос на speech_end = {max_audio_length-(i+1)*frame_length-partial_frame_length}")
                        silence_frames += 1
                        if silence_frames >= min_silence_frames:
                            break
                    else:
                        silence_frames = 0
                        logger.debug(f"Найден ГОЛОС на speech_end = {max_audio_length-i*frame_length-partial_frame_length}")
            except Exception as e:
                logger.error(f"Ошибка VAD - {e}"
                            f"\nframe_rate = {frame_rate}"
                            f"\nframe_length = {frame_length}"
                            f"\nframe_index = {i}"
                            f"\nframe_length_actual = {len(frame)}")
                raise
        else:
            no_silent = True
        try:
            # Вычисление speech_end
            if no_silent:
                speech_end = max_audio_length
            elif not partial_frame_length:
                speech_end = max_audio_length - (i + 1) * frame_length
            else:
                speech_end = max_audio_length - i * frame_length
        except Exception as e:
            print(e)
        else:
            separation_time = int(speech_end * 1000 / silero_bitrate)

            # Todo - в качестве оптимизации расхода памяти в audio_to_asr и audio_overlap можно хранить не аудио а время начала и окончания чанка.
            audio_to_asr[socket_id].append(audio_buffer[socket_id][:separation_time])
            audio_overlap[socket_id] = audio_buffer[socket_id][separation_time:]

            logger.debug(f"Передано на ASR аудио продолжительностью {audio_to_asr[socket_id][-1].duration_seconds}")
            logger.debug(f"Передано в перекрытие аудио продолжительностью {audio_overlap[socket_id].duration_seconds}")
            audio_buffer[socket_id] = AudioSegment.silent(1, frame_rate)

    return


def samples_padding(samples, sample_rate = settings.BASE_SAMPLE_RATE, duration = settings.MAX_OVERLAP_DURATION) -> np.ndarray:

    max_samples_len = int(duration * sample_rate)
    # Выравнивание до максимальной длины
    if len(samples) < max_samples_len:
        # Дополнение тишиной (нулями) до нужной длины
        padded_samples = np.zeros(max_samples_len, dtype=np.float32)
        padded_samples[:len(samples)] = samples
    elif len(samples) > max_samples_len:
        # Обрезка до максимальной длины
        logger.warning(f"Аудио длиной {len(samples) / settings.BASE_SAMPLE_RATE:.2f} сек. "
                       f"превышает MAX_OVERLAP_DURATION ({settings.MAX_OVERLAP_DURATION} сек.). "
                       f"Будет обрезано до {max_samples_len} семплов.")
        padded_samples = samples[:max_samples_len]

    else:
        # Идеальный размер
        padded_samples = samples

    return padded_samples


async def find_last_speech_position_v2(session, is_last_chunk):
    """
    Версия find_last_speech_position без глобальных dict.
    Работает с полями session: audio_buffer, audio_overlap, audio_to_asr.
    """
    import time
    vad_start = time.perf_counter()

    if is_last_chunk:
        last_audio = session.audio_overlap + session.audio_buffer
        for i in range(0, len(last_audio), settings.MAX_OVERLAP_DURATION * 1000):
            session.audio_to_asr.append(
                last_audio[i:min(i + settings.MAX_OVERLAP_DURATION * 1000, len(last_audio))]
            )
        logger.debug(
            "VAD last_chunk for %s: split into %d segments, total=%.3f sec, elapsed=%.3f sec",
            session.client_id,
            len(session.audio_to_asr),
            last_audio.duration_seconds,
            time.perf_counter() - vad_start,
        )
        return

    # --- Не последний чанк: ищем паузу ---
    session.audio_buffer = session.audio_overlap + session.audio_buffer
    frame_rate = session.audio_buffer.frame_rate
    silero_bitrate = 16000

    if not session.audio_buffer:
        logger.error("Ошибка: audio_buffer пустой")
        raise ValueError("audio_buffer не может быть пустым")

    if session.audio_buffer.frame_rate != silero_bitrate:
        audio_for_vad = await async_resample_audiosegment(session.audio_buffer, silero_bitrate)
    else:
        audio_for_vad = session.audio_buffer

    logger.debug(
        "VAD start for %s: buffer=%.3f sec, overlap=%.3f sec, frame_rate=%d",
        session.client_id,
        session.audio_buffer.duration_seconds,
        session.audio_overlap.duration_seconds,
        frame_rate,
    )

    # Добавляем overlap к VAD-аудио для поиска границы (как в оригинале)
    audio_for_vad = session.audio_overlap + audio_for_vad

    try:
        audio = get_np_array_samples_float32(audio_for_vad.raw_data, audio_for_vad.sample_width)
        logger.debug(f"Аудио для VAD: длина={len(audio)}, min={np.min(audio)}, max={np.max(audio)}")
    except Exception as e:
        logger.error(f"Ошибка в get_np_array_samples_float32: {e}")
        raise

    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        logger.error("Обнаружены NaN или бесконечные значения в audio")
        raise ValueError("Некорректные значения в audio")

    duration_seconds = 0.5
    frame_length = 512 if audio_for_vad.frame_rate == 16000 else 256

    if frame_length is None:
        raise ValueError("для VAD Поддерживаются только фреймрейты 8000 или 16000 Гц")

    frame_duration = frame_length / frame_rate
    min_silence_frames = int(duration_seconds / frame_duration)
    max_audio_length = len(audio) if len(audio) < settings.MAX_OVERLAP_DURATION * silero_bitrate else settings.MAX_OVERLAP_DURATION * silero_bitrate
    partial_frame_length = 0

    frames = [audio[i:i + frame_length] for i in range(int(len(audio) // 3), max_audio_length, frame_length)]
    logger.debug(f"Создано фреймов: {len(frames)}, frame_length={frame_length}")

    silence_frames = 0
    await vad.reset_state()
    vad_state = vad.state

    no_silent = False
    for i, frame in enumerate(reversed(frames)):
        vad.state = vad_state
        try:
            if len(frame) < frame_length:
                partial_frame_length = len(frame)
                logger.debug(f"Пропущен неполный фрейм: длина={partial_frame_length}")
                continue
            else:
                speech_prob, vad_state = await vad.is_speech(frame, audio_for_vad.frame_rate)
                if speech_prob < vad.prob_level:
                    silence_frames += 1
                    if silence_frames >= min_silence_frames:
                        break
                else:
                    silence_frames = 0
        except Exception as e:
            logger.error(f"Ошибка VAD - {e}"
                        f"\nframe_rate = {frame_rate}"
                        f"\nframe_length = {frame_length}"
                        f"\nframe_index = {i}"
                        f"\nframe_length_actual = {len(frame)}")
            raise
    else:
        no_silent = True

    try:
        if no_silent:
            speech_end = max_audio_length
        elif not partial_frame_length:
            speech_end = max_audio_length - (i + 1) * frame_length
        else:
            speech_end = max_audio_length - i * frame_length
    except Exception as e:
        logger.error(f"Ошибка вычисления speech_end: {e}")
        speech_end = max_audio_length
        no_silent = True

    separation_time = int(speech_end * 1000 / silero_bitrate)
    asr_segment = session.audio_buffer[:separation_time]
    overlap_segment = session.audio_buffer[separation_time:]

    session.audio_to_asr.append(asr_segment)
    session.audio_overlap = overlap_segment if overlap_segment.duration_seconds > 0 else AudioSegment.silent(1, frame_rate)
    session.audio_buffer = AudioSegment.silent(1, frame_rate)

    logger.debug(
        "VAD done for %s: no_silent=%s, speech_end=%d, separation_time=%d ms, "
        "asr_segment=%.3f sec, overlap=%.3f sec, elapsed=%.3f sec",
        session.client_id,
        no_silent,
        speech_end,
        separation_time,
        asr_segment.duration_seconds,
        session.audio_overlap.duration_seconds,
        time.perf_counter() - vad_start,
    )

    return
