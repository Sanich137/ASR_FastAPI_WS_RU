import config
from utils.do_logging import logger
from utils.bytes_to_samples_audio import get_np_array_samples_float32

from utils.pre_start_init import (audio_overlap,
                                  audio_buffer,
                                  audio_to_asr,
                                  )
from pydub import AudioSegment
from VoiceActivityDetector import vad
from utils.resamppling import resample_audiosegment

import numpy as np
from pydub import AudioSegment

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
        # for i in last_audio[::config.MAX_OVERLAP_DURATION*1000]:
        #     audio_to_asr[socket_id].append(last_audio[:i])

        for i in range(0, len(last_audio), config.MAX_OVERLAP_DURATION*1000):
            audio_to_asr[socket_id].append(last_audio[i:min(i + config.MAX_OVERLAP_DURATION*1000, len(last_audio))])



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
            audio_for_vad = await resample_audiosegment(audio_buffer[socket_id], silero_bitrate)
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
        max_audio_length = len(audio) if len(audio) < config.MAX_OVERLAP_DURATION*silero_bitrate else config.MAX_OVERLAP_DURATION*silero_bitrate

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


def samples_padding(samples, sample_rate = config.BASE_SAMPLE_RATE, duration = config.MAX_OVERLAP_DURATION) -> np.ndarray:

    max_samples_len = int(duration * sample_rate)
    # Выравнивание до максимальной длины
    if len(samples) < max_samples_len:
        # Дополнение тишиной (нулями) до нужной длины
        padded_samples = np.zeros(max_samples_len, dtype=np.float32)
        padded_samples[:len(samples)] = samples
    elif len(samples) > max_samples_len:
        # Обрезка до максимальной длины
        logger.warning(f"Аудио длиной {len(samples) / config.BASE_SAMPLE_RATE:.2f} сек. "
                       f"превышает MAX_OVERLAP_DURATION ({config.MAX_OVERLAP_DURATION} сек.). "
                       f"Будет обрезано до {max_samples_len} семплов.")
        padded_samples = samples[:max_samples_len]

    else:
        # Идеальный размер
        padded_samples = samples

    return padded_samples