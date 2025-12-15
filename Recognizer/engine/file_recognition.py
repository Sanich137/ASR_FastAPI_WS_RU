import time

from pydub import AudioSegment
import config
import asyncio
import uuid
from utils.tokens_to_Result import process_asr_json_deprecated, process_gigaam_asr
from utils.pre_start_init import (
    posted_and_downloaded_audio,
    audio_buffer,
    audio_overlap,
    audio_to_asr,
    audio_duration,
)
from utils.do_logging import logger
from utils.chunk_doing import find_last_speech_position
from utils.resamppling import sync_resample_audiosegment
from Recognizer.engine.stream_recognition import simple_recognise, recognise_w_speed_correction, simple_recognise_batch
from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer.engine.echoe_clearing import remove_echo
from Diarisation.diarazer import do_diarizing
from threading import Lock

# Глобальный лок для потокобезопасности
audio_lock = Lock()

def process_file(tmp_path, params):
    process_file_start = time.perf_counter()
    res = False
    diarized = False
    error_description = str()

    result = {
        "success": res,
        "error_description": error_description,
        "raw_data": dict(),
        "sentenced_data": dict(),
    }

    post_id = str(uuid.uuid4())
    logger.debug(f'Принят новый "post_file" id = {post_id}')

    try:
        with audio_lock:
            if params.make_mono:
                posted_and_downloaded_audio[post_id] = AudioSegment.from_file(tmp_path).set_channels(1)
            else:
                posted_and_downloaded_audio[post_id] = AudioSegment.from_file(tmp_path)
    except Exception as e:
        error_description += f"Error loading audio file: {e}"
        logger.error(error_description)
        result["success"] = False
        result["error_description"] = error_description
        return result

    # Проверка длины переданного на распознавание аудио
    try:
        with audio_lock:
            if posted_and_downloaded_audio[post_id].duration_seconds < 5:
                logger.debug(f"На вход передано аудио короче 5 секунд. Будет дополнено тишиной ещё 5 сек.")
                posted_and_downloaded_audio[post_id] += AudioSegment.silent(duration=5, frame_rate=config.BASE_SAMPLE_RATE)
    except Exception as e:
        error_description += f"Error len_fixing_file: {e}"
        logger.error(error_description)
        result["success"] = False
        result["error_description"] = error_description
        return result

    # Приводим фреймрейт к фреймрейту модели
    print(f"Начало проверки фреймрейта {(time.perf_counter()-process_file_start):.4f} сек.")
    try:
        with audio_lock:
            if posted_and_downloaded_audio[post_id].frame_rate != config.BASE_SAMPLE_RATE:
                posted_and_downloaded_audio[post_id] = sync_resample_audiosegment(
                                                                        audio_data=posted_and_downloaded_audio[post_id],
                                                                        target_sample_rate=config.BASE_SAMPLE_RATE)
                print(f"Корректировка фреймрейта {(time.perf_counter() - process_file_start):.4f} сек.")
    except KeyError as e_key:
        error_description = f"Ошибка обращения по ключу {post_id} при изменения фреймрейта - {e_key}"
        logger.error(error_description)
        result["success"] = False
        result['error_description'] = str(error_description)
        return result

    except Exception as e:
        error_description = f"Ошибка изменения фреймрейта - {e}"
        logger.error(error_description)
        result["success"] = False
        result['error_description'] = str(error_description)
        return result

    # Обрабатываем чанки с аудио по N секунд
    for n_channel, mono_data in enumerate(posted_and_downloaded_audio[post_id].split_to_mono()):
        time_chunks_start = time.perf_counter()
        # Подготовительные действия
        try:
            with audio_lock:
                audio_buffer[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
                audio_overlap[post_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
                audio_duration[post_id] = 0
        except Exception as e:
                error_description = f"Ошибка изменения фреймрейта - {e}"
                logger.error(error_description)
                result["success"] = False
                result['error_description'] = str(error_description)
                return result

        result["raw_data"].update({f"channel_{n_channel + 1}": list()})

        # Основной процесс перебора чанков для распознавания
        overlaps = list(mono_data[::config.MAX_OVERLAP_DURATION * 1000])  # Чанки аудио для распознавания
        total_chunks = len(overlaps)  # Количество чанков, для поиска последнего
        audio_to_asr[post_id] = list()
        for idx, overlap in enumerate(overlaps):
            is_last_chunk = (idx == total_chunks - 1) # Если чанк последний
            with audio_lock:
                if (audio_overlap[post_id].duration_seconds + overlap.duration_seconds) < config.MAX_OVERLAP_DURATION:
                    silent_secs = config.MAX_OVERLAP_DURATION - (audio_overlap[post_id].duration_seconds + overlap.duration_seconds)
                    overlap += AudioSegment.silent(silent_secs, frame_rate=config.BASE_SAMPLE_RATE)
                audio_buffer[post_id] = overlap
                asyncio.run(find_last_speech_position(post_id, is_last_chunk)) # Последний чанк обрабатывается иначе.

        if params.use_batch:
            logger.info("Запрошен батчинг")
            list_asr_result_wo_conf = simple_recognise_batch(audio_to_asr[post_id],params.batch_size)  # --> list

            for _, asr_result_wo_conf in enumerate(list_asr_result_wo_conf):

                asr_result = process_gigaam_asr(asr_result_wo_conf, audio_duration[post_id])

                result["raw_data"][f"channel_{n_channel + 1}"].append(asr_result)
                with audio_lock:
                    audio_duration[post_id] += audio_to_asr[post_id][_].duration_seconds
                res = True
                logger.debug(asr_result)
        else:
            for audio_asr in audio_to_asr[post_id]:
                try:
                    # Снижаем скорость аудио по необходимости
                    if params.do_auto_speech_speed_correction or params.speech_speed_correction_multiplier != 1:
                        logger.debug("Будут использованы механизмы анализа скорости речи и замедления аудио")
                        asr_result_wo_conf, speed, multiplier = asyncio.run(recognise_w_speed_correction(audio_asr,
                                                                            can_slow_down=True,
                                                                            multiplier=params.speech_speed_correction_multiplier))
                        params.speech_speed_correction_multiplier = multiplier
                    else:
                        # Производим распознавание
                        asr_result_wo_conf = asyncio.run(simple_recognise(audio_asr))

                except Exception as e:
                    logger.error(f"Error ASR audio - {e}")
                    error_description = f"Error ASR audio - {e}"
                else:
                    asr_result = process_gigaam_asr(asr_result_wo_conf, audio_duration[post_id])

                    result["raw_data"][f"channel_{n_channel + 1}"].append(asr_result)

                    with audio_lock:
                        audio_duration[post_id] += audio_asr.duration_seconds
                    res = True
                    logger.debug(asr_result)

            with audio_lock:
                try:
                    del audio_overlap[post_id]
                    del audio_buffer[post_id]
                    del audio_to_asr[post_id]
                    del audio_duration[post_id]
                except Exception as e:
                    error_description = f"Ошибка при очистке данных - {e}"
                    logger.error(error_description)
                    result["success"] = False
                    result['error_description'] = str(error_description)
    del mono_data
    del overlaps

    if params.do_echo_clearing:
        try:
            result["raw_data"] = asyncio.run(remove_echo(result["raw_data"]))

        except Exception as e:
            logger.error(f"Error echo clearing - {e}")
            error_description = f"Error echo clearing - {e}"
            res = False

    if params.do_diarization and not config.CAN_DIAR:
        error_description += "Diarization is not available.\n"
        logger.error("Запрошена диаризация, но она не доступна.")
        params.do_diarization = False
    # Проверяем возможность диаризации. Если здесь стерео-канал, то диаризацию выключаем.
    elif params.do_diarization and len(posted_and_downloaded_audio[post_id].split_to_mono()) != 1:
        error_description += f"Only mono diarization available.\n"
        logger.warn("При запрошенной диаризации аудио имеет более одного аудио-канала. Диаризация будет выключена.")
        params.do_diarization = False

    if params.do_diarization:
        try:
            result["diarized_data"] = asyncio.run(do_diarizing(
                file_id=str(post_id), asr_raw_data=result["raw_data"], diar_vad_sensity=params.diar_vad_sensity
            ))
        except Exception as e:
            logger.error(f"do_diarizing - {e}")
            error_description = f"do_diarizing - {e}"
            res = False
        else:
            diarized = True

    if params.do_dialogue:
        data_to_do_sensitizing = result["diarized_data"] if diarized else result["raw_data"]
        try:
            result["sentenced_data"] = asyncio.run(do_sensitizing(
                input_asr_json=data_to_do_sensitizing, do_punctuation=params.do_punctuation
                                                                    )
                                                    )
        except Exception as e:
            logger.error(f"do_sensitizing - {e}")
            error_description = f"do_sensitizing - {e}"
            res = False
        else:
            if not params.keep_raw:
                result["raw_data"].clear()
    else:
        result["sentenced_data"].clear()

    result["error_description"] = error_description
    result["success"] = res

    with audio_lock:
        del posted_and_downloaded_audio[post_id]

    logger.debug(result)
    return result
