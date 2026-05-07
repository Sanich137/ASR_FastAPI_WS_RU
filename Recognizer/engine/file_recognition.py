import time
from pydub import AudioSegment
from config import settings
import asyncio
import logging

from services.recognition_session import FileRecognitionSession
from utils.chunk_doing import find_last_speech_position_v2
from utils.resamppling import sync_resample_audiosegment
from Recognizer.engine.stream_recognition import simple_recognise, recognise_w_speed_correction, simple_recognise_batch
from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer.engine.echoe_clearing import remove_echo
from Diarisation.diarazer import do_diarizing

logger = logging.getLogger(__name__)

def process_file(session=None, recognizer=None, punctuator=None, diarizer=None, tmp_path=None, params=None):
    # Legacy adapter: поддержка старых роутов, передающих tmp_path + params
    if session is None and tmp_path is not None and params is not None:
        session = FileRecognitionSession(params=params)
        session.tmp_path = tmp_path
    elif session is None:
        raise TypeError("process_file() требует либо 'session', либо оба 'tmp_path' и 'params'")

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

    post_id = session.post_id
    params = session.params
    logger.debug(f'Принят новый "post_file" id = {post_id}')

    try:
        if params.make_mono:
            audio_input = AudioSegment.from_file(session.tmp_path).set_channels(1)
        else:
            audio_input = AudioSegment.from_file(session.tmp_path)
    except Exception as e:
        error_description += f"Error loading audio file: {e}"
        logger.error(error_description)
        result["success"] = False
        result["error_description"] = error_description
        return result

    # Проверка длины переданного на распознавание аудио
    try:
        if audio_input.duration_seconds < 5:
            logger.debug(f"На вход передано аудио короче 5 секунд. Будет дополнено тишиной ещё 5 сек.")
            audio_input += AudioSegment.silent(duration=5, frame_rate=settings.BASE_SAMPLE_RATE)
    except Exception as e:
        error_description += f"Error len_fixing_file: {e}"
        logger.error(error_description)
        result["success"] = False
        result["error_description"] = error_description
        return result

    # Приводим фреймрейт к фреймрейту модели
    logger.debug(f"Начало проверки фреймрейта {(time.perf_counter()-process_file_start):.4f} сек.")
    try:
        if audio_input.frame_rate != settings.BASE_SAMPLE_RATE:
            audio_input = sync_resample_audiosegment(
                audio_data=audio_input,
                target_sample_rate=settings.BASE_SAMPLE_RATE
            )
            logger.debug(f"Корректировка фреймрейта {(time.perf_counter() - process_file_start):.4f} сек.")
    except Exception as e:
        error_description = f"Ошибка изменения фреймрейта - {e}"
        logger.error(error_description)
        result["success"] = False
        result['error_description'] = str(error_description)
        return result

    # Обрабатываем чанки с аудио по N секунд
    session.collected_asr_res = {}
    for n_channel, mono_data in enumerate(audio_input.split_to_mono()):
        time_chunks_start = time.perf_counter()
        # Подготовительные действия
        session.audio_buffer = AudioSegment.silent(1, frame_rate=settings.BASE_SAMPLE_RATE)
        session.audio_overlap = AudioSegment.silent(1, frame_rate=settings.BASE_SAMPLE_RATE)
        session.audio_to_asr = []
        session.audio_duration = 0.0

        session.collected_asr_res[f"channel_{n_channel + 1}"] = []

        # Основной процесс перебора чанков для распознавания
        overlaps = list(mono_data[::settings.MAX_OVERLAP_DURATION * 1000])  # Чанки аудио для распознавания
        total_chunks = len(overlaps)  # Количество чанков, для поиска последнего
        for idx, overlap in enumerate(overlaps):
            is_last_chunk = (idx == total_chunks - 1)  # Если чанк последний
            if (session.audio_overlap.duration_seconds + overlap.duration_seconds) < settings.MAX_OVERLAP_DURATION:
                silent_secs = settings.MAX_OVERLAP_DURATION - (session.audio_overlap.duration_seconds + overlap.duration_seconds)
                overlap += AudioSegment.silent(silent_secs, frame_rate=settings.BASE_SAMPLE_RATE)
            session.audio_buffer = overlap
            asyncio.run(find_last_speech_position_v2(session, is_last_chunk))  # Последний чанк обрабатывается иначе.

        if params.use_batch:
            logger.info("Запрошен батчинг")
            list_asr_result_wo_conf = simple_recognise_batch(session.audio_to_asr, params.batch_size, recognizer)  # --> list

            for idx, asr_result_wo_conf in enumerate(list_asr_result_wo_conf):
                asr_result = recognizer.apply_postprocessing(asr_result_wo_conf, session.audio_duration)

                session.collected_asr_res[f"channel_{n_channel + 1}"].append(asr_result)
                session.audio_duration += session.audio_to_asr[idx].duration_seconds
                res = True
                logger.debug(asr_result)
        else:
            for audio_asr in session.audio_to_asr:
                try:
                    # Снижаем скорость аудио по необходимости
                    if params.do_auto_speech_speed_correction or params.speech_speed_correction_multiplier != 1:
                        logger.debug("Будут использованы механизмы анализа скорости речи и замедления аудио")
                        asr_result_wo_conf, speed, multiplier = asyncio.run(recognise_w_speed_correction(
                            audio_data=audio_asr,
                            can_slow_down=True,
                            multiplier=params.speech_speed_correction_multiplier,
                            recognizer=recognizer)
                        )
                        params.speech_speed_correction_multiplier = multiplier
                    else:
                        # Производим распознавание
                        asr_result_wo_conf = asyncio.run(simple_recognise(audio_asr, recognizer))

                except Exception as e:
                    logger.error(f"Error ASR audio - {e}")
                    error_description = f"Error ASR audio - {e}"
                else:
                    asr_result = recognizer.apply_postprocessing(asr_result_wo_conf, session.audio_duration)

                    session.collected_asr_res[f"channel_{n_channel + 1}"].append(asr_result)

                    session.audio_duration += audio_asr.duration_seconds
                    res = True
                    logger.debug(asr_result)
    del mono_data
    del overlaps

    if params.do_echo_clearing:
        try:
            session.collected_asr_res = asyncio.run(remove_echo(session.collected_asr_res))

        except Exception as e:
            logger.error(f"Error echo clearing - {e}")
            error_description = f"Error echo clearing - {e}"
            res = False

    if params.do_diarization and not settings.CAN_DIAR:
        error_description += "Diarization is not available.\n"
        logger.error("Запрошена диаризация, но она не доступна.")
        params.do_diarization = False
    # Проверяем возможность диаризации. Если здесь стерео-канал, то диаризацию выключаем.
    elif params.do_diarization and len(audio_input.split_to_mono()) != 1:
        error_description += f"Only mono diarization available.\n"
        logger.warn("При запрошенной диаризации аудио имеет более одного аудио-канала. Диаризация будет выключена.")
        params.do_diarization = False

    if params.do_diarization:
        try:
            result["diarized_data"] = asyncio.run(do_diarizing(
                file_id=str(post_id),
                asr_raw_data=session.collected_asr_res,
                diar_vad_sensity=params.diar_vad_sensity,
                diarizer=diarizer,
            ))
        except Exception as e:
            logger.error(f"do_diarizing - {e}")
            error_description = f"do_diarizing - {e}"
            res = False
        else:
            diarized = True

    if params.do_dialogue:
        data_to_do_sensitizing = result["diarized_data"] if diarized else session.collected_asr_res
        try:
            result["sentenced_data"] = asyncio.run(do_sensitizing(
                input_asr_json=data_to_do_sensitizing,
                do_punctuation=params.do_punctuation,
                punctuator=punctuator
            ))
        except Exception as e:
            logger.error(f"do_sensitizing - {e}")
            error_description = f"do_sensitizing - {e}"
            res = False
        else:
            if not params.keep_raw:
                session.collected_asr_res.clear()
    else:
        result["sentenced_data"].clear()

    result["error_description"] = error_description
    result["success"] = res

    # Переносим raw_data из сессии в результат для обратной совместимости ответа
    result["raw_data"] = session.collected_asr_res

    logger.debug(result)
    return result
