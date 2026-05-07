"""
Модуль services/asr_pipeline.py
Содержит бизнес-логику потокового распознавания речи (ASR) через WebSocket:
накопление аудио, VAD-разделение по паузам, распознавание, постпроцессинг,
накопление результатов и формирование финального диалога с пунктуацией.
"""

import logging
from io import BytesIO

from pydub import AudioSegment

from config import settings
from models.ws_models import (
    WSResultMessage,
    WSRecognitionData,
    WSWordItem,
    WSErrorMessage,
)
from services.ws_session import AudioSession, SessionState
from services.ws_manager import ConnectionManager
from utils.resamppling import async_resample_audiosegment
from utils.tokens_to_Result import process_single_token_vocab_output
from utils.chunk_doing import find_last_speech_position_v2
from Recognizer.engine.stream_recognition import simple_recognise
from Recognizer.engine.sentensizer import do_sensitizing

logger = logging.getLogger(__name__)


async def process_audio_stream_chunk(
    session: AudioSession,
    chunk_bytes: bytes,
    recognizer,
    punctuator,
    manager: ConnectionManager,
) -> None:
    """
    Обрабатывает входящий чанк аудио в потоковом режиме.

    Алгоритм:
      1. Проверяет чётность байтов (дополняет до чётного при необходимости).
      2. Создаёт AudioSegment из raw bytes (PCM16) или через from_file для других форматов.
      3. Ресемплит до BASE_SAMPLE_RATE, приводит к моно.
      4. Накапливает в session.audio_buffer.
      5. При достижении MAX_OVERLAP_DURATION вызывает find_last_speech_position
         (через временную синхронизацию с глобальными dict для сохранения точной логики VAD).
      6. Распознаёт готовый сегмент через simple_recognise.
      7. Применяет process_single_token_vocab_output со сдвигом времени.
      8. Отправляет результат клиенту (или silence partial при пустом тексте).

    Args:
        session: Текущая аудио-сессия (содержит audio_buffer, audio_overlap и т.д.).
        chunk_bytes: Сырые байты аудио от клиента.
        recognizer: Экземпляр Recognizer.
        punctuator: Экземпляр SbertPuncCaseOnnx (не используется в чанке, передаётся для единообразия).
        manager: Менеджер WebSocket-соединений для отправки ответов.
    """
    try:
        # --- 1. Проверка чётности ---
        if len(chunk_bytes) % 2 != 0:
            chunk_bytes += bytes(2 - (len(chunk_bytes) % 2))

        # --- 2. Создание AudioSegment ---
        audio_format = session.config.audio_format if session.config else "pcm16"
        sample_rate = session.config.sample_rate if session.config else settings.BASE_SAMPLE_RATE

        if audio_format == "pcm16":
            audiosegment_chunk = AudioSegment(
                chunk_bytes,
                frame_rate=sample_rate,
                sample_width=2,
                channels=1,
            )
        else:
            buffer = BytesIO(chunk_bytes)
            buffer.seek(0)
            audiosegment_chunk = AudioSegment.from_file(buffer)

        # --- 3. Ресемплинг и моно ---
        if audiosegment_chunk.frame_rate != settings.BASE_SAMPLE_RATE:
            audiosegment_chunk = await async_resample_audiosegment(audiosegment_chunk, settings.BASE_SAMPLE_RATE)
        if audiosegment_chunk.channels != 1:
            audiosegment_chunk = audiosegment_chunk.set_channels(1)

        # --- 4. Накопление в буфер ---
        session.audio_buffer += audiosegment_chunk
        session.last_activity = __import__("time").time()

        # --- 5. Проверка порога VAD ---
        combined_duration = (session.audio_overlap + session.audio_buffer).duration_seconds
        if combined_duration < settings.MAX_OVERLAP_DURATION:
            return

        # --- 5a. VAD-разделение через find_last_speech_position_v2 ---
        try:
            await find_last_speech_position_v2(session, is_last_chunk=False)
        except Exception as exc:
            logger.exception("VAD error for %s: %s", session.client_id, exc)
            session.state = SessionState.error
            error_msg = WSErrorMessage(
                code="vad_error",
                message=f"VAD processing error: {exc}",
                is_fatal=False,
            )
            try:
                await manager.send_message(session.client_id, error_msg)
            except Exception:
                pass
            return

        # --- 6. Распознавание последнего сегмента ---
        if not session.audio_to_asr:
            return

        segment = session.audio_to_asr[-1]
        if segment.duration_seconds <= 0:
            return

        asr_result = await simple_recognise(segment, recognizer=recognizer)
        asr_result_words = process_single_token_vocab_output(asr_result, session.audio_duration)
        session.audio_duration += segment.duration_seconds

        # Накопление для финального диалога
        session.ws_collected_asr_res[f"channel_{1}"].append(asr_result_words)

        # --- 7. Отправка результата ---
        text = asr_result_words.get("data", {}).get("text", "")
        is_silence = len(text) == 0 or text == " "

        if is_silence:
            if session.wait_null_answers:
                msg = WSResultMessage(
                    channel_name=session.channel_name,
                    silence=True,
                    data=WSRecognitionData(),
                    error=None,
                    last_message=False,
                )
                await manager.send_message(session.client_id, msg)
            else:
                logger.debug("Silence partial skipped (wait_null_answers=False)")
        else:
            words = asr_result_words.get("data", {}).get("result", [])
            ws_words = [
                WSWordItem(conf=w["conf"], start=w["start"], end=w["end"], word=w["word"])
                for w in words
            ]
            data = WSRecognitionData(result=ws_words, text=text)
            msg = WSResultMessage(
                channel_name=session.channel_name,
                silence=False,
                data=data,
                error=None,
                last_message=False,
            )
            await manager.send_message(session.client_id, msg)

    except Exception as exc:
        logger.exception("ASR pipeline chunk error for %s: %s", session.client_id, exc)
        session.state = SessionState.error
        error_msg = WSErrorMessage(
            code="asr_pipeline_error",
            message=f"Chunk processing error: {exc}",
            is_fatal=False,
        )
        try:
            await manager.send_message(session.client_id, error_msg)
        except Exception:
            pass


async def process_final_audio(
    session: AudioSession,
    recognizer,
    punctuator,
    manager: ConnectionManager,
) -> None:
    """
    Обрабатывает финальный буфер аудио по получении EOF/EOS.

    Алгоритм:
      1. Объединяет audio_overlap + audio_buffer.
      2. Если длительность < 2 сек — дополняет тишиной до минимума.
      3. Распознаёт, постпроцессинг, добавляет в ws_collected_asr_res.
      4. Если do_dialogue — вызывает do_sensitizing с пунктуацией.
      5. Формирует и отправляет final WSResultMessage.

    Args:
        session: Текущая аудио-сессия.
        recognizer: Экземпляр Recognizer.
        punctuator: Экземпляр SbertPuncCaseOnnx.
        manager: Менеджер WebSocket-соединений.
    """
    try:
        # --- 1. Объединение остатков ---
        final_audio = session.audio_overlap + session.audio_buffer
        session.audio_to_asr.append(final_audio)
        logger.debug("Final audio duration: %.3f sec", final_audio.duration_seconds)

        # --- 2. Дополнение тишиной при необходимости ---
        if final_audio.duration_seconds < 2:
            final_audio = final_audio + AudioSegment.silent(1000, frame_rate=settings.BASE_SAMPLE_RATE)
            session.audio_to_asr[-1] = final_audio
            logger.debug("Final audio padded with silence to %.3f sec", final_audio.duration_seconds)

        # --- 3. Распознавание ---
        last_asr_result = await simple_recognise(final_audio, recognizer=recognizer)
        last_result = process_single_token_vocab_output(last_asr_result, session.audio_duration)
        session.ws_collected_asr_res[f"channel_{1}"].append(last_result)

        # --- 4. Определение silence ---
        text = last_result.get("data", {}).get("text", "")
        is_silence = len(text) == 0 or text == " "
        if is_silence:
            last_result = None

        # --- 5. Построение диалога / пунктуация ---
        sentenced_data = None
        if session.do_dialogue:
            try:
                sentenced_data = await do_sensitizing(
                    session.ws_collected_asr_res,
                    session.do_punctuation,
                    punctuator=punctuator,
                )
            except Exception as exc:
                logger.error("do_sensitizing error: %s", exc)
                sentenced_data = None

        # --- 6. Формирование ответа ---
        if is_silence:
            data = WSRecognitionData()
        else:
            words = last_result.get("data", {}).get("result", [])
            ws_words = [
                WSWordItem(conf=w["conf"], start=w["start"], end=w["end"], word=w["word"])
                for w in words
            ]
            data = WSRecognitionData(result=ws_words, text=text)

        final_msg = WSResultMessage(
            channel_name=session.channel_name,
            silence=is_silence,
            data=data,
            error=None,
            last_message=True,
            sentenced_data=sentenced_data,
        )
        await manager.send_message(session.client_id, final_msg)
        session.state = SessionState.completed

    except Exception as exc:
        logger.exception("ASR pipeline final error for %s: %s", session.client_id, exc)
        session.state = SessionState.error
        error_msg = WSErrorMessage(
            code="asr_pipeline_final_error",
            message=f"Final processing error: {exc}",
            is_fatal=False,
        )
        try:
            await manager.send_message(session.client_id, error_msg)
        except Exception:
            pass
