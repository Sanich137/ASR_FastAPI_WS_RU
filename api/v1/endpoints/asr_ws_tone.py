# -*- coding: utf-8 -*-
"""
WebSocket-роут /api/v1/asr/ws-stream — НАСТОЯЩИЙ потоковый риалтайм на нативном T-one.

В отличие от /api/v1/asr/ws (офлайн псевдо-стрим GigaAM: копит до MAX_OVERLAP_DURATION и
распознаёт целым куском), здесь аудио скармливается модели кадрами по 300 мс с сохранением
состояния, а фразы отдаются по мере их завершения детектором границ T-one (~0.3-1 c).

Особенности:
  - Автодетект протокола (services/ws_protocol): понимает и legacy ({config}/{eof}/raw bytes),
    и новый ({type:config/audio_chunk/eos/ping}). Ответы в формате WSResultMessage —
    его поля (silence/data/last_message) совместимы с legacy asterisk-socket-server.
  - Потоковый ресемплинг источник->8 кГц (T-one фиксирован на 8 кГц).
  - Времена фраз накопительны от первого пакета (по объёму поданного аудио) — корректны
    для поканального мёржа на стороне клиента.
  - БД не используется (в отличие от asr_ws.py) — эндпоинт независим от наличия таблиц.
"""

import logging
import uuid

from fastapi import APIRouter, WebSocket

from config import settings
from models.ws_models import (
    WSResultMessage,
    WSRecognitionData,
    WSPongMessage,
    WSMessageType,
)
from services.ws_manager import ConnectionManager
from services.ws_protocol import detect
from utils.tone_stream import take_frames, flush_tail, phrase_to_data, StreamResampler
from Recognizer.tone_engine import get_tone_pipeline

router = APIRouter(prefix="/asr", tags=["ASR"])
logger = logging.getLogger(__name__)


def _result_message(phrase, channel_name: str, last: bool = False) -> WSResultMessage:
    return WSResultMessage(
        type=WSMessageType.final_result if last else WSMessageType.partial_result,
        channel_name=channel_name,
        silence=False,
        data=WSRecognitionData(**phrase_to_data(phrase)),
        last_message=last,
    )


@router.websocket("/ws-stream")
async def websocket_tone_stream(websocket: WebSocket):
    manager: ConnectionManager = websocket.app.state.ws_manager
    client_id = str(uuid.uuid4())

    if not await manager.connect(websocket, client_id):
        return  # лимит соединений исчерпан, ConnectionManager уже закрыл сокет

    # Предзагруженный в lifespan движок (готов сразу после старта); фолбэк - ленивая загрузка
    pipeline = getattr(websocket.app.state, "tone_pipeline", None) or get_tone_pipeline()
    state = None
    buf = bytearray()
    channel_name = "Null"
    resampler = StreamResampler(settings.TONE_SAMPLE_RATE)  # проходной, пока не пришёл config

    logger.debug("[tone] new stream %s", client_id)

    try:
        while True:
            try:
                message = await websocket.receive()
            except Exception as exc:
                logger.debug("[tone] receive error %s: %s", client_id, exc)
                break

            evt = detect(message)

            if evt.kind == "disconnect":
                logger.info("[tone] disconnect %s (%s)", channel_name, client_id)
                break

            if evt.kind == "config":
                channel_name = evt.channel_name or "Null"
                sr = evt.sample_rate or settings.TONE_SAMPLE_RATE
                resampler = StreamResampler(sr)
                if sr != settings.TONE_SAMPLE_RATE:
                    logger.info("[tone] %s: sample_rate=%s, включён ресемплинг к 8 кГц", channel_name, sr)
                logger.info("[tone] config received for channel %s", channel_name)
                continue

            if evt.kind == "ping":
                await manager.send_message(client_id, WSPongMessage())
                continue

            if evt.kind == "audio":
                # Время T-one считается по объёму поданного аудио (от первого пакета),
                # ресемплинг сохраняет длительность — метки остаются корректными.
                buf.extend(resampler.process(evt.audio or b""))
                try:
                    for samples in take_frames(buf):
                        phrases, state = pipeline.forward(samples, state)
                        for phrase in phrases:
                            if phrase.text:
                                await manager.send_message(client_id, _result_message(phrase, channel_name))
                except Exception as exc:
                    logger.error("[tone] recognize error %s (%s): %s", channel_name, client_id, exc)
                continue

            if evt.kind == "eos":
                logger.info("[tone] EOS for channel %s", channel_name)
                break

            # evt.kind == "ignore" — молча пропускаем

        # --- финализация: дослать хвост ресемплера и буфера, закрыть фразы ---
        last_phrase = None
        try:
            buf.extend(resampler.process(b"", last=True))
            tail = flush_tail(buf)
            final_phrases = []
            if tail is not None:
                phrases, state = pipeline.forward(tail, state, is_last=True)
                final_phrases.extend(phrases)
            fin_phrases, state = pipeline.finalize(state)
            final_phrases.extend(fin_phrases)
            final_phrases = [p for p in final_phrases if p.text]
            # все, кроме последней, шлём обычными; последнюю пометим last_message
            for p in final_phrases[:-1]:
                await manager.send_message(client_id, _result_message(p, channel_name))
            if final_phrases:
                last_phrase = final_phrases[-1]
        except Exception as exc:
            logger.error("[tone] finalize error %s (%s): %s", channel_name, client_id, exc)

        if last_phrase is not None:
            await manager.send_message(client_id, _result_message(last_phrase, channel_name, last=True))
        else:
            await manager.send_message(client_id, WSResultMessage(
                type=WSMessageType.final_result,
                channel_name=channel_name,
                silence=True,
                data=WSRecognitionData(),
                last_message=True,
            ))
    finally:
        await manager.disconnect(client_id)
        logger.info("[tone] closed %s (%s)", channel_name, client_id)
