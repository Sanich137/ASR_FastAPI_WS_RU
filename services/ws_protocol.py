# -*- coding: utf-8 -*-
"""
Автодетект и нормализация входящих WebSocket-сообщений.

Поддерживает ДВА протокола в одном эндпоинте:
  - legacy (asterisk-socket-server):  {"config": {...}, "channelName": ...}
                                       сырые PCM16 байты (binary frame)
                                       {"eof": 1}
  - новый (api/v1):                   {"type": "config", "sample_rate": ..., ...}
                                       {"type": "audio_chunk", "audio_base64": ...} | binary frame
                                       {"type": "eos"} / {"type": "ping"}

Возвращает единое событие WSEvent, чтобы эндпоинт не зависел от формата.
"""

import base64
from dataclasses import dataclass
from typing import Optional

import ujson

from models.ws_models import (
    parse_ws_message,
    WSBaseMessage,
    WSConfigMessage,
    WSEosMessage,
)


@dataclass
class WSEvent:
    kind: str                      # "config" | "audio" | "eos" | "ping" | "disconnect" | "ignore"
    audio: Optional[bytes] = None
    sample_rate: Optional[int] = None
    channel_name: Optional[str] = None
    wait_null_answers: bool = False
    raw: Optional[dict] = None     # исходный распарсенный JSON (для расширений)


def detect(message: dict) -> WSEvent:
    """
    Нормализует одно сообщение из ws.receive() (dict с ключами type/text/bytes).
    """
    if message.get("type") == "websocket.disconnect":
        return WSEvent("disconnect")

    # Бинарный фрейм аудио — одинаков в обоих протоколах
    raw_bytes = message.get("bytes")
    if raw_bytes:
        return WSEvent("audio", audio=raw_bytes)

    text = message.get("text")
    if not text:
        return WSEvent("ignore")

    try:
        d = ujson.loads(text)
    except Exception:
        return WSEvent("ignore", raw=None)

    if not isinstance(d, dict):
        return WSEvent("ignore")

    # --- legacy ---
    if isinstance(d.get("config"), dict):
        cfg = d["config"]
        return WSEvent(
            "config",
            sample_rate=cfg.get("sample_rate"),
            channel_name=d.get("channelName") or cfg.get("channelName"),
            wait_null_answers=bool(cfg.get("wait_null_answers", False)),
            raw=d,
        )
    if "eof" in d or "eos" in d:
        return WSEvent("eos", raw=d)

    # --- новый протокол (дискриминатор type) ---
    mtype = d.get("type")
    if mtype == "config":
        return WSEvent(
            "config",
            sample_rate=d.get("sample_rate"),
            channel_name=d.get("channel_name"),
            wait_null_answers=bool(d.get("wait_null_answers", False)),
            raw=d,
        )
    if mtype == "audio_chunk":
        b64 = d.get("audio_base64")
        try:
            audio = base64.b64decode(b64) if b64 else b""
        except Exception:
            audio = b""
        return WSEvent("audio", audio=audio, raw=d)
    if mtype in ("eos", "eof"):
        return WSEvent("eos", raw=d)
    if mtype == "ping":
        return WSEvent("ping", raw=d)

    return WSEvent("ignore", raw=d)


def normalize_to_ws_message(text: str | bytes) -> Optional[WSBaseMessage]:
    """
    Приводит текстовое WS-сообщение ЛЮБОГО протокола к каноническому pydantic-объекту
    (WSConfigMessage / WSEosMessage / WSAudioMessage / ...). Понимает:
      - новый протокол (дискриминатор type) — через parse_ws_message;
      - legacy ({"config": {...}, "channelName": ...} и {"eof": 1}).
    Возвращает None, если распознать не удалось.

    Используется в офлайн-эндпоинтах (/api/v1/asr/ws и legacy /ws), чтобы они принимали
    оба протокола без изменения нижележащей логики.
    """
    # 1) новый протокол
    try:
        return parse_ws_message(text)
    except Exception:
        pass

    # 2) legacy
    try:
        d = ujson.loads(text)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None

    if isinstance(d.get("config"), dict):
        cfg = d["config"]
        try:
            return WSConfigMessage(
                sample_rate=int(cfg.get("sample_rate") or 16000),
                wait_null_answers=bool(cfg.get("wait_null_answers", True)),
                do_dialogue=bool(cfg.get("do_dialogue", False)),
                do_punctuation=bool(cfg.get("do_punctuation", False)),
                audio_format=cfg.get("audio_format", "pcm16"),
                channel_name=d.get("channelName") or cfg.get("channelName"),
            )
        except Exception:
            return None
    if "eof" in d or "eos" in d:
        return WSEosMessage()
    return None
