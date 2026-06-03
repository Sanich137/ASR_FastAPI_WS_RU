# -*- coding: utf-8 -*-
"""
Хелперы для потокового распознавания через T-one (tone.StreamingCTCPipeline).

T-one ест ровно settings.TONE_CHUNK_SAMPLES (2400) семплов int32 @ 8 кГц за один forward()
и отдаёт фразы (TextPhrase) с пофразовыми таймингами. Здесь:
  - потоковый ресемплинг источник->8 кГц (stateful, без артефактов на стыках);
  - нарезка входящего PCM16-потока на кадры нужного размера;
  - синтез пословного result'а из фразы (клиент ждёт data.result = массив слов со
    start/end, а T-one даёт только границы фразы).
"""

import numpy as np
import soxr

from config import settings

# Размер одного кадра T-one в байтах PCM16 (2 байта на семпл, моно)
BYTES_PER_FRAME = settings.TONE_CHUNK_SAMPLES * 2


class StreamResampler:
    """
    Stateful ресемплер источник->8 кГц для непрерывного потока PCM16 (моно).

    Состояние фильтра сохраняется между чанками (soxr.ResampleStream), поэтому нет
    артефактов на стыках пакетов, а суммарная длительность сохраняется - значит
    времена T-one (считаются по поданным 8к-семплам) продолжают соответствовать
    реальному времени аудио. Если источник уже 8 кГц - проходной режим.
    """
    def __init__(self, in_rate: int, out_rate: int = settings.TONE_SAMPLE_RATE):
        self.enabled = int(in_rate) != int(out_rate)
        if self.enabled:
            self._rs = soxr.ResampleStream(in_rate, out_rate, 1, dtype="int16", quality="HQ")

    def process(self, pcm16_bytes: bytes, last: bool = False) -> bytes:
        if not self.enabled:
            return pcm16_bytes
        x = np.frombuffer(pcm16_bytes, dtype=np.int16)
        y = self._rs.resample_chunk(x, last=last)
        return y.astype(np.int16).tobytes()


def take_frames(buf: bytearray) -> list:
    """
    Достаёт из буфера все полные кадры по settings.TONE_CHUNK_SAMPLES семплов.

    Извлечённые байты удаляются из buf, неполный «хвост» остаётся для следующего вызова.

    :param buf: bytearray с накопленным сырым PCM16 (моно, 8 кГц)
    :return: список np.ndarray (dtype=int32, shape=(TONE_CHUNK_SAMPLES,))
    """
    frames = []
    while len(buf) >= BYTES_PER_FRAME:
        raw = bytes(buf[:BYTES_PER_FRAME])
        del buf[:BYTES_PER_FRAME]
        frames.append(np.frombuffer(raw, dtype=np.int16).astype(np.int32))
    return frames


def flush_tail(buf: bytearray) -> np.ndarray | None:
    """
    Дополняет остаток буфера тишиной до полного кадра (для финального forward с is_last).

    :param buf: bytearray с остатком (< BYTES_PER_FRAME)
    :return: np.ndarray (int32, shape=(TONE_CHUNK_SAMPLES,)) или None, если остатка нет
    """
    if len(buf) == 0:
        return None
    samples = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.int32)
    buf.clear()
    pad = settings.TONE_CHUNK_SAMPLES - len(samples)
    if pad > 0:
        samples = np.pad(samples, (0, pad))
    return samples[:settings.TONE_CHUNK_SAMPLES]


def phrase_to_words(text: str, start: float, end: float) -> list:
    """
    Раскладывает текст фразы на слова, равномерно распределяя тайминги по [start, end].

    T-one не отдаёт пословных таймингов, поэтому распределяем линейно. Для построения
    предложений на стороне клиента важны относительные паузы - при равномерном
    распределении внутри фразы пауз нет, границы предложений совпадут с границами
    фраз T-one, что и требуется.
    """
    words = text.split()
    if not words:
        return []
    span = max(end - start, 0.0)
    step = span / len(words)
    result = []
    for i, word in enumerate(words):
        result.append({
            "conf": 1.0,
            "start": round(start + i * step, 2),
            "end": round(start + (i + 1) * step, 2),
            "word": word,
        })
    return result


def phrase_to_data(phrase) -> dict:
    """
    Приводит TextPhrase к {"result": [...], "word-list], "text": ...} для WSRecognitionData.

    :param phrase: tone.TextPhrase (text, start_time, end_time)
    :return: {"result": [{conf,start,end,word}...], "text": "..."}
    """
    return {
        "result": phrase_to_words(phrase.text, phrase.start_time, phrase.end_time),
        "text": phrase.text,
    }
