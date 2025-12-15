from pydub import AudioSegment

import ujson
import config
import uuid
from io import BytesIO
import subprocess


from utils.pre_start_init import app
from fastapi import WebSocket, WebSocketException
from utils.do_logging import logger
from utils.chunk_doing import find_last_speech_position
from utils.pre_start_init import audio_buffer, audio_overlap, audio_to_asr, audio_duration,ws_collected_asr_res
from utils.send_messages import send_messages
from utils.tokens_to_Result import process_asr_json_deprecated, process_gigaam_asr
from utils.resamppling import async_resample_audiosegment

from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer.engine.stream_recognition import simple_recognise


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    wait_null_answers=True
    client_id = uuid.uuid4()
    logger.debug(f'Принят новый сокет id = {client_id}')
    audio_buffer[client_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
    audio_overlap[client_id] = AudioSegment.silent(1, frame_rate=config.BASE_SAMPLE_RATE)
    audio_duration[client_id] = 0
    audio_to_asr[client_id] = list()
    ws_collected_asr_res[client_id] = {f"channel_{1}": list()}
    do_dialogue = False
    do_punctuation = False
    audio_format = 'raw'
    sample_rate = config.BASE_SAMPLE_RATE  # Если не получен фреймрейт в конфиге сокета, по попытается принять с конфигом модели.
    sentenced_data = None
    error_description = None

    await ws.accept()
    channel_name = str()

    while True:
        try:
            message = await ws.receive()
        except Exception as wse:
            logger.error(f"receive WebSocketException - {wse}")
            return

        if isinstance(message, dict) and message.get('text'):
            try:
                if message.get('text') and 'config' in message.get('text'):
                    json_cfg = ujson.loads(message.get('text'))['config']
                    audio_format = json_cfg.get("audio_format", 'pcm16')
                    sample_rate = json_cfg.get('sample_rate')
                    wait_null_answers = json_cfg.get('wait_null_answers', wait_null_answers)
                    do_dialogue = json_cfg.get("do_dialogue", False)
                    do_punctuation = json_cfg.get("do_punctuation", False)
                    try:
                        channel_name = message.get('text').get("channelName")
                    except Exception as e:
                        channel_name = "Null"
                        logger.debug("ChannelName not parsed")
                    logger.info(f"Task received, config -  {message.get('text')}")
                    continue

                elif message.get('text') and 'eof' in message.get('text'):
                    logger.info(f"EOF received in channel {channel_name}")
                    break
                else:
                    logger.error(f"Can`t recognise  text part of  message {message.get('text')} in channel {channel_name}")

            except Exception as e:
                logger.error(f'Error text message compiling. Message:{message} - error:{e} in channel {channel_name}')
        elif isinstance(message, dict) and message.get('bytes'):
            try:
                # Получаем новый чанк с данными
                chunk = message.get('bytes')

                if audio_format == 'pcm16':
                    # Переводим чанк в объект Audiosegment
                    audiosegment_chunk = AudioSegment(
                        chunk,
                        frame_rate = sample_rate,  # Частота дискретизации
                        sample_width = 2,   # Ширина сэмпла (2 байта для int16)
                        channels = 1        # Количество каналов. По умолчанию - 1, Моно.
                    )
                else:
                    try:
                        buffer = BytesIO(chunk)
                        buffer.seek(0)
                        # buffer.write()
                        audiosegment_chunk = AudioSegment.from_file(buffer)

                    except Exception as e:
                        logger.error(f"Ошибка принятия аудио - {e} in channel {channel_name}")
                    else:
                        logger.debug(f"Чанк принят и распознан in channel {channel_name}")

                # Приводим фреймрейт к фреймрейту модели
                if audiosegment_chunk.frame_rate != config.BASE_SAMPLE_RATE:
                    audiosegment_chunk = await async_resample_audiosegment(audiosegment_chunk, config.BASE_SAMPLE_RATE)

                if audiosegment_chunk.channels != 1:
                    audiosegment_chunk = audiosegment_chunk.set_channels(1)

                # Копим буфер
                audio_buffer[client_id] += audiosegment_chunk

                # Накопили больше нормы
                if (audio_overlap[client_id]+audio_buffer[client_id]).duration_seconds >= config.MAX_OVERLAP_DURATION:
                    # Проверяем новый чанк перед объединением (там же режем хвост и добавляем его при необходимости)
                    await find_last_speech_position(client_id, is_last_chunk=False)
                else:
                    continue
            except Exception as e:
                logger.error(f"AcceptWaveform error - {e} in channel {channel_name}")
            else:
                try:
                    asr_result = await simple_recognise(audio_to_asr[client_id][-1])
                    asr_result_words = await process_asr_json_deprecated(asr_result, audio_duration[client_id])
                    audio_duration[client_id] += audio_to_asr[client_id][-1].duration_seconds
                    logger.debug(asr_result_words)

                    # Копим ответы для пунктуации
                    ws_collected_asr_res[client_id][f"channel_{1}"].append(asr_result_words)

                except Exception as e:
                    logger.error(f"recognizer.get_result(stream()) error - {e}")
                else:
                    if len(asr_result_words.get("data").get("text")) == 0 or asr_result_words.get("data").get("text") == ' ':
                        if wait_null_answers:
                            if not await send_messages(ws, _silence = True, _data = None, _error = None, _channel_name=channel_name):
                                logger.error(f"send_message not ok work canceled")
                                try:
                                    del audio_overlap[client_id]
                                    del audio_buffer[client_id]
                                    del audio_to_asr[client_id]
                                    del audio_duration[client_id]
                                    del ws_collected_asr_res[client_id]
                                except Exception as e:
                                    logger.error(f"error clearing globals after abnormal closing socket - {e}")
                                return
                            # await asyncio.sleep(0.01)
                        else:
                            logger.debug("sending silence partials skipped")
                            continue
                    else:
                        if not await send_messages(ws, _silence=False, _data=asr_result_words, _error=None, _channel_name = channel_name):
                            logger.error(f"send_message not ok work canceled")
                            try:
                                del audio_overlap[client_id]
                                del audio_buffer[client_id]
                                del audio_to_asr[client_id]
                                del audio_duration[client_id]
                                del ws_collected_asr_res[client_id]
                            except Exception as e:
                                logger.error(f"error clearing globals after abnormal closing socket - {e}")
                            return
        elif isinstance(message, dict) and message.get('type') == "websocket.disconnect":
            description = f"Channel {channel_name} closed from outside"
            logger.error(description)
            break
        else:
            error_description = f"Can`t parse message - {message} in channel {channel_name}"
            logger.error(error_description)

            if not await send_messages(ws, _silence=False, _data=None, _error=error_description, _channel_name=channel_name):
                logger.error(f"send_message not ok work canceled in channel {channel_name}")
                try:
                    del audio_overlap[client_id]
                    del audio_buffer[client_id]
                    del audio_to_asr[client_id]
                    del audio_duration[client_id]
                    del ws_collected_asr_res[client_id]
                except Exception as e:
                    logger.error(f"error clearing globals after abnormal closing socket - {e} in channel {channel_name}")
                return

    # Передаём на распознавание собранный не полный буфер
    # перевод в семплы для распознавания.
    audio_to_asr[client_id].append(audio_overlap[client_id] + audio_buffer[client_id])
    logger.debug(f'итоговое сообщение - {audio_to_asr[client_id][-1].duration_seconds} секунд')

    try:
        try:
            if audio_to_asr[client_id][-1].duration_seconds < 2:
                audio_to_asr[client_id][-1] = audio_to_asr[client_id][-1] + AudioSegment.silent(1000, frame_rate=sample_rate)
        except Exception as e:
            logger.error(f"Ошибка дополнения тишиной последнего чанка - {e} in channel {channel_name}")
            last_result = None
            error_description = f"Ошибка дополнения тишиной последнего чанка - {e} in channel {channel_name}"
        else:
            last_asr_result_w_conf = await simple_recognise(audio_to_asr[client_id][-1])
            last_result = await process_asr_json_deprecated(last_asr_result_w_conf, audio_duration[client_id])
            logger.debug(f'Последний результат {last_result.get("data").get("text")} in channel {channel_name}')

            # Добавляем ответ для пунктуации
            ws_collected_asr_res[client_id][f"channel_{1}"].append(last_result)


    except Exception as e:
        logger.error(f"last_asr_result_w_conf error - {e}")

    else:
        if len(last_result.get("data").get("text")) == 0:
            is_silence = True
            last_result = None
        elif last_result.get("data").get("text") == ' ':
            is_silence = True
            last_result = None
        else:
            logger.debug(last_result)
            is_silence = False

        if do_dialogue:
            try:
                sentenced_data = await do_sensitizing(ws_collected_asr_res[client_id], do_punctuation)
            except Exception as e:
                logger.error(f"await do_sensitizing - {e}")
                error_description = f"do_sensitizing - {e}"

        #
        if not await send_messages(ws, _silence=is_silence, _data=last_result, _error=error_description, _last_message=True,
                                   _sentenced_data=sentenced_data, _channel_name=channel_name):
            logger.error(f"send_message not ok work canceled in channel {channel_name}")
            try:
                del audio_overlap[client_id]
                del audio_buffer[client_id]
                del audio_to_asr[client_id]
                del audio_duration[client_id]
                del ws_collected_asr_res[client_id]
            except Exception as e:
                logger.error(f"error clearing globals after abnormal closing socket - {e} in channel {channel_name}")
            return

    logger.info(f"Closing connection {channel_name}")
    await ws.close()

    try:
        del audio_overlap[client_id]
        del audio_buffer[client_id]
        del audio_to_asr[client_id]
        del audio_duration[client_id]
        del ws_collected_asr_res[client_id]
    except Exception as e:
        logger.error(f"error clearing globals after NORMAL closing socket - {e} in channel {channel_name}")
    return
