from pydub import AudioSegment

import ujson
import config
import uuid
from io import BytesIO

from utils.pre_start_init import app
from fastapi import WebSocket, WebSocketException
from utils.do_logging import logger
from utils.pre_start_init import ws_collected_asr_res

streaming_recognizers = {}

from utils.send_messages import send_messages
from utils.resamppling import async_resample_audiosegment

from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer import pool


@app.websocket("/ws_streaming")
async def websocket_streaming(ws: WebSocket):
    wait_null_answers = True
    client_id = uuid.uuid4()
    logger.debug(f'Принят новый сокет потокового распознавания id = {client_id}')
    ws_collected_asr_res[client_id] = {f"channel_{1}": list()}
    streaming_recognizers[client_id] = pool.acquire()
    do_dialogue = False
    do_punctuation = False
    audio_format = 'raw'
    sample_rate = config.BASE_SAMPLE_RATE
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
                chunk = message.get('bytes')

                if audio_format == 'pcm16':
                    audiosegment_chunk = AudioSegment(
                        chunk,
                        frame_rate = sample_rate,
                        sample_width = 2,
                        channels = 1
                    )
                else:
                    try:
                        buffer = BytesIO(chunk)
                        buffer.seek(0)
                        audiosegment_chunk = AudioSegment.from_file(buffer)
                    except Exception as e:
                        logger.error(f"Ошибка принятия аудио - {e} in channel {channel_name}")
                    else:
                        logger.debug(f"Чанк принят и распознан in channel {channel_name}")

                if audiosegment_chunk.frame_rate != config.STREAM_BASE_SAMPLE_RATE:
                    audiosegment_chunk = await async_resample_audiosegment(audiosegment_chunk, config.BASE_SAMPLE_RATE)

                if audiosegment_chunk.channels != 1:
                    audiosegment_chunk = audiosegment_chunk.set_channels(1)

                streaming_recognizers[client_id].push_audiosegment(audiosegment_chunk)
                new_text = streaming_recognizers[client_id].get_new_text()

                if new_text:
                    asr_result_words = {
                        "data": {
                            "text": new_text,
                            "words": []
                        }
                    }
                    ws_collected_asr_res[client_id][f"channel_{1}"].append(asr_result_words)
                    logger.debug(asr_result_words)
                    if not await send_messages(ws, _silence=False, _data=asr_result_words, _error=None, _channel_name=channel_name):
                        logger.error(f"send_message not ok work canceled")
                        try:
                            del ws_collected_asr_res[client_id]
                            pool.release(streaming_recognizers.pop(client_id, None))
                        except Exception as e:
                            logger.error(f"error clearing globals after abnormal closing socket - {e}")
                        return
                else:
                    if wait_null_answers:
                        if not await send_messages(ws, _silence=True, _data=None, _error=None, _channel_name=channel_name):
                            logger.error(f"send_message not ok work canceled")
                            try:
                                del ws_collected_asr_res[client_id]
                                pool.release(streaming_recognizers.pop(client_id, None))
                            except Exception as e:
                                logger.error(f"error clearing globals after abnormal closing socket - {e}")
                            return
                    else:
                        logger.debug("sending silence partials skipped")
                        continue
            except Exception as e:
                logger.error(f"AcceptWaveform error - {e} in channel {channel_name}")
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
                    del ws_collected_asr_res[client_id]
                except Exception as e:
                    logger.error(f"error clearing globals after abnormal closing socket - {e} in channel {channel_name}")
                return

    try:
        final_text = streaming_recognizers[client_id].flush()
        if final_text:
            last_result = {
                "data": {
                    "text": final_text,
                    "words": []
                }
            }
            ws_collected_asr_res[client_id][f"channel_{1}"].append(last_result)
            is_silence = False
            logger.debug(f'Последний результат {final_text} in channel {channel_name}')
        else:
            is_silence = True
            last_result = None
    except Exception as e:
        logger.error(f"Ошибка финализации потокового распознавания - {e} in channel {channel_name}")
        last_result = None
        error_description = f"Ошибка финализации потокового распознавания - {e} in channel {channel_name}"
        is_silence = True

    if do_dialogue and last_result:
        try:
            sentenced_data = await do_sensitizing(ws_collected_asr_res[client_id], do_punctuation)
        except Exception as e:
            logger.error(f"await do_sensitizing - {e}")
            error_description = f"do_sensitizing - {e}"

    if not await send_messages(ws, _silence=is_silence, _data=last_result, _error=error_description, _last_message=True,
                               _sentenced_data=sentenced_data, _channel_name=channel_name):
        logger.error(f"send_message not ok work canceled in channel {channel_name}")
        try:
            del ws_collected_asr_res[client_id]
            pool.release(streaming_recognizers.pop(client_id, None))
        except Exception as e:
            logger.error(f"error clearing globals after abnormal closing socket - {e} in channel {channel_name}")
        return

    logger.info(f"Closing connection {channel_name}")
    await ws.close()

    try:
        del ws_collected_asr_res[client_id]
        pool.release(streaming_recognizers.pop(client_id, None))
    except Exception as e:
        logger.error(f"error clearing globals after NORMAL closing socket - {e} in channel {channel_name}")
    return
