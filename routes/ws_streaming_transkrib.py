from pydub import AudioSegment

import asyncio
import queue
import ujson
import config
import uuid
from io import BytesIO

from utils.pre_start_init import app
from fastapi import WebSocket, WebSocketException
from utils.do_logging import logger
streaming_handles = {}

from utils.send_messages import send_messages
from utils.resamppling import async_resample_audiosegment

from Recognizer.engine.sentensizer import do_sensitizing
from Recognizer import pool


async def _result_forwarder(client_id, handle, ws, session_config):
    """Читает результаты из output_queue worker'а и отправляет клиенту."""
    collected = []
    try:
        while True:
            try:
                msg = await asyncio.get_event_loop().run_in_executor(None, handle.output_queue.get)
            except Exception as e:
                logger.error(f"Forwarder error reading queue: {e}")
                break

            msg_type = msg.get("type")
            if msg_type == "done":
                break

            data = msg.get("data", {})
            if not data or not data.get("text"):
                continue

            asr_result = {
                "data": {
                    "text": data.get("text", ""),
                    "result": data.get("words", [])
                }
            }
            collected.append(asr_result)

            is_last = (msg_type == "final")
            sentenced_data = {}
            if is_last and session_config.get("do_dialogue"):
                try:
                    sentenced_data = await do_sensitizing({f"channel_{1}": collected}, session_config.get("do_punctuation"))
                except Exception as e:
                    logger.error(f"await do_sensitizing - {e}")

            if not await send_messages(
                ws,
                _silence=False,
                _data=asr_result,
                _error=None,
                _last_message=is_last,
                _sentenced_data=sentenced_data,
                _channel_name=session_config.get("channel_name", "Null"),
            ):
                logger.error(f"send_message not ok in forwarder")
                break
    finally:
        # Закрываем WebSocket, чтобы гарантировать выход из ws.receive() в роуте
        # и корректное выполнение finally с pool.release
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/ws_streaming")
async def websocket_streaming(ws: WebSocket):
    client_id = uuid.uuid4()
    logger.debug(f'Принят новый сокет потокового распознавания id = {client_id}')
    handle = pool.acquire()
    streaming_handles[client_id] = handle

    session_config = {"do_dialogue": False, "do_punctuation": False, "channel_name": "Null"}
    audio_format = 'raw'
    sample_rate = config.BASE_SAMPLE_RATE
    close_reason = "unknown"

    await ws.accept()

    forwarder_task = asyncio.create_task(
        _result_forwarder(client_id, handle, ws, session_config)
    )

    try:
        while True:
            try:
                message = await ws.receive()
            except Exception as wse:
                logger.error(f"receive WebSocketException - {wse}")
                close_reason = f"receive_error:{wse}"
                break

            if isinstance(message, dict) and message.get('text'):
                try:
                    if message.get('text') and 'config' in message.get('text'):
                        json_cfg = ujson.loads(message.get('text'))['config']
                        audio_format = json_cfg.get("audio_format", 'pcm16')
                        sample_rate = json_cfg.get('sample_rate')
                        session_config["do_dialogue"] = json_cfg.get("do_dialogue", False)
                        session_config["do_punctuation"] = json_cfg.get("do_punctuation", False)
                        session_config["channel_name"] = json_cfg.get("channelName", "Null")
                        if session_config["channel_name"] == "Null":
                            logger.debug("ChannelName not parsed")
                        logger.info(f"Task received, config -  {message.get('text')}")
                        continue

                    elif message.get('text') and 'eof' in message.get('text'):
                        logger.info(f"EOF received in channel {session_config['channel_name']}")
                        try:
                            handle.input_queue.put({"type": "eof"}, block=False)
                        except queue.Full:
                            logger.warning(f"Input queue full for {client_id}")
                        close_reason = "eof_received"
                        break
                    else:
                        logger.error(f"Can`t recognise  text part of  message {message.get('text')} in channel {session_config['channel_name']}")

                except Exception as e:
                    logger.error(f'Error text message compiling. Message:{message} - error:{e} in channel {session_config["channel_name"]}')
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
                            logger.error(f"Ошибка принятия аудио - {e} in channel {session_config['channel_name']}")
                            continue
                        else:
                            logger.debug(f"Чанк принят и распознан in channel {session_config['channel_name']}")

                    if audiosegment_chunk.frame_rate != config.STREAM_BASE_SAMPLE_RATE:
                        audiosegment_chunk = await async_resample_audiosegment(audiosegment_chunk,
                                                                               config.STREAM_BASE_SAMPLE_RATE)

                    if audiosegment_chunk.channels != 1:
                        audiosegment_chunk = audiosegment_chunk.set_channels(1)

                    try:
                        handle.input_queue.put({"type": "chunk", "payload": audiosegment_chunk}, block=False)
                    except queue.Full:
                        logger.warning(f"Input queue full for {client_id}, dropping chunk")
                except Exception as e:
                    logger.error(f"AcceptWaveform error - {e} in channel {session_config['channel_name']}")
            elif isinstance(message, dict) and message.get('type') == "websocket.disconnect":
                description = f"Channel {session_config['channel_name']} closed from outside"
                logger.error(description)
                try:
                    handle.input_queue.put({"type": "disconnect"}, block=False)
                except queue.Full:
                    logger.warning(f"Input queue full for {client_id}, cannot send disconnect")
                close_reason = "websocket_disconnect"
                break
            else:
                error_description = f"Can`t parse message - {message} in channel {session_config['channel_name']}"
                logger.error(error_description)

                if not await send_messages(ws, _silence=False, _data=None, _error=error_description, _channel_name=session_config['channel_name']):
                    logger.error(f"send_message not ok work canceled in channel {session_config['channel_name']}")
                    try:
                        handle.input_queue.put({"type": "disconnect"}, block=False)
                    except queue.Full:
                        logger.warning(f"Input queue full for {client_id}, cannot send disconnect")
                    close_reason = "send_messages_failed"
                    break
    finally:
        # Даём worker'у время дочитать оставшиеся чанки перед EOF
        if handle and close_reason == "eof_received":
            try:
                drain_start = asyncio.get_event_loop().time()
                while (
                    not handle.input_queue.empty()
                    and (asyncio.get_event_loop().time() - drain_start) < 3.0
                ):
                    await asyncio.sleep(0.05)
                if handle.worker.is_alive():
                    await asyncio.get_event_loop().run_in_executor(
                        None, handle.worker.join, 2.0
                    )
                logger.info(
                    f"Post-EOF drain for {client_id}: "
                    f"input_queue={handle.input_queue.qsize()} "
                    f"worker_alive={handle.worker.is_alive()}"
                )
            except Exception as e:
                logger.warning(f"Post-EOF drain error for {client_id}: {e}")

        # Ждём завершения forwarder'а
        try:
            await asyncio.wait_for(forwarder_task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning(f"Forwarder timeout for {client_id}, cancelling")
            forwarder_task.cancel()
            try:
                await forwarder_task
            except asyncio.CancelledError:
                pass

        # Очистка
        handle = streaming_handles.get(client_id)
        if handle:
            try:
                logger.info(
                    f"Closing connection {session_config['channel_name']} | "
                    f"reason={close_reason} | "
                    f"input_queue={handle.input_queue.qsize()} | "
                    f"output_queue={handle.output_queue.qsize()} | "
                    f"recognizer_mode={handle.recognizer._mode} | "
                    f"pre_roll={len(handle.recognizer._pre_roll_buffer)} | "
                    f"vad_samples={len(handle.recognizer._vad_samples)} | "
                    f"vad_frame_count={handle.recognizer._vad_frame_count} | "
                    f"in_speech={handle.recognizer._in_speech}"
                )
            except Exception as e:
                logger.error(f"Error logging close state: {e}")
        pool.release(handle, send_disconnect=(close_reason != "eof_received"))
        try:
            del streaming_handles[client_id]
        except KeyError:
            pass

        try:
            await ws.close()
        except Exception:
            pass
