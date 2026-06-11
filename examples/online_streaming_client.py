import asyncio
import time
from pathlib import Path
import websockets
import ujson
from pydub import AudioSegment
import argparse
import logging


class OnlineASRStreamingClient:
    def __init__(self, uri, frame_rate=8000, chunk_duration=0.1,pause_emulation=0.1):
        self.uri = uri

        self.frame_rate = frame_rate
        self.buffer_size = int(frame_rate * chunk_duration * 2)  # 16-bit samples
        self.pause_emulation = pause_emulation
        self._active_channels = set()

    async def stream_audio(self, file_path, wait_null_answers=False):
        """Потоковая передача аудио на /ws_streaming с обработкой ответов"""
        self._active_channels.clear()
        channels = self._prepare_audio(file_path)

        # Отправляем все каналы параллельно
        await asyncio.gather(
            *[
                self._stream_channel(idx + 1, ch, wait_null_answers)
                for idx, ch in enumerate(channels)
            ]
        )

    async def _stream_channel(self, channel_index: int, sound: AudioSegment, wait_null_answers: bool):
        """Отправка одного канала на сервер и обработка ответов"""
        self._active_channels.add(channel_index)
        try:
            async with websockets.connect(
                    self.uri,
                    ping_interval=None,
                    close_timeout=2
            ) as websocket:
                await self._send_config(websocket, sound.frame_rate, wait_null_answers)

                await asyncio.gather(
                    self._stream_data(websocket, sound, channel_index),
                    self._handle_responses(websocket, channel_index)
                )
        finally:
            self._active_channels.discard(channel_index)

    def _prepare_audio(self, file_path):
        """Подготовка аудиофайла: возвращает список AudioSegment по каналам"""
        sound = AudioSegment.from_file(str(file_path))
        if sound.frame_rate != self.frame_rate:
            logging.warning(f"Конвертация FR {sound.frame_rate} → {self.frame_rate}")
            sound = sound.set_frame_rate(self.frame_rate)

        if sound.channels == 1:
            return [sound]

        # Разбиваем стерео/мульти-канал на отдельные моно-каналы
        channels = []
        for i in range(sound.channels):
            ch = sound.split_to_mono()[i]
            channels.append(ch)
        return channels

    async def _send_config(self, websocket, sample_rate, wait_null_answers):
        """Отправка конфигурации серверу"""
        config = {
            "sample_rate": sample_rate,
            "wait_null_answers": wait_null_answers,
            "audio_format": "pcm16",
            "language": "ru"
        }
        await websocket.send(ujson.dumps({"config": config}))
        logging.info("Configuration sent")

    async def _stream_data(self, websocket, sound, channel_index: int):
        """Потоковая передача аудиоданных"""
        try:
            data = sound.raw_data
            chunk_size = self.buffer_size

            for i in range(0, len(data), chunk_size):
                if channel_index not in self._active_channels:
                    break

                chunk = data[i:i + chunk_size]
                await websocket.send(chunk)
                await asyncio.sleep(self.pause_emulation)

            # Отправка EOF после завершения стриминга
            await websocket.send(ujson.dumps({"eof": True}))
            logging.info(f"Канал_{channel_index}: EOF sent")

            # Даём серверу время отправить последние ответы
            await asyncio.sleep(0.01)

        except Exception as e:
            logging.error(f"Канал_{channel_index}: Ошибка при отправке аудио: {e}")
            self._active_channels.discard(channel_index)

    async def _handle_responses(self, websocket, channel_index: int):
        """Обработка ответов сервера"""
        try:
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    result = ujson.loads(response)

                    if result.get("data") and result["data"].get("text"):
                        if result.get("last_message", False):
                            print(f"[Канал_{channel_index} Финальный] {result['data']['text']}")
                            logging.info(f"Канал_{channel_index}: Получено последнее сообщение")
                            break
                        else:
                            print(f"[Канал_{channel_index} Частичный] {result['data']['text']}")
                    elif result.get("last_message"):
                        logging.info(f"Канал_{channel_index}: Получено последнее сообщение (пустое)")
                        break

                except asyncio.TimeoutError:
                    if channel_index not in self._active_channels:
                        break
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logging.info(f"Канал_{channel_index}: Соединение закрыто сервером")
                    break
                except Exception as e:
                    logging.error(f"Канал_{channel_index}: Ошибка обработки ответа: {e}")
                    continue

        finally:
            self._active_channels.discard(channel_index)


if __name__ == "__main__":
    s_time = time.time()
    # Для запуска из IDE
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.uri = "ws://192.168.101.28:49153/ws_streaming"
    args.file = "orig.wav"
    args.frame_rate = 16000
    args.chunk_duration = 5
    args.pause_emulation = 0.05

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    client = OnlineASRStreamingClient(
        args.uri,
        frame_rate=args.frame_rate,
        chunk_duration=args.chunk_duration,
        pause_emulation=args.pause_emulation
    )

    try:
        asyncio.run(client.stream_audio(Path(args.file)))
    except KeyboardInterrupt:
        logging.info("Прервано пользователем")
    except Exception as e:
        logging.error(f"Ошибка клиента: {e}")
    finally:
        logging.info("Клиент завершил работу")
        logging.info(f"Время выполнения задачи - {time.time()-s_time}c.")
