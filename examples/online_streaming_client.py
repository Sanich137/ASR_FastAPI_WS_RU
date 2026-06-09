import asyncio
from pathlib import Path
import websockets
import ujson
from pydub import AudioSegment
import argparse
import logging


class OnlineASRStreamingClient:
    def __init__(self, uri, frame_rate=8000, buffer_size_sec=0.5):
        self.uri = uri
        self.frame_rate = frame_rate
        self.buffer_size = int(frame_rate * buffer_size_sec * 2)  # 16-bit samples
        self.active = False

    async def stream_audio(self, file_path, wait_null_answers=False):
        """Потоковая передача аудио на /ws_streaming с обработкой ответов"""
        self.active = True
        sound = self._prepare_audio(file_path)

        async with websockets.connect(
                self.uri,
                ping_interval=None,
                close_timeout=2
        ) as websocket:
            # Отправка конфигурации
            await self._send_config(websocket, sound.frame_rate, wait_null_answers)

            # Потоковая передача данных и обработка ответов параллельно
            await asyncio.gather(
                self._stream_data(websocket, sound),
                self._handle_responses(websocket)
            )

    def _prepare_audio(self, file_path):
        """Подготовка аудиофайла"""
        sound = AudioSegment.from_file(str(file_path))
        if sound.frame_rate != self.frame_rate:
            logging.warning(f"Конвертация FR {sound.frame_rate} → {self.frame_rate}")
            sound = (sound.set_frame_rate(self.frame_rate))
        return sound.set_channels(1)

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

    async def _stream_data(self, websocket, sound):
        """Потоковая передача аудиоданных"""
        try:
            data = sound.raw_data
            chunk_size = self.buffer_size

            for i in range(0, len(data), chunk_size):
                if not self.active:
                    break

                chunk = data[i:i + chunk_size]
                await websocket.send(chunk)
                await asyncio.sleep(0.5)

            # Отправка EOF после завершения стриминга
            await websocket.send(ujson.dumps({"eof": True}))
            logging.info("EOF sent")

            # Даём серверу время отправить последние ответы
            await asyncio.sleep(0.01)

        except Exception as e:
            logging.error(f"Ошибка при отправке аудио: {e}")
            self.active = False

    async def _handle_responses(self, websocket):
        """Обработка ответов сервера"""
        try:
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    result = ujson.loads(response)

                    if result.get("data") and result["data"].get("text"):
                        if result.get("last_message", False):
                            print(f"[Финальный] {result['data']['text']}")
                            logging.info("Получено последнее сообщение")
                            break
                        else:
                            print(f"[Частичный] {result['data']['text']}")
                    elif result.get("last_message"):
                        logging.info("Получено последнее сообщение (пустое)")
                        break

                except asyncio.TimeoutError:
                    if not self.active:
                        break
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logging.info("Соединение закрыто сервером")
                    break
                except Exception as e:
                    logging.error(f"Ошибка обработки ответа: {e}")
                    continue

        finally:
            self.active = False


if __name__ == "__main__":

    # Для запуска из IDE
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.uri = "ws://192.168.101.28:49153/ws_streaming"
    args.file = "test.wav"
    args.frame_rate = 16000
    args.buffer_size = 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    client = OnlineASRStreamingClient(
        args.uri,
        frame_rate=args.frame_rate,
        buffer_size_sec=args.buffer_size
    )

    try:
        asyncio.run(client.stream_audio(Path(args.file)))
    except KeyboardInterrupt:
        logging.info("Прервано пользователем")
    except Exception as e:
        logging.error(f"Ошибка клиента: {e}")
    finally:
        logging.info("Клиент завершил работу")
