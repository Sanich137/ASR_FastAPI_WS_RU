import logging
import asyncio

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler, DirMovedEvent, FileMovedEvent
from pathlib import Path

from config import settings

from models.fast_api_models import PostFileRequest
from utils.save_asr_to_md import save_to_file


def send_file_to_asr(event, file_path):
    from Recognizer.engine.file_recognition import process_file
    if not event.is_directory:
        file_params = PostFileRequest()
        file_params.speech_speed_correction_multiplier = 1
        file_params.do_diarization = settings.CAN_DIAR
        file_params.do_punctuation = settings.CAN_PUNCTUATE
        file_params.do_dialogue = True
        file_params.do_echo_clearing = True
        file_params.make_mono = settings.MAKE_MONO
        file_params.diar_vad_sensity = 2
        file_params.do_auto_speech_speed_correction = True
        file_params.keep_raw = False

        asr_data = process_file(
            tmp_path=file_path,
            params=file_params
        )
        # # Имя файла без расширения
        # file_name = file_path.stem

        if asr_data["success"]:
            asyncio.run(save_to_file(asr_data, file_path))

            if settings.DELETE_LOCAL_FILE_AFTR_ASR:
                try:
                    file_path.unlink()
                    logging.info(f"Файл {file_path} удалён")
                except FileNotFoundError:
                    logging.info(f"Не могу удалить файл {file_path} - не существует")
                except PermissionError:
                    logging.error(f"Нет прав на удаление файла {file_path}")
                except Exception as e:
                    logging.error(f"Ошибка при удалении файла: {e}")
                finally:
                    logging.info(f"Работы с файлом {file_path} завершены")

class FileHandler(FileSystemEventHandler):

    def on_created(self, event):
        # Todo - перенести paths в отдельный файл и импортировать его по необходимости
        file_path = Path(event.src_path)
        logging.info(f"Получено сообщение о новом файле {event.src_path}")
        if file_path.suffix not in settings.AUDIOEXTENTIONS:
            logging.info(f"Файл {file_path.name} пропущен, т.к. не аудио формат")
        else:
            send_file_to_asr(event,file_path)

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        file_path = Path(event.dest_path)
        logging.info(f"Получено сообщение о переименовании  файла {event.dest_path}")

        if file_path.suffix not in settings.AUDIOEXTENTIONS:
            logging.info(f"Файл {file_path.name} пропущен, т.к. не аудио формат")
        else:
            send_file_to_asr(event, file_path)

def start_file_watcher(file_path: str) -> Observer:
    event_handler = FileHandler()
    # observer = Observer()
    observer = PollingObserver()
    observer.schedule(event_handler=event_handler, path=file_path, recursive=True)
    observer.start()
    logging.info(f"Включили отслеживание состояния папки {file_path}")

    return observer