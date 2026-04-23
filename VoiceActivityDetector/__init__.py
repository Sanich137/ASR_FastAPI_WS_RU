from config import settings
from utils.pre_start_init import paths
from utils.do_logging import logger
import requests


if not paths.get("vad_model_path").exists():
    logger.info("Модель silero_vad.onnx отсутствует. Предпринимаем попытку скачать её.")
    url = "https://github.com/snakers4/silero-vad/blob/v5.0/files/silero_vad.onnx?raw=true"
    output_file = str(paths.get("vad_model_path"))

    try:
        response = requests.get(url, stream=True)
    except Exception as e:
        logger.error(f"Ошибка выполнения запроса на скачивание модели - {e}")
    else:
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
            logger.info("Модель silero_vad.onnx успешно загружена")
        else:
            logger.error(f"Ошибка при скачивании файла. Статус: {response.status_code}")
else:
    logger.info("Будет использован имеющийся файл Silero_VADv5")

if not paths.get("vad_model_path").exists():
    logger.error("Работа сервиса без VAD невозможна")
    raise FileExistsError
else:
    from .do_vad import SileroVAD
    vad=SileroVAD(paths.get("vad_model_path"), use_gpu= settings.VAD_WITH_GPU)
    # Нужно наблюдать за результатом работы в многопотоке (если он будет)
    # Если будут сбои, то переводить создание класса в отдельный процесс.
    vad.set_mode(settings.VAD_SENSITIVITY)
