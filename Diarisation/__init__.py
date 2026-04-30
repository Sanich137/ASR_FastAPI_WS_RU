from config import settings
from utils.pre_start_init import paths
from VoiceActivityDetector import vad
import requests
import logging
from starlette.requests import HTTPConnection
logger = logging.getLogger(__name__)


def ensure_diar_model() -> bool:
    """Проверяет наличие модели диаризации и при необходимости скачивает её."""
    if not settings.CAN_DIAR:
        return False
    if not paths.get("diar_speaker_model_path").exists():
        logger.error(f"Модель для диаризации не найдена. Предпринимаются попытки скачать {settings.DIAR_MODEL_NAME}")
        output_path = paths.get("diar_speaker_model_path")
        api_url = "https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree"

        try:
            response = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            target_file = settings.DIAR_MODEL_NAME
            file_data = next((item for item in response.json()["Data"] if item["Key"] == target_file), None)

            if not file_data:
                logger.error(f"Модели с именем {settings.DIAR_MODEL_NAME} в списке возможных для загрузки нет.")
                onnx_models_with_size = [
                    (item['Key'].split(".")[0], item['Size'] // (1024 * 1024))
                    for item in response.json()["Data"]
                    if item['Key'].endswith('.onnx')
                ]
                logger.info("Доступны для скачивания следующие модели:")
                for name, size in sorted(onnx_models_with_size):
                    logger.info(f"{name} - {size} MB")
                raise Exception("Файл не найден в API")

            download_url = file_data["Url"]
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            logger.error(f"Файл модели не сохранён. Ошибка - {e}")
            return False
        else:
            logger.info(f"Модель успешно загружена : {output_path}")
    else:
        logger.debug(f"Будет использован имеющийся файл {settings.DIAR_MODEL_NAME}")

    if not paths.get("diar_speaker_model_path").exists():
        logger.error(f"Модель для Диаризации отсутствует. Диаризация выключена.")
        logger.error(f"Скачайте модель со страницы 'https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md' "
                     f"и поместите по адресу: {str(paths.get('diar_speaker_model_path'))}")
        return False
    return True


def get_diarizer(conn: HTTPConnection):
    return conn.app.state.diarizer
