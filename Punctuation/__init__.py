from utils.pre_start_init import paths
from utils.do_logging import logger
from .punctuate import SbertPuncCaseOnnx
from config import settings

try:
    sbertpunc = SbertPuncCaseOnnx(paths.get("punctuation_model_path"), use_gpu = settings.PUNCTUATE_WITH_GPU)
except Exception as e:
    logger.error(f"Error getting punctuation model - {e}")
else:
    logger.info(f'Успешно загружена модель Пунктуации')