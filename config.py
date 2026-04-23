import os
from datetime import date
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator


class Settings(BaseSettings):
    """
    Конфигурация приложения на базе pydantic-settings.
    Все значения могут быть переопределены через переменные окружения
    или файл .env в корне проекта.
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # server settings
    HOST: str = '0.0.0.0'
    PORT: int = 49153

    # Model settings
    # Vosk5SmallStreaming  Vosk5 Gigaam Whisper Gigaam_rnnt, "gigaam-v3-rnnt", "gigaam-v3-ctc"
    MODEL_NAME: str = "gigaam-v3-ctc"
    # Стрим из астериска отдаёт только 8к
    BASE_SAMPLE_RATE: int = 16000
    PROVIDER: str = "CUDA"
    NUM_THREADS: int = 0

    # HuggingFace Hub settings
    HF_HOME: str = "./models"

    # Logger settings
    LOGGING_LEVEL: str = 'DEBUG'
    LOGGING_FORMAT: str = '#%(levelname)-8s %(filename)s [LINE:%(lineno)d] [%(asctime)s]  %(message)s'
    FILENAME: str = Field(default_factory=lambda: f'logs/ASR-{date.today()}.log')
    FILEMODE: str = 'a'
    # Срок хранения логов в днях
    LOG_BACKUP_COUNT: int = 180
    IS_PROD: bool = True

    # Recognition settings
    # Максимальная продолжительность буфера аудио (зависит от модели) приемлемый диапазон 10-15 сек.
    # Для Vosk, для Гига СТС можно больше.
    MAX_OVERLAP_DURATION: int = 30
    # Пока не менять
    RECOGNITION_ATTEMPTS: int = 1
    # Нормальное количество токенов в секунду. При превышении этого значения становится
    # возможным автоматически замедлять скорость речи для улучшения распознавания. В реальной речи, как правило, находится
    # в интервале от 13 до 25.
    SPEECH_PER_SEC_NORM_RATE: int = 18
    MAKE_MONO: bool = False
    USE_BATCH: bool = True
    # Размер батча для распознавания аудио.
    ASR_BATCH_SIZE: int = 8

    # Vad settings
    # 1 to 5 Higher - more words.
    VAD_SENSITIVITY: int = 3
    VAD_WITH_GPU: bool = False

    # Sentensize settings
    # Параметр определяет как мелко будет биться текст на предложения. Чем меньше значение,
    # тем более короткие будут предложения. В среднем в одном предложении 10 слов.
    # То есть, по длительности каждая десятая пауза означает конец предложения или мысли.
    # Влияет на пунктуацию выражений.
    BETWEEN_WORDS_PERCENTILE: int = 80

    # Punctuate settings
    CAN_PUNCTUATE: bool = True
    PUNCTUATE_WITH_GPU: bool = False

    # Diarisation settings
    CAN_DIAR: bool = False
    # Разных моделей для диаризации много.
    # [('cnceleb_resnet34', 25), ('cnceleb_resnet34_LM', 25), ('voxblink2_samresnet100', 191), ('voxblink2_samresnet100_ft', 191),
    # ('voxblink2_samresnet34', 96), ('voxblink2_samresnet34_ft', 96), ('voxceleb_CAM++', 27), ('voxceleb_CAM++_LM', 27),
    # ('voxceleb_ECAPA1024', 56), ('voxceleb_ECAPA1024_LM', 56), ('voxceleb_ECAPA512', 23), ('voxceleb_ECAPA512_LM', 23),
    # ('voxceleb_gemini_dfresnet114_LM', 24), ('voxceleb_resnet152_LM', 75), ('voxceleb_resnet221_LM', 90),
    # ('voxceleb_resnet293_LM', 109), ('voxceleb_resnet34', 25), ('voxceleb_resnet34_LM', 25)]
    DIAR_MODEL_NAME: str = "voxblink2_samresnet100_ft"
    DIAR_WITH_GPU: bool = False
    # Для значений меньше 1 будут использованы все доступные ядра.
    # При значении от 1 - указанное число ядер CPU. Работает только при DIAR_WITH_GPU False
    CPU_WORKERS: int = 0
    # Ширина Батча для процесса извлечения эмбеддингов с GPU.
    # Оптимально от 4 до 16. Дальнейшее увеличение приводит к неоправданному расходу памяти.
    DIAR_GPU_BATCH_SIZE: int = 2

    # Speed speech correction
    # Инструменты управления распознаванием быстрой речи. Включено
    DO_SPEED_SPEECH_CORRECTION: bool = True
    # 1 - обычная скорость, меньше - медленнее, больше - быстрее
    SPEED_SPEECH_CORRECTION_MULTIPLIER: float = 1.0

    # Local file recognition
    # Настройки сервиса локального распознавания.
    DO_LOCAL_FILE_RECOGNITIONS: bool = False
    DELETE_LOCAL_FILE_AFTR_ASR: bool = False
    HUMAN_FORMAT_MD_FILE: bool = False

    @field_validator(
        'IS_PROD', 'MAKE_MONO', 'USE_BATCH', 'VAD_WITH_GPU',
        'CAN_PUNCTUATE', 'PUNCTUATE_WITH_GPU', 'CAN_DIAR',
        'DIAR_WITH_GPU', 'DO_SPEED_SPEECH_CORRECTION',
        'DO_LOCAL_FILE_RECOGNITIONS', 'DELETE_LOCAL_FILE_AFTR_ASR',
        'HUMAN_FORMAT_MD_FILE',
        mode='before'
    )
    @classmethod
    def _int_to_bool(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v == 1
        if isinstance(v, str):
            return int(v) == 1
        return bool(v)

    @model_validator(mode='after')
    def _compute_derived(self):
        # HF_HOME должен быть установлен ДО импорта библиотек, использующих HuggingFace Hub
        os.environ["HF_HOME"] = self.HF_HOME

        # К имени модели диаризации всегда добавляем расширение .onnx
        if not self.DIAR_MODEL_NAME.endswith('.onnx'):
            self.DIAR_MODEL_NAME = self.DIAR_MODEL_NAME + '.onnx'

        # DIAR_WITH_GPU актуален только при GPU-провайдерах
        if self.DIAR_WITH_GPU and self.PROVIDER not in ["CUDA", "TENSORRT"]:
            self.DIAR_WITH_GPU = False

        # VAD_WITH_GPU актуален только при GPU-провайдерах
        if self.VAD_WITH_GPU and self.PROVIDER not in ["CUDA", "TENSORRT"]:
            self.VAD_WITH_GPU = False

        # PUNCTUATE_WITH_GPU актуален только при GPU-провайдерах
        if self.PUNCTUATE_WITH_GPU and self.PROVIDER not in ["CUDA", "TENSORRT"]:
            self.PUNCTUATE_WITH_GPU = False

        return self


# Единственный экземпляр настроек
settings = Settings()

# ==============================================================================
# Обратная совместимость: экспортируем атрибуты на уровень модуля.
# Все остальные модули проекта могут продолжать использовать `config.HOST`,
# `config.PORT` и т.д. без изменений.
# ==============================================================================
HOST = settings.HOST
PORT = settings.PORT
MODEL_NAME = settings.MODEL_NAME
BASE_SAMPLE_RATE = settings.BASE_SAMPLE_RATE
PROVIDER = settings.PROVIDER
NUM_THREADS = settings.NUM_THREADS
HF_HOME = settings.HF_HOME
LOGGING_LEVEL = settings.LOGGING_LEVEL
LOGGING_FORMAT = settings.LOGGING_FORMAT
FILENAME = settings.FILENAME
FILEMODE = settings.FILEMODE
LOG_BACKUP_COUNT = settings.LOG_BACKUP_COUNT
IS_PROD = settings.IS_PROD
MAX_OVERLAP_DURATION = settings.MAX_OVERLAP_DURATION
RECOGNITION_ATTEMPTS = settings.RECOGNITION_ATTEMPTS
SPEECH_PER_SEC_NORM_RATE = settings.SPEECH_PER_SEC_NORM_RATE
MAKE_MONO = settings.MAKE_MONO
USE_BATCH = settings.USE_BATCH
ASR_BATCH_SIZE = settings.ASR_BATCH_SIZE
VAD_SENSITIVITY = settings.VAD_SENSITIVITY
VAD_WITH_GPU = settings.VAD_WITH_GPU
BETWEEN_WORDS_PERCENTILE = settings.BETWEEN_WORDS_PERCENTILE
CAN_PUNCTUATE = settings.CAN_PUNCTUATE
PUNCTUATE_WITH_GPU = settings.PUNCTUATE_WITH_GPU
CAN_DIAR = settings.CAN_DIAR
DIAR_MODEL_NAME = settings.DIAR_MODEL_NAME
DIAR_WITH_GPU = settings.DIAR_WITH_GPU
CPU_WORKERS = settings.CPU_WORKERS
DIAR_GPU_BATCH_SIZE = settings.DIAR_GPU_BATCH_SIZE
DO_SPEED_SPEECH_CORRECTION = settings.DO_SPEED_SPEECH_CORRECTION
SPEED_SPEECH_CORRECTION_MULTIPLIER = settings.SPEED_SPEECH_CORRECTION_MULTIPLIER
DO_LOCAL_FILE_RECOGNITIONS = settings.DO_LOCAL_FILE_RECOGNITIONS
DELETE_LOCAL_FILE_AFTR_ASR = settings.DELETE_LOCAL_FILE_AFTR_ASR
HUMAN_FORMAT_MD_FILE = settings.HUMAN_FORMAT_MD_FILE

AUDIOEXTENTIONS = [
    # Основные форматы
    '.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a', '.wma', '.aiff', '.alac',
    # Менее распространённые форматы
    '.ape', '.opus', '.amr', '.au', '.mid', '.midi', '.ac3', '.dts', '.ra', '.rm', '.voc',
    # Форматы для сжатия и профессионального аудио
    '.dsd', '.pcm', '.raw', '.tta', '.webm', '.3ga', '.8svx', '.cda',
    # Форматы с потерями и без потерь
    '.mp2', '.mp1', '.gsm', '.vox', '.dss', '.mka', '.tak', '.ofr', '.spx',
    # Игровые аудиоформаты
    '.xm', '.mod', '.s3m', '.it', '.nsf',
    # Редкие/устаревшие форматы
    '.669', '.mtm', '.med', '.far', '.umx'
]
