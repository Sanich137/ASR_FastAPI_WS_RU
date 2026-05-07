import logging
import os
from datetime import date, timedelta
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
    ALLOWED_HOSTS: list[str] = ["*"]
    CORS_ORIGINS: list[str] = ["*"]
    TRUSTED_PROXIES: list[str] = ["*"]

    # Model settings
    # Vosk5SmallStreaming  Vosk5 Gigaam Whisper Gigaam_rnnt, "gigaam-v3-rnnt", "gigaam-v3-ctc"
    MODEL_NAME: str = "gigaam-v3-ctc"
    # Стрим из астериска отдаёт только 8к
    BASE_SAMPLE_RATE: int = 16000
    PROVIDER: str = "CPU"
    NUM_THREADS: int = 0

    # HuggingFace Hub settings
    HF_HOME: str = "./models"

    # Logger settings
    LOGGING_LEVEL: str = 'DEBUG'
    LOGGING_FORMAT: str = '#%(levelname)-8s %(filename)s [LINE:%(lineno)d] [%(asctime)s]  %(message)s'
    FILENAME: str = Field(default_factory=lambda: f'logs/ASR-started_{date.today()}')
    FILEMODE: str = 'a'
    # Срок хранения логов в днях
    LOG_BACKUP_COUNT: int = 60
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

    # Auth / JWT settings
    SECRET_KEY: str = Field(default="change-me-in-production-32-chars-long", min_length=32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 дней
    BCRYPT_ROUNDS: int = 12

    # Quota settings
    GUEST_DAILY_QUOTA: int = 10

    # WebSocket settings
    WS_MAX_CONNECTIONS: int = 100
    WS_MAX_BUFFER_DURATION_SEC: float = 300.0
    WS_IDLE_TIMEOUT_SEC: float = 60.0
    WS_PING_TIMEOUT_SEC: float = 20.0
    WS_MAX_MESSAGE_SIZE_MB: float = 10.0
    WS_MAX_SESSION_DURATION_SEC: float = 300.0
    WS_STATUS_BROADCAST_INTERVAL_SEC: float = 5.0
    WS_STATUS_GPU_OVERLOAD_THRESHOLD_PCT: float = 90.0
    WS_STATUS_BUSY_CONNECTIONS_THRESHOLD_PCT: float = 80.0

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

        # Предупреждение о небезопасной CORS-конфигурации в продакшене
        if self.IS_PROD and self.CORS_ORIGINS == ["*"]:
            logging.warning(
                "SECURITY WARNING: CORS_ORIGINS is set to ['*'] in production (IS_PROD=True). "
                "This is insecure when allow_credentials=True. "
                "Please specify explicit origins in CORS_ORIGINS."
            )

        # Валидация SECRET_KEY для production
        if self.IS_PROD and len(self.SECRET_KEY) < 32:
            raise ValueError(
                "SECURITY ERROR: SECRET_KEY must be at least 32 characters long in production (IS_PROD=True). "
                "Please set a strong SECRET_KEY environment variable."
            )

        # Warning при использовании дефолтного SECRET_KEY в production
        if self.IS_PROD and self.SECRET_KEY == "change-me-in-production-32-chars-long":
            logging.warning(
                "SECURITY WARNING: Using default SECRET_KEY in production (IS_PROD=True). "
                "Please set a strong unique SECRET_KEY environment variable."
            )

        return self


# Единственный экземпляр настроек
settings = Settings()



AUDIOEXTENTIONS = [
    # Основные форматы
    'mp3', 'wav', 'aac', 'ogg', 'flac', 'm4a', 'wma', 'aiff', 'alac',
    # Менее распространённые форматы
    'ape', 'opus', 'amr', 'au', 'mid', 'midi', 'ac3', 'dts', 'ra', 'rm', 'voc',
    # Форматы для сжатия и профессионального аудио
    'dsd', 'pcm', 'raw', 'tta', 'webm', '3ga', '8svx', 'cda',
    # Форматы с потерями и без потерь
    'mp2', 'mp1', 'gsm', 'vox', 'dss', 'mka', 'tak', 'ofr', 'spx',
    # Игровые аудиоформаты
    'xm', 'mod', 's3m', 'it', 'nsf',
    # Редкие/устаревшие форматы
    '669', 'mtm', 'med', 'far', 'umx'
]

# Описание WebSocket для OpenAPI
WS_DESCRIPTION = """
## WebSocket Endpoint — `/api/v1/asr/ws` (актуальный протокол)

### Пример конфигурации

Отправьте JSON с конфигурацией:

```json
{
    "type": "config",
    "sample_rate": 16000,
    "audio_format": "pcm16",
    "audio_transport": "json_base64",
    "wait_null_answers": true,
    "do_dialogue": false,
    "do_punctuation": false,
    "channel_name": "channel_1"
}
```

### Пример передачи аудио (JSON + base64)

```json
{
    "type": "audio_chunk",
    "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
    "seq_num": 0
}
```

### Пример передачи аудио (Binary)

Используется при `audio_transport: "binary"`. Отправляйте WebSocket **binary frame** напрямую (без JSON-обёртки). Сервер читает его через `receive_bytes()`.

### Пример завершения потока (EOS)

```json
{
    "type": "eos"
}
```

### Формат ответа

```json
{
    "type": "partial_result",
    "channel_name": "channel_1",
    "silence": false,
    "data": {
        "result": [
            {"conf": 1.0, "start": 116.48, "end": 116.76, "word": "владимир"},
            {"conf": 1.0, "start": 116.92, "end": 117.48, "word": "анатольевич"}
        ],
        "text": "владимир анатольевич"
    },
    "error": null,
    "last_message": false,
    "sentenced_data": null
}
```

При завершении (`last_message: true`) и включённых `do_dialogue` + `do_punctuation`:

```json
{
    "type": "final_result",
    "channel_name": "channel_1",
    "silence": false,
    "data": {
        "result": [...],
        "text": "..."
    },
    "error": null,
    "last_message": true,
    "sentenced_data": {
        "raw_text_sentenced_recognition": "channel_1: Ничьих, не требуя ... мои.\\nchannel_1: У Лукоморья дуб зеленый.",
        "list_of_sentenced_recognitions": [...],
        "full_text_only": ["..."]
    }
}
```

---

## WebSocket Endpoint — `/ws` (legacy, deprecated)

> ⚠️ **Deprecated**: этот endpoint сохраняется для обратной совместимости. Используйте `/api/v1/asr/ws`.

### Пример конфигурации (legacy)

```json
{
    "config": {
        "audio_format": "pcm16",
        "sample_rate": 16000,
        "wait_null_answers": true,
        "do_dialogue": false,
        "do_punctuation": false,
        "channelName": "channel_1"
    }
}
```

### Пример передачи данных (legacy)

```json
{
    "text": "eof"
}
```

### Ответы (legacy)

```json
{
    "channel_name": "Null",
    "silence": false,
    "data": {
        "result": [
            {"conf": 1.0, "start": 116.48, "end": 116.76, "word": "владимир"},
            {"conf": 1.0, "start": 116.92, "end": 117.48, "word": "анатольевич"}
        ],
        "text": "владимир анатольевич"
    },
    "error": null,
    "last_message": false,
    "sentenced_data": {}
}
```
"""
