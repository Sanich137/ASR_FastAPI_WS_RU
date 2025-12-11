import os
import datetime
from tarfile import DEFAULT_FORMAT

# server settings
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 49153))

# Model settings
MODEL_NAME = os.getenv('MODEL_NAME', "gigaam-v3-ctc")  ## Vosk5SmallStreaming  Vosk5 Gigaam Whisper Gigaam_rnnt, "gigaam-v3-rnnt", "gigaam-v3-ctc"
BASE_SAMPLE_RATE = int(os.getenv('BASE_SAMPLE_RATE', 16000))  # Стрим из астериска отдаёт только 8к
PROVIDER = os.getenv('PROVIDER',"CUDA")
NUM_THREADS = int(os.getenv('NUM_THREADS', 0))

# HuggingFaceHubSettings
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./models")

# Logger settings
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'DEBUG')
LOGGING_FORMAT = os.getenv('LOGGING_FORMAT', u'#%(levelname)-8s %(filename)s [LINE:%(lineno)d] [%(asctime)s]  %(message)s')
FILENAME = os.getenv('FILENAME', f'logs/ASR-{datetime.datetime.now().date()}.log')
FILEMODE = os.getenv('FILEMODE', 'a')
LOG_BACKUP_COUNT = os.getenv('LOG_BACKUP_COUNT', 180)   # Срок хранения логов в днях
IS_PROD = True if int(os.getenv('IS_PROD', 1))==1 else False

# Recognition_settings
MAX_OVERLAP_DURATION = int(os.getenv('MAX_OVERLAP_DURATION', 30))  # Максимальная продолжительность буфера аудио (зависит от модели) приемлемый диапазон 10-15 сек. Для Vosk, для Гига СТС можно больше.
RECOGNITION_ATTEMPTS = 1  # Пока не менять
SPEECH_PER_SEC_NORM_RATE = 18  # Нормальное количество токенов в секунду. При превышении этого значения становится
# возможным автоматически замедлять скорость речи для улучшения распознавания. В реальной речи, как правило, находится
# в интервале от 13 до 25.
MAKE_MONO = True if int(os.getenv('MAKE_MONO', 0)) == 1 else False
USE_BATCH = True if int(os.getenv('USE_BATCH', 1)) == 1 else False
ASR_BATCH_SIZE = int(os.getenv('ASR_BATCH_SIZE', 16))   # Размер батча для распознавания аудио.

# Vad_settings
VAD_SENSITIVITY = int(os.getenv('VAD_SENSE', 3))  # 1 to 5 Higher - more words.
VAD_WITH_GPU = True if int(os.getenv('VAD_WITH_GPU', 0)) == 1 else False

# Sentensize_settings
BETWEEN_WORDS_PERCENTILE = int(os.getenv('BETWEEN_WORDS_PERCENTILE', 80)) # Параметр определяет как мелко будет биться
# текст на предложения. Чем меньше значение, тем более короткие будут предложения. В среднем в одном предложении 10 слов.
# То есть, по длительности каждая десятая пауза означает конец предложения или мысли. Влияет на пунктуацию выражений.

# Punctuate_settings
CAN_PUNCTUATE = True if int(os.getenv('CAN_PUNCTUATE', 1)) == 1 else False
PUNCTUATE_WITH_GPU = True if int(os.getenv('PUNCTUATE_WITH_GPU', 0)) == 1 else False

# Diarisation_settings
CAN_DIAR = True if int(os.getenv('CAN_DIAR', 0)) == 1 else False
DIAR_MODEL_NAME = str(os.getenv('DIAR_MODEL_NAME', "voxblink2_samresnet100_ft")+".onnx")
DIAR_WITH_GPU = True if int(os.getenv('DIAR_WITH_GPU', 0)) == 1 else False
CPU_WORKERS = int(os.getenv('CPU_WORKERS', 0)) # Для значений меньше 1 будут использованы все доступные ядра.
# При значении от 1 - указанное число ядер CPU. Работает только при DIAR_WITH_GPU False
DIAR_GPU_BATCH_SIZE = int(os.getenv('DIAR_GPU_BATCH_SIZE', 2))  # Ширина Батча для процесса извлечения эмбеддингов с GPU.
# Оптимально от 4 до 16. Дальнейшее увеличение приводит к неоправданному расходу памяти.

# Разных моделей для диаризации много.
# [('cnceleb_resnet34', 25), ('cnceleb_resnet34_LM', 25), ('voxblink2_samresnet100', 191), ('voxblink2_samresnet100_ft', 191),
# ('voxblink2_samresnet34', 96), ('voxblink2_samresnet34_ft', 96), ('voxceleb_CAM++', 27), ('voxceleb_CAM++_LM', 27),
# ('voxceleb_ECAPA1024', 56), ('voxceleb_ECAPA1024_LM', 56), ('voxceleb_ECAPA512', 23), ('voxceleb_ECAPA512_LM', 23),
# ('voxceleb_gemini_dfresnet114_LM', 24), ('voxceleb_resnet152_LM', 75), ('voxceleb_resnet221_LM', 90),
# ('voxceleb_resnet293_LM', 109), ('voxceleb_resnet34', 25), ('voxceleb_resnet34_LM', 25)]

# Инструменты управления распознаванием быстрой речи.
DO_SPEED_SPEECH_CORRECTION = True if int(os.getenv('USE_SPEED_SPEECH_CORRECTION', 1)) == 1 else False  # Включено

# 1 - обычная скорость, меньше - медленнее, больше - быстрее
SPEED_SPEECH_CORRECTION_MULTIPLIER = float(os.getenv('SPEED_SPEECH_CORRECTION_MULTIPLIER', 1))

# Настройки сервиса локального распознавания.
DO_LOCAL_FILE_RECOGNITIONS = True if int(os.getenv('DO_LOCAL_FILE_RECOGNITIONS', 0)) == 1 else False
DELETE_LOCAL_FILE_AFTR_ASR = True if int(os.getenv('DELETE_LOCAL_FILE_AFTR_ASR', 0)) == 1 else False
HUMAN_FORMAT_MD_FILE = True if int(os.getenv('HUMAN_FORMAT_MD_FILE', 0)) == 1 else False


AUDIOEXTENTIONS =  [
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


print(f"Using '{LOGGING_LEVEL}' LOGGING_LEVEL")