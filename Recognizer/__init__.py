import multiprocessing
import numpy as np
from sympy.physics.units import giga

from utils.do_logging import logger
from . import engine
import config
try:
    import tensorrt
except Exception as e:
    logger.error(f"Попытка импорта tensorrt завершилась ошибкой. {e}. Функционал TensorrtExecutionProvider может быть недоступен.")
import onnx_asr
import onnxruntime as ort
from onnx_asr.loader import PreprocessorRuntimeConfig, OnnxSessionOptions

recognizer = None
session_options = ort.SessionOptions()
session_options.log_severity_level = 4  # Выключаем подробный лог
session_options.enable_profiling = False
session_options.enable_mem_pattern = True  # True в диаризации
session_options.enable_mem_reuse = True  # True в диаризации
session_options.enable_cpu_mem_arena = True  # True в диаризации
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.inter_op_num_threads = 0
session_options.intra_op_num_threads = 0
session_options.add_session_config_entry("session.disable_prepacking", "1")  # Отключаем дублирование весов
session_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


TENSORRT_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
CUDA_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU_providers = ["CPUExecutionProvider"]


match config.PROVIDER:
    case "TENSORRT":
        preprocessor_providers = encoding_providers = resampler_providers = TENSORRT_providers
        logger.info("Using TENSORRT provider")
        cpu_preprocessing = False

    case "CUDA":
        preprocessor_providers = encoding_providers = resampler_providers = CUDA_providers
        logger.info("Using CUDA provider")
        cpu_preprocessing = False
    case _ :
        preprocessor_providers = encoding_providers = resampler_providers = CPU_providers
        logger.info("Using CPU provider")
        cpu_preprocessing = True

# Некоторые модели не поддерживают TensorrtExecutionProvider или поддерживают его частично. Чистим
if "vosk" in config.MODEL_NAME:
    try:
        preprocessor_providers.remove("TensorrtExecutionProvider")
        resampler_providers.remove("TensorrtExecutionProvider")
    except ValueError:
        pass
elif "giga" in config.MODEL_NAME:
    try:
        resampler_providers.remove("TensorrtExecutionProvider")
    except ValueError:
        pass
elif "whisper" in config.MODEL_NAME:
    try:
        encoding_providers.remove("TensorrtExecutionProvider")
    except ValueError:
        pass
elif "fastconformer" in config.MODEL_NAME:
    try:
        encoding_providers.remove("TensorrtExecutionProvider")
    except ValueError:
        pass



preprocessor_settings = PreprocessorRuntimeConfig()
preprocessor_settings.update({"providers":CPU_providers if cpu_preprocessing else preprocessor_providers})
preprocessor_settings.update({"sess_options":session_options})
preprocessor_settings.update({"max_concurrent_workers":multiprocessing.cpu_count()})

resampler_settings = OnnxSessionOptions()
resampler_settings.update({"providers":CPU_providers if cpu_preprocessing else resampler_providers})
resampler_settings.update({"sess_options":session_options})

recognizer = onnx_asr.load_model(model=config.MODEL_NAME,
                                 providers=encoding_providers,
                                 sess_options=session_options,
                                 cpu_preprocessing=cpu_preprocessing,
                                 preprocessor_config=preprocessor_settings,
                                 resampler_config=resampler_settings
                                 ).with_timestamps()

try:
    audio = np.random.randn(int(config.MAX_OVERLAP_DURATION * config.BASE_SAMPLE_RATE)).astype(np.float32)
    recognizer.recognize([audio])
except Exception as e:
    logger.error("Ошибка при прогреве модели. Сервис работать не будет. Возможно, модель не поддерживает выбранный провайдер.")
else:
    logger.info(f"Успешно загружена ASR модель {config.MODEL_NAME}.")
