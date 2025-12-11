import multiprocessing
import numpy as np
from utils.do_logging import logger
from . import engine
import config

try:
    import tensorrt
except Exception as e:
    logger.error(f"Попытка импорта tensorrt завершилась ошибкой. {e}. Функционал TensorrtExecutionProvider может быть недоступен.")

import onnx_asr
import onnxruntime as ort

from onnx_asr.loader import PreprocessorRuntimeConfig

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


match config.PROVIDER:
    case "CUDA":
        preprocessor_providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
        encoding_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("Using CUDA provider")
        cpu_preprocessing = False
    case "TENSORRT":
        preprocessor_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        encoding_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        cpu_preprocessing = False
        print("Using TENSORRT provider")
    case True :
        encoding_providers = ["CPUExecutionProvider"]
        print("Using CPU provider")
        cpu_preprocessing = True

preprocessor_settings = PreprocessorRuntimeConfig()
preprocessor_settings.update({"providers":preprocessor_providers})
preprocessor_settings.update({"sess_options":session_options})
preprocessor_settings.update({"max_concurrent_workers":multiprocessing.cpu_count()})


recognizer = onnx_asr.load_model(model=config.MODEL_NAME,
                                 providers=encoding_providers,
                                 sess_options=session_options,
                                 cpu_preprocessing=cpu_preprocessing,
                                 preprocessor_config=preprocessor_settings,
                                 ).with_timestamps()

audio = np.random.randn(int(config.MAX_OVERLAP_DURATION * config.BASE_SAMPLE_RATE)).astype(np.float32)
recognizer.recognize([audio])
logger.info(f"Прогрета и будет использована модель {config.MODEL_NAME} прогрета.")
