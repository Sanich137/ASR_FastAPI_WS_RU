import multiprocessing
import os
from . import engine
from utils.pre_start_init import paths
from utils.do_logging import logger
import config
import onnx_asr
import onnxruntime as ort

from onnx_asr.loader import PreprocessorRuntimeConfig

logger.info(f"{config.MODEL_NAME} chosen model.")

recognizer = None

session_options = ort.SessionOptions()
session_options.log_severity_level = 4  # Выключаем подробный лог
session_options.enable_profiling = False
session_options.enable_mem_pattern = False  # True в диаризации
session_options.enable_mem_reuse = False  # True в диаризации
session_options.enable_cpu_mem_arena = False  # True в диаризации
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.inter_op_num_threads = 0
session_options.intra_op_num_threads = 0
session_options.add_session_config_entry("session.disable_prepacking", "1")  # Отключаем дублирование весов
session_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if config.PROVIDER=="CUDA" else ["CPUExecutionProvider"]


preprocessor_settings = PreprocessorRuntimeConfig()
preprocessor_settings.update({"providers":providers})
preprocessor_settings.update({"sess_options":session_options})
preprocessor_settings.update({"max_concurrent_workers":multiprocessing.cpu_count()})


recognizer = onnx_asr.load_model(model=config.MODEL_NAME,
                            providers=providers,
                            sess_options=session_options,
                            cpu_preprocessing=False,
                            preprocessor_config=preprocessor_settings,
                            ).with_timestamps()



logger.debug(f"Model {config.MODEL_NAME} ready to start!")
