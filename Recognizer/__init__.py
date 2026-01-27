import multiprocessing
import numpy as np

from utils.do_logging import logger
from utils import tokens_to_Result
from . import engine
import config
import onnxruntime as ort
import onnx_asr
from onnx_asr.loader import PreprocessorRuntimeConfig, OnnxSessionOptions


TENSORRT_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
CUDA_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU_providers = ["CPUExecutionProvider"]



class Recognizer:
    def __init__(self):
        self.model_name = config.MODEL_NAME
        self._post_processor = tokens_to_Result.process_single_token_vocab_output
        self.preprocessor_providers = list()
        self.encoding_providers = list()
        self.resampler_providers = list()
        self.cpu_preprocessing = False

        match config.PROVIDER:

            case "TENSORRT":
                self.preprocessor_providers = self.encoding_providers = self.resampler_providers = TENSORRT_providers
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None, )
                try:
                    import tensorrt_libs
                except Exception as e:
                    logger.error(
                        f"Ошибка импорта tensorrt. {e}. Функционал TensorrtExecutionProvider будет недоступен.")
                    self.preprocessor_providers = self.encoding_providers = self.resampler_providers = CUDA_providers
                logger.info(f"Using {self.preprocessor_providers[0]} provider")
            case "CUDA":
                self.preprocessor_providers = self.encoding_providers = self.resampler_providers = CUDA_providers
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None, )
                logger.info(f"Using {self.preprocessor_providers[0]} provider")
            case _ :
                self.preprocessor_providers = self.encoding_providers = self.resampler_providers = CPU_providers
                logger.info("Using CPU provider")
                self.cpu_preprocessing = True

        # Некоторые модели не поддерживают TensorrtExecutionProvider или поддерживают его частично. Чистим
        if "vosk" in self.model_name:
            if "TensorrtExecutionProvider" in self.preprocessor_providers:
                 self.preprocessor_providers.remove("TensorrtExecutionProvider")
            if "TensorrtExecutionProvider" in self.resampler_providers:
                 self.resampler_providers.remove("TensorrtExecutionProvider")
            self._post_processor = tokens_to_Result.process_multi_tokens_vocab_output
        elif "t-one" in self.model_name:
            if "TensorrtExecutionProvider" in self.resampler_providers:
                self.resampler_providers.remove("TensorrtExecutionProvider")
            self._post_processor = tokens_to_Result.process_single_token_vocab_output
        elif "giga" in self.model_name:
            if "TensorrtExecutionProvider" in self.resampler_providers:
                self.resampler_providers.remove("TensorrtExecutionProvider")
            self._post_processor = tokens_to_Result.process_single_token_vocab_output
        elif "whisper" in self.model_name:
            if "TensorrtExecutionProvider" in self.encoding_providers:
                self.encoding_providers.remove("TensorrtExecutionProvider")
            self._post_processor = tokens_to_Result.process_multi_tokens_vocab_output
        elif "fastconformer" in self.model_name:
            if "TensorrtExecutionProvider" in self.encoding_providers:
                self.encoding_providers.remove("TensorrtExecutionProvider")
            self._post_processor = tokens_to_Result.process_multi_tokens_vocab_output
        elif "parakeet" in self.model_name:
            if "TensorrtExecutionProvider" in self.resampler_providers:
                self.resampler_providers.remove("TensorrtExecutionProvider")
            self._post_processor = tokens_to_Result.process_single_token_vocab_output

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


        preprocessor_settings = PreprocessorRuntimeConfig()
        preprocessor_settings.update({"providers":CPU_providers if self.cpu_preprocessing else self.preprocessor_providers})
        preprocessor_settings.update({"sess_options":session_options})
        preprocessor_settings.update({"max_concurrent_workers":multiprocessing.cpu_count()})

        resampler_settings = OnnxSessionOptions()
        resampler_settings.update({"providers":CPU_providers if self.cpu_preprocessing else self.resampler_providers})
        resampler_settings.update({"sess_options":session_options})

        self._recognizer = onnx_asr.load_model(model=self.model_name,
                                         providers=self.encoding_providers,
                                         sess_options=session_options,
                                         preprocessor_config=preprocessor_settings,
                                         resampler_config=resampler_settings,
                                         ).with_timestamps()

        try:
            audio = np.random.randn(int(config.MAX_OVERLAP_DURATION * config.BASE_SAMPLE_RATE)).astype(np.float32)
            self._recognizer.recognize([audio])
        except Exception as e:
            logger.error("Ошибка при прогреве модели. Сервис работать не будет. Возможно, модель не поддерживает выбранный провайдер.")
        else:
            logger.info(f"Успешно загружена ASR модель {self.model_name}. ")

    def __getattr__(self, name):

        return getattr(self._recognizer, name)

    def apply_postprocessing(self, *params) -> list:
        """
        Применяет выбранный при инициализации постобработчик к результатам.
        """
        logger.debug(f"Применяется постобработка текста для модели '{self.model_name}'")
        return self._post_processor(*params)

recognizer = Recognizer()
