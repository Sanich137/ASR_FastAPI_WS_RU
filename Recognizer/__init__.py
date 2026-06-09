import multiprocessing
import queue
from typing import Optional

import numpy as np
from pydub import AudioSegment
from utils.pre_start_init import paths


from utils.do_logging import logger
from utils import tokens_to_Result
from . import engine
import config
import onnxruntime as ort
import onnx_asr
from onnx_asr.loader import PreprocessorRuntimeConfig, OnnxSessionOptions
from Recognizer.temporary_folder.icefall_streaming_asr import IcefallStreamingASR



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


class StreamingRecognizer:
    """
    Адаптер потокового распознавания на базе IcefallStreamingASR.
    Принимает AudioSegment (как весь остальной проект) и возвращает текст.
    """

    def __init__(self):
        self._asr = IcefallStreamingASR(
            model_dir=str(paths["streaming_model_dir"]),
            tokens_path=str(paths["streaming_tokens_path"]),
            bpe_model_path=str(paths.get("streaming_bpe_path")),
            sample_rate=16000,
            num_mel_bins=80,
            num_threads=4,
            use_beam=True,
            beam_size=15,
            lm_scale=0.3,
            backoff_id=500,
            length_norm=True,
            unigram_vocab_path=str(paths.get("streaming_unigram_vocab_path")) if paths.get("streaming_unigram_vocab_path") else None,
            ngram_lm_path=str(paths.get("streaming_ngram_lm_path")) if paths.get("streaming_ngram_lm_path") else None,
            hotwords_path=str(paths.get("streaming_hotwords_path")) if paths.get("streaming_hotwords_path") else None,
            hotword_bonus=1.0,
        )

    def reset(self):
        self._asr.reset()

    def push_audiosegment(self, segment: AudioSegment):
        """Принимает AudioSegment (моно, 16 кГц) и отправляет в потоковый ASR."""
        samples = segment.get_array_of_samples()
        self._asr.push_audio_chunk(samples, sample_width=segment.sample_width, channels=segment.channels)

    def get_new_text(self) -> str:
        """Забирает накопленные текстовые чанки."""
        chunks = []
        while True:
            chunk = self._asr.get_transcript_chunk(timeout=0.0)
            if chunk is None:
                break
            chunks.append(chunk)
        return "".join(chunks)

    def flush(self) -> str:
        """Финализирует распознавание и возвращает оставшийся текст."""
        self._asr.flush()
        return self.get_new_text()

    @property
    def full_text(self) -> str:
        return self._asr.text


class StreamingRecognizerPool:
    """
    Пул переиспользуемых экземпляров StreamingRecognizer.
    Позволяет избежать дорогостоящей загрузки ONNX-модели
    при каждом новом WebSocket-соединении.
    """

    def __init__(self, size: int = 5):
        self._pool = queue.Queue(maxsize=size)
        for _ in range(size):
            self._pool.put(StreamingRecognizer())

    def acquire(self) -> StreamingRecognizer:
        try:
            return self._pool.get(block=False)
        except queue.Empty:
            logger.warning("StreamingRecognizer pool exhausted, creating new instance")
            return StreamingRecognizer()

    def release(self, recognizer: Optional[StreamingRecognizer]):
        if recognizer is None:
            return
        recognizer.reset()
        try:
            self._pool.put(recognizer, block=False)
        except queue.Full:
            pass


pool = StreamingRecognizerPool(size=5)
