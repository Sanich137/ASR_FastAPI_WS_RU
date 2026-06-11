import multiprocessing
import queue
import threading
import uuid
from typing import Optional

import numpy as np
from pydub import AudioSegment
from utils.pre_start_init import paths
from VoiceActivityDetector.do_vad import SileroVAD


from utils.do_logging import logger
from utils import tokens_to_Result
from . import engine
import config
import onnxruntime as ort
import onnx_asr
from onnx_asr.loader import PreprocessorRuntimeConfig, OnnxSessionOptions
from Recognizer.engine.icefall_streaming_asr import IcefallStreamingASR



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

    def __init__(self, use_vad: bool = True, use_beam: bool = False,):
        self._use_vad = use_vad
        self._asr = IcefallStreamingASR(
            model_dir=str(paths["streaming_model_dir"]),
            tokens_path=str(paths["streaming_tokens_path"]),
            bpe_model_path=str(paths.get("streaming_bpe_path")),
            sample_rate=16000,
            num_mel_bins=80,
            num_threads=4,
            use_beam=use_beam,
            beam_size=5,
            lm_scale=0.2,
            backoff_id=500,
            length_norm=False,
            hotwords_path=str(paths.get("streaming_hotwords_path")),
            hotword_bonus=0.0,
        )
        self._mode = 'vad' if self._use_vad else 'stream'
        if self._use_vad:
            self._vad = SileroVAD(paths.get("vad_model_path"), use_gpu=config.VAD_WITH_GPU)
            self._vad.set_mode(config.VAD_SENSITIVITY)
            self._vad_samples = np.array([], dtype=np.int16)
            self._pre_roll_buffer = []
            self._in_speech = False
            self._vad_frame_count = 0

    def reset(self):
        self._asr.reset()
        self._mode = 'vad' if self._use_vad else 'stream'
        if self._use_vad:
            self._vad.reset_state_sync()
            self._vad_samples = np.array([], dtype=np.int16)
            self._pre_roll_buffer = []
            self._in_speech = False
            self._vad_frame_count = 0

    def push_audiosegment(self, segment: AudioSegment):
        """Принимает AudioSegment (моно, 16 кГц) и отправляет в потоковый ASR."""
        if not self._use_vad:
            samples = segment.get_array_of_samples()
            self._asr.push_audio_chunk(samples, sample_width=segment.sample_width, channels=segment.channels)
            return
        samples = np.array(segment.get_array_of_samples(), dtype=np.int16)
        if segment.channels > 1:
            samples = samples.reshape((-1, segment.channels)).mean(axis=1).astype(np.int16)
        if self._mode == 'stream':
            self._asr.push_audio_chunk(samples, sample_width=2, channels=1)
            return
        self._process_vad(samples)

    def _process_vad(self, samples: np.ndarray):
        # samples: int16, mono, 16kHz
        if len(self._vad_samples) == 0:
            self._vad_samples = samples
        else:
            self._vad_samples = np.concatenate((self._vad_samples, samples))
        frame_size = self._vad.frame_size

        while len(self._vad_samples) >= frame_size:
            frame = self._vad_samples[:frame_size]
            self._vad_samples = self._vad_samples[frame_size:]

            frame_float = frame.astype(np.float32) / 32768.0
            prob, state = self._vad.is_speech_sync(frame_float, self._vad.sample_rate)
            self._vad.state = state

            if prob > self._vad.prob_level:
                # Начало речи: отправляем pre-roll + текущий фрейм в ASR
                if self._pre_roll_buffer:
                    speech_chunk = np.concatenate(self._pre_roll_buffer + [frame])
                else:
                    speech_chunk = frame
                self._asr.push_audio_chunk(speech_chunk, sample_width=2, channels=1)
                self._mode = 'stream'
                self._asr.mark_stream_resume()
                self._pre_roll_buffer = []
                # Отправляем остаток уже накопленных сэмплов напрямую в ASR
                if len(self._vad_samples) > 0:
                    self._asr.push_audio_chunk(self._vad_samples, sample_width=2, channels=1)
                    self._vad_samples = np.array([], dtype=np.int16)
                return
            else:
                # Тишина: накапливаем pre-roll (макс ~1 сек = 32 фрейма)
                self._pre_roll_buffer.append(frame)
                if len(self._pre_roll_buffer) > 32:
                    self._pre_roll_buffer.pop(0)
            self._vad_frame_count += 1

    def get_new_text(self) -> dict:
        """Забирает накопленные текстовые чанки."""
        words = []
        texts = []
        while True:
            chunk = self._asr.get_transcript_chunk(timeout=0.0)
            if chunk is None:
                break
            if isinstance(chunk, dict):
                words.append({
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "word": chunk["text"]
                })
                texts.append(chunk["text"])
            else:
                texts.append(str(chunk))
        # Endpointing: если ASR сигнализирует о конце речи, возвращаемся в VAD
        if self._use_vad and self._mode == 'stream' and self._asr.endpoint_triggered:
            logger.info(
                f"StreamingRecognizer endpoint triggered, "
                f"switching to vad (pending={len(self._asr._pending_tokens)})"
            )
            self._mode = 'vad'
            self._asr.endpoint_triggered = False
            self._pre_roll_buffer = []
            self._vad_samples = np.array([], dtype=np.int16)
        return {
            "text": " ".join(texts),
            "words": words
        }

    def flush(self) -> dict:
        """Финализирует распознавание и возвращает оставшийся текст."""
        logger.info(
            f"StreamingRecognizer flush: mode={self._mode} "
            f"pending={len(self._asr._pending_tokens)} "
            f"fbank_ready={self._asr.fbank.num_frames_ready()} "
            f"processed={self._asr.processed_frames}"
        )
        self._asr.flush()
        result = self.get_new_text()
        logger.info(
            f"StreamingRecognizer flush result: "
            f"text_len={len(result.get('text', ''))} "
            f"words={len(result.get('words', []))}"
        )
        return result

    @property
    def full_text(self) -> str:
        return self._asr.text


class ClientSessionHandle:
    def __init__(self, worker, input_queue, output_queue, recognizer):
        self.worker = worker
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.recognizer = recognizer


class StreamingClientWorker(threading.Thread):
    def __init__(self, recognizer, input_queue, output_queue, client_id):
        super().__init__(daemon=True)
        self.recognizer = recognizer
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.client_id = client_id
        self._stop_event = threading.Event()

    def run(self):
        logger.info(f"Worker {self.client_id} started")
        while not self._stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            msg_type = item.get("type")

            if msg_type == "chunk":
                payload = item.get("payload")
                if payload is not None:
                    self.recognizer.push_audiosegment(payload)
                    result = self.recognizer.get_new_text()
                    if result and result.get("text"):
                        self.output_queue.put({"type": "partial", "data": result})

            elif msg_type == "eof":
                logger.info(
                    f"Worker {self.client_id} received EOF, "
                    f"input_queue_remaining={self.input_queue.qsize()}"
                )
                result = self.recognizer.flush()
                logger.info(
                    f"Worker {self.client_id} flush done: "
                    f"text_len={len(result.get('text', ''))} "
                    f"words={len(result.get('words', []))}"
                )
                self.output_queue.put({"type": "final", "data": result})
                self.output_queue.put({"type": "done"})
                logger.info(f"Worker {self.client_id} sent final+done, breaking")
                break

            elif msg_type == "disconnect":
                self.recognizer.reset()
                self.output_queue.put({"type": "done"})
                break

        logger.info(f"Worker {self.client_id} finished")

    def stop(self):
        self._stop_event.set()


class StreamingRecognizerPool:
    """
    Пул переиспользуемых экземпляров StreamingRecognizer.
    Позволяет избежать дорогостоящей загрузки ONNX-модели
    при каждом новом WebSocket-соединении.
    """

    def __init__(self, size: int = 5):
        self._pool = queue.Queue(maxsize=size)
        for _ in range(size):
            self._pool.put(StreamingRecognizer(use_vad=True, use_beam=True))

    def acquire(self) -> ClientSessionHandle:
        try:
            recognizer = self._pool.get(block=False)
        except queue.Empty:
            logger.warning("StreamingRecognizer pool exhausted, creating new instance")
            recognizer = StreamingRecognizer(use_vad=True, use_beam=True)

        input_q = queue.Queue(maxsize=50)
        output_q = queue.Queue(maxsize=100)
        client_id = str(uuid.uuid4())
        worker = StreamingClientWorker(recognizer, input_q, output_q, client_id)
        worker.start()
        return ClientSessionHandle(worker, input_q, output_q, recognizer)

    def release(self, handle: Optional[ClientSessionHandle], send_disconnect: bool = True):
        if handle is None:
            return
        if send_disconnect:
            try:
                handle.input_queue.put({"type": "disconnect"}, block=False)
            except queue.Full:
                pass
        handle.worker.join(timeout=2.0)
        if handle.worker.is_alive():
            if not send_disconnect:
                # При EOF worker должен был завершиться сам, но не успел — шлём disconnect
                try:
                    handle.input_queue.put({"type": "disconnect"}, block=False)
                except queue.Full:
                    pass
            handle.worker.stop()
            handle.worker.join(timeout=1.0)
        handle.recognizer.reset()
        try:
            self._pool.put(handle.recognizer, block=False)
        except queue.Full:
            pass


pool = StreamingRecognizerPool(size=2)
