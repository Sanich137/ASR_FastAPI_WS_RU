# -*- coding: utf-8 -*-
from utils.do_logging import logger
from utils.files_whatcher import start_file_watcher
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
import config
import threading
import gc


BASE_DIR = Path(__file__).resolve().parent.parent
paths = {
    "vosk_small_streaming_tokens_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "lang" / "tokens.txt",
    "vosk_small_streaming_encoder_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "am" / "encoder.onnx",
    "vosk_small_streaming_decoder_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "am" / "decoder.onnx",
    "vosk_small_streaming_joiner_path": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "am" / "joiner.onnx",
    "vosk_small_streaming_bpe_vocab": BASE_DIR / "models" / "vosk-model-small-streaming-ru" / "lang" / "bpe.model",

    "vosk_full_tokens_path": BASE_DIR / "models" / "vosk-model-ru" / "lang" / "tokens.txt",
    "vosk_full_encoder_path": BASE_DIR / "models" / "vosk-model-ru" / "am-onnx" / "encoder.onnx",
    "vosk_full_decoder_path": BASE_DIR / "models" / "vosk-model-ru" / "am-onnx" / "decoder.onnx",
    "vosk_full_joiner_path": BASE_DIR / "models" / "vosk-model-ru" / "am-onnx" / "joiner.onnx",
    "vosk_full_bpe_vocab": BASE_DIR / "models" / "vosk-model-ru" / "lang" / "bpe.model",

    "gigaam_tokens_path": BASE_DIR / "models" / "GigaAMv2_CTC_RU_ASR_for_sherpa_onnx" / "tokens.txt",
    "gigaam_encoder_path": BASE_DIR / "models" / "GigaAMv2_CTC_RU_ASR_for_sherpa_onnx" / "GigaAMv2_ctc_public.onnx",

    "gigaam_rnnt_encoder_path": BASE_DIR / "models" / "GigaAMv2_RNNT_RU_ASR_for_sherpa_onnx" / "encoder.onnx",
    "gigaam_rnnt_decoder_path": BASE_DIR / "models" / "GigaAMv2_RNNT_RU_ASR_for_sherpa_onnx" / "decoder.onnx",
    "gigaam_rnnt_joiner_path": BASE_DIR / "models" / "GigaAMv2_RNNT_RU_ASR_for_sherpa_onnx" / "joiner.onnx",
    "gigaam_rnnt_tokens_path": BASE_DIR / "models" / "GigaAMv2_RNNT_RU_ASR_for_sherpa_onnx" / "tokens.txt",

    # "whisper_tokens_path": BASE_DIR / "models" / "sherpa-onnx-whisper-medium" / "medium-tokens.txt",
    # "whisper_encoder_path": BASE_DIR / "models" / "sherpa-onnx-whisper-medium" / "medium-encoder.onnx",
    # "whisper_decoder_path": BASE_DIR / "models" / "sherpa-onnx-whisper-medium" / "medium-decoder.onnx",
    # "whisper_encoder_path": BASE_DIR / "models" / "sherpa-onnx-whisper-small" / "small-encoder.int8.onnx",
    # "whisper_decoder_path": BASE_DIR / "models" / "sherpa-onnx-whisper-small" / "small-decoder.int8.onnx",
    # Todo - говорят производительность Whisper пофиксили. Протестить!

    "punctuation_model_path": BASE_DIR / "models" / "sbert_punc_case_ru_onnx",
    "vad_model_path": BASE_DIR / "models" / "VAD_silero_v5" / "silero_vad.onnx",
    "diar_speaker_model_path": BASE_DIR / "models" / "DIARISATION_model" / f"{config.DIAR_MODEL_NAME}",
    "streaming_model_dir": BASE_DIR / "models" / "Vosk_054",
    "streaming_tokens_path": BASE_DIR / "models" / "Vosk_054" / "tokens.txt",
    "streaming_bpe_path": BASE_DIR / "models" / "Vosk_054" / "bpe.model",
    "streaming_unigram_vocab_path": BASE_DIR / "models" / "Vosk_054" / "unigram_500.vocab",
    "streaming_hotwords_path": BASE_DIR / "models" / "Vosk_054" / "hotwords.txt",
    "streaming_ngram_lm_path": BASE_DIR / "models" / "Vosk_054" / "2gram.fst.txt",

    "BASE_DIR": BASE_DIR,
    "test_file": BASE_DIR /'trash'/'111.wav',
    "trash_folder": BASE_DIR / 'trash',
    "local_recognition_folder": BASE_DIR / 'local_asr' / 'to_asr',
    "result_local_recognition_folder": BASE_DIR / 'local_asr' / 'after_asr',
}

from collections import defaultdict

# Глобальные переменные
audio_overlap = defaultdict()
audio_buffer = defaultdict()
audio_to_asr = defaultdict()
audio_duration = defaultdict(float)
ws_collected_asr_res = defaultdict()
posted_and_downloaded_audio = defaultdict()

# Устанавливаем новые пороги сборщика мусора
gc.set_threshold(500,  # быстрые файлы было 700
                 5,    # средне выживающие файлы было 10
                 5)    # долгожители было 10.


@asynccontextmanager
async def lifespan(app: FastAPI):
    # on_start
    logger.debug("Приложение FastAPI запущено")
    global observer
    if config.DO_LOCAL_FILE_RECOGNITIONS:
        observer_thread = threading.Thread(
            target=lambda: start_file_watcher(file_path=str(paths.get("local_recognition_folder"))),
            daemon=True
        )
        observer_thread.start()
        logger.info("File watcher started")

    yield  # Здесь приложение работает

app = FastAPI(lifespan=lifespan,
              version="1.0",
              docs_url='/docs',
              root_path='/root',
              title='ASR on SHERPA-ONNX'
              )
