# -*- coding: utf-8 -*-
"""
Ленивый синглтон потокового движка T-one (tone.StreamingCTCPipeline).

Грузится один раз при первом обращении (а не при импорте), чтобы не тянуть модель,
пока потоковый эндпоинт не используется, и не вмешиваться в загрузку офлайн-модели
GigaAM. Модель и KenLM качаются с HF (t-tech/T-one) в каталог HF_HOME.

Провайдер акустической модели:
  - по умолчанию CPU (так задумано библиотекой; для чанков по 300 мс это обычно оптимально);
  - при settings.STREAM_WITH_GPU=1 и GPU-провайдере собираем pipeline вручную с
    CUDAExecutionProvider (даёт возможность померить, выгоден ли GPU на коротких чанках).
"""

import logging

from config import settings

logger = logging.getLogger(__name__)

_pipeline = None

CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _build_gpu_pipeline():
    """Собирает StreamingCTCPipeline с CUDA-сессией акустической модели."""
    import onnxruntime as ort
    from tone import StreamingCTCPipeline, DecoderType
    from tone.onnx_wrapper import StreamingCTCModel
    from tone.logprob_splitter import StreamingLogprobSplitter
    from tone.decoder import GreedyCTCDecoder, BeamSearchCTCDecoder

    model_path = StreamingCTCModel.download_from_hugging_face()
    sess = ort.InferenceSession(model_path, providers=CUDA_PROVIDERS)
    model = StreamingCTCModel(sess)
    splitter = StreamingLogprobSplitter()
    if str(settings.TONE_DECODER).lower() == "greedy":
        decoder = GreedyCTCDecoder()
    else:
        decoder = BeamSearchCTCDecoder.from_hugging_face()
    logger.info("Использован провайдер %s для T-one", sess.get_providers()[0])
    return StreamingCTCPipeline(model, splitter, decoder)


def get_tone_pipeline():
    """Возвращает singleton StreamingCTCPipeline, загружая его при первом вызове."""
    global _pipeline
    if _pipeline is None:
        from tone import StreamingCTCPipeline, DecoderType

        decoder_type = (DecoderType.GREEDY
                        if str(settings.TONE_DECODER).lower() == "greedy"
                        else DecoderType.BEAM_SEARCH)
        logger.info("Загрузка потоковой модели T-one (decoder=%s, gpu=%s)...",
                    decoder_type.value, settings.STREAM_WITH_GPU)
        if settings.STREAM_WITH_GPU:
            try:
                _pipeline = _build_gpu_pipeline()
            except Exception as exc:
                logger.error("Не удалось поднять T-one на GPU (%s), откат на CPU", exc)
                _pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=decoder_type)
        else:
            _pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=decoder_type)
        logger.info("Потоковая модель T-one загружена.")
    return _pipeline
