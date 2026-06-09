#!/usr/bin/env python3
"""
Streaming ASR wrapper for Icefall ONNX transducer models.
"""

import logging
import queue
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from Recognizer.temporary_folder.icefall_onnx import IcefallOnnxASR

log = logging.getLogger(__name__)

try:
    import kaldi_native_fbank as knf

    HAS_KNF = True
except ImportError:
    HAS_KNF = False
    knf = None  # type: ignore


class FbankExtractor:
    def __init__(self, sample_rate: int, num_mel_bins: int = 80) -> None:
        if not HAS_KNF:
            raise ImportError(
                "kaldi-native-fbank is required. Install it with:\n"
                "  pip install kaldi-native-fbank"
            )
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 3e-5
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = sample_rate
        opts.mel_opts.num_bins = num_mel_bins
        self.fbank = knf.OnlineFbank(opts)
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins

    def accept_waveform(self, waveform: np.ndarray) -> None:
        self.fbank.accept_waveform(self.sample_rate, waveform.tolist())

    def num_frames_ready(self) -> int:
        return self.fbank.num_frames_ready

    def get_frames(self, start: int, end: int) -> np.ndarray:
        frames = []
        for i in range(start, end):
            frames.append(self.fbank.get_frame(i))
        return np.stack(frames, axis=0).astype(np.float32)

    def input_finished(self) -> None:
        self.fbank.input_finished()


class IcefallStreamingASR:
    """
    High-level streaming ASR interface for Icefall ONNX models.
    """

    def __init__(
        self,
        model_dir: str,
        tokens_path: str,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        num_threads: int = 4,
        encoder_name: str = "encoder.chunk64.onnx",
        decoder_name: str = "decoder.chunk64.onnx",
        joiner_name: str = "joiner.chunk64.onnx",
        bpe_model_path: Optional[str] = None,
        use_beam: bool = False,
        beam_size: int = 5,
        ngram_lm_path: Optional[str] = None,
        lm_scale: float = 0.0,
        backoff_id: int = 0,
        length_norm: bool = False,
        unigram_vocab_path: Optional[str] = None,
        frame_shift_s: float = 0.01,
        hotwords_path: Optional[str] = None,
        hotword_bonus: float = 0.0,
    ) -> None:
        self.model = IcefallOnnxASR(
            encoder_path=str(Path(model_dir) / encoder_name),
            decoder_path=str(Path(model_dir) / decoder_name),
            joiner_path=str(Path(model_dir) / joiner_name),
            tokens_path=tokens_path,
            num_threads=num_threads,
            bpe_model_path=bpe_model_path,
            beam_size=beam_size,
            use_beam=use_beam,
            ngram_lm_path=ngram_lm_path,
            lm_scale=lm_scale,
            backoff_id=backoff_id,
            length_norm=length_norm,
            unigram_vocab_path=unigram_vocab_path,
            frame_shift_s=frame_shift_s,
            hotwords_path=hotwords_path,
            hotword_bonus=hotword_bonus,
        )
        self.sample_rate = sample_rate
        self.fbank = FbankExtractor(sample_rate, num_mel_bins)
        # chunk_size is taken from the ONNX model graph if available
        self.chunk_size = (
            self.model.expected_chunk_size
            if self.model.expected_chunk_size is not None
            else 64
        )
        if self.model.expected_chunk_size is not None:
            log.info(f"Using chunk_size={self.chunk_size} from ONNX model")
        self.processed_frames = 0
        self.output_queue: queue.Queue[str] = queue.Queue()
        self._text_so_far = ""
        self._token_details: List[Dict[str, Any]] = []
        self._pending_tokens: List[Dict[str, Any]] = []
        self._last_token_real_time: float = 0.0

    def reset(self) -> None:
        self.model.reset()
        self.fbank = FbankExtractor(self.sample_rate, self.fbank.num_mel_bins)
        self.processed_frames = 0
        self._text_so_far = ""
        self._token_details: List[Dict[str, Any]] = []
        self._pending_tokens = []
        self._last_token_real_time = 0.0
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

    def push_audio_chunk(self, audio_chunk, sample_width: int = 2, channels: int = 1) -> None:
        samples = np.array(audio_chunk)
        if sample_width == 2:
            samples = samples.astype(np.float32) / 32768.0
        elif sample_width == 4:
            samples = samples.astype(np.float32) / 2147483648.0
        else:
            samples = samples.astype(np.float32)

        if channels > 1:
            samples = samples.reshape((-1, channels)).mean(axis=1)

        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        waveform = samples.flatten().astype(np.float32)
        self.fbank.accept_waveform(waveform)
        self._process_ready_frames()

    def _process_ready_frames(self) -> None:
        ready = self.fbank.num_frames_ready()
        while ready - self.processed_frames >= self.chunk_size:
            start = self.processed_frames
            end = start + self.chunk_size
            features = self.fbank.get_frames(start, end)  # (chunk_size, num_mel_bins)
            features = features[np.newaxis, ...]  # (1, chunk_size, num_mel_bins)
            encoder_out = self.model._run_encoder(features)
            token_results = self.model.decode_encoder_out(encoder_out, frame_offset=start)
            if token_results:
                self._pending_tokens.extend(token_results)
                self._last_token_real_time = time.time()
                self._commit_pending()
            self.processed_frames = end
            ready = self.fbank.num_frames_ready()

    def flush(self) -> None:
        """Process any remaining frames after the audio stream ends."""
        self.fbank.input_finished()
        ready = self.fbank.num_frames_ready()

        # Обрабатываем оставшиеся полные чанки
        while ready - self.processed_frames >= self.chunk_size:
            start = self.processed_frames
            end = start + self.chunk_size
            features = self.fbank.get_frames(start, end)
            features = features[np.newaxis, ...]
            encoder_out = self.model._run_encoder(features)
            token_results = self.model.decode_encoder_out(encoder_out, frame_offset=start)
            if token_results:
                self._pending_tokens.extend(token_results)
                self._last_token_real_time = time.time()
                self._commit_pending()
            self.processed_frames = end
            ready = self.fbank.num_frames_ready()

        # Обрабатываем хвост с паддингом до chunk_size
        if ready > self.processed_frames:
            start = self.processed_frames
            end = ready
            features = self.fbank.get_frames(start, end)
            if features.shape[0] < self.chunk_size:
                pad = np.zeros(
                    (self.chunk_size - features.shape[0], features.shape[1]),
                    dtype=np.float32,
                )
                features = np.concatenate([features, pad], axis=0)
            features = features[np.newaxis, ...]
            encoder_out = self.model._run_encoder(features)
            token_results = self.model.decode_encoder_out(encoder_out, frame_offset=start)
            if token_results:
                self._pending_tokens.extend(token_results)
                self._last_token_real_time = time.time()
            self._flush_pending()
            self.processed_frames = end

    def _commit_pending(self) -> None:
        """Emit complete words from _pending_tokens to the output queue."""
        if not self._pending_tokens:
            return

        # Find the last token that starts a new word (SentencePiece ▁ prefix)
        last_word_start = -1
        for i in range(len(self._pending_tokens) - 1, -1, -1):
            if self._pending_tokens[i]["text"].startswith("▁"):
                last_word_start = i
                break

        if last_word_start > 0:
            completed = self._pending_tokens[:last_word_start]
            self._pending_tokens = self._pending_tokens[last_word_start:]

            text = self.model.decode_token_ids([r["token"] for r in completed])
            if text:
                self.output_queue.put_nowait(text)
                self._text_so_far += text
                self._token_details.extend(completed)

    def _flush_pending(self) -> None:
        """Forcefully emit all pending tokens (used on timeout or flush)."""
        if not self._pending_tokens:
            return
        completed = self._pending_tokens
        self._pending_tokens = []
        text = self.model.decode_token_ids([r["token"] for r in completed])
        if text:
            self.output_queue.put_nowait(text)
            self._text_so_far += text
            self._token_details.extend(completed)

    def get_transcript_chunk(self, timeout: float = 0.1) -> Optional[str]:
        if self._pending_tokens and (time.time() - self._last_token_real_time) > 0.5:
            self._flush_pending()
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def text(self) -> str:
        return self._text_so_far

    @property
    def token_details(self) -> Dict[str, List[Any]]:
        """Return timestamps, tokens and logprobs in the requested format."""
        return {
            "timestamps": [r["timestamp"] for r in self._token_details],
            "tokens": [r["text"] for r in self._token_details],
            "logprobs": [r["logprob"] for r in self._token_details],
        }
