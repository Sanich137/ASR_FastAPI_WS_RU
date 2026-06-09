#!/usr/bin/env python3
"""
ONNX Runtime inference for Icefall Zipformer2 transducer models.
"""

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

try:
    import sentencepiece as spm

    HAS_SPM = True
except ImportError:
    HAS_SPM = False
    spm = None  # type: ignore

log = logging.getLogger(__name__)


class NgramFstLM:
    """
    Minimal bigram LM loader from an OpenFST text file (e.g. 2gram.fst.txt).
    Assumes the FST is an acceptor (ilabel == olabel) and uses tropical weights.
    """

    def __init__(self, fst_path: str, backoff_id: int = 0) -> None:
        self.backoff_id = backoff_id
        self.unigrams: Dict[int, float] = {}
        self.bigrams: Dict[Tuple[int, int], float] = {}
        # state -> last token (for states reachable from start with 1 token)
        self._state_to_token: Dict[int, int] = {}

        with open(fst_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 5:
                    src, dst, ilabel, olabel, weight = parts
                    src = int(src)
                    dst = int(dst)
                    ilabel = int(ilabel)
                    weight = float(weight)
                    if ilabel == self.backoff_id:
                        # backoff arc – ignore in this minimal version
                        continue
                    if src == 0:
                        # unigram arc from start state
                        self.unigrams[ilabel] = -weight
                        self._state_to_token[dst] = ilabel
                    elif src in self._state_to_token:
                        prev_token = self._state_to_token[src]
                        self.bigrams[(prev_token, ilabel)] = -weight
                        self._state_to_token[dst] = ilabel
                elif len(parts) == 2:
                    # final state – ignore for scoring
                    pass

    def score(self, prev_token: int, token: int) -> float:
        """Return log-prob (natural log) of token given prev_token."""
        if (prev_token, token) in self.bigrams:
            return self.bigrams[(prev_token, token)]
        if token in self.unigrams:
            return self.unigrams[token]
        return -99.0  # very unlikely


class UnigramVocabLM:
    """
    Minimal unigram LM loader from a SentencePiece .vocab file.
    Each line: <token>\\t<logprob>
    """

    def __init__(self, vocab_path: str) -> None:
        self.scores: Dict[str, float] = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    token = parts[0]
                    try:
                        score = float(parts[1])
                        self.scores[token] = score
                    except ValueError:
                        continue

    def score(self, token: str) -> float:
        """Return log-prob of token (negative for rare tokens, near 0 for frequent)."""
        return self.scores.get(token, -10.0)


class IcefallOnnxASR:
    """
    Low-level ONNX encoder/decoder/joiner runner for Icefall transducer models.
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        joiner_path: str,
        tokens_path: str,
        num_threads: int = 4,
        bpe_model_path: Optional[str] = None,
        beam_size: int = 5,
        use_beam: bool = False,
        ngram_lm_path: Optional[str] = None,
        lm_scale: float = 0.0,
        backoff_id: int = 0,
        length_norm: bool = False,
        unigram_vocab_path: Optional[str] = None,
        frame_shift_s: float = 0.01,
        hotwords_path: Optional[str] = None,
        hotword_bonus: float = 0.0,
    ) -> None:
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = num_threads
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(encoder_path, sess_opts, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, sess_opts, providers=providers)
        self.joiner = ort.InferenceSession(joiner_path, sess_opts, providers=providers)

        # Tokens
        self.tokens: Dict[int, str] = {}
        with open(tokens_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        idx = int(parts[-1])
                        token = parts[0]
                        self.tokens[idx] = token
                    except ValueError:
                        continue
        # Auto-detect blank id from tokens.txt
        self.blank_id = 0
        for idx, tok in self.tokens.items():
            if tok in ("<blk>", "<blank>", "<eps>"):
                self.blank_id = idx
                log.info("Detected blank_id=%d for token %r", idx, tok)
                break
        else:
            log.warning("Could not detect blank token in %s; assuming blank_id=0", tokens_path)

        self.vocab_size = max(self.tokens.keys()) + 1 if self.tokens else 500
        # subsampling_factor will be inferred once from the actual ONNX encoder output
        self.subsampling_factor: Optional[float] = None

        # Optional SentencePiece BPE decoder
        self.bpe_processor: Optional[Any] = None
        if bpe_model_path and os.path.isfile(bpe_model_path) and HAS_SPM:
            self.bpe_processor = spm.SentencePieceProcessor()
            self.bpe_processor.Load(bpe_model_path)
            log.info("Loaded SentencePiece model from %s", bpe_model_path)

            # Consistency check between BPE model and tokens.txt
            if self.tokens:
                mismatches = 0
                checked = 0
                for idx, tok in self.tokens.items():
                    if idx < 0 or idx >= self.bpe_processor.GetPieceSize():
                        continue
                    sp_tok = self.bpe_processor.IdToPiece(idx)
                    if sp_tok != tok:
                        mismatches += 1
                        if mismatches <= 3:
                            log.warning(
                                "BPE/token mismatch at id %d: tokens.txt=%r, bpe.model=%r",
                                idx, tok, sp_tok,
                            )
                    checked += 1
                if checked > 0:
                    if mismatches / checked > 0.1:
                        log.error(
                            "BPE model and tokens.txt have %.1f%% mismatches. "
                            "Decoding will likely produce garbage.",
                            100.0 * mismatches / checked,
                        )
                    elif mismatches > 0:
                        log.warning(
                            "BPE model and tokens.txt have %d/%d mismatches.",
                            mismatches, checked,
                        )
                    else:
                        log.info(
                            "BPE model and tokens.txt are consistent (%d tokens checked).",
                            checked,
                        )
        elif bpe_model_path:
            log.warning("BPE model not found or sentencepiece not installed: %s", bpe_model_path)

        # Introspection
        self._enc_in_names = [inp.name for inp in self.encoder.get_inputs()]
        self._enc_out_names = [out.name for out in self.encoder.get_outputs()]
        self._dec_in_names = [inp.name for inp in self.decoder.get_inputs()]
        self._dec_out_names = [out.name for out in self.decoder.get_outputs()]
        self._join_in_names = [inp.name for inp in self.joiner.get_inputs()]
        self._join_out_names = [out.name for out in self.joiner.get_outputs()]

        log.debug("Encoder inputs: %s", self._enc_in_names)
        log.debug("Encoder outputs: %s", self._enc_out_names)
        log.debug("Decoder inputs: %s", self._dec_in_names)
        log.debug("Joiner inputs: %s", self._join_in_names)

        # Feature dim and expected time dimension from the ONNX graph
        self.feature_dim = 80
        self.expected_chunk_size = None
        for inp in self.encoder.get_inputs():
            if inp.name == "x" and len(inp.shape) == 3:
                if inp.shape[2] is not None:
                    self.feature_dim = inp.shape[2]
                if inp.shape[1] is not None:
                    self.expected_chunk_size = int(inp.shape[1])
                break

        # Decoder context size (e.g., 2 for bigram history)
        # From checkpoint epoch-32-avg-2.pt: context_size=2
        self.decoder_context_size = 2
        for inp in self.decoder.get_inputs():
            if inp.name == "y" and len(inp.shape) == 2:
                if isinstance(inp.shape[1], int):
                    self.decoder_context_size = inp.shape[1]
                break

        self.beams: Optional[List[Dict[str, Any]]] = None
        self._beam_prev_len = 0
        self.reset()
        self.max_symbols_per_step = 5
        self.beam_size = beam_size
        self.use_beam = use_beam
        self.length_norm = length_norm
        self.lm_scale = lm_scale
        self.frame_shift_s = frame_shift_s
        self.ngram_lm: Optional[NgramFstLM] = None
        if ngram_lm_path and os.path.isfile(ngram_lm_path):
            self.ngram_lm = NgramFstLM(ngram_lm_path, backoff_id=backoff_id)
            log.info("Loaded n-gram LM from %s", ngram_lm_path)
        else:
            if ngram_lm_path:
                log.warning("n-gram LM file not found: %s", ngram_lm_path)

        self.unigram_lm: Optional[UnigramVocabLM] = None
        if unigram_vocab_path and os.path.isfile(unigram_vocab_path):
            self.unigram_lm = UnigramVocabLM(unigram_vocab_path)
            log.info("Loaded unigram vocab LM from %s", unigram_vocab_path)
        else:
            if unigram_vocab_path:
                log.warning("Unigram vocab file not found: %s", unigram_vocab_path)

        self.hotword_bonus = hotword_bonus
        self.hotword_tuples: List[Tuple[int, ...]] = []
        if hotwords_path and os.path.isfile(hotwords_path):
            if self.bpe_processor is not None:
                with open(hotwords_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Also split by spaces to add individual words as hotwords
                        words = line.split()
                        for word in words:
                            token_ids = self.bpe_processor.EncodeAsIds(word)
                            if token_ids:
                                self.hotword_tuples.append(tuple(token_ids))
                                log.info(
                                    "Hotword %r -> token_ids=%s tokens=%s",
                                    word,
                                    token_ids,
                                    [self.tokens.get(i, "?") for i in token_ids],
                                )
                        # Additionally add the full phrase as a hotword
                        token_ids = self.bpe_processor.EncodeAsIds(line)
                        if token_ids and len(words) > 1:
                            self.hotword_tuples.append(tuple(token_ids))
                            log.info(
                                "Hotword phrase %r -> token_ids=%s tokens=%s",
                                line,
                                token_ids,
                                [self.tokens.get(i, "?") for i in token_ids],
                            )
                log.info("Loaded %d hotwords from %s", len(self.hotword_tuples), hotwords_path)
            else:
                log.warning("Hotwords require BPE model; skipping hotwords loading.")
        elif hotwords_path:
            log.warning("Hotwords file not found: %s", hotwords_path)

    def reset(self) -> None:
        """Reset all states for a new session."""
        self._enc_states: Dict[str, np.ndarray] = {}
        for inp in self.encoder.get_inputs():
            if inp.name == "x":
                continue
            shape = [s if isinstance(s, (int, np.integer)) else 1 for s in inp.shape]
            if "len" in inp.name or "frames" in inp.name:
                self._enc_states[inp.name] = np.zeros(shape, dtype=np.int64)
            else:
                self._enc_states[inp.name] = np.zeros(shape, dtype=np.float32)

        self._dec_states: Optional[Dict[str, np.ndarray]] = None
        self._dec_context = np.full((1, self.decoder_context_size), self.blank_id, dtype=np.int64)
        self.prev_token = self.blank_id

        self.beams = None
        self._beam_prev_len = 0

    @staticmethod
    def _log_softmax(x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
        return x - x_max - np.log(sum_e_x)

    def _run_encoder(self, x: np.ndarray) -> np.ndarray:
        """
        x: (1, T, C)
        Returns encoder_out: (1, T', enc_dim)
        """
        feed = {"x": x}
        feed.update(self._enc_states)
        outputs = self.encoder.run(None, feed)
        encoder_out = outputs[0]
        if self.subsampling_factor is None and x.shape[1] > 0 and encoder_out.shape[1] > 0:
            self.subsampling_factor = x.shape[1] / encoder_out.shape[1]
            log.info("Inferred subsampling_factor from ONNX encoder: %.2f", self.subsampling_factor)
        state_idx = 0
        for name in self._enc_in_names:
            if name == "x":
                continue
            self._enc_states[name] = outputs[1 + state_idx]
            state_idx += 1
        return encoder_out

    def _run_decoder(self) -> np.ndarray:
        feed = {"y": self._dec_context}
        if self._dec_states is not None:
            feed.update(self._dec_states)
        outputs = self.decoder.run(None, feed)
        out_map = {name: out for name, out in zip(self._dec_out_names, outputs)}
        self._dec_states = {}
        for name in self._dec_out_names:
            if name != "decoder_out":
                self._dec_states[name] = out_map[name]
        return out_map.get("decoder_out", outputs[0])

    def _run_decoder_state(
        self,
        context: np.ndarray,
        states: Optional[Dict[str, np.ndarray]],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        feed = {"y": context}
        if states is not None:
            feed.update(states)
        outputs = self.decoder.run(None, feed)
        out_map = {name: out for name, out in zip(self._dec_out_names, outputs)}
        new_states = {}
        for name in self._dec_out_names:
            if name != "decoder_out":
                new_states[name] = out_map[name]
        return out_map.get("decoder_out", outputs[0]), new_states

    def _run_joiner(self, encoder_out: np.ndarray, decoder_out: np.ndarray) -> np.ndarray:
        feed = {}
        for name in self._join_in_names:
            if "encoder" in name:
                feed[name] = encoder_out
            elif "decoder" in name:
                feed[name] = decoder_out
        outputs = self.joiner.run(None, feed)
        out_map = {name: out for name, out in zip(self._join_out_names, outputs)}
        return out_map.get("logits", outputs[0])

    def decode_token_ids(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs to text using BPE if available."""
        if self.bpe_processor is not None:
            return self.bpe_processor.DecodeIds(token_ids)
        return "".join(self.tokens.get(tid, "") for tid in token_ids)

    def decode_encoder_out(
        self,
        encoder_out: np.ndarray,
        frame_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Decode encoder output using either greedy search or modified beam search.
        encoder_out: (1, T, enc_dim)
        Returns list of token dicts with keys: token, text, logprob, timestamp.
        """
        if not self.use_beam:
            return self._decode_encoder_out_greedy(encoder_out, frame_offset)
        return self._decode_encoder_out_beam(encoder_out, frame_offset)

    def _decode_encoder_out_greedy(
        self,
        encoder_out: np.ndarray,
        frame_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        T = encoder_out.shape[1]
        results: List[Dict[str, Any]] = []
        for t in range(T):
            enc_t = encoder_out[:, t, :]
            symbols_emitted = 0
            while symbols_emitted < self.max_symbols_per_step:
                dec_out = self._run_decoder()
                logits = self._run_joiner(enc_t, dec_out)
                logprobs = self._log_softmax(logits)
                token = int(np.argmax(logits))
                token_logprob = float(logprobs[0, token])
                if token == self.blank_id:
                    break
                symbols_emitted += 1
                self.prev_token = token
                self._dec_context = np.roll(self._dec_context, -1, axis=1)
                self._dec_context[0, -1] = token
                text = self.tokens.get(token, "")
                if self.subsampling_factor is not None:
                    frame_idx = frame_offset + t * self.subsampling_factor
                else:
                    frame_idx = frame_offset + t
                timestamp = frame_idx * self.frame_shift_s
                results.append(
                    {
                        "token": token,
                        "text": text,
                        "logprob": token_logprob,
                        "timestamp": timestamp,
                    }
                )
        return results

    def _decode_encoder_out_beam(
        self,
        encoder_out: np.ndarray,
        frame_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        T = encoder_out.shape[1]
        blank_id = self.blank_id

        if self.beams is None:
            self.beams = [
                {
                    "score": 0.0,
                    "tokens": [],
                    "texts": [],
                    "logprobs": [],
                    "timestamps": [],
                    "dec_context": np.full((1, self.decoder_context_size), blank_id, dtype=np.int64),
                    "dec_states": None,
                }
            ]

        beams = self.beams

        for t in range(T):
            enc_t = encoder_out[:, t, :]  # (1, enc_dim)
            next_beams: List[Dict[str, Any]] = []

            active = beams

            for _ in range(self.max_symbols_per_step):
                if not active:
                    break

                all_candidates: List[Dict[str, Any]] = []

                for hyp in active:
                    dec_out, new_states = self._run_decoder_state(
                        hyp["dec_context"], hyp["dec_states"]
                    )
                    logits = self._run_joiner(enc_t, dec_out)
                    logprobs = self._log_softmax(logits)[0]  # (vocab_size,)

                    # Blank: advance to next frame, keep decoder state unchanged
                    blank_score = hyp["score"] + float(logprobs[blank_id])
                    next_beams.append(
                        {
                            "score": blank_score,
                            "tokens": list(hyp["tokens"]),
                            "texts": list(hyp["texts"]),
                            "logprobs": list(hyp["logprobs"]),
                            "timestamps": list(hyp["timestamps"]),
                            "dec_context": hyp["dec_context"].copy(),
                            "dec_states": hyp["dec_states"],
                        }
                    )

                    # Non-blank expansions
                    other_logprobs = logprobs.copy()
                    other_logprobs[blank_id] = -np.inf
                    top_k = min(self.beam_size, len(logprobs) - 1)
                    if top_k > 0:
                        top_indices = np.argpartition(-other_logprobs, top_k - 1)[:top_k]
                        top_indices = top_indices[np.argsort(-other_logprobs[top_indices])]
                        for idx in top_indices:
                            idx = int(idx)
                            score = hyp["score"] + float(logprobs[idx])
                            if self.ngram_lm is not None and self.lm_scale != 0.0:
                                prev_token = hyp["tokens"][-1] if hyp["tokens"] else blank_id
                                lm_score = self.ngram_lm.score(prev_token, idx)
                                score += self.lm_scale * lm_score
                            if self.unigram_lm is not None and self.lm_scale != 0.0:
                                token_text = self.tokens.get(idx, "")
                                score += self.lm_scale * self.unigram_lm.score(token_text)
                            if self.hotword_tuples and self.hotword_bonus != 0.0:
                                new_tokens = hyp["tokens"] + [idx]
                                for hw in self.hotword_tuples:
                                    if len(new_tokens) >= len(hw) and tuple(new_tokens[-len(hw):]) == hw:
                                        score += self.hotword_bonus
                                        log.debug(
                                            "Hotword bonus +%.2f for prefix=%s hotword=%s",
                                            self.hotword_bonus,
                                            "".join(hyp["texts"]),
                                            "".join(self.tokens.get(i, "") for i in hw),
                                        )
                                        break
                                    # Debug: show progress towards hotword
                                    if len(new_tokens) < len(hw):
                                        prefix_match = True
                                        for i in range(len(new_tokens)):
                                            if new_tokens[i] != hw[i]:
                                                prefix_match = False
                                                break
                                        if prefix_match and idx == hw[len(new_tokens) - 1]:
                                            log.debug(
                                                "Hotword progress: prefix=%s next_expected=%s hotword=%s",
                                                "".join(hyp["texts"]),
                                                self.tokens.get(hw[len(new_tokens) - 1], "?"),
                                                "".join(self.tokens.get(i, "") for i in hw),
                                            )
                            new_context = np.roll(hyp["dec_context"], -1, axis=1)
                            new_context[0, -1] = idx
                            if self.subsampling_factor is not None:
                                frame_idx = frame_offset + t * self.subsampling_factor
                            else:
                                frame_idx = frame_offset + t
                            timestamp = frame_idx * self.frame_shift_s
                            token_text = self.tokens.get(idx, "")
                            all_candidates.append(
                                {
                                    "score": score,
                                    "tokens": hyp["tokens"] + [idx],
                                    "texts": hyp["texts"] + [token_text],
                                    "logprobs": hyp["logprobs"] + [float(logprobs[idx])],
                                    "timestamps": hyp["timestamps"] + [timestamp],
                                    "dec_context": new_context,
                                    "dec_states": new_states,
                                }
                            )

                if not all_candidates:
                    break

                sort_key = (
                    (lambda x: x["score"] / max(1, len(x["tokens"])))
                    if self.length_norm
                    else (lambda x: x["score"])
                )
                all_candidates.sort(key=sort_key, reverse=True)
                active = all_candidates[: self.beam_size]

            next_beams.extend(active)

            # Merge duplicate prefixes with log-sum-exp
            merged: Dict[Tuple[int, ...], Dict[str, Any]] = {}
            for hyp in next_beams:
                key = tuple(hyp["tokens"])
                if key in merged:
                    old = merged[key]
                    a = old["score"]
                    b = hyp["score"]
                    if a >= b:
                        merged_score = a + math.log1p(math.exp(b - a))
                    else:
                        merged_score = b + math.log1p(math.exp(a - b))
                    old["score"] = merged_score
                    if b > a:
                        old["tokens"] = hyp["tokens"]
                        old["texts"] = hyp["texts"]
                        old["logprobs"] = hyp["logprobs"]
                        old["timestamps"] = hyp["timestamps"]
                        old["dec_context"] = hyp["dec_context"]
                        old["dec_states"] = hyp["dec_states"]
                else:
                    merged[key] = hyp

            beams = list(merged.values())
            sort_key = (
                (lambda x: x["score"] / max(1, len(x["tokens"])))
                if self.length_norm
                else (lambda x: x["score"])
            )
            beams.sort(key=sort_key, reverse=True)
            beams = beams[: self.beam_size]

        self.beams = beams

        if not beams:
            return []

        best = beams[0]
        # Return only newly emitted tokens since last call
        new_tokens = best["tokens"][self._beam_prev_len :]
        new_texts = best["texts"][self._beam_prev_len :]
        new_logprobs = best["logprobs"][self._beam_prev_len :]
        new_timestamps = best["timestamps"][self._beam_prev_len :]
        self._beam_prev_len = len(best["tokens"])

        return [
            {
                "token": tok,
                "text": txt,
                "logprob": lp,
                "timestamp": ts,
            }
            for tok, txt, lp, ts in zip(
                new_tokens, new_texts, new_logprobs, new_timestamps
            )
        ]
