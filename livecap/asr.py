"""Streaming ASR wrapper around Kyutai's stt-1b-en_fr (moshi 0.2.13).

Consumes AudioChunk (livecap/contract.py, 16 kHz mono) and turns it into a
stream of AsrToken, one per 80 ms Mimi frame (12.5 Hz), by running the
model's step-by-step streaming API (mimi.encode -> lm_gen.step) instead of
the file-mode batch pass used in moshi/run_inference.py.

Key facts verified before writing this module (see plan sent to the team
lead, approved 2026-07-18):
    - Mimi (the audio codec) runs natively at 24 kHz / 12.5 Hz, frame_size =
      1920 samples (confirmed via the checkpoint's config.json:
      mimi.sample_rate=24000, mimi.frame_rate=12.5). CONTRACT.md fixes
      AudioChunk at 16 kHz, so this module resamples 16 kHz -> 24 kHz
      internally with _LinearResampler before framing.
    - sphn.resample (a transitive dependency of moshi, used in run_inference
      for one-shot file reads) is NOT chunk-safe: it is a fixed-block-size
      resampler that rounds every call's output up to a multiple of 1536
      samples regardless of input length, so independent calls on small
      streaming chunks do not concatenate into the same signal as one
      whole-file call. _LinearResampler below is a phase-continuous
      streaming replacement (numpy only, no new dependency), verified
      against a whole-signal np.interp reference (max abs error ~1e-11
      across irregular chunk sizes).
    - The checkpoint (kyutai/stt-1b-en_fr) has dep_q=0: no audio codebooks
      to decode, only a single text token per step. Token ids 0 (<unk>) and
      3 (<pad>) are the model's "no new text this step" marker -- this is
      the concrete mechanism behind Kyutai's advertised "semantic VAD"
      (confirmed: neither the HF model card nor the transformers docs
      describe a separate VAD channel/head; the text token stream itself
      carries it, exactly as used in moshi/run_inference.py).
    - stt_config.audio_delay_seconds=0.5: text lags audio by ~0.5 s inside
      the model. flush() drains this at end-of-stream by feeding trailing
      silence.
"""

from __future__ import annotations

import os

# Must be set before any torch/moshi import: Triton is absent on Windows, and
# without this moshi silently exits 0 after loading the model with zero
# transcript (verified on this machine's POC, see CLAUDE.md gotchas).
os.environ.setdefault("NO_TORCH_COMPILE", "1")

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from moshi.models import LMGen, loaders

from livecap.contract import AudioChunk

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "kyutai/stt-1b-en_fr"
CONTRACT_SAMPLE_RATE = 16_000  # AudioChunk.samples rate, per livecap/contract.py
# The model's own "no new text" marker tokens (see module docstring).
NO_TEXT_TOKEN_IDS = frozenset({0, 3})


@dataclass(frozen=True)
class AsrToken:
    """One 80 ms streaming step's output.

    Internal hand-off between asr.py and segmenter.py: NOT part of
    CONTRACT.md. segmenter.py consumes a stream of these and only produces a
    CONTRACT.md TranscriptSegment once a run of tokens stabilizes.

    Attributes:
        text: decoded text piece for this step ("" when is_silence is True,
            or when the model piece itself renders to nothing).
        is_silence: True when the model emitted a no-text marker token
            (id 0 or 3) this step -- the model's implicit VAD signal.
        timestamp: time.monotonic() when this step was produced.
    """

    text: str
    is_silence: bool
    timestamp: float


class _LinearResampler:
    """Phase-continuous streaming linear-interpolation resampler.

    Maintains a fractional read position and a small tail of unconsumed
    input samples across calls, so that feeding it arbitrary-sized chunks
    produces the same result (up to float32 precision) as resampling the
    whole concatenated signal in one call. See module docstring for why a
    stateless one-shot resampler (sphn.resample) cannot be used here.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self._ratio = src_rate / dst_rate  # input samples consumed per output sample
        self._pos = 0.0  # fractional read position into the pending buffer
        self._tail: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

    def push(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        """Resample one chunk of input samples, returning 0+ output samples."""
        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)
        buf = np.concatenate([self._tail, samples]) if self._tail.size else samples
        n_out = int((len(buf) - 1 - self._pos) / self._ratio) + 1
        if n_out <= 0:
            self._tail = buf
            return np.zeros(0, dtype=np.float32)
        idx = self._pos + np.arange(n_out) * self._ratio
        out = np.interp(idx, np.arange(len(buf)), buf).astype(np.float32)
        consumed_pos = self._pos + n_out * self._ratio
        keep_from = int(consumed_pos)
        self._tail = buf[keep_from:]
        self._pos = consumed_pos - keep_from
        return out


class KyutaiStreamingAsr:
    """Loads Kyutai stt-1b-en_fr once (warm) and transcribes AudioChunk streams."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        hf_repo: str = DEFAULT_HF_REPO,
        warmup_steps: int = 8,
    ) -> None:
        self.device = torch.device(device)
        logger.info("loading Kyutai STT checkpoint %s", hf_repo)
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(hf_repo)
        self.mimi = checkpoint_info.get_mimi(device=self.device)
        self.text_tokenizer = checkpoint_info.get_text_tokenizer()
        lm = checkpoint_info.get_moshi(device=self.device, dtype=torch.bfloat16)
        self.lm_gen = LMGen(lm, **checkpoint_info.lm_gen_config)
        logger.info("Kyutai STT checkpoint loaded on %s", self.device)

        self._frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self._resampler = _LinearResampler(CONTRACT_SAMPLE_RATE, self.mimi.sample_rate)
        self._pending: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._first_frame = True

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self._warmup(warmup_steps)

    def _warmup(self, steps: int) -> None:
        """Run silent frames to pay the CUDA kernel JIT cost once, then reset
        the streaming state so the warm-up context does not leak into the
        first real audio frame (which still needs the double-step below)."""
        if steps <= 0:
            return
        silence = torch.zeros(
            (1, self.mimi.channels, self._frame_size), device=self.device
        )
        with torch.no_grad():
            for _ in range(steps):
                codes = self.mimi.encode(silence)
                self.lm_gen.step(codes)
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self._first_frame = True
        logger.debug("asr warm-up done (%d steps)", steps)

    def push_chunk(self, chunk: AudioChunk) -> list[AsrToken]:
        """Feed one AudioChunk (16 kHz mono, arbitrary length). Returns one
        AsrToken per complete 80 ms Mimi frame now available; 0 tokens if the
        chunk was smaller than a frame (buffered internally for next call)."""
        resampled = self._resampler.push(chunk.samples)
        self._pending = (
            np.concatenate([self._pending, resampled])
            if self._pending.size
            else resampled
        )
        tokens: list[AsrToken] = []
        while self._pending.shape[0] >= self._frame_size:
            frame = self._pending[: self._frame_size]
            self._pending = self._pending[self._frame_size :]
            tokens.append(self._step(frame))
        return tokens

    def _step(self, frame: NDArray[np.float32]) -> AsrToken:
        frame_t = (
            torch.from_numpy(frame).to(self.device).view(1, self.mimi.channels, -1)
        )
        step_start = time.monotonic()
        with torch.no_grad():
            codes = self.mimi.encode(frame_t)
            if self._first_frame:
                # The first slice of codes must be seen twice by the
                # transformer, otherwise it is silently replaced by the
                # model's initial tokens -- same quirk documented in
                # moshi/run_inference.py InferenceState.run() ("Ensure that
                # the first slice of codes is properly seen by the
                # transformer as otherwise the first slice is replaced by
                # the initial tokens"), simplified here for dep_q=0.
                self.lm_gen.step(codes)
                self._first_frame = False
            out = self.lm_gen.step(codes)
        step_ms = (time.monotonic() - step_start) * 1000
        logger.debug("asr step: %.2f ms", step_ms)

        text = ""
        is_silence = True
        if out is not None:
            token_id = int(out[0, 0].cpu().item())
            if token_id not in NO_TEXT_TOKEN_IDS:
                piece = self.text_tokenizer.id_to_piece(token_id)
                text = piece.replace("▁", " ")
                is_silence = False
        return AsrToken(text=text, is_silence=is_silence, timestamp=time.monotonic())

    def flush(self, tail_seconds: float = 1.5) -> list[AsrToken]:
        """Push trailing silence to drain the model's ~0.5 s internal delay
        (stt_config.audio_delay_seconds) at the end of a stream."""
        n = int(tail_seconds * CONTRACT_SAMPLE_RATE)
        silence_chunk = AudioChunk(
            samples=np.zeros(n, dtype=np.float32), timestamp=time.monotonic()
        )
        return self.push_chunk(silence_chunk)
