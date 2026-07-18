"""GPU integration test for livecap.asr + livecap.segmenter, plus a CPU-only
regression test for the streaming resampler design decision.

The @pytest.mark.gpu tests load the real Kyutai stt-1b-en_fr checkpoint and
require a CUDA GPU (registered in pytest.ini; run `pytest -m gpu` to select
them explicitly, or plain `pytest` which runs everything including them on
a machine that has the GPU, as this one does).

They replay the two reference wavs from the POC (already 16 kHz mono,
matching livecap/contract.py's AudioChunk rate) through the real streaming
pipeline (KyutaiStreamingAsr.push_chunk -> Segmenter.feed), reconstitute
the transcript, and check it against keywords taken verbatim from the
POC's own transcripts (test_fr.transcript.txt / test_en.transcript.txt),
tolerant to minor wording differences between the file-mode POC run and
this streaming re-implementation.

The wavs are deliberately split into chunk sizes that are NOT a round
multiple of 80 ms or of the internal 24 kHz Mimi frame (997, 1301, 733,
1600, 421 samples), per the coordination note from t-capture: capture.py
(task 2) hands asr.py blocks of <= 80 ms that are not guaranteed to be
frame-aligned once resampled, so push_chunk's internal buffering must
bridge incomplete frames across calls without losing or duplicating audio.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from livecap.asr import CONTRACT_SAMPLE_RATE, KyutaiStreamingAsr, _LinearResampler
from livecap.contract import AudioChunk
from livecap.segmenter import Segmenter

logger = logging.getLogger(__name__)

POC_DIR = Path(__file__).resolve().parent.parent.parent / "livecap-poc"
RAGGED_CHUNK_SIZES = (997, 1301, 733, 1600, 421)


def _iter_ragged_chunks(
    samples: np.ndarray, start_timestamp: float
) -> list[AudioChunk]:
    chunks: list[AudioChunk] = []
    i = 0
    size_idx = 0
    t = start_timestamp
    while i < len(samples):
        size = RAGGED_CHUNK_SIZES[size_idx % len(RAGGED_CHUNK_SIZES)]
        size_idx += 1
        block = samples[i : i + size]
        chunks.append(
            AudioChunk(samples=np.asarray(block, dtype=np.float32), timestamp=t)
        )
        i += size
        t += len(block) / CONTRACT_SAMPLE_RATE
    return chunks


def _transcribe_wav(asr: KyutaiStreamingAsr, wav_path: Path) -> tuple[str, float]:
    samples, samplerate = sf.read(wav_path, dtype="float32", always_2d=False)
    assert samplerate == CONTRACT_SAMPLE_RATE, (
        f"{wav_path.name} is {samplerate} Hz, expected {CONTRACT_SAMPLE_RATE} Hz "
        "(livecap/contract.py AudioChunk rate)"
    )
    segmenter = Segmenter()
    segments_text: list[str] = []
    step_count = 0

    def _consume(tokens: list) -> None:
        nonlocal step_count
        for token in tokens:
            step_count += 1
            segment = segmenter.feed(token)
            if segment is not None:
                segments_text.append(segment.text)

    start = time.monotonic()
    for chunk in _iter_ragged_chunks(samples, start):
        _consume(asr.push_chunk(chunk))
    _consume(asr.flush())
    elapsed = time.monotonic() - start

    final = segmenter.flush(time.monotonic())
    if final is not None:
        segments_text.append(final.text)

    ms_per_step = (elapsed * 1000 / step_count) if step_count else 0.0
    return " ".join(segments_text), ms_per_step


@pytest.fixture(scope="module")
def asr_engine() -> KyutaiStreamingAsr:
    return KyutaiStreamingAsr(device="cuda")


def test_linear_resampler_matches_whole_signal_reference() -> None:
    """White-box regression test for the streaming resample design decision
    documented in asr.py: sphn.resample is not chunk-safe (verified while
    planning this task -- independent calls on small chunks round each
    output up to a multiple of 1536 samples and do not concatenate into the
    same signal as one whole-file call). _LinearResampler is the numpy-only
    fallback; pin its correctness against a whole-signal np.interp
    reference, fed through the same ragged chunking used by the GPU tests
    below."""
    rng = np.random.RandomState(0)
    signal = rng.uniform(-1, 1, size=CONTRACT_SAMPLE_RATE * 3).astype(np.float32)

    dst_rate = 24_000
    ratio = CONTRACT_SAMPLE_RATE / dst_rate
    n_out_ref = int((len(signal) - 1) / ratio) + 1
    idx_ref = np.arange(n_out_ref) * ratio
    reference = np.interp(idx_ref, np.arange(len(signal)), signal).astype(np.float32)

    resampler = _LinearResampler(CONTRACT_SAMPLE_RATE, dst_rate)
    streamed = np.concatenate(
        [resampler.push(chunk.samples) for chunk in _iter_ragged_chunks(signal, 0.0)]
    )

    n = min(len(reference), len(streamed))
    assert n > 0
    assert np.abs(reference[:n] - streamed[:n]).max() < 1e-6


@pytest.mark.gpu
def test_streaming_transcript_matches_french_poc_keywords(
    asr_engine: KyutaiStreamingAsr,
) -> None:
    transcript, ms_per_step = _transcribe_wav(asr_engine, POC_DIR / "test_fr.wav")
    lowered = transcript.lower()
    logger.info("FR transcript: %r (%.2f ms/step)", transcript, ms_per_step)

    for keyword in ("réunion", "power app", "sharepoint", "migration"):
        assert keyword in lowered, (
            f"expected keyword {keyword!r} in transcript: {transcript!r}"
        )
    # POC measured 16-18 ms/step on this GPU (RTX 5080); generous ceiling
    # to avoid flakiness from first-call CUDA variance while still
    # catching a real performance regression.
    assert ms_per_step < 200, f"{ms_per_step:.2f} ms/step exceeds sanity ceiling"


@pytest.mark.gpu
def test_streaming_transcript_matches_english_poc_keywords(
    asr_engine: KyutaiStreamingAsr,
) -> None:
    transcript, ms_per_step = _transcribe_wav(asr_engine, POC_DIR / "test_en.wav")
    lowered = transcript.lower()
    logger.info("EN transcript: %r (%.2f ms/step)", transcript, ms_per_step)

    for keyword in ("meeting", "power apps", "sharepoint", "migration"):
        assert keyword in lowered, (
            f"expected keyword {keyword!r} in transcript: {transcript!r}"
        )
    assert ms_per_step < 200, f"{ms_per_step:.2f} ms/step exceeds sanity ceiling"
