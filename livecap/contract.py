"""Shared contract between LiveCap Local pipeline stages.

This module is the single source of truth for the dataclasses and queues
every other module imports. Do not redefine these shapes elsewhere; import
from here.

Pipeline and module ownership:
    livecap/capture.py    (tache 2) produces AudioChunk        -> audio_q
    livecap/asr.py        (tache 3) consumes audio_q, runs Kyutai stt-1b-en_fr
    livecap/segmenter.py  (tache 3) produces TranscriptSegment -> segment_q
    livecap/translator.py (tache 4) consumes segment_q, produces TranslatedSegment -> ui_q
    livecap/overlay.py    (tache 5) consumes ui_q, renders subtitles
    main.py               (tache 6) wires the queues and threads together

See CONTRACT.md for the human-readable version of this document and for the
config.yaml schema.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AudioChunk:
    """One block of mixed, resampled audio pushed by capture.py onto audio_q.

    Attributes:
        samples: mono float32 PCM samples at 16 kHz, shape (n_samples,).
        timestamp: time.monotonic() seconds at the start of this chunk. Used
            downstream to compute TranscriptSegment.t_start/t_end.
    """

    samples: NDArray[np.float32]
    timestamp: float


@dataclass(frozen=True)
class TranscriptSegment:
    """One stabilized transcript segment, pushed by segmenter.py onto segment_q.

    Attributes:
        text: transcribed text in the source language, already stripped.
        lang: detected source language code, "fr" or "en".
        t_start: time.monotonic() seconds at segment start.
        t_end: time.monotonic() seconds at the stabilization point (silence
            > 700 ms or terminal punctuation, per PRD segmentation rule).
        final: always True for segments pushed onto segment_q — segmenter.py
            only emits stabilized segments; unstable streaming partials stay
            internal to asr.py/segmenter.py and never cross this queue.
    """

    text: str
    lang: str
    t_start: float
    t_end: float
    final: bool


@dataclass(frozen=True)
class TranslatedSegment:
    """One translated segment, pushed by translator.py onto ui_q.

    Attributes:
        source: the TranscriptSegment this translation was produced from.
        text: translated text in the configured target language. If
            translation failed (Ollama timeout/error), translator.py still
            pushes a TranslatedSegment with text == "" so overlay.py can fall
            back to displaying the VO line alone, per PRD (never block the
            display).
        latency_ms: wall-clock translation latency in milliseconds, 0.0 when
            text == "" (translation was skipped/failed).
    """

    source: TranscriptSegment
    text: str
    latency_ms: float


def make_audio_queue(maxsize: int = 64) -> "queue.Queue[AudioChunk]":
    """Queue capture.py (producer) feeds and asr.py (consumer) drains."""
    return queue.Queue(maxsize=maxsize)


def make_segment_queue(maxsize: int = 32) -> "queue.Queue[TranscriptSegment]":
    """Queue segmenter.py (producer) feeds and translator.py (consumer) drains."""
    return queue.Queue(maxsize=maxsize)


def make_ui_queue(maxsize: int = 32) -> "queue.Queue[TranslatedSegment]":
    """Queue translator.py (producer) feeds and overlay.py (consumer) drains."""
    return queue.Queue(maxsize=maxsize)
