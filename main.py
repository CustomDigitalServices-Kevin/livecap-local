"""LiveCap Local entry point: wires capture -> asr+segmenter -> translator ->
overlay through the three CONTRACT.md queues, one thread per stage, plus a
``--input-wav`` mode that short-circuits capture for the scripted E2E test.

Pipeline (see CONTRACT.md for the full interface reference)::

    [AudioCapture | WavFeederThread] --AudioChunk--> audio_q
        --> AsrSegmenterThread (KyutaiStreamingAsr + Segmenter) --TranscriptSegment--> segment_q
        --> TranslatorThread (livecap.translator) --TranslatedSegment--> ui_q
        --> UiFanoutThread --> TranscriptLogger + overlay_ui_q --> LiveOverlay (Qt main thread)

Design notes not obvious from the code alone:

- Every module in livecap/ (capture, asr, segmenter, translator, overlay,
  config, contract) is imported, never redefined. The only new pipeline
  code lives here and in ``livecap/translog.py``.
- ``ui_q`` (the real CONTRACT.md queue, written by TranslatorThread) has
  exactly one consumer: ``UiFanoutThread``. It logs the transcript (if
  enabled) and re-publishes each item onto a *second*, main.py-owned queue
  (``overlay_ui_q``) that ``Overlay``/``LiveOverlay`` polls on the Qt main
  thread. This keeps overlay.py's single-consumer ``ui_q`` polling design
  intact (it never needs to know about logging) while still letting
  main.py observe every segment exactly once.
- Progressive VO (Segmenter's ``on_partial`` callback, CONTRACT.md line 37:
  partials must never reach ``segment_q``): the callback runs on
  AsrSegmenterThread and pushes plain strings onto a bounded, drop-oldest
  ``partial_q``. ``LiveOverlay`` (below) subclasses ``Overlay`` to add a
  *second* QTimer that drains ``partial_q`` and appends one extra render
  line -- it never touches ``Overlay``'s own ``ui_q``/history logic, so
  overlay.py itself is untouched.
- ``--input-wav`` reuses ``audio_q`` (via ``WavFeederThread``) instead of
  bypassing the queue layer entirely, so the same ``AsrSegmenterThread`` /
  ``TranslatorThread`` / ``UiFanoutThread`` code path is exercised as in
  live mode -- this is a real pipeline integration test, not a separate
  one-off transcription script. Only ``AudioCapture`` (the WASAPI capture
  module, task 2) is short-circuited, per the PRD.
- End-of-stream (wav mode) is a ``threading.Event`` (``end_of_stream``)
  shared between the feeder and AsrSegmenterThread, not a sentinel smuggled
  through ``audio_q`` -- ``audio_q`` stays strictly ``Queue[AudioChunk]``
  per CONTRACT.md. The same flush code path also runs on Ctrl+C/window
  close in live mode, so trailing speech is never lost on shutdown either.
- Ctrl+C under a running Qt event loop: CPython only delivers a signal
  between bytecode instructions, and Qt's C++ event loop blocks there
  indefinitely. The fix is the standard documented idiom -- a short-period
  QTimer with a no-op Python callback forces the interpreter back into
  bytecode regularly so the SIGINT handler actually runs. See
  https://machinekoder.com/how-to-not-shoot-yourself-in-the-foot-using-python-qt/
  and https://riverbankcomputing.com/pipermail/pyqt/2015-March/035673.html.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Must be set before any torch/moshi import. livecap/asr.py already sets
# this (os.environ.setdefault) as soon as it is imported, but main.py is the
# actual process entry point, so set it here too, before importing
# livecap.asr, per CONTRACT.md's "a poser dans le code ... pas seulement
# dans le launcher" -- defense in depth, not a duplicate requirement.
os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import soundfile as sf

from livecap.asr import AsrToken, KyutaiStreamingAsr
from livecap.capture import (
    BLOCK_MS,
    TARGET_RATE,
    AudioCapture,
    CaptureError,
    resample_mono_to_16k,
)
from livecap.config import ConfigError, LiveCapConfig, load_config
from livecap.contract import (
    AudioChunk,
    TranscriptSegment,
    TranslatedSegment,
    make_audio_queue,
    make_segment_queue,
    make_ui_queue,
)
from livecap.segmenter import Segmenter
from livecap.translator import OllamaUnavailableError, TranslatorThread
from livecap.translog import TranscriptLogger

logger = logging.getLogger("livecap.main")

DEFAULT_CONFIG_PATH = "config.yaml"
STAGE_JOIN_TIMEOUT_S = 5.0
DRAIN_POLL_INTERVAL_S = 0.05
# Generous ceiling for "wait until segment_q/ui_q empties" in --input-wav
# mode: must comfortably exceed one Ollama translation call
# (TRANSLATE_TIMEOUT_S = 5s in livecap/translator.py) plus scheduling slack.
DRAIN_TIMEOUT_S = 8.0
PARTIAL_QUEUE_MAXSIZE = 8
# Generous ceiling to wait for AsrSegmenterThread.ready (cold model load +
# CUDA JIT warm-up measured at 2-5 s on RTX 5080; a producer starting before
# this drops audio into a full audio_q -- see AsrSegmenterThread docstring).
ASR_READY_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="LiveCap Local -- sous-titrage et traduction temps reel 100% local.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"chemin du fichier config.yaml (defaut: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--input-wav",
        default=None,
        metavar="FICHIER.wav",
        help=(
            "court-circuite la capture WASAPI : decoupe ce fichier wav en "
            "AudioChunk de 80 ms et le rejoue dans le pipeline (asr.flush() "
            "en fin de fichier). Utilise par l'E2E scripte."
        ),
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="mode headless (sans fenetre PyQt6) : pour l'E2E scripte / debogage sans GPU d'affichage.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="device torch pour le modele ASR (defaut: cuda)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="niveau de log (defaut: INFO)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Pipeline metrics (returned to callers -- main() CLI and tests/test_e2e.py)
# ---------------------------------------------------------------------------


@dataclass
class PipelineMetrics:
    """Observability collected while the pipeline runs.

    ``first_segment_to_translation_ms`` is the PRD/task-6-requested latency:
    time between the first stabilized TranscriptSegment reaching segment_q
    and the first TranslatedSegment reaching ui_q. ``None`` until both have
    happened at least once.
    """

    first_final_segment_at: float | None = None
    first_translation_at: float | None = None
    segments_seen: int = 0
    translations_seen: int = 0

    @property
    def first_segment_to_translation_ms(self) -> float | None:
        if self.first_final_segment_at is None or self.first_translation_at is None:
            return None
        return (self.first_translation_at - self.first_final_segment_at) * 1000.0


@dataclass
class PipelineResult:
    """Returned by run_input_wav_pipeline() for callers (CLI, test_e2e.py)."""

    metrics: PipelineMetrics
    transcript_path: Path | None
    translated_segments: list[TranslatedSegment] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1 (asr.py + segmenter.py): AsrSegmenterThread
# ---------------------------------------------------------------------------


class AsrSegmenterThread(threading.Thread):
    """Consumes audio_q, runs KyutaiStreamingAsr + Segmenter, produces
    TranscriptSegment onto segment_q.

    The model (KyutaiStreamingAsr) is constructed inside run(), not
    __init__, so building this object (e.g. for argument-parsing tests) is
    cheap and does not touch the GPU.

    Shutdown/end-of-stream share one code path: the loop exits either when
    stop() was called (live mode: Ctrl+C / window close) or when
    end_of_stream is set AND audio_q has been fully drained (--input-wav
    mode, set by WavFeederThread once every chunk has been pushed). Either
    way, a final asr.flush() + segmenter.flush() runs once before the
    thread exits, so trailing speech is never silently dropped.
    """

    def __init__(
        self,
        audio_q: "queue.Queue[AudioChunk]",
        segment_q: "queue.Queue[TranscriptSegment]",
        metrics: PipelineMetrics,
        *,
        on_partial: Callable[[str], None] | None = None,
        device: str = "cuda",
        poll_timeout: float = 0.1,
        end_of_stream: threading.Event | None = None,
    ) -> None:
        super().__init__(name="AsrSegmenterThread", daemon=True)
        self._audio_q = audio_q
        self._segment_q = segment_q
        self._metrics = metrics
        self._on_partial = on_partial
        self._device = device
        self._poll_timeout = poll_timeout
        self._end_of_stream = end_of_stream
        self._stop_event = threading.Event()
        # Set once KyutaiStreamingAsr has finished loading (~2-5 s cold,
        # confirmed while integrating this thread with WavFeederThread: an
        # audio producer that starts before this event is set fills the
        # bounded audio_q and starts dropping chunks -- real audio loss, not
        # a live-capture "stale chunk" tradeoff. Producers (WavFeederThread,
        # AudioCapture) must wait on this before their own start().
        self.ready = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        asr = KyutaiStreamingAsr(device=self._device)
        segmenter = Segmenter(on_partial=self._on_partial)
        self.ready.set()
        try:
            while True:
                try:
                    chunk = self._audio_q.get(timeout=self._poll_timeout)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    if self._end_of_stream is not None and self._end_of_stream.is_set():
                        break
                    continue
                self._process_tokens(asr.push_chunk(chunk), segmenter)
                if self._stop_event.is_set():
                    break
            # Final flush: drains the model's internal audio_delay_seconds
            # and stabilizes any pending buffered text (see asr.py / segmenter.py).
            self._process_tokens(asr.flush(), segmenter)
            final = segmenter.flush(time.monotonic())
            if final is not None:
                self._publish(final)
        finally:
            logger.info("AsrSegmenterThread arrete proprement.")

    def _process_tokens(self, tokens: list[AsrToken], segmenter: Segmenter) -> None:
        for token in tokens:
            segment = segmenter.feed(token)
            if segment is not None:
                self._publish(segment)

    def _publish(self, segment: TranscriptSegment) -> None:
        if self._metrics.first_final_segment_at is None:
            self._metrics.first_final_segment_at = time.monotonic()
        self._metrics.segments_seen += 1
        try:
            self._segment_q.put(segment, timeout=self._poll_timeout)
        except queue.Full:
            logger.warning("segment_q pleine, segment perdu: %r", segment.text)


# ---------------------------------------------------------------------------
# --input-wav feeder (bypasses AudioCapture, per the PRD)
# ---------------------------------------------------------------------------


def chunk_wav_file(wav_path: str | Path, block_ms: int = BLOCK_MS) -> list[AudioChunk]:
    """Read a wav file and slice it into <= block_ms AudioChunk, 16 kHz mono.

    Downmixes multi-channel audio (mean across channels, matching
    livecap.capture's own downmix policy) and resamples via
    livecap.capture.resample_mono_to_16k if the file isn't already 16 kHz
    (the two POC reference wavs are already 16 kHz mono, per
    tests/test_asr_integration.py's own assertion, so this is a defensive
    path for any other wav a user might pass to --input-wav).
    """
    samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.float32)
    if sample_rate != TARGET_RATE:
        samples = resample_mono_to_16k(samples, sample_rate)

    frame_size = max(1, int(TARGET_RATE * block_ms / 1000))
    chunks: list[AudioChunk] = []
    t = time.monotonic()
    for start in range(0, samples.size, frame_size):
        block = samples[start : start + frame_size]
        chunks.append(AudioChunk(samples=block, timestamp=t))
        t += block.size / TARGET_RATE
    return chunks


class WavFeederThread(threading.Thread):
    """Pushes chunk_wav_file(wav_path) onto audio_q, then sets end_of_stream.

    Runs as fast as the bounded audio_q allows (blocking put, providing
    natural backpressure against AsrSegmenterThread) rather than pacing
    itself to real time -- this is a file replay for testing/analysis, not
    a live capture, so there is no reason to wait out the file's actual
    duration.
    """

    def __init__(
        self,
        wav_path: str | Path,
        audio_q: "queue.Queue[AudioChunk]",
        end_of_stream: threading.Event,
        *,
        put_timeout: float = 2.0,
    ) -> None:
        super().__init__(name="WavFeederThread", daemon=True)
        self._wav_path = wav_path
        self._audio_q = audio_q
        self._end_of_stream = end_of_stream
        self._put_timeout = put_timeout

    def run(self) -> None:
        chunks = chunk_wav_file(self._wav_path)
        logger.info(
            "Wav %s decoupe en %d chunks de %d ms.",
            self._wav_path,
            len(chunks),
            BLOCK_MS,
        )
        for chunk in chunks:
            try:
                self._audio_q.put(chunk, timeout=self._put_timeout)
            except queue.Full:
                logger.warning("audio_q pleine et non vidangeable : chunk wav perdu")
        self._end_of_stream.set()


# ---------------------------------------------------------------------------
# Stage 3: UI fan-out (single ui_q consumer: logs, forwards to the overlay)
# ---------------------------------------------------------------------------


class UiFanoutThread(threading.Thread):
    """Sole consumer of the real ui_q: logs the transcript and re-publishes
    each TranslatedSegment onto overlay_ui_q (if the overlay is enabled).

    Also records PipelineMetrics.first_translation_at and appends to
    ``collected`` (used by tests/test_e2e.py to inspect every segment
    without re-parsing the transcript file).
    """

    def __init__(
        self,
        ui_q: "queue.Queue[TranslatedSegment]",
        transcript_logger: TranscriptLogger,
        metrics: PipelineMetrics,
        *,
        overlay_ui_q: "queue.Queue[TranslatedSegment] | None" = None,
        poll_timeout: float = 0.1,
    ) -> None:
        super().__init__(name="UiFanoutThread", daemon=True)
        self._ui_q = ui_q
        self._transcript_logger = transcript_logger
        self._metrics = metrics
        self._overlay_ui_q = overlay_ui_q
        self._poll_timeout = poll_timeout
        self._stop_event = threading.Event()
        self.collected: list[TranslatedSegment] = []

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while True:
            try:
                segment = self._ui_q.get(timeout=self._poll_timeout)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            if self._metrics.first_translation_at is None:
                self._metrics.first_translation_at = time.monotonic()
            self._metrics.translations_seen += 1
            self.collected.append(segment)
            self._transcript_logger.log(segment)
            if self._overlay_ui_q is not None:
                try:
                    self._overlay_ui_q.put_nowait(segment)
                except queue.Full:
                    logger.warning("overlay_ui_q pleine, segment non affiche")
        logger.info("UiFanoutThread arrete proprement.")


# ---------------------------------------------------------------------------
# Overlay integration (LiveOverlay subclasses Overlay -- overlay.py is
# imported and extended, never modified; see module docstring)
# ---------------------------------------------------------------------------


def _make_bounded_putter(q: "queue.Queue[str]") -> Callable[[str], None]:
    """Drop-oldest putter for partial_q, mirroring AudioCapture._push's
    policy in livecap/capture.py: a stale partial line is worse than a
    dropped one for a *progressive* display that will be overwritten by
    the next partial anyway."""

    def _put(text: str) -> None:
        try:
            q.put_nowait(text)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(text)
            except queue.Full:
                pass

    return _put


def _build_live_overlay_class():
    """Returns the LiveOverlay class. Defined lazily inside a function so
    that importing main.py (e.g. from tests/test_main.py, or --no-overlay
    runs) never requires PyQt6 to construct a QApplication/display -- only
    --no-overlay=False code paths import PyQt6 at all."""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QColor

    from livecap.overlay import (
        CURRENT_ALPHA,
        DEFAULT_PAUSE_SHORTCUT,
        DEFAULT_POLL_INTERVAL_MS,
        VO_COLOR,
        Overlay,
        _RenderLine,
    )

    class LiveOverlay(Overlay):
        """Overlay (task 5) + a progressive VO partial line fed by
        Segmenter.on_partial via partial_q (see module docstring)."""

        def __init__(
            self,
            config,
            ui_q,
            partial_q,
            poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
            pause_shortcut: str = DEFAULT_PAUSE_SHORTCUT,
            parent=None,
        ) -> None:
            # Set before super().__init__(): Overlay.__init__ (base class,
            # task 5) calls self._relayout() -> self._build_render_lines()
            # as its last step, which dispatches to THIS subclass's override
            # below -- so _partial_text must already exist by then, or that
            # first call raises AttributeError.
            self._partial_q = partial_q
            self._partial_text = ""
            super().__init__(
                config,
                ui_q,
                poll_interval_ms=poll_interval_ms,
                pause_shortcut=pause_shortcut,
                parent=parent,
            )
            self._partial_timer = QTimer(self)
            self._partial_timer.setInterval(poll_interval_ms)
            self._partial_timer.timeout.connect(self._poll_partial_queue)
            self._partial_timer.start()

        def _poll_partial_queue(self) -> None:
            if self._paused:
                return
            latest = None
            updated = False
            while True:
                try:
                    latest = self._partial_q.get_nowait()
                except queue.Empty:
                    break
                updated = True
            if updated:
                self._partial_text = latest or ""
                self._relayout()
                self.update()

        def _poll_ui_queue(self) -> None:
            before = self._current
            super()._poll_ui_queue()
            if self._current is not before and self._partial_text:
                # A stabilized segment just landed: the progressive partial
                # line it was built from is now stale (its final text is in
                # _current), so drop it to avoid rendering a duplicate line.
                self._partial_text = ""
                self._relayout()
                self.update()

        def _build_render_lines(self) -> list:
            lines = super()._build_render_lines()
            if self._partial_text:
                color = QColor(VO_COLOR)
                color.setAlpha(CURRENT_ALPHA)
                lines.append(_RenderLine(self._partial_text, color))
            return lines

        def shutdown(self) -> None:
            self._partial_timer.stop()
            super().shutdown()

    return LiveOverlay


def _wire_settings_panel(overlay, translator_thread, config):
    """Build the gear panel and connect it to the live pipeline.

    Every change is applied immediately (translator setters are thread-safe,
    the overlay font is a Qt-thread call) and then persisted to
    settings.local.yaml. A failed save is logged, never fatal: the setting is
    already in effect, only its persistence was lost.

    Returns the panel, which the caller must keep referenced -- it is a
    top-level widget with no Qt parent, so dropping the reference would let
    Python garbage-collect it and the gear button would open nothing.
    """
    from livecap import config as config_module
    from livecap.settings_panel import SettingsPanel

    panel = SettingsPanel(
        model=config.translation.model,
        context_turns=config.translation.context_turns,
        target_lang=config.target_lang,
        font_size=config.overlay.font_size,
    )

    def persist() -> None:
        try:
            # Resolved through the module on every call, not bound once at
            # import: that keeps the save point patchable in tests, and a
            # test that cannot substitute this call would silently assert
            # nothing.
            config_module.save_settings_override(panel.current_settings())
        except (OSError, ConfigError) as exc:
            logger.warning(
                "Reglage applique mais non enregistre (%s): %s", type(exc).__name__, exc
            )

    def on_model(model: str) -> None:
        translator_thread.set_model(model)
        # The dialogue history was produced by the previous model; a different
        # model should not be primed with another one's phrasing.
        translator_thread.clear_context()
        persist()

    def on_context(turns: int) -> None:
        translator_thread.set_context_turns(turns)
        persist()

    def on_lang(lang: str) -> None:
        translator_thread.set_target_lang(lang)
        persist()

    def on_font(size: int) -> None:
        overlay.set_font_size(size)
        persist()

    panel.model_changed.connect(on_model)
    panel.context_turns_changed.connect(on_context)
    panel.target_lang_changed.connect(on_lang)
    panel.font_size_changed.connect(on_font)
    panel.reset_geometry_requested.connect(overlay.reset_position)

    def toggle_panel() -> None:
        if panel.isVisible():
            panel.hide()
        else:
            panel.show_next_to(overlay)

    overlay.settings_requested.connect(toggle_panel)
    # Qt.Tool popups are not closed by the app quitting on their own.
    overlay.close_requested.connect(panel.close)
    return panel


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class Pipeline:
    """Owns every stage's handle and runs the shutdown sequence exactly
    once (idempotent -- may be triggered by SIGINT, Qt's aboutToQuit, and
    the normal end-of-run path)."""

    def __init__(self) -> None:
        self.capture: AudioCapture | None = None
        self.wav_feeder: WavFeederThread | None = None
        self.asr_thread: AsrSegmenterThread | None = None
        self.translator_thread: TranslatorThread | None = None
        self.fanout_thread: UiFanoutThread | None = None
        self.transcript_logger: TranscriptLogger | None = None
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        logger.info("Arret du pipeline en cours...")
        if self.capture is not None:
            self.capture.stop()
        if self.wav_feeder is not None:
            self.wav_feeder.join(timeout=STAGE_JOIN_TIMEOUT_S)
        if self.asr_thread is not None:
            self.asr_thread.stop()
            self.asr_thread.join(timeout=STAGE_JOIN_TIMEOUT_S)
            if self.asr_thread.is_alive():
                logger.warning(
                    "AsrSegmenterThread n'a pas rejoint dans le delai imparti."
                )
        if self.translator_thread is not None:
            self.translator_thread.stop()
            self.translator_thread.join(timeout=STAGE_JOIN_TIMEOUT_S)
            if self.translator_thread.is_alive():
                logger.warning(
                    "TranslatorThread n'a pas rejoint dans le delai imparti."
                )
        if self.fanout_thread is not None:
            self.fanout_thread.stop()
            self.fanout_thread.join(timeout=STAGE_JOIN_TIMEOUT_S)
            if self.fanout_thread.is_alive():
                logger.warning("UiFanoutThread n'a pas rejoint dans le delai imparti.")
        if self.transcript_logger is not None:
            self.transcript_logger.close()
        logger.info("Pipeline arrete proprement, aucun thread zombie attendu.")


def _wait_until_empty(q: "queue.Queue", timeout: float) -> bool:
    """Poll q.empty() until True or timeout. Returns whether it drained."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if q.empty():
            return True
        time.sleep(DRAIN_POLL_INTERVAL_S)
    return q.empty()


def run_input_wav_pipeline(
    config: LiveCapConfig,
    wav_path: str | Path,
    *,
    device: str = "cuda",
) -> PipelineResult:
    """Runs the full pipeline against one wav file, headless, and returns
    once every stage has drained -- used by the CLI (--input-wav
    --no-overlay) and directly imported by tests/test_e2e.py for
    deterministic, non-CLI control over a single run.

    Always headless/synchronous (no Qt event loop): manual visual
    verification of --input-wav against the real overlay is a separate,
    independent code path, run_with_overlay() below.
    """
    audio_q = make_audio_queue()
    segment_q = make_segment_queue()
    ui_q = make_ui_queue()
    metrics = PipelineMetrics()
    transcript_logger = TranscriptLogger(config.log_transcript)
    transcript_logger.open()

    end_of_stream = threading.Event()
    asr_thread = AsrSegmenterThread(
        audio_q,
        segment_q,
        metrics,
        device=device,
        end_of_stream=end_of_stream,
    )
    translator_thread = TranslatorThread(
        segment_q,
        ui_q,
        target_lang=config.target_lang,
        model=config.translation.model,
        context_turns=config.translation.context_turns,
    )
    fanout_thread = UiFanoutThread(ui_q, transcript_logger, metrics)
    wav_feeder = WavFeederThread(wav_path, audio_q, end_of_stream)

    asr_thread.start()
    translator_thread.start()
    fanout_thread.start()
    if not asr_thread.ready.wait(timeout=ASR_READY_TIMEOUT_S):
        logger.warning(
            "AsrSegmenterThread pas pret apres %.0f s, feeding quand meme (risque de perte audio).",
            ASR_READY_TIMEOUT_S,
        )
    wav_feeder.start()

    wav_feeder.join(timeout=STAGE_JOIN_TIMEOUT_S)
    asr_thread.join(timeout=STAGE_JOIN_TIMEOUT_S)
    if not _wait_until_empty(segment_q, DRAIN_TIMEOUT_S):
        logger.warning("segment_q non vidangee dans le delai imparti.")
    translator_thread.stop()
    translator_thread.join(timeout=STAGE_JOIN_TIMEOUT_S)
    if not _wait_until_empty(ui_q, DRAIN_TIMEOUT_S):
        logger.warning("ui_q non vidangee dans le delai imparti.")
    fanout_thread.stop()
    fanout_thread.join(timeout=STAGE_JOIN_TIMEOUT_S)
    transcript_logger.close()

    return PipelineResult(
        metrics=metrics,
        transcript_path=transcript_logger.path,
        translated_segments=fanout_thread.collected,
    )


def run_with_overlay(
    config: LiveCapConfig, *, input_wav: str | None, device: str
) -> int:
    """Live/overlay mode: builds the Qt app + LiveOverlay, wires SIGINT and
    window-close to Pipeline.shutdown(), and runs the Qt event loop on the
    main thread (PyQt6 requires this -- see overlay.py's own docstring)."""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    LiveOverlay = _build_live_overlay_class()

    audio_q = make_audio_queue()
    segment_q = make_segment_queue()
    ui_q = make_ui_queue()
    overlay_ui_q = make_ui_queue()
    partial_q: "queue.Queue[str]" = queue.Queue(maxsize=PARTIAL_QUEUE_MAXSIZE)
    metrics = PipelineMetrics()

    transcript_logger = TranscriptLogger(config.log_transcript)
    transcript_logger.open()

    pipeline = Pipeline()
    pipeline.transcript_logger = transcript_logger

    end_of_stream = threading.Event() if input_wav else None
    on_partial = _make_bounded_putter(partial_q)
    asr_thread = AsrSegmenterThread(
        audio_q,
        segment_q,
        metrics,
        on_partial=on_partial,
        device=device,
        end_of_stream=end_of_stream,
    )
    translator_thread = TranslatorThread(
        segment_q,
        ui_q,
        target_lang=config.target_lang,
        model=config.translation.model,
        context_turns=config.translation.context_turns,
    )
    fanout_thread = UiFanoutThread(
        ui_q, transcript_logger, metrics, overlay_ui_q=overlay_ui_q
    )

    pipeline.asr_thread = asr_thread
    pipeline.translator_thread = translator_thread
    pipeline.fanout_thread = fanout_thread

    if input_wav:
        pipeline.wav_feeder = WavFeederThread(input_wav, audio_q, end_of_stream)
    else:
        pipeline.capture = AudioCapture(config.devices, audio_q)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    overlay = LiveOverlay(config.overlay, overlay_ui_q, partial_q)

    def _shutdown_and_quit(*_args: object) -> None:
        pipeline.shutdown()
        app.quit()

    # SIGINT-under-Qt idiom (see module docstring for the source citations):
    # a short no-op QTimer keeps the interpreter ticking through bytecode so
    # Python's own SIGINT handler actually gets to run.
    signal.signal(signal.SIGINT, lambda *_: _shutdown_and_quit())
    sigint_pump = QTimer()
    sigint_pump.timeout.connect(lambda: None)
    sigint_pump.start(200)

    app.aboutToQuit.connect(pipeline.shutdown)
    # The overlay's X button / Ctrl+Alt+Q must tear down the whole pipeline,
    # not just hide the window (the capture and ASR threads are not owned by
    # Qt and would otherwise keep the process alive).
    overlay.close_requested.connect(_shutdown_and_quit)
    settings_panel = _wire_settings_panel(overlay, translator_thread, config)

    asr_thread.start()
    translator_thread.start()
    fanout_thread.start()
    # Wait for the model to finish loading before starting either audio
    # producer: starting earlier fills the bounded audio_q and drops audio
    # (see AsrSegmenterThread.ready docstring -- found while integrating
    # WavFeederThread against a real GPU run).
    if not asr_thread.ready.wait(timeout=ASR_READY_TIMEOUT_S):
        logger.warning(
            "AsrSegmenterThread pas pret apres %.0f s, demarrage de la capture quand meme (risque de perte audio).",
            ASR_READY_TIMEOUT_S,
        )

    try:
        if pipeline.capture is not None:
            pipeline.capture.start()
    except CaptureError as exc:
        logger.error("Capture audio impossible: %s", exc)
        pipeline.shutdown()
        return 1

    if pipeline.wav_feeder is not None:
        pipeline.wav_feeder.start()

    overlay.show()
    exit_code = app.exec()
    pipeline.shutdown()  # idempotent: no-op if aboutToQuit already ran it
    return exit_code


def run(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 success)."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("Configuration invalide (%s): %s", args.config, exc)
        return 1

    if args.no_overlay:
        if not args.input_wav:
            logger.error(
                "--no-overlay exige --input-wav (pas de mode headless live pour l'instant)."
            )
            return 1
        try:
            result = run_input_wav_pipeline(config, args.input_wav, device=args.device)
        except OllamaUnavailableError as exc:
            logger.error("Ollama indisponible: %s", exc)
            return 1
        logger.info(
            "Termine. %d segments, %d traductions, latence 1er segment->1ere traduction: %s ms. Transcript: %s",
            result.metrics.segments_seen,
            result.metrics.translations_seen,
            f"{result.metrics.first_segment_to_translation_ms:.1f}"
            if result.metrics.first_segment_to_translation_ms is not None
            else "n/a",
            result.transcript_path,
        )
        return 0

    try:
        return run_with_overlay(config, input_wav=args.input_wav, device=args.device)
    except OllamaUnavailableError as exc:
        logger.error("Ollama indisponible: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(run())
