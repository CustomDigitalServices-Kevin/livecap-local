"""Unit tests for main.py's integration logic that do NOT require a GPU or
Ollama: CLI parsing, wav chunking (downmix/resample), the UI fan-out
thread, the wav feeder thread, Pipeline shutdown idempotency, and the
LiveOverlay progressive-partial-line subclass (via pytest-qt, no real
overlay.py changes -- see main.py's module docstring).

GPU-requiring pipeline behavior (the real Kyutai model + live Ollama) is
covered by tests/test_e2e.py (@pytest.mark.gpu).
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import main
from livecap.config import OverlayConfig
from livecap.contract import TranscriptSegment, TranslatedSegment, make_ui_queue
from livecap.translog import TranscriptLogger

# --- CLI argument parsing ----------------------------------------------------


def test_parse_args_defaults() -> None:
    args = main.parse_args([])
    assert args.config == main.DEFAULT_CONFIG_PATH
    assert args.input_wav is None
    assert args.no_overlay is False
    assert args.device == "cuda"
    assert args.log_level == "INFO"


def test_parse_args_input_wav_and_no_overlay() -> None:
    args = main.parse_args(
        [
            "--config",
            "custom.yaml",
            "--input-wav",
            "foo.wav",
            "--no-overlay",
            "--device",
            "cpu",
        ]
    )
    assert args.config == "custom.yaml"
    assert args.input_wav == "foo.wav"
    assert args.no_overlay is True
    assert args.device == "cpu"


def test_parse_args_rejects_invalid_log_level() -> None:
    with pytest.raises(SystemExit):
        main.parse_args(["--log-level", "VERBOSE"])


# --- chunk_wav_file: pure signal-processing path, no GPU --------------------


def test_chunk_wav_file_slices_into_80ms_blocks_from_real_fixture() -> None:
    fixture = Path(__file__).parent / "fixtures" / "test_fr.wav"
    chunks = main.chunk_wav_file(fixture)

    assert len(chunks) > 1
    expected_frame_size = int(main.TARGET_RATE * main.BLOCK_MS / 1000)
    for chunk in chunks[:-1]:
        assert chunk.samples.shape == (expected_frame_size,)
        assert chunk.samples.dtype == np.float32
    assert chunks[-1].samples.size <= expected_frame_size
    # timestamps strictly increase (monotonic replay clock)
    for a, b in zip(chunks, chunks[1:]):
        assert b.timestamp >= a.timestamp


def test_chunk_wav_file_downmixes_and_resamples_non_native_wav(tmp_path: Path) -> None:
    """A stereo 8 kHz wav (neither the capture pipeline's native rate nor
    channel count) must come out mono at 16 kHz, matching AudioChunk's
    contract (livecap/contract.py)."""
    duration_s = 0.5
    src_rate = 8_000
    n = int(duration_s * src_rate)
    t = np.linspace(0, duration_s, n, endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 220 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    wav_path = tmp_path / "stereo_8k.wav"
    sf.write(str(wav_path), stereo, src_rate)

    chunks = main.chunk_wav_file(wav_path)

    assert all(chunk.samples.ndim == 1 for chunk in chunks)
    total_samples = sum(chunk.samples.size for chunk in chunks)
    expected = round(n * main.TARGET_RATE / src_rate)
    # resample_mono_to_16k (livecap/capture.py) rounds to the nearest sample count.
    assert abs(total_samples - expected) <= 1


# --- PipelineMetrics ---------------------------------------------------------


def test_pipeline_metrics_latency_none_until_both_timestamps_set() -> None:
    metrics = main.PipelineMetrics()
    assert metrics.first_segment_to_translation_ms is None

    metrics.first_final_segment_at = 10.0
    assert metrics.first_segment_to_translation_ms is None

    metrics.first_translation_at = 10.25
    assert metrics.first_segment_to_translation_ms == pytest.approx(250.0)


# --- WavFeederThread ---------------------------------------------------------


def test_wav_feeder_thread_pushes_all_chunks_and_sets_end_of_stream(
    tmp_path: Path,
) -> None:
    duration_s = 0.24  # 3 x 80ms blocks at 16 kHz
    n = int(duration_s * main.TARGET_RATE)
    signal = (
        0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, n, endpoint=False))
    ).astype(np.float32)
    wav_path = tmp_path / "tone_16k_mono.wav"
    sf.write(str(wav_path), signal, main.TARGET_RATE)

    audio_q: "queue.Queue" = queue.Queue()
    end_of_stream = threading.Event()
    feeder = main.WavFeederThread(wav_path, audio_q, end_of_stream)
    feeder.start()
    feeder.join(timeout=5.0)

    assert not feeder.is_alive()
    assert end_of_stream.is_set()
    total = 0
    while not audio_q.empty():
        total += audio_q.get_nowait().samples.size
    assert total == n


# --- UiFanoutThread ----------------------------------------------------------


def _translated(text: str, translation: str, lang: str = "en") -> TranslatedSegment:
    source = TranscriptSegment(text=text, lang=lang, t_start=0.0, t_end=1.0, final=True)
    return TranslatedSegment(source=source, text=translation, latency_ms=12.0)


def test_ui_fanout_thread_logs_and_forwards_to_overlay_queue(tmp_path: Path) -> None:
    ui_q: "queue.Queue" = queue.Queue()
    overlay_ui_q = make_ui_queue()
    transcript_logger = TranscriptLogger(enabled=True, directory=tmp_path)
    transcript_logger.open()
    metrics = main.PipelineMetrics()

    fanout = main.UiFanoutThread(
        ui_q, transcript_logger, metrics, overlay_ui_q=overlay_ui_q, poll_timeout=0.02
    )
    fanout.start()
    try:
        seg = _translated("Hello world", "Bonjour le monde")
        ui_q.put(seg)
        forwarded = overlay_ui_q.get(timeout=2.0)
        assert forwarded is seg
    finally:
        fanout.stop()
        fanout.join(timeout=2.0)
    transcript_logger.close()

    assert fanout.collected == [seg]
    assert metrics.first_translation_at is not None
    assert metrics.translations_seen == 1

    content = transcript_logger.path.read_text(encoding="utf-8")
    assert "Hello world" in content
    assert "Bonjour le monde" in content


def test_ui_fanout_thread_without_overlay_still_logs(tmp_path: Path) -> None:
    ui_q: "queue.Queue" = queue.Queue()
    transcript_logger = TranscriptLogger(enabled=False, directory=tmp_path)
    transcript_logger.open()  # no-op, disabled
    metrics = main.PipelineMetrics()

    fanout = main.UiFanoutThread(ui_q, transcript_logger, metrics, poll_timeout=0.02)
    fanout.start()
    try:
        seg = _translated("solo", "seul")
        ui_q.put(seg)
        deadline = time.monotonic() + 2.0
        while not fanout.collected and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        fanout.stop()
        fanout.join(timeout=2.0)

    assert fanout.collected == [seg]
    assert transcript_logger.path is None  # disabled logger never opens a file


# --- _wait_until_empty --------------------------------------------------------


def test_wait_until_empty_true_once_drained() -> None:
    q: "queue.Queue" = queue.Queue()
    q.put(1)

    def _drain_soon() -> None:
        time.sleep(0.05)
        q.get_nowait()

    threading.Thread(target=_drain_soon, daemon=True).start()
    assert main._wait_until_empty(q, timeout=2.0) is True


def test_wait_until_empty_false_on_timeout() -> None:
    q: "queue.Queue" = queue.Queue()
    q.put(1)
    assert main._wait_until_empty(q, timeout=0.1) is False


# --- Pipeline.shutdown() idempotency -----------------------------------------


class _FakeStage:
    def __init__(self) -> None:
        self.stop_calls = 0
        self.join_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1

    def is_alive(self) -> bool:
        return False


def test_pipeline_shutdown_idempotent(tmp_path: Path) -> None:
    pipeline = main.Pipeline()
    asr_stage = _FakeStage()
    translator_stage = _FakeStage()
    fanout_stage = _FakeStage()
    pipeline.asr_thread = asr_stage  # type: ignore[assignment]
    pipeline.translator_thread = translator_stage  # type: ignore[assignment]
    pipeline.fanout_thread = fanout_stage  # type: ignore[assignment]
    transcript_logger = TranscriptLogger(enabled=True, directory=tmp_path)
    transcript_logger.open()
    pipeline.transcript_logger = transcript_logger

    pipeline.shutdown()
    pipeline.shutdown()  # must be a no-op the second time

    assert asr_stage.stop_calls == 1
    assert translator_stage.stop_calls == 1
    assert fanout_stage.stop_calls == 1


# --- LiveOverlay: progressive partial VO line (pytest-qt, no GPU) ------------


def _overlay_config(**overrides: object) -> OverlayConfig:
    base = dict(position="bottom-center", font_size=22, click_through=False, history=3)
    base.update(overrides)
    return OverlayConfig(**base)  # type: ignore[arg-type]


def test_live_overlay_shows_partial_line_after_current_segment(qtbot: object) -> None:
    LiveOverlay = main._build_live_overlay_class()
    ui_q = make_ui_queue()
    partial_q: "queue.Queue[str]" = queue.Queue()
    overlay = LiveOverlay(_overlay_config(), ui_q, partial_q, poll_interval_ms=20)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]

    seg = _translated("Bonjour", "Hello")
    ui_q.put(seg)
    qtbot.waitUntil(lambda: overlay._current is seg, timeout=2000)  # type: ignore[attr-defined]

    partial_q.put("Comment allez-")
    qtbot.waitUntil(lambda: overlay._partial_text == "Comment allez-", timeout=2000)  # type: ignore[attr-defined]

    texts = [line.text for line in overlay._render_lines]
    assert texts[-1] == "Comment allez-"
    assert "Bonjour" in texts
    assert "Hello" in texts
    overlay.shutdown()


def test_live_overlay_clears_stale_partial_when_final_segment_lands(
    qtbot: object,
) -> None:
    LiveOverlay = main._build_live_overlay_class()
    ui_q = make_ui_queue()
    partial_q: "queue.Queue[str]" = queue.Queue()
    overlay = LiveOverlay(_overlay_config(), ui_q, partial_q, poll_interval_ms=20)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]

    partial_q.put("Bonjour à")
    qtbot.waitUntil(lambda: overlay._partial_text == "Bonjour à", timeout=2000)  # type: ignore[attr-defined]
    assert overlay._render_lines[-1].text == "Bonjour à"  # type: ignore[attr-defined]

    seg = _translated("Bonjour à tous.", "Hello everyone.")
    ui_q.put(seg)
    qtbot.waitUntil(lambda: overlay._current is seg, timeout=2000)  # type: ignore[attr-defined]

    assert overlay._partial_text == ""  # type: ignore[attr-defined]
    texts = [line.text for line in overlay._render_lines]
    assert texts.count("Bonjour à") == 0
    overlay.shutdown()


def test_live_overlay_partial_ignored_while_paused(qtbot: object) -> None:
    LiveOverlay = main._build_live_overlay_class()
    ui_q = make_ui_queue()
    partial_q: "queue.Queue[str]" = queue.Queue()
    overlay = LiveOverlay(_overlay_config(), ui_q, partial_q, poll_interval_ms=20)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]
    overlay.show()
    qtbot.waitExposed(overlay)  # type: ignore[attr-defined]

    overlay.toggle_pause()
    partial_q.put("ne doit pas s'afficher")
    qtbot.wait(150)  # type: ignore[attr-defined]

    assert overlay._partial_text == ""  # type: ignore[attr-defined]
    assert not partial_q.empty()
    overlay.shutdown()


# --- cablage du panneau de reglages (roue dentee) --------------------------


class _FakeTranslator:
    """Double du TranslatorThread: enregistre ce que le panneau lui demande."""

    def __init__(self) -> None:
        self.model: str | None = None
        self.context_turns: int | None = None
        self.target_lang: str | None = None
        self.cleared = 0

    def set_model(self, model: str) -> None:
        self.model = model

    def set_context_turns(self, turns: int) -> None:
        self.context_turns = turns

    def set_target_lang(self, lang: str) -> None:
        self.target_lang = lang

    def clear_context(self) -> None:
        self.cleared += 1


def _config_for_panel() -> "main.LiveCapConfig":
    from livecap.config import (
        DevicesConfig,
        LiveCapConfig,
        TranslationConfig,
    )

    return LiveCapConfig(
        target_lang="fr",
        devices=DevicesConfig(loopback="auto", mic="auto"),
        overlay=OverlayConfig(
            position="bottom-center", font_size=22, click_through=False, history=3
        ),
        log_transcript=False,
        translation=TranslationConfig(model="qwen3:4b-instruct", context_turns=0),
    )


@pytest.fixture
def panel_env(qtbot: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Overlay reel + faux translator, avec persistance redirigee vers tmp_path."""
    from livecap.overlay import Overlay

    overlay = Overlay(_config_for_panel().overlay, make_ui_queue(), poll_interval_ms=20)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]
    translator = _FakeTranslator()

    saved: list[dict] = []

    def fake_save(settings: dict, **kwargs: object) -> Path:
        saved.append(settings)
        return tmp_path / "settings.local.yaml"

    monkeypatch.setattr("livecap.config.save_settings_override", fake_save)
    panel = main._wire_settings_panel(overlay, translator, _config_for_panel())
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    yield overlay, translator, panel, saved
    overlay.shutdown()


def test_gear_signal_opens_and_closes_the_panel(panel_env) -> None:
    overlay, _translator, panel, _saved = panel_env
    assert not panel.isVisible()

    overlay.settings_requested.emit()
    assert panel.isVisible()

    overlay.settings_requested.emit()
    assert not panel.isVisible()


def test_panel_model_change_reaches_the_translator(panel_env) -> None:
    _overlay, translator, panel, saved = panel_env

    panel.model_combo.setCurrentIndex(1)

    from livecap.translator import MODEL_CHOICES

    assert translator.model == MODEL_CHOICES[1][0]
    # Changer de modele doit purger l'historique de l'ancien.
    assert translator.cleared == 1
    assert saved, "le changement doit etre persiste"


def test_panel_context_change_reaches_the_translator(panel_env) -> None:
    _overlay, translator, panel, saved = panel_env

    panel.context_spin.setValue(3)

    assert translator.context_turns == 3
    assert saved


def test_panel_lang_change_reaches_the_translator(panel_env) -> None:
    _overlay, translator, panel, saved = panel_env

    panel.lang_combo.setCurrentIndex(panel.lang_combo.findData("en"))

    assert translator.target_lang == "en"
    assert saved


def test_panel_font_change_reaches_the_overlay(panel_env) -> None:
    overlay, _translator, panel, saved = panel_env

    panel.font_spin.setValue(30)

    assert overlay._font.pointSize() == 30
    assert saved


def test_panel_reset_button_clears_manual_geometry(panel_env) -> None:
    overlay, _translator, panel, _saved = panel_env
    overlay.move(400, 300)
    overlay._remember_manual_anchor()
    overlay._manual_size = (500, 200)

    panel.reset_geometry_requested.emit()

    assert overlay._manual_anchor is None
    assert overlay._manual_size is None


def test_failed_persistence_does_not_break_the_setting(
    panel_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un disque plein ne doit pas empecher le reglage de s'appliquer.

    Le compteur d'appels n'est pas decoratif: sans lui, ce test passe meme
    si la sauvegarde n'est jamais tentee (verifie en cassant volontairement
    la resolution de save_settings_override), donc il n'assurerait rien.
    """
    _overlay, translator, panel, _saved = panel_env
    calls: list[dict] = []

    def boom(settings: dict, **kwargs: object) -> Path:
        calls.append(settings)
        raise OSError("disque plein")

    monkeypatch.setattr("livecap.config.save_settings_override", boom)

    panel.context_spin.setValue(2)

    assert calls, "la sauvegarde doit avoir ete tentee"
    # L'OSError est absorbee: le reglage reste applique.
    assert translator.context_turns == 2
