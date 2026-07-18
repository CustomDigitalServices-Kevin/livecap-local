"""Tests for livecap.overlay: geometry math, config application, queue
consumption/history, and pause via pytest-qt (qtbot).
"""

from __future__ import annotations

import queue

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence

from livecap.config import OverlayConfig
from livecap.contract import TranscriptSegment, TranslatedSegment, make_ui_queue
from livecap.overlay import Overlay, compute_geometry

SCREEN_W = 1920
SCREEN_H = 1080


def _config(**overrides: object) -> OverlayConfig:
    base = dict(position="bottom-center", font_size=22, click_through=False, history=3)
    base.update(overrides)
    return OverlayConfig(**base)  # type: ignore[arg-type]


def _translated(text: str, translation: str, lang: str = "en") -> TranslatedSegment:
    source = TranscriptSegment(text=text, lang=lang, t_start=0.0, t_end=1.0, final=True)
    return TranslatedSegment(source=source, text=translation, latency_ms=42.0)


# --- compute_geometry (pure function, no Qt widget needed) ------------------


def test_compute_geometry_top_left() -> None:
    x, y = compute_geometry("top-left", 200, 100, SCREEN_W, SCREEN_H, margin=24)
    assert (x, y) == (24, 24)


def test_compute_geometry_top_center() -> None:
    x, y = compute_geometry("top-center", 200, 100, SCREEN_W, SCREEN_H, margin=24)
    assert (x, y) == ((SCREEN_W - 200) // 2, 24)


def test_compute_geometry_top_right() -> None:
    x, y = compute_geometry("top-right", 200, 100, SCREEN_W, SCREEN_H, margin=24)
    assert (x, y) == (SCREEN_W - 24 - 200, 24)


def test_compute_geometry_bottom_left() -> None:
    x, y = compute_geometry("bottom-left", 200, 100, SCREEN_W, SCREEN_H, margin=24)
    assert (x, y) == (24, SCREEN_H - 24 - 100)


def test_compute_geometry_bottom_center() -> None:
    x, y = compute_geometry("bottom-center", 200, 100, SCREEN_W, SCREEN_H, margin=24)
    assert (x, y) == ((SCREEN_W - 200) // 2, SCREEN_H - 24 - 100)


def test_compute_geometry_bottom_right() -> None:
    x, y = compute_geometry("bottom-right", 200, 100, SCREEN_W, SCREEN_H, margin=24)
    assert (x, y) == (SCREEN_W - 24 - 200, SCREEN_H - 24 - 100)


def test_compute_geometry_bottom_anchor_fixed_as_height_grows() -> None:
    """Growing height must keep the bottom edge (y + height) fixed."""
    x1, y1 = compute_geometry("bottom-center", 200, 100, SCREEN_W, SCREEN_H, margin=24)
    x2, y2 = compute_geometry("bottom-center", 200, 250, SCREEN_W, SCREEN_H, margin=24)
    assert y1 + 100 == y2 + 250


def test_compute_geometry_invalid_position_raises() -> None:
    with pytest.raises(ValueError, match="position"):
        compute_geometry("middle", 100, 100, SCREEN_W, SCREEN_H)


# --- Overlay widget: config application -------------------------------------


def test_overlay_applies_window_flags(qtbot: object) -> None:
    ui_q = make_ui_queue()
    overlay = Overlay(_config(click_through=False), ui_q)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]

    flags = overlay.windowFlags()
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert flags & Qt.WindowType.Tool
    assert not (flags & Qt.WindowType.WindowTransparentForInput)
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    overlay.shutdown()


def test_overlay_click_through_sets_transparent_for_input(qtbot: object) -> None:
    ui_q = make_ui_queue()
    overlay = Overlay(_config(click_through=True), ui_q)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]

    assert overlay.windowFlags() & Qt.WindowType.WindowTransparentForInput
    overlay.shutdown()


# --- Overlay widget: ui_q consumption / history -----------------------------


def test_overlay_renders_segment_from_queue(qtbot: object) -> None:
    ui_q = make_ui_queue()
    overlay = Overlay(_config(), ui_q, poll_interval_ms=20)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]

    seg = _translated("Hello world", "Bonjour le monde")
    ui_q.put(seg)

    qtbot.waitUntil(lambda: overlay._current is seg, timeout=2000)  # type: ignore[attr-defined]

    texts = [line.text for line in overlay._render_lines]
    assert "Hello world" in texts
    assert "Bonjour le monde" in texts
    overlay.shutdown()


def test_overlay_fallback_vo_only_segment_renders_single_line(qtbot: object) -> None:
    ui_q = make_ui_queue()
    overlay = Overlay(_config(), ui_q, poll_interval_ms=20)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]

    seg = _translated("Bonjour", "")  # empty text = fallback (timeout/same-language)
    ui_q.put(seg)
    qtbot.waitUntil(lambda: overlay._current is seg, timeout=2000)  # type: ignore[attr-defined]

    assert len(overlay._render_lines) == 1
    assert overlay._render_lines[0].text == "Bonjour"
    overlay.shutdown()


def test_overlay_moves_previous_current_into_history(qtbot: object) -> None:
    ui_q = make_ui_queue()
    overlay = Overlay(_config(history=3), ui_q, poll_interval_ms=20)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]

    seg1 = _translated("one", "un")
    ui_q.put(seg1)
    qtbot.waitUntil(lambda: overlay._current is seg1, timeout=2000)  # type: ignore[attr-defined]

    seg2 = _translated("two", "deux")
    ui_q.put(seg2)
    qtbot.waitUntil(lambda: overlay._current is seg2, timeout=2000)  # type: ignore[attr-defined]

    assert overlay._history == [seg1]
    overlay.shutdown()


def test_overlay_history_bounded_by_config(qtbot: object) -> None:
    ui_q = make_ui_queue()
    overlay = Overlay(_config(history=1), ui_q, poll_interval_ms=20)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]

    segments = [_translated(f"seg{i}", f"trad{i}") for i in range(3)]
    for seg in segments:
        ui_q.put(seg)
        qtbot.waitUntil(lambda seg=seg: overlay._current is seg, timeout=2000)  # type: ignore[attr-defined]

    # Only the most recently displaced segment (seg1) should remain in history.
    assert overlay._history == [segments[1]]
    assert overlay._current is segments[2]
    overlay.shutdown()


def test_overlay_history_zero_keeps_only_current(qtbot: object) -> None:
    ui_q = make_ui_queue()
    overlay = Overlay(_config(history=0), ui_q, poll_interval_ms=20)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]

    seg1 = _translated("one", "un")
    ui_q.put(seg1)
    qtbot.waitUntil(lambda: overlay._current is seg1, timeout=2000)  # type: ignore[attr-defined]
    seg2 = _translated("two", "deux")
    ui_q.put(seg2)
    qtbot.waitUntil(lambda: overlay._current is seg2, timeout=2000)  # type: ignore[attr-defined]

    assert overlay._history == []
    overlay.shutdown()


# --- Overlay widget: pause/resume shortcut ---------------------------------


def test_overlay_pause_stops_consuming_queue(qtbot: object) -> None:
    ui_q = make_ui_queue()
    overlay = Overlay(_config(), ui_q, poll_interval_ms=20)
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]
    overlay.show()
    qtbot.waitExposed(overlay)  # type: ignore[attr-defined]

    overlay.toggle_pause()
    assert overlay._paused is True

    seg = _translated("Hello", "Bonjour")
    ui_q.put(seg)
    qtbot.wait(150)  # type: ignore[attr-defined]
    assert overlay._current is None
    assert not ui_q.empty()

    overlay.toggle_pause()
    assert overlay._paused is False
    qtbot.waitUntil(lambda: overlay._current is seg, timeout=2000)  # type: ignore[attr-defined]
    overlay.shutdown()


def test_overlay_shortcut_configured_and_wired_to_toggle_pause(qtbot: object) -> None:
    """Shortcut wiring, exercised without a real key event.

    A genuine qtbot.keyClick() only reaches QShortcut if Qt considers the
    window/application "active" (WindowShortcut/ApplicationShortcut
    context). In this sandboxed environment QApplication never receives
    real OS window activation (isActiveWindow()/QApplication.activeWindow()
    stay False/None even right after show()+waitExposed), so any such test
    would be exercising the test harness's window-manager access rather than
    this module's logic. What IS verified here, without depending on
    activation: the shortcut carries the configured key sequence, and its
    activated signal is wired to toggle_pause (emitting it directly, the
    same signal QShortcut itself would emit once Qt's activation gate lets
    the key event through).
    """
    ui_q = make_ui_queue()
    overlay = Overlay(_config(), ui_q, poll_interval_ms=20, pause_shortcut="Ctrl+Alt+P")
    qtbot.addWidget(overlay)  # type: ignore[attr-defined]

    assert overlay._pause_shortcut.key() == QKeySequence("Ctrl+Alt+P")
    assert overlay._pause_shortcut.context() == Qt.ShortcutContext.ApplicationShortcut

    overlay._pause_shortcut.activated.emit()
    assert overlay._paused is True

    overlay.shutdown()
