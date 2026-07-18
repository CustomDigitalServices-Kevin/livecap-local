"""PyQt6 overlay window for LiveCap Local.

Frameless, always-on-top, optionally click-through subtitle overlay that
renders the current VO + translated line plus a short history, driven by
ui_q (TranslatedSegment, see CONTRACT.md). Positioned/sized/styled from
livecap.config.OverlayConfig -- imported from there, never redefined here.

Threading: ui_q is filled by TranslatorThread from a background thread, but
Qt widgets may only be touched from the thread that owns them. This module
therefore never touches ui_q anywhere except a QTimer callback running on the
Qt main thread (see _poll_ui_queue); it never spins its own consumer thread.

Layout: segments are drawn in a single reading-order column, oldest history
line first, current segment last, regardless of the configured anchor
corner -- only the anchor point itself (computed by compute_geometry) is
kept fixed as the block grows/shrinks with new segments. Each segment
becomes 1 line (VO only, when translation is "" -- failed/timed-out/
same-language, per CONTRACT.md) or 2 lines (VO + translation). Lines wider
than the overlay are elided with "..." rather than wrapped: segmenter.py
stabilizes on punctuation/silence so segments are expected to be
sentence-sized, and full word-wrap is judged out of scope for a subtitle
overlay (documented trade-off, not a hidden gap).

Pause/resume: a QShortcut (default Ctrl+Alt+P) toggles a local pause flag
that stops draining ui_q. Its context is Qt.ShortcutContext.ApplicationShortcut
so it fires as long as the LiveCap Local process itself has input focus, but
it still cannot fire while a DIFFERENT application (e.g. Teams) is focused.
A true SYSTEM-WIDE hotkey (so pause also works while Teams has focus) would
need either the Win32 RegisterHotKey/WM_HOTKEY pair -- which
Qt6's own bug tracker documents as unreliable via QWidget.nativeEvent on
Qt 6.2-6.5 (https://forum.qt.io/topic/144158/global-hotkeys-and-nativeevent-is-not-working-in-qt6)
-- or a new third-party dependency (keyboard/pynput/pywin32), none of which
are in requirements.txt. Given that documented unreliability and the
no-new-dependency constraint from the task, this module ships the local
QShortcut only; global pause is a known, documented limitation (see also
README.md).
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass

from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QShortcut,
)
from PyQt6.QtWidgets import QWidget

from livecap.config import OVERLAY_POSITIONS, OverlayConfig
from livecap.contract import TranslatedSegment

logger = logging.getLogger(__name__)

MAX_WIDTH_RATIO = 0.75  # fraction of screen width the overlay may occupy
MARGIN_PX = 24
LINE_SPACING_PX = 6
OUTLINE_WIDTH_RATIO = 0.10  # stroke width as a fraction of font_size
VO_COLOR = QColor(255, 255, 255)
TRANSLATION_COLOR = QColor(255, 221, 87)
OUTLINE_COLOR = QColor(0, 0, 0)
CURRENT_ALPHA = 255
HISTORY_ALPHA = 150
DEFAULT_PAUSE_SHORTCUT = "Ctrl+Alt+P"
DEFAULT_POLL_INTERVAL_MS = 50
FALLBACK_SCREEN_RECT = QRect(0, 0, 1920, 1080)  # used only if no screen is reported


def compute_geometry(
    position: str,
    width: int,
    height: int,
    screen_width: int,
    screen_height: int,
    margin: int = MARGIN_PX,
) -> tuple[int, int]:
    """Return the (x, y) top-left corner keeping `position`'s anchor fixed.

    top-*: anchored at the top (y = margin), grows downward as height grows.
    bottom-*: anchored at the bottom (y = screen_height - margin - height).
    *-left / *-center / *-right: same idea horizontally.
    """
    if position not in OVERLAY_POSITIONS:
        raise ValueError(
            f"position={position!r} invalide, attendu un de {OVERLAY_POSITIONS}"
        )
    vertical, _, horizontal = position.partition("-")

    y = margin if vertical == "top" else screen_height - margin - height

    if horizontal == "left":
        x = margin
    elif horizontal == "center":
        x = (screen_width - width) // 2
    else:  # "right"
        x = screen_width - margin - width

    return x, y


@dataclass(frozen=True)
class _RenderLine:
    text: str
    color: QColor


class Overlay(QWidget):
    """Borderless always-on-top subtitle overlay. See module docstring."""

    def __init__(
        self,
        config: OverlayConfig,
        ui_q: "queue.Queue[TranslatedSegment]",
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        pause_shortcut: str = DEFAULT_PAUSE_SHORTCUT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._ui_q = ui_q
        self._current: TranslatedSegment | None = None
        self._history: list[TranslatedSegment] = []
        self._render_lines: list[_RenderLine] = []
        self._paused = False

        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        if config.click_through:
            flags |= Qt.WindowType.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Click-through overlays must never steal focus from Teams when shown.
        self.setAttribute(
            Qt.WidgetAttribute.WA_ShowWithoutActivating, config.click_through
        )

        self._font = QFont("Segoe UI", config.font_size, QFont.Weight.Bold)

        self._pause_shortcut = QShortcut(QKeySequence(pause_shortcut), self)
        # ApplicationShortcut rather than the default WindowShortcut: a
        # borderless Qt.WindowType.Tool overlay is not reliably reported as
        # the OS "active window" by every window manager/test harness, so
        # scope the shortcut to "the LiveCap Local process has input focus"
        # instead of "this exact top-level widget is active".
        self._pause_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._pause_shortcut.activated.connect(self.toggle_pause)

        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._poll_ui_queue)
        self._timer.start()

        self._relayout()

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        logger.info("Overlay pause=%s", self._paused)

    def shutdown(self) -> None:
        """Stop the polling timer. Call before closing/destroying the widget."""
        self._timer.stop()

    def _poll_ui_queue(self) -> None:
        if self._paused:
            return
        updated = False
        while True:
            try:
                segment = self._ui_q.get_nowait()
            except queue.Empty:
                break
            if self._current is not None:
                self._history.append(self._current)
                if self._config.history <= 0:
                    self._history.clear()
                elif len(self._history) > self._config.history:
                    self._history = self._history[-self._config.history :]
            self._current = segment
            updated = True
        if updated:
            self._relayout()
            self.update()

    def _iter_display_segments(self) -> list[tuple[TranslatedSegment, int]]:
        displayed = [(seg, HISTORY_ALPHA) for seg in self._history]
        if self._current is not None:
            displayed.append((self._current, CURRENT_ALPHA))
        return displayed

    def _build_render_lines(self) -> list[_RenderLine]:
        lines: list[_RenderLine] = []
        for seg, alpha in self._iter_display_segments():
            vo_color = QColor(VO_COLOR)
            vo_color.setAlpha(alpha)
            lines.append(_RenderLine(seg.source.text, vo_color))
            if seg.text:
                tr_color = QColor(TRANSLATION_COLOR)
                tr_color.setAlpha(alpha)
                lines.append(_RenderLine(seg.text, tr_color))
        return lines

    def _relayout(self) -> None:
        self._render_lines = self._build_render_lines()
        metrics = QFontMetrics(self._font)
        line_height = metrics.height() + LINE_SPACING_PX
        self._line_height = line_height

        screen = QGuiApplication.primaryScreen()
        screen_geo = (
            screen.availableGeometry() if screen is not None else FALLBACK_SCREEN_RECT
        )

        max_width = int(screen_geo.width() * MAX_WIDTH_RATIO)
        content_width = max(
            (metrics.horizontalAdvance(line.text) for line in self._render_lines),
            default=0,
        )
        width = (
            min(max_width, content_width + 2 * MARGIN_PX)
            if self._render_lines
            else max_width
        )
        height = max(len(self._render_lines), 1) * line_height + MARGIN_PX

        self.resize(width, height)
        x, y = compute_geometry(
            self._config.position,
            width,
            height,
            screen_geo.width(),
            screen_geo.height(),
        )
        self.move(screen_geo.x() + x, screen_geo.y() + y)

    def paintEvent(self, event: object) -> None:  # noqa: ARG002 (Qt override signature)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        metrics = QFontMetrics(self._font)
        stroke_width = max(1, int(self._config.font_size * OUTLINE_WIDTH_RATIO))
        pen = QPen(OUTLINE_COLOR)
        pen.setWidth(stroke_width)

        y = metrics.ascent() + MARGIN_PX // 2
        max_text_width = max(1, self.width() - 2 * MARGIN_PX)
        for line in self._render_lines:
            elided = metrics.elidedText(
                line.text, Qt.TextElideMode.ElideRight, max_text_width
            )
            path = QPainterPath()
            path.addText(MARGIN_PX, y, self._font, elided)
            painter.setPen(pen)
            painter.setBrush(line.color)
            painter.drawPath(path)
            y += self._line_height
        painter.end()

    def closeEvent(self, event: object) -> None:  # noqa: ARG002 (Qt override signature)
        self.shutdown()
