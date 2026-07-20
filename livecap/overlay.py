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
than the overlay are WRAPPED at word boundaries (see wrap_text), so one
segment may occupy several visual rows. Earlier versions elided them with
"..." instead, which was tenable only while the panel sized itself to its
content; once it became resizable, narrowing it silently hid speech.

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
import math
import queue
from dataclasses import dataclass, replace

from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, QTimer, pyqtSignal
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
MAX_HEIGHT_RATIO = 0.45  # fraction of screen height the overlay may occupy
MARGIN_PX = 24
LINE_SPACING_PX = 6
OUTLINE_WIDTH_RATIO = 0.10  # stroke width as a fraction of font_size
VO_COLOR = QColor(255, 255, 255)
TRANSLATION_COLOR = QColor(255, 221, 87)
OUTLINE_COLOR = QColor(0, 0, 0)
BACKGROUND_COLOR = QColor(0, 0, 0, 235)  # opaque black panel behind the text
BACKGROUND_RADIUS_PX = 10
CURRENT_ALPHA = 255
HISTORY_ALPHA = 200
CHROME_COLOR = QColor(255, 255, 255, 130)  # close button + resize grip strokes
CHROME_HOVER_COLOR = QColor(255, 120, 120)
CLOSE_BUTTON_PX = 18  # side of the square hit area, top-right of the panel
GEAR_BUTTON_PX = 18  # side of the square hit area, left of the close button
GEAR_TEETH = 8
RESIZE_GRIP_PX = 18  # side of the square hit area, bottom-right of the panel
MIN_WIDTH_PX = 240
MIN_HEIGHT_PX = 60
DEFAULT_PAUSE_SHORTCUT = "Ctrl+Alt+P"
DEFAULT_RESET_POSITION_SHORTCUT = "Ctrl+Alt+R"
DEFAULT_QUIT_SHORTCUT = "Ctrl+Alt+Q"
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


def wrap_text(text: str, metrics: QFontMetrics, max_width: int) -> list[str]:
    """Split `text` into visual lines that each fit within `max_width` px.

    Breaks on spaces; a single word wider than max_width is broken between
    characters rather than left to overflow (long URLs, agglutinated words).
    Returns [""] for empty input so an empty line still occupies one row.

    Wrapping rather than eliding: the overlay used to elide with "..." on the
    assumption its width always matched its content, which stopped being true
    once the panel became resizable -- narrowing it silently hid speech.
    """
    if not text:
        return [""]
    if metrics.horizontalAdvance(text) <= max_width:
        return [text]

    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}" if current else word
        if metrics.horizontalAdvance(candidate) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        # The word alone may still overflow: break it by characters.
        while metrics.horizontalAdvance(word) > max_width:
            cut = len(word)
            while cut > 1 and metrics.horizontalAdvance(word[:cut]) > max_width:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    if current:
        lines.append(current)
    return lines or [""]


class Overlay(QWidget):
    """Borderless always-on-top subtitle overlay. See module docstring."""

    #: Emitted when the user asks to close the overlay (X button or quit
    #: shortcut). main.py connects it to QApplication.quit so the whole
    #: pipeline shuts down through the existing aboutToQuit handler.
    #: Closing the widget is NOT relied on to end the process: Qt only
    #: auto-quits on "the last visible primary window"
    #: (https://doc.qt.io/qt-6/qguiapplication.html#quitOnLastWindowClosed-prop),
    #: and --no-overlay/embedded callers may own other windows, so the quit
    #: is requested explicitly rather than inferred from window bookkeeping.
    close_requested = pyqtSignal()

    #: Emitted when the gear button is clicked. main.py owns the settings
    #: panel, because applying a setting means reaching into the translator
    #: thread -- something this widget must not know about.
    settings_requested = pyqtSignal()

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
        # Same lines after word-wrapping to the current width; this is what is
        # scrolled and painted (see _relayout).
        self._wrapped_lines: list[_RenderLine] = []
        self._paused = False
        # Scroll state: _scroll_offset is the index of the first VISIBLE render
        # line. _stick_to_bottom keeps the newest segment in view until the user
        # scrolls up; scrolling back to the bottom re-arms the auto-follow.
        self._scroll_offset = 0
        self._visible_lines = 1
        self._stick_to_bottom = True
        # Drag state: _manual_anchor is None until the user moves the panel by
        # hand; from then on _relayout honours it instead of config.position.
        self._drag_origin = None
        self._manual_anchor: tuple[int, int] | None = None
        # Resize state: _manual_size overrides the content-driven width/height
        # once the user drags the bottom-right grip.
        self._resize_origin: tuple[QPoint, int, int] | None = None
        self._manual_size: tuple[int, int] | None = None
        self._close_hovered = False
        self._gear_hovered = False
        self.setMouseTracking(True)  # needed for close-button hover feedback

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

        self._reset_position_shortcut = QShortcut(
            QKeySequence(DEFAULT_RESET_POSITION_SHORTCUT), self
        )
        self._reset_position_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._reset_position_shortcut.activated.connect(self.reset_position)

        self._quit_shortcut = QShortcut(QKeySequence(DEFAULT_QUIT_SHORTCUT), self)
        self._quit_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._quit_shortcut.activated.connect(self.request_close)

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

        # The panel grows with the transcript until MAX_HEIGHT_RATIO of the
        # screen, then stops growing and becomes a scrollable viewport. Once
        # the user has resized it by hand, their size wins outright.
        if self._manual_size is not None:
            width, manual_height = self._manual_size
            max_visible = max(1, (manual_height - MARGIN_PX) // line_height)
        else:
            max_height = int(screen_geo.height() * MAX_HEIGHT_RATIO)
            max_visible = max(1, (max_height - MARGIN_PX) // line_height)

        # Wrap AFTER the width is settled: a narrowed panel turns one segment
        # into several visual rows, and it is those rows that scrolling,
        # height and painting all count -- not the logical lines.
        text_width = max(1, width - 2 * MARGIN_PX)
        self._wrapped_lines = [
            _RenderLine(chunk, line.color)
            for line in self._render_lines
            for chunk in wrap_text(line.text, metrics, text_width)
        ]

        total_lines = max(len(self._wrapped_lines), 1)
        self._visible_lines = min(total_lines, max_visible)
        max_offset = max(0, len(self._wrapped_lines) - self._visible_lines)
        if self._stick_to_bottom:
            self._scroll_offset = max_offset
        else:
            self._scroll_offset = min(self._scroll_offset, max_offset)
        if self._manual_size is not None:
            height = self._manual_size[1]
        else:
            height = self._visible_lines * line_height + MARGIN_PX

        self.resize(width, height)
        if self._manual_anchor is not None:
            # The user dragged the panel: keep THEIR anchor, not the configured
            # one, so the block still grows away from the same edge.
            ax, ay = self._manual_anchor
            y = ay if self._config.position.startswith("top") else ay - height
            self.move(ax, y)
            return
        x, y = compute_geometry(
            self._config.position,
            width,
            height,
            screen_geo.width(),
            screen_geo.height(),
        )
        self.move(screen_geo.x() + x, screen_geo.y() + y)

    def _remember_manual_anchor(self) -> None:
        """Freeze the current position as the anchor used by _relayout."""
        if self._config.position.startswith("top"):
            self._manual_anchor = (self.x(), self.y())
        else:
            self._manual_anchor = (self.x(), self.y() + self.height())

    def reset_position(self) -> None:
        """Forget the dragged position AND size, back to the configured defaults."""
        self._manual_anchor = None
        self._manual_size = None
        self._relayout()
        self.update()

    def set_font_size(self, font_size: int) -> None:
        """Change the subtitle font size live (settings panel)."""
        if font_size <= 0:
            raise ValueError("font_size doit etre un entier > 0")
        if font_size == self._font.pointSize():
            return
        self._font = QFont("Segoe UI", font_size, QFont.Weight.Bold)
        # OUTLINE_WIDTH_RATIO is applied to config.font_size in paintEvent, so
        # the config object has to follow or the outline would keep the old
        # thickness. OverlayConfig is frozen, hence the replacement.
        self._config = replace(self._config, font_size=font_size)
        self._relayout()
        self.update()

    def request_close(self) -> None:
        """Ask the host application to shut down, then hide the overlay."""
        logger.info("Fermeture demandee depuis l'overlay")
        self.close_requested.emit()
        self.close()

    def close_button_rect(self) -> QRect:
        """Hit area of the X button (top-right corner of the panel)."""
        return QRect(
            self.width() - CLOSE_BUTTON_PX - MARGIN_PX // 3,
            MARGIN_PX // 3,
            CLOSE_BUTTON_PX,
            CLOSE_BUTTON_PX,
        )

    def gear_button_rect(self) -> QRect:
        """Hit area of the settings gear, immediately left of the X."""
        close_rect = self.close_button_rect()
        return QRect(
            close_rect.left() - GEAR_BUTTON_PX - MARGIN_PX // 4,
            close_rect.top(),
            GEAR_BUTTON_PX,
            GEAR_BUTTON_PX,
        )

    def resize_grip_rect(self) -> QRect:
        """Hit area of the resize grip (bottom-right corner of the panel)."""
        return QRect(
            self.width() - RESIZE_GRIP_PX,
            self.height() - RESIZE_GRIP_PX,
            RESIZE_GRIP_PX,
            RESIZE_GRIP_PX,
        )

    def mousePressEvent(self, event) -> None:  # noqa: ANN001 (Qt override signature)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        if self.close_button_rect().contains(pos):
            self.request_close()
            event.accept()
            return
        if self.gear_button_rect().contains(pos):
            self.settings_requested.emit()
            event.accept()
            return
        if self.resize_grip_rect().contains(pos):
            self._resize_origin = (
                event.globalPosition().toPoint(),
                self.width(),
                self.height(),
            )
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            event.accept()
            return
        self._drag_origin = (
            event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        )
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001 (Qt override signature)
        pos = event.position().toPoint()
        if self._resize_origin is not None:
            if not (event.buttons() & Qt.MouseButton.LeftButton):
                return
            origin, start_w, start_h = self._resize_origin
            delta = event.globalPosition().toPoint() - origin
            self._manual_size = (
                max(MIN_WIDTH_PX, start_w + delta.x()),
                max(MIN_HEIGHT_PX, start_h + delta.y()),
            )
            self._relayout()
            self.update()
            event.accept()
            return
        if self._drag_origin is not None:
            if not (event.buttons() & Qt.MouseButton.LeftButton):
                return
            self.move(event.globalPosition().toPoint() - self._drag_origin)
            event.accept()
            return
        # Idle hover: highlight the X and show the resize cursor on the grip.
        hovered = self.close_button_rect().contains(pos)
        gear_hovered = self.gear_button_rect().contains(pos)
        if hovered != self._close_hovered or gear_hovered != self._gear_hovered:
            self._close_hovered = hovered
            self._gear_hovered = gear_hovered
            self.update()
        if self.resize_grip_rect().contains(pos):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif hovered or gear_hovered:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001 (Qt override signature)
        if self._resize_origin is not None:
            self._resize_origin = None
            self.unsetCursor()
            # Resizing from the bottom-right moves the bottom edge, so the
            # anchor must follow or the next relayout would snap it back.
            self._remember_manual_anchor()
            event.accept()
            return
        if self._drag_origin is None:
            return
        self._drag_origin = None
        self.unsetCursor()
        self._remember_manual_anchor()
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: ANN001 (Qt override signature)
        if self._close_hovered or self._gear_hovered:
            self._close_hovered = False
            self._gear_hovered = False
            self.update()
        self.unsetCursor()

    def scroll_by(self, lines: int) -> None:
        """Scroll the transcript by `lines` render lines (negative = older)."""
        max_offset = max(0, len(self._wrapped_lines) - self._visible_lines)
        new_offset = max(0, min(max_offset, self._scroll_offset + lines))
        if new_offset == self._scroll_offset:
            return
        self._scroll_offset = new_offset
        # Re-arm auto-follow only once the user is back at the newest line.
        self._stick_to_bottom = new_offset >= max_offset
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: ANN001 (Qt override signature)
        # angleDelta is in 1/8 degree units; one notch = 120 units = 3 lines,
        # matching the Qt default scroll step.
        notches = event.angleDelta().y() / 120.0
        self.scroll_by(int(-notches * 3) or (-1 if notches > 0 else 1))
        event.accept()

    def paintEvent(self, event: object) -> None:  # noqa: ARG002 (Qt override signature)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(BACKGROUND_COLOR)
        painter.drawRoundedRect(self.rect(), BACKGROUND_RADIUS_PX, BACKGROUND_RADIUS_PX)
        metrics = QFontMetrics(self._font)
        stroke_width = max(1, int(self._config.font_size * OUTLINE_WIDTH_RATIO))
        pen = QPen(OUTLINE_COLOR)
        pen.setWidth(stroke_width)

        y = metrics.ascent() + MARGIN_PX // 2
        visible = self._wrapped_lines[
            self._scroll_offset : self._scroll_offset + self._visible_lines
        ]
        for line in visible:
            path = QPainterPath()
            path.addText(MARGIN_PX, y, self._font, line.text)
            painter.setPen(pen)
            painter.setBrush(line.color)
            painter.drawPath(path)
            y += self._line_height

        self._paint_chrome(painter)
        painter.end()

    def _paint_chrome(self, painter: QPainter) -> None:
        """Draw the close button and the resize grip over the panel."""
        painter.setBrush(Qt.BrushStyle.NoBrush)

        close_rect = self.close_button_rect()
        close_pen = QPen(CHROME_HOVER_COLOR if self._close_hovered else CHROME_COLOR)
        close_pen.setWidth(2)
        painter.setPen(close_pen)
        inset = 4
        painter.drawLine(
            close_rect.left() + inset,
            close_rect.top() + inset,
            close_rect.right() - inset,
            close_rect.bottom() - inset,
        )
        painter.drawLine(
            close_rect.right() - inset,
            close_rect.top() + inset,
            close_rect.left() + inset,
            close_rect.bottom() - inset,
        )

        gear_rect = self.gear_button_rect()
        gear_pen = QPen(CHROME_HOVER_COLOR if self._gear_hovered else CHROME_COLOR)
        gear_pen.setWidth(2)
        painter.setPen(gear_pen)
        center = QPointF(gear_rect.center()) + QPointF(0.5, 0.5)
        outer_r = gear_rect.width() / 2.0 - 1.0
        inner_r = outer_r * 0.55
        # Teeth as radial spokes plus a hub circle: at 18 px a literal gear
        # outline turns to mush, spokes stay legible.
        for tooth in range(GEAR_TEETH):
            angle = math.pi * 2 * tooth / GEAR_TEETH
            dx, dy = math.cos(angle), math.sin(angle)
            painter.drawLine(
                QPointF(center.x() + dx * inner_r, center.y() + dy * inner_r),
                QPointF(center.x() + dx * outer_r, center.y() + dy * outer_r),
            )
        painter.drawEllipse(center, inner_r * 0.85, inner_r * 0.85)

        grip_rect = self.resize_grip_rect()
        grip_pen = QPen(CHROME_COLOR)
        grip_pen.setWidth(2)
        painter.setPen(grip_pen)
        for offset in (4, 9, 14):
            painter.drawLine(
                grip_rect.right() - offset,
                grip_rect.bottom() - 2,
                grip_rect.right() - 2,
                grip_rect.bottom() - offset,
            )

    def closeEvent(self, event: object) -> None:  # noqa: ARG002 (Qt override signature)
        self.shutdown()
