"""Settings panel opened by the overlay's gear button.

A frameless always-on-top popup listing what can be changed mid-meeting:
translation model, dialogue-context depth, target language, and subtitle
font size. It owns no state of its own -- every change is pushed out through
a signal, and the host (main.py) decides what to apply and what to persist.
That keeps this module free of any dependency on the translator thread, so
it stays testable without Ollama.

Layout uses plain Qt widgets rather than the hand-painted style of
overlay.py: this is a form, not a subtitle surface, and widgets bring
keyboard navigation and accessibility for free. It is styled dark to match
the overlay it hangs off.

Kept OUT of overlay.py deliberately: overlay.py is on the hot path (it
repaints on every segment) and is already the largest module here.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from livecap.config import TARGET_LANGS
from livecap.translator import MAX_CONTEXT_TURNS, MODEL_CHOICES

logger = logging.getLogger(__name__)

MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 72

_LANG_LABELS = {"fr": "Francais", "en": "Anglais"}

STYLESHEET = """
QFrame#settingsRoot {
    background-color: #131313;
    border: 1px solid #3a3a3a;
    border-radius: 10px;
}
QLabel { color: #e8e8e8; }
QLabel#hint { color: #9a9a9a; }
QLabel#title { font-weight: bold; }
QComboBox, QSpinBox {
    background-color: #1f1f1f;
    color: #ffffff;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 3px 6px;
}
QComboBox QAbstractItemView {
    background-color: #1f1f1f;
    color: #ffffff;
    selection-background-color: #3d5a80;
}
QPushButton {
    background-color: #262626;
    color: #ffffff;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 4px 12px;
}
QPushButton:hover { background-color: #333333; }
"""


class SettingsPanel(QWidget):
    """Popup form. Emits one signal per change; applies nothing itself."""

    model_changed = pyqtSignal(str)
    context_turns_changed = pyqtSignal(int)
    target_lang_changed = pyqtSignal(str)
    font_size_changed = pyqtSignal(int)
    reset_geometry_requested = pyqtSignal()

    def __init__(
        self,
        *,
        model: str,
        context_turns: int,
        target_lang: str,
        font_size: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setStyleSheet(STYLESHEET)

        root = QFrame(self)
        root.setObjectName("settingsRoot")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        grid = QGridLayout(root)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        title = QLabel("Reglages LiveCap")
        title.setObjectName("title")
        grid.addWidget(title, 0, 0, 1, 2)

        grid.addWidget(QLabel("Modele"), 1, 0)
        self.model_combo = QComboBox()
        for model_id, label, description in MODEL_CHOICES:
            self.model_combo.addItem(f"{label} - {model_id}", model_id)
            self.model_combo.setItemData(
                self.model_combo.count() - 1, description, Qt.ItemDataRole.ToolTipRole
            )
        self._select_by_data(self.model_combo, model)
        self.model_combo.currentIndexChanged.connect(self._emit_model)
        grid.addWidget(self.model_combo, 1, 1)

        self.model_hint = QLabel(self._model_description(model))
        self.model_hint.setObjectName("hint")
        self.model_hint.setWordWrap(True)
        grid.addWidget(self.model_hint, 2, 0, 1, 2)

        grid.addWidget(QLabel("Contexte"), 3, 0)
        self.context_spin = QSpinBox()
        self.context_spin.setRange(0, MAX_CONTEXT_TURNS)
        self.context_spin.setValue(max(0, min(context_turns, MAX_CONTEXT_TURNS)))
        self.context_spin.setSuffix(" tours")
        self.context_spin.valueChanged.connect(self.context_turns_changed.emit)
        grid.addWidget(self.context_spin, 3, 1)

        context_hint = QLabel(
            "Phrases precedentes envoyees au modele. Aide surtout les termes "
            "ambigus (board = conseil). 0 = comportement d'origine."
        )
        context_hint.setObjectName("hint")
        context_hint.setWordWrap(True)
        grid.addWidget(context_hint, 4, 0, 1, 2)

        grid.addWidget(QLabel("Traduire vers"), 5, 0)
        self.lang_combo = QComboBox()
        for lang in TARGET_LANGS:
            self.lang_combo.addItem(_LANG_LABELS.get(lang, lang), lang)
        self._select_by_data(self.lang_combo, target_lang)
        self.lang_combo.currentIndexChanged.connect(self._emit_lang)
        grid.addWidget(self.lang_combo, 5, 1)

        grid.addWidget(QLabel("Taille du texte"), 6, 0)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(MIN_FONT_SIZE, MAX_FONT_SIZE)
        self.font_spin.setValue(max(MIN_FONT_SIZE, min(font_size, MAX_FONT_SIZE)))
        self.font_spin.setSuffix(" px")
        self.font_spin.valueChanged.connect(self.font_size_changed.emit)
        grid.addWidget(self.font_spin, 6, 1)

        reset_button = QPushButton("Reinitialiser position et taille")
        reset_button.clicked.connect(self.reset_geometry_requested.emit)
        grid.addWidget(reset_button, 7, 0, 1, 2)

        close_button = QPushButton("Fermer ce panneau")
        close_button.clicked.connect(self.hide)
        grid.addWidget(close_button, 8, 0, 1, 2)

        self.setFixedWidth(360)

    @staticmethod
    def _select_by_data(combo: QComboBox, value: str) -> None:
        """Select the entry whose userData is `value`, or leave index 0.

        A config pointing at a model absent from MODEL_CHOICES (hand-edited
        config.yaml) must not crash the panel -- it simply shows the first
        entry, and nothing is emitted until the user actually picks one.
        """
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _model_description(model_id: str) -> str:
        for candidate, _label, description in MODEL_CHOICES:
            if candidate == model_id:
                return description
        return ""

    def _emit_model(self) -> None:
        model_id = self.model_combo.currentData()
        if not model_id:
            return
        self.model_hint.setText(self._model_description(model_id))
        self.model_changed.emit(model_id)

    def _emit_lang(self) -> None:
        lang = self.lang_combo.currentData()
        if lang:
            self.target_lang_changed.emit(lang)

    def current_settings(self) -> dict[str, object]:
        """Snapshot of the form, in the shape save_settings_override expects."""
        return {
            "target_lang": self.lang_combo.currentData(),
            "overlay": {"font_size": self.font_spin.value()},
            "translation": {
                "model": self.model_combo.currentData(),
                "context_turns": self.context_spin.value(),
            },
        }

    def show_next_to(self, anchor: QWidget) -> None:
        """Show the panel just above/below `anchor`, kept inside the screen."""
        self.adjustSize()
        screen = anchor.screen()
        available = (
            screen.availableGeometry() if screen is not None else anchor.geometry()
        )

        x = anchor.x() + anchor.width() - self.width()
        # Prefer above the overlay; fall back below when there is no room.
        y = anchor.y() - self.height() - 8
        if y < available.top():
            y = anchor.y() + anchor.height() + 8

        x = max(available.left(), min(x, available.right() - self.width()))
        y = max(available.top(), min(y, available.bottom() - self.height()))
        self.move(x, y)
        self.show()
        self.raise_()
