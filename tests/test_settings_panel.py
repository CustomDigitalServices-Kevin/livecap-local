"""Tests for livecap.settings_panel: initial state, emitted signals, and the
settings snapshot handed to save_settings_override.

The panel deliberately applies nothing itself -- it only emits -- so these
tests need neither Ollama nor a translator thread.
"""

from __future__ import annotations

from livecap.settings_panel import MAX_FONT_SIZE, MIN_FONT_SIZE, SettingsPanel
from livecap.translator import MAX_CONTEXT_TURNS, MODEL_CHOICES


def _panel(**overrides: object) -> SettingsPanel:
    kwargs: dict[str, object] = {
        "model": MODEL_CHOICES[0][0],
        "context_turns": 0,
        "target_lang": "fr",
        "font_size": 22,
    }
    kwargs.update(overrides)
    return SettingsPanel(**kwargs)  # type: ignore[arg-type]


def test_panel_reflects_initial_settings(qtbot: object) -> None:
    panel = _panel(model=MODEL_CHOICES[1][0], context_turns=3, target_lang="en")
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    assert panel.model_combo.currentData() == MODEL_CHOICES[1][0]
    assert panel.context_spin.value() == 3
    assert panel.lang_combo.currentData() == "en"


def test_panel_lists_every_model_choice(qtbot: object) -> None:
    panel = _panel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    listed = [panel.model_combo.itemData(i) for i in range(panel.model_combo.count())]
    assert listed == [choice[0] for choice in MODEL_CHOICES]


def test_unknown_model_does_not_crash_the_panel(qtbot: object) -> None:
    """A hand-edited config.yaml pointing at an unlisted model must still open."""
    panel = _panel(model="some:model-not-in-the-catalogue")
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    # Falls back to the first entry rather than raising or showing an empty combo.
    assert panel.model_combo.currentIndex() == 0


def test_context_spin_is_bounded(qtbot: object) -> None:
    panel = _panel(context_turns=999)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    assert panel.context_spin.maximum() == MAX_CONTEXT_TURNS
    assert panel.context_spin.value() == MAX_CONTEXT_TURNS


def test_font_size_is_bounded(qtbot: object) -> None:
    panel = _panel(font_size=999)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    assert panel.font_spin.maximum() == MAX_FONT_SIZE
    assert panel.font_spin.value() == MAX_FONT_SIZE
    assert panel.font_spin.minimum() == MIN_FONT_SIZE


def test_changing_model_emits_signal(qtbot: object) -> None:
    panel = _panel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    emitted: list[str] = []
    panel.model_changed.connect(emitted.append)

    panel.model_combo.setCurrentIndex(1)

    assert emitted == [MODEL_CHOICES[1][0]]


def test_changing_context_emits_signal(qtbot: object) -> None:
    panel = _panel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    emitted: list[int] = []
    panel.context_turns_changed.connect(emitted.append)

    panel.context_spin.setValue(2)

    assert emitted == [2]


def test_changing_lang_emits_signal(qtbot: object) -> None:
    panel = _panel(target_lang="fr")
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    emitted: list[str] = []
    panel.target_lang_changed.connect(emitted.append)

    index = panel.lang_combo.findData("en")
    panel.lang_combo.setCurrentIndex(index)

    assert emitted == ["en"]


def test_changing_font_emits_signal(qtbot: object) -> None:
    panel = _panel(font_size=22)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    emitted: list[int] = []
    panel.font_size_changed.connect(emitted.append)

    panel.font_spin.setValue(30)

    assert emitted == [30]


def test_current_settings_shape_matches_config_sections(qtbot: object) -> None:
    """The snapshot must nest exactly like config.yaml, or the merge misplaces keys."""
    panel = _panel(model=MODEL_CHOICES[1][0], context_turns=4, target_lang="en")
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.font_spin.setValue(28)

    assert panel.current_settings() == {
        "target_lang": "en",
        "overlay": {"font_size": 28},
        "translation": {
            "model": MODEL_CHOICES[1][0],
            "context_turns": 4,
        },
    }


def test_model_hint_follows_selection(qtbot: object) -> None:
    panel = _panel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    assert panel.model_hint.text() == MODEL_CHOICES[0][2]

    panel.model_combo.setCurrentIndex(1)

    assert panel.model_hint.text() == MODEL_CHOICES[1][2]
