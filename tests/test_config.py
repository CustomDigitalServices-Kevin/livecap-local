"""Tests for livecap.config: happy path, missing keys, invalid values."""

from __future__ import annotations

from pathlib import Path

import pytest

import yaml

from livecap.config import (
    ConfigError,
    load_config,
    parse_config,
    save_settings_override,
)
from livecap.translator import MAX_CONTEXT_TURNS
from livecap.translator import MODEL as DEFAULT_MODEL

VALID_RAW = {
    "target_lang": "fr",
    "devices": {"loopback": "auto", "mic": "auto"},
    "overlay": {
        "position": "bottom-center",
        "font_size": 22,
        "click_through": False,
        "history": 3,
    },
    "log_transcript": False,
}


def _valid_raw() -> dict:
    """Copie independante de VALID_RAW (les tests ci-dessous la mutent)."""
    return {
        "target_lang": VALID_RAW["target_lang"],
        "devices": dict(VALID_RAW["devices"]),
        "overlay": dict(VALID_RAW["overlay"]),
        "log_transcript": VALID_RAW["log_transcript"],
    }


def _with_override(section: str | None, key: str, value: object) -> dict:
    """Return a deep-ish copy of VALID_RAW with one key overridden."""
    raw = {
        "target_lang": VALID_RAW["target_lang"],
        "devices": dict(VALID_RAW["devices"]),
        "overlay": dict(VALID_RAW["overlay"]),
        "log_transcript": VALID_RAW["log_transcript"],
    }
    if section is None:
        raw[key] = value
    else:
        raw[section][key] = value
    return raw


def test_parse_config_happy_path() -> None:
    cfg = parse_config(VALID_RAW)
    assert cfg.target_lang == "fr"
    assert cfg.devices.loopback == "auto"
    assert cfg.devices.mic == "auto"
    assert cfg.overlay.position == "bottom-center"
    assert cfg.overlay.font_size == 22
    assert cfg.overlay.click_through is False
    assert cfg.overlay.history == 3
    assert cfg.log_transcript is False


def test_parse_config_target_lang_en() -> None:
    raw = _with_override(None, "target_lang", "en")
    assert parse_config(raw).target_lang == "en"


def test_parse_config_mic_none() -> None:
    raw = _with_override("devices", "mic", "none")
    assert parse_config(raw).devices.mic == "none"


@pytest.mark.parametrize(
    "section,key",
    [
        (None, "target_lang"),
        (None, "devices"),
        (None, "overlay"),
        (None, "log_transcript"),
        ("devices", "loopback"),
        ("devices", "mic"),
        ("overlay", "position"),
        ("overlay", "font_size"),
        ("overlay", "click_through"),
        ("overlay", "history"),
    ],
)
def test_parse_config_missing_key_raises(section: str | None, key: str) -> None:
    raw = {
        "target_lang": VALID_RAW["target_lang"],
        "devices": dict(VALID_RAW["devices"]),
        "overlay": dict(VALID_RAW["overlay"]),
        "log_transcript": VALID_RAW["log_transcript"],
    }
    if section is None:
        del raw[key]
    else:
        del raw[section][key]
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_parse_config_invalid_target_lang() -> None:
    raw = _with_override(None, "target_lang", "de")
    with pytest.raises(ConfigError, match="target_lang"):
        parse_config(raw)


def test_parse_config_invalid_overlay_position() -> None:
    raw = _with_override("overlay", "position", "middle")
    with pytest.raises(ConfigError, match="overlay.position"):
        parse_config(raw)


def test_parse_config_invalid_font_size_zero() -> None:
    raw = _with_override("overlay", "font_size", 0)
    with pytest.raises(ConfigError, match="font_size"):
        parse_config(raw)


def test_parse_config_invalid_font_size_bool() -> None:
    # bool is a subclass of int in Python; must be rejected explicitly.
    raw = _with_override("overlay", "font_size", True)
    with pytest.raises(ConfigError, match="font_size"):
        parse_config(raw)


def test_parse_config_invalid_history_negative() -> None:
    raw = _with_override("overlay", "history", -1)
    with pytest.raises(ConfigError, match="history"):
        parse_config(raw)


def test_parse_config_devices_not_a_table() -> None:
    raw = _with_override(None, "devices", "auto")
    with pytest.raises(ConfigError, match="devices"):
        parse_config(raw)


def test_parse_config_empty_loopback_string() -> None:
    raw = _with_override("devices", "loopback", "")
    with pytest.raises(ConfigError, match="loopback"):
        parse_config(raw)


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="introuvable"):
        load_config(tmp_path / "does-not-exist.yaml")


def test_load_config_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="vide"):
        load_config(empty)


def test_load_config_reads_repo_default(tmp_path: Path) -> None:
    """The repo's own config.yaml at the project root must be valid."""
    repo_config = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg = load_config(repo_config)
    assert cfg.target_lang in ("fr", "en")


# --- section translation et persistance des reglages ----------------------


def test_translation_section_defaults_when_absent() -> None:
    """Un config.yaml anterieur au panneau reste valide."""
    raw = _valid_raw()
    raw.pop("translation", None)
    config = parse_config(raw)

    assert config.translation.model == DEFAULT_MODEL
    assert config.translation.context_turns == 0


def test_translation_section_is_parsed() -> None:
    raw = _valid_raw()
    raw["translation"] = {"model": "qwen3:14b-q4_K_M", "context_turns": 3}
    config = parse_config(raw)

    assert config.translation.model == "qwen3:14b-q4_K_M"
    assert config.translation.context_turns == 3


def test_translation_context_turns_out_of_range_rejected() -> None:
    raw = _valid_raw()
    raw["translation"] = {"context_turns": MAX_CONTEXT_TURNS + 1}
    with pytest.raises(ConfigError, match="context_turns"):
        parse_config(raw)


def test_translation_model_must_be_non_empty_string() -> None:
    raw = _valid_raw()
    raw["translation"] = {"model": ""}
    with pytest.raises(ConfigError, match="translation.model"):
        parse_config(raw)


def test_overrides_file_is_merged_over_config(tmp_path: Path) -> None:
    base = tmp_path / "config.yaml"
    base.write_text(yaml.safe_dump(_valid_raw()), encoding="utf-8")
    override = tmp_path / "settings.local.yaml"
    override.write_text(
        yaml.safe_dump({"translation": {"context_turns": 4}, "target_lang": "en"}),
        encoding="utf-8",
    )

    config = load_config(base, overrides_path=override)

    assert config.target_lang == "en"
    assert config.translation.context_turns == 4
    # Les cles non surchargees viennent toujours de config.yaml.
    assert config.overlay.position == _valid_raw()["overlay"]["position"]


def test_missing_overrides_file_is_not_an_error(tmp_path: Path) -> None:
    base = tmp_path / "config.yaml"
    base.write_text(yaml.safe_dump(_valid_raw()), encoding="utf-8")

    config = load_config(base, overrides_path=tmp_path / "absent.yaml")

    assert config.target_lang == _valid_raw()["target_lang"]


def test_save_settings_override_round_trips(tmp_path: Path) -> None:
    base = tmp_path / "config.yaml"
    base.write_text(yaml.safe_dump(_valid_raw()), encoding="utf-8")
    override = tmp_path / "settings.local.yaml"

    save_settings_override(
        {"translation": {"model": "qwen3:14b-q4_K_M", "context_turns": 2}},
        overrides_path=override,
        base_path=base,
    )

    assert override.is_file()
    config = load_config(base, overrides_path=override)
    assert config.translation.model == "qwen3:14b-q4_K_M"
    assert config.translation.context_turns == 2


def test_save_settings_override_merges_with_existing(tmp_path: Path) -> None:
    """Enregistrer la police ne doit pas effacer le modele deja enregistre."""
    base = tmp_path / "config.yaml"
    base.write_text(yaml.safe_dump(_valid_raw()), encoding="utf-8")
    override = tmp_path / "settings.local.yaml"

    save_settings_override(
        {"translation": {"model": "qwen3:14b-q4_K_M"}},
        overrides_path=override,
        base_path=base,
    )
    save_settings_override(
        {"overlay": {"font_size": 30}}, overrides_path=override, base_path=base
    )

    config = load_config(base, overrides_path=override)
    assert config.translation.model == "qwen3:14b-q4_K_M"
    assert config.overlay.font_size == 30


def test_save_settings_override_rejects_invalid_value(tmp_path: Path) -> None:
    """Une valeur invalide ne doit jamais atterrir dans le fichier."""
    base = tmp_path / "config.yaml"
    base.write_text(yaml.safe_dump(_valid_raw()), encoding="utf-8")
    override = tmp_path / "settings.local.yaml"

    with pytest.raises(ConfigError):
        save_settings_override(
            {"translation": {"context_turns": 99}},
            overrides_path=override,
            base_path=base,
        )

    assert not override.exists()
