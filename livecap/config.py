"""Load and validate config.yaml for LiveCap Local.

Schema (canonical example in config.yaml, prose in CONTRACT.md):
    target_lang: "fr" | "en"
    devices:
      loopback: "auto" | <device name substring>
      mic: "auto" | "none" | <device name substring>
    overlay:
      position: one of OVERLAY_POSITIONS
      font_size: int > 0
      click_through: bool
      history: int >= 0
    log_transcript: bool
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# livecap.translator owns the model catalogue and the context cap; it imports
# only livecap.contract, so importing it here creates no cycle.
from livecap.translator import MAX_CONTEXT_TURNS
from livecap.translator import MODEL as DEFAULT_MODEL

OVERLAY_POSITIONS = (
    "top-left",
    "top-center",
    "top-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
)
TARGET_LANGS = ("fr", "en")

#: Where the overlay's gear panel persists what the user picked. Kept apart
#: from config.yaml because rewriting that file through yaml.safe_dump would
#: strip its comments, which are its documentation.
SETTINGS_OVERRIDE_FILE = "settings.local.yaml"


class ConfigError(ValueError):
    """Raised when config.yaml is missing a required key or has an invalid value."""


@dataclass(frozen=True)
class DevicesConfig:
    loopback: str
    mic: str


@dataclass(frozen=True)
class OverlayConfig:
    position: str
    font_size: int
    click_through: bool
    history: int


@dataclass(frozen=True)
class TranslationConfig:
    """Settings the overlay's gear panel can change at runtime.

    Defaults match the shipped behaviour: the fast 4B model and no dialogue
    context, so an existing config.yaml without a `translation:` section keeps
    working unchanged.
    """

    model: str = DEFAULT_MODEL
    context_turns: int = 0


@dataclass(frozen=True)
class LiveCapConfig:
    target_lang: str
    devices: DevicesConfig
    overlay: OverlayConfig
    log_transcript: bool
    translation: TranslationConfig = TranslationConfig()


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        label = f"{section}.{key}" if section else key
        raise ConfigError(f"{label} manquant dans config.yaml")
    return mapping[key]


def _parse_devices(raw: dict[str, Any]) -> DevicesConfig:
    loopback = _require(raw, "loopback", "devices")
    mic = _require(raw, "mic", "devices")
    if not isinstance(loopback, str) or not loopback:
        raise ConfigError(
            "devices.loopback doit etre une chaine non vide ('auto' ou nom de peripherique)"
        )
    if not isinstance(mic, str) or not mic:
        raise ConfigError(
            "devices.mic doit etre une chaine non vide ('auto', 'none' ou nom de peripherique)"
        )
    return DevicesConfig(loopback=loopback, mic=mic)


def _parse_overlay(raw: dict[str, Any]) -> OverlayConfig:
    position = _require(raw, "position", "overlay")
    font_size = _require(raw, "font_size", "overlay")
    click_through = _require(raw, "click_through", "overlay")
    history = _require(raw, "history", "overlay")

    if position not in OVERLAY_POSITIONS:
        raise ConfigError(
            f"overlay.position={position!r} invalide, attendu un de {OVERLAY_POSITIONS}"
        )
    if not isinstance(font_size, int) or isinstance(font_size, bool) or font_size <= 0:
        raise ConfigError("overlay.font_size doit etre un entier > 0")
    if not isinstance(click_through, bool):
        raise ConfigError("overlay.click_through doit etre un booleen")
    if not isinstance(history, int) or isinstance(history, bool) or history < 0:
        raise ConfigError("overlay.history doit etre un entier >= 0")

    return OverlayConfig(
        position=position,
        font_size=font_size,
        click_through=click_through,
        history=history,
    )


def _parse_translation(raw: dict[str, Any]) -> TranslationConfig:
    """Parse the optional `translation:` section (absent = shipped defaults)."""
    model = raw.get("model", DEFAULT_MODEL)
    context_turns = raw.get("context_turns", 0)

    if not isinstance(model, str) or not model:
        raise ConfigError("translation.model doit etre une chaine non vide")
    if (
        not isinstance(context_turns, int)
        or isinstance(context_turns, bool)
        or not 0 <= context_turns <= MAX_CONTEXT_TURNS
    ):
        raise ConfigError(
            f"translation.context_turns doit etre un entier entre 0 et {MAX_CONTEXT_TURNS}"
        )

    return TranslationConfig(model=model, context_turns=context_turns)


def parse_config(raw: dict[str, Any]) -> LiveCapConfig:
    """Validate an already-parsed config mapping and return a typed LiveCapConfig.

    Raises ConfigError with a precise message on the first missing key or
    invalid value encountered.
    """
    target_lang = _require(raw, "target_lang", "")
    if target_lang not in TARGET_LANGS:
        raise ConfigError(f"target_lang={target_lang!r} invalide, attendu 'fr' ou 'en'")

    devices_raw = _require(raw, "devices", "")
    if not isinstance(devices_raw, dict):
        raise ConfigError("devices doit etre une table (loopback, mic)")
    devices = _parse_devices(devices_raw)

    overlay_raw = _require(raw, "overlay", "")
    if not isinstance(overlay_raw, dict):
        raise ConfigError(
            "overlay doit etre une table (position, font_size, click_through, history)"
        )
    overlay = _parse_overlay(overlay_raw)

    log_transcript = _require(raw, "log_transcript", "")
    if not isinstance(log_transcript, bool):
        raise ConfigError("log_transcript doit etre un booleen")

    # Optional section: a config.yaml predating the settings panel is valid.
    translation_raw = raw.get("translation", {})
    if not isinstance(translation_raw, dict):
        raise ConfigError("translation doit etre une table (model, context_turns)")
    translation = _parse_translation(translation_raw)

    return LiveCapConfig(
        target_lang=target_lang,
        devices=devices,
        overlay=overlay,
        log_transcript=log_transcript,
        translation=translation,
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return `base` with `override` applied, recursing into nested tables."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path = "config.yaml",
    overrides_path: str | Path | None = SETTINGS_OVERRIDE_FILE,
) -> LiveCapConfig:
    """Read config.yaml, apply the user overrides file, and validate the result.

    `overrides_path` (settings.local.yaml by default) holds what the overlay's
    gear panel saved. It is merged on top of config.yaml and may be absent.
    Pass overrides_path=None to load config.yaml alone.

    Raises ConfigError if config.yaml is missing, empty, not a YAML mapping, or
    if the merged result fails validation.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"fichier de configuration introuvable : {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} est vide ou n'est pas une table YAML valide")

    if overrides_path is not None:
        override_file = Path(overrides_path)
        if override_file.is_file():
            with override_file.open("r", encoding="utf-8") as f:
                overrides = yaml.safe_load(f)
            if isinstance(overrides, dict):
                raw = _deep_merge(raw, overrides)
            elif overrides is not None:
                raise ConfigError(f"{override_file} n'est pas une table YAML valide")

    return parse_config(raw)


def save_settings_override(
    settings: dict[str, Any],
    overrides_path: str | Path = SETTINGS_OVERRIDE_FILE,
    base_path: str | Path = "config.yaml",
) -> Path:
    """Persist the gear panel's choices, merged into any existing overrides.

    Written to a SEPARATE file rather than back into config.yaml on purpose:
    round-tripping config.yaml through yaml.safe_dump would strip every
    comment in it, and those comments are the file's documentation.

    Returns the path written. Raises OSError on write failure -- the caller
    decides whether a failed save is fatal (it is not: the in-memory settings
    already took effect).
    """
    override_file = Path(overrides_path)
    existing: dict[str, Any] = {}
    if override_file.is_file():
        with override_file.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            existing = loaded

    merged = _deep_merge(existing, settings)
    # Validate before writing, so a bad value can never persist into a file
    # that would then break the next startup.
    with Path(base_path).open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    parse_config(_deep_merge(base if isinstance(base, dict) else {}, merged))

    tmp = override_file.with_suffix(override_file.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(
            "# Reglages enregistres depuis le panneau de l'overlay (roue dentee).\n"
            "# Fusionne par-dessus config.yaml au demarrage. Supprimer ce fichier\n"
            "# remet les valeurs de config.yaml.\n"
        )
        yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=True)
    tmp.replace(override_file)  # atomic: never leaves a half-written file
    return override_file
