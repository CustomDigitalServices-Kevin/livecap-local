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

OVERLAY_POSITIONS = (
    "top-left",
    "top-center",
    "top-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
)
TARGET_LANGS = ("fr", "en")


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
class LiveCapConfig:
    target_lang: str
    devices: DevicesConfig
    overlay: OverlayConfig
    log_transcript: bool


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

    return LiveCapConfig(
        target_lang=target_lang,
        devices=devices,
        overlay=overlay,
        log_transcript=log_transcript,
    )


def load_config(path: str | Path = "config.yaml") -> LiveCapConfig:
    """Read config.yaml from disk, parse it as YAML, and validate it.

    Raises ConfigError if the file is missing, empty, not a YAML mapping, or
    fails validation.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"fichier de configuration introuvable : {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} est vide ou n'est pas une table YAML valide")

    return parse_config(raw)
