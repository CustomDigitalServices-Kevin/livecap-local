"""End-to-end test for the full LiveCap Local pipeline (task 6 acceptance
criteria): replay a known FR wav and a known EN wav through
main.run_input_wav_pipeline() -- capture -> asr+segmenter -> translator ->
overlay-fanout/transcript, exactly as --input-wav --no-overlay drives it --
and check both the VO transcript AND its translation for expected keywords,
plus the transcript file actually written to disk.

GPU + Ollama required (both mark this @pytest.mark.gpu, registered in
pytest.ini): the ASR model needs CUDA, and translation needs a running
Ollama server on :11434 with qwen3:4b-instruct pulled. Ollama's health is
checked once at collection time via check_ollama_health(); every test here
is skipped (not failed) with an explicit reason if it is not reachable, per
the task-6 instruction to document that skip rather than fail the suite.

Wav fixtures are tests/fixtures/test_fr.wav and tests/fixtures/test_en.wav,
copied verbatim from the POC (C:\\dev\\Antigravity\\tools\\livecap-poc,
2026-07-18) -- same files tests/test_asr_integration.py already replays for
the ASR+segmenter-only integration test, but through main.py's full
pipeline here instead.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

import main
from livecap.config import DevicesConfig, LiveCapConfig, OverlayConfig
from livecap.translator import check_ollama_health

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OLLAMA_UNAVAILABLE_REASON = (
    "Ollama non demarre localement sur :11434 (health check /api/tags)"
)

# Keyword sets the task-6 brief asks for (reunion/meeting, SharePoint,
# migration, Power Apps), checked on BOTH the VO transcript and its
# translation. "power app(s)" is split singular/plural per direction: the
# reference audio (tests/fixtures/test_fr.wav, test_en.wav, from the POC)
# genuinely says "Power App" (singular) in the FR recording and "Power Apps"
# (plural) in the EN one -- confirmed against tests/test_asr_integration.py's
# own already-validated FR ("power app") / EN ("power apps") keyword lists,
# and against Ollama's translation output, which preserves whichever form
# was spoken rather than normalizing it.
FR_AUDIO_VO_KEYWORDS = ("réunion", "sharepoint", "migration", "power app")
FR_AUDIO_TRANSLATION_KEYWORDS = ("meeting", "sharepoint", "migration", "power app")
EN_AUDIO_VO_KEYWORDS = ("meeting", "sharepoint", "migration", "power apps")
EN_AUDIO_TRANSLATION_KEYWORDS = ("réunion", "sharepoint", "migration", "power apps")

# Ceiling for the first-final-segment -> first-translation latency: must
# comfortably clear one Ollama call (TRANSLATE_TIMEOUT_S = 5s in
# livecap/translator.py) plus scheduling slack, without being so loose it
# stops catching a real regression.
MAX_FIRST_TRANSLATION_LATENCY_MS = 6000.0

pytestmark = pytest.mark.skipif(
    not check_ollama_health(), reason=OLLAMA_UNAVAILABLE_REASON
)


def _config(target_lang: str) -> LiveCapConfig:
    return LiveCapConfig(
        target_lang=target_lang,
        devices=DevicesConfig(loopback="auto", mic="none"),
        overlay=OverlayConfig(
            position="bottom-center", font_size=22, click_through=False, history=3
        ),
        log_transcript=True,
    )


def _vo_text(result: main.PipelineResult) -> str:
    return " ".join(seg.source.text for seg in result.translated_segments)


def _translation_text(result: main.PipelineResult) -> str:
    return " ".join(seg.text for seg in result.translated_segments if seg.text)


def _assert_keywords(haystack: str, keywords: tuple[str, ...], label: str) -> None:
    lowered = haystack.lower()
    missing = [kw for kw in keywords if kw not in lowered]
    assert not missing, f"{label}: mots-cles absents {missing!r} dans {haystack!r}"


@pytest.mark.gpu
def test_e2e_french_wav_transcribed_and_translated_to_english() -> None:
    """FR wav -> VO francais + traduction anglaise, loggees, mots-cles presents des deux cotes."""
    result = main.run_input_wav_pipeline(
        _config(target_lang="en"), FIXTURES_DIR / "test_fr.wav", device="cuda"
    )

    assert result.translated_segments, "aucun segment traduit produit pour test_fr.wav"
    vo_text = _vo_text(result)
    translation_text = _translation_text(result)
    logger.info("FR->EN VO: %r", vo_text)
    logger.info("FR->EN traduction: %r", translation_text)

    _assert_keywords(vo_text, FR_AUDIO_VO_KEYWORDS, "VO francaise")
    _assert_keywords(
        translation_text, FR_AUDIO_TRANSLATION_KEYWORDS, "traduction anglaise"
    )

    latency_ms = result.metrics.first_segment_to_translation_ms
    assert latency_ms is not None, (
        "latence 1er segment -> 1ere traduction jamais mesuree"
    )
    logger.info(
        "FR->EN latence 1er segment final -> 1ere traduction: %.1f ms", latency_ms
    )
    assert latency_ms < MAX_FIRST_TRANSLATION_LATENCY_MS

    assert result.transcript_path is not None and result.transcript_path.is_file()
    transcript_content = result.transcript_path.read_text(encoding="utf-8").lower()
    for keyword in FR_AUDIO_VO_KEYWORDS:
        assert keyword in transcript_content, (
            f"transcript fichier: mot-cle {keyword!r} absent"
        )


@pytest.mark.gpu
def test_e2e_english_wav_transcribed_and_translated_to_french() -> None:
    """EN wav -> VO anglais + traduction francaise, loggees, mots-cles presents des deux cotes."""
    result = main.run_input_wav_pipeline(
        _config(target_lang="fr"), FIXTURES_DIR / "test_en.wav", device="cuda"
    )

    assert result.translated_segments, "aucun segment traduit produit pour test_en.wav"
    vo_text = _vo_text(result)
    translation_text = _translation_text(result)
    logger.info("EN->FR VO: %r", vo_text)
    logger.info("EN->FR traduction: %r", translation_text)

    _assert_keywords(vo_text, EN_AUDIO_VO_KEYWORDS, "VO anglaise")
    _assert_keywords(
        translation_text, EN_AUDIO_TRANSLATION_KEYWORDS, "traduction francaise"
    )

    latency_ms = result.metrics.first_segment_to_translation_ms
    assert latency_ms is not None, (
        "latence 1er segment -> 1ere traduction jamais mesuree"
    )
    logger.info(
        "EN->FR latence 1er segment final -> 1ere traduction: %.1f ms", latency_ms
    )
    assert latency_ms < MAX_FIRST_TRANSLATION_LATENCY_MS

    assert result.transcript_path is not None and result.transcript_path.is_file()
    transcript_content = result.transcript_path.read_text(encoding="utf-8").lower()
    for keyword in EN_AUDIO_VO_KEYWORDS:
        assert keyword in transcript_content, (
            f"transcript fichier: mot-cle {keyword!r} absent"
        )


@pytest.mark.gpu
def test_cli_input_wav_no_overlay_exits_zero(tmp_path: Path) -> None:
    """Exercises the real CLI entry point (python main.py --input-wav ... --no-overlay),
    not just the internal function -- covers argparse + run()'s error handling +
    logging setup, which the two tests above (direct function calls) skip.

    Uses a throwaway config.yaml (target_lang=en, log_transcript=false) in tmp_path
    to avoid depending on / mutating the repo's own config.yaml.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "target_lang: en\n"
        "devices:\n  loopback: auto\n  mic: none\n"
        "overlay:\n  position: bottom-center\n  font_size: 22\n  click_through: false\n  history: 3\n"
        "log_transcript: false\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            "--config",
            str(config_path),
            "--input-wav",
            str(FIXTURES_DIR / "test_fr.wav"),
            "--no-overlay",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    logger.info("CLI stdout/stderr:\n%s", proc.stderr)
    assert proc.returncode == 0, f"exit code {proc.returncode}, stderr:\n{proc.stderr}"
    assert "Termine." in proc.stderr
