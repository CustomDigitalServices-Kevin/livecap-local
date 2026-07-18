"""Tests for livecap.translog: TranscriptLogger (task 6's transcript file
writer, used by main.py's UiFanoutThread). Pure pytest, no GPU."""

from __future__ import annotations

from pathlib import Path

from livecap.contract import TranscriptSegment, TranslatedSegment
from livecap.translog import TranscriptLogger


def _translated(text: str, translation: str, lang: str = "en") -> TranslatedSegment:
    source = TranscriptSegment(text=text, lang=lang, t_start=0.0, t_end=1.0, final=True)
    return TranslatedSegment(source=source, text=translation, latency_ms=42.0)


def test_disabled_logger_is_a_full_noop(tmp_path: Path) -> None:
    logger = TranscriptLogger(enabled=False, directory=tmp_path)
    logger.open()
    logger.log(_translated("Hello", "Bonjour"))
    logger.close()

    assert logger.path is None
    assert list(tmp_path.iterdir()) == []


def test_open_creates_directory_and_timestamped_file(tmp_path: Path) -> None:
    target_dir = tmp_path / "transcripts"
    started = _dt_at(2026, 7, 18, 14, 32)
    logger = TranscriptLogger(enabled=True, directory=target_dir, now=started)
    logger.open()

    assert target_dir.is_dir()
    assert logger.path == target_dir / "2026-07-18_1432.txt"
    assert logger.path.is_file()
    logger.close()


def test_open_is_idempotent(tmp_path: Path) -> None:
    logger = TranscriptLogger(enabled=True, directory=tmp_path)
    logger.open()
    first_path = logger.path
    logger.open()  # must not reopen/truncate or change the path
    assert logger.path == first_path
    logger.close()


def test_log_writes_vo_and_translation_lines(tmp_path: Path) -> None:
    logger = TranscriptLogger(enabled=True, directory=tmp_path)
    logger.open()
    logger.log(_translated("Hello world", "Bonjour le monde", lang="en"))
    logger.close()

    content = logger.path.read_text(encoding="utf-8")
    assert "(en) Hello world" in content
    assert "-> Bonjour le monde" in content


def test_log_vo_only_when_translation_empty(tmp_path: Path) -> None:
    """Fallback (timeout/same-language) per CONTRACT.md: text == "" -> VO alone."""
    logger = TranscriptLogger(enabled=True, directory=tmp_path)
    logger.open()
    logger.log(_translated("Bonjour", "", lang="fr"))
    logger.close()

    content = logger.path.read_text(encoding="utf-8")
    assert "(fr) Bonjour" in content
    assert "->" not in content


def test_log_before_open_is_a_noop(tmp_path: Path) -> None:
    logger = TranscriptLogger(enabled=True, directory=tmp_path)
    logger.log(_translated("never written", "jamais ecrit"))  # open() never called
    assert list(tmp_path.iterdir()) == []


def test_close_is_idempotent(tmp_path: Path) -> None:
    logger = TranscriptLogger(enabled=True, directory=tmp_path)
    logger.open()
    logger.close()
    logger.close()  # must not raise


def test_context_manager_opens_and_closes(tmp_path: Path) -> None:
    with TranscriptLogger(enabled=True, directory=tmp_path) as logger:
        logger.log(_translated("one", "un"))
        path = logger.path
    assert path is not None
    assert "one" in path.read_text(encoding="utf-8")


def _dt_at(year: int, month: int, day: int, hour: int, minute: int):
    import datetime as dt

    return dt.datetime(year, month, day, hour, minute)
