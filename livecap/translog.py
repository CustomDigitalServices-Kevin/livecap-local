"""Timestamped transcript logging for LiveCap Local (task 6, main.py).

Writes one line (VO) plus an optional second line (translation, when a
translation was actually produced) per TranslatedSegment consumed from
ui_q, to a UTF-8 text file named ``transcripts/YYYY-MM-DD_HHMM.txt`` --
the timestamp is wall-clock (``datetime.now()``), unlike
``TranscriptSegment.t_start``/``t_end`` which are ``time.monotonic()``
values (per ``livecap/contract.py``) and not meaningful to a human reading
the file after the fact.

Enabled by ``config.yaml``'s ``log_transcript: true`` (see CONTRACT.md and
``livecap/config.py``). When disabled, every method is a no-op so callers
in main.py do not need to branch on the flag themselves.

Thread-safety: ``log()`` is called from main.py's single UI fan-out thread
(the sole consumer of the real ``ui_q``, see main.py module docstring), so
no locking is required for ordering; a lock is still held around the
file write itself so ``log()`` stays safe to call from any thread if that
assumption ever changes, at negligible cost.
"""

from __future__ import annotations

import datetime as _dt
import logging
import threading
from pathlib import Path
from types import TracebackType

from livecap.contract import TranslatedSegment

logger = logging.getLogger(__name__)

DEFAULT_TRANSCRIPTS_DIR = Path("transcripts")
_FILENAME_FORMAT = "%Y-%m-%d_%H%M"
_LINE_TIME_FORMAT = "%H:%M:%S"


class TranscriptLogger:
    """Appends VO + translation lines to a timestamped transcript file.

    No-op when ``enabled`` is False (default ``config.yaml``), so main.py
    can always construct one and call ``log()`` unconditionally.
    """

    def __init__(
        self,
        enabled: bool,
        directory: str | Path = DEFAULT_TRANSCRIPTS_DIR,
        *,
        now: _dt.datetime | None = None,
    ) -> None:
        self._enabled = enabled
        self._directory = Path(directory)
        self._lock = threading.Lock()
        self._file = None
        self._path: Path | None = None
        self._now_override = now  # test seam: deterministic filename

    @property
    def path(self) -> Path | None:
        """Path of the open transcript file, or None if disabled/not opened."""
        return self._path

    def open(self) -> None:
        """Create transcripts/ (if needed) and open the timestamped file.

        No-op if disabled or already open. Raises OSError if the directory
        cannot be created or the file cannot be opened (caller's
        responsibility to surface -- there is no silent fallback for a
        transcript the user explicitly asked for).
        """
        if not self._enabled or self._file is not None:
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        started = self._now_override or _dt.datetime.now()
        self._path = self._directory / f"{started.strftime(_FILENAME_FORMAT)}.txt"
        self._file = self._path.open("a", encoding="utf-8")
        self._file.write(
            f"# LiveCap Local -- transcript demarre {started.strftime(_LINE_TIME_FORMAT)}\n"
        )
        self._file.flush()
        logger.info("Transcript log: %s", self._path)

    def log(self, segment: TranslatedSegment) -> None:
        """Append one TranslatedSegment. No-op if disabled or not open."""
        if not self._enabled or self._file is None:
            return
        stamp = _dt.datetime.now().strftime(_LINE_TIME_FORMAT)
        source = segment.source
        with self._lock:
            self._file.write(f"[{stamp}] ({source.lang}) {source.text}\n")
            if segment.text:
                self._file.write(f"           -> {segment.text}\n")
            self._file.flush()

    def close(self) -> None:
        """Close the transcript file. No-op if disabled or not open. Idempotent."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> "TranscriptLogger":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
