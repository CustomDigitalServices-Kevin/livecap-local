"""Stabilizes the AsrToken stream (livecap/asr.py) into TranscriptSegment
(livecap/contract.py, pushed onto segment_q).

Segmentation rule (CONTRACT.md / PRD): a segment is stabilized on terminal
punctuation (. ! ?) OR on a silence run > 700 ms. Silence is counted in
AUDIO time (number of consecutive AsrToken.is_silence frames x 80 ms), not
wall-clock time: asr.py's Mimi frame is exactly 80 ms of audio per step
regardless of how fast the GPU runs the step, so this stays correct both
live (where audio_q is paced in real time by capture.py) and in a fast
file-replay (main.py --input-wav, task 6 / tests/test_asr_integration.py),
where the GPU can process many frames per wall-clock millisecond -- a
wall-clock silence threshold would simply never fire in that mode.

Partial (unstable) text is exposed only through an optional callback for
future progressive display (task 5); per CONTRACT.md line 37 it must never
be built into a TranscriptSegment or reach segment_q.

Language detection: the model (kyutai/stt-1b-en_fr) does not emit any
explicit language signal (verified: no language special token in its
sentencepiece vocab, and neither the HF model card nor the transformers
docs mention one). detect_lang() below is therefore a lightweight fr/en
heuristic (stopwords + accented-character ratio), not real language
identification -- adequate for this project's fr/en-only scope.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from livecap.asr import AsrToken
from livecap.contract import TranscriptSegment

SILENCE_THRESHOLD_S = 0.700
FRAME_DURATION_S = 0.080  # matches asr.py's Mimi frame (24000 Hz / 12.5 Hz)
TERMINAL_PUNCTUATION = (".", "!", "?")

_WORD_RE = re.compile(r"[a-zàâäéèêëîïôöùûüçœ]+")

_FR_STOPWORDS = frozenset(
    {
        "le",
        "la",
        "les",
        "de",
        "des",
        "du",
        "un",
        "une",
        "et",
        "est",
        "vous",
        "nous",
        "que",
        "qui",
        "pour",
        "dans",
        "avec",
        "sur",
        "pas",
        "ne",
        "se",
        "ce",
        "cette",
        "ces",
        "au",
        "aux",
        "sont",
        "fait",
        "faire",
        "plus",
        "tout",
        "bien",
        "être",
        "avoir",
        "comme",
        "mais",
        "donc",
        "alors",
        "merci",
        "bonjour",
        "ensemble",
        "planning",
        "reunion",
        "réunion",
    }
)
_EN_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "is",
        "are",
        "of",
        "to",
        "in",
        "for",
        "with",
        "on",
        "not",
        "this",
        "that",
        "these",
        "those",
        "we",
        "you",
        "will",
        "be",
        "have",
        "as",
        "but",
        "so",
        "thank",
        "hello",
        "good",
        "today",
        "everyone",
        "joining",
        "schedule",
    }
)
_FR_ACCENTED_CHARS = "àâäéèêëîïôöùûüçœ"


def detect_lang(text: str) -> str:
    """Heuristic fr/en guess for a stabilized segment. Ties default to "en"."""
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    fr_score = sum(1 for w in words if w in _FR_STOPWORDS)
    fr_score += sum(lowered.count(c) for c in _FR_ACCENTED_CHARS) * 2
    en_score = sum(1 for w in words if w in _EN_STOPWORDS)
    return "fr" if fr_score > en_score else "en"


class Segmenter:
    """Consumes AsrToken one at a time, emits stabilized TranscriptSegment."""

    def __init__(self, on_partial: Callable[[str], None] | None = None) -> None:
        """on_partial: optional callback invoked with the current (unstable)
        buffered text every time it grows -- for future progressive display
        (task 5). Never used to build a TranscriptSegment."""
        self._on_partial = on_partial
        self._buffer_text = ""
        self._silence_s = 0.0
        self._t_start: float | None = None

    def feed(self, token: AsrToken) -> TranscriptSegment | None:
        """Consume one AsrToken; return a stabilized TranscriptSegment, or
        None if the current segment is still accumulating."""
        if token.is_silence:
            self._silence_s += FRAME_DURATION_S
            if self._buffer_text.strip() and self._silence_s >= SILENCE_THRESHOLD_S:
                return self._stabilize(token.timestamp)
            return None

        self._silence_s = 0.0
        was_empty = not self._buffer_text
        self._buffer_text += token.text
        if was_empty and self._buffer_text:
            self._t_start = token.timestamp
        if self._buffer_text and self._on_partial is not None:
            self._on_partial(self._buffer_text.strip())

        stripped = self._buffer_text.rstrip()
        if stripped and stripped[-1] in TERMINAL_PUNCTUATION:
            return self._stabilize(token.timestamp)
        return None

    def flush(self, timestamp: float) -> TranscriptSegment | None:
        """Force-stabilize any pending buffered text (end of stream)."""
        if self._buffer_text.strip():
            return self._stabilize(timestamp)
        return None

    def _stabilize(self, t_end: float) -> TranscriptSegment | None:
        text = self._buffer_text.strip()
        t_start = self._t_start if self._t_start is not None else t_end
        self._buffer_text = ""
        self._silence_s = 0.0
        self._t_start = None
        if not text:
            return None
        return TranscriptSegment(
            text=text, lang=detect_lang(text), t_start=t_start, t_end=t_end, final=True
        )
