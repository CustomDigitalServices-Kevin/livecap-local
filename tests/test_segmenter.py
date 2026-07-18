"""Tests for livecap.segmenter: punctuation/silence stabilization rules,
empty tokens, partial vs final segments, language heuristic, edge cases.

Pure pytest, no GPU: AsrToken is a plain dataclass, so the Segmenter can be
driven directly with hand-built tokens without loading any model.
"""

from __future__ import annotations

from livecap.asr import AsrToken
from livecap.segmenter import FRAME_DURATION_S, Segmenter, detect_lang

SILENCE_FRAMES_UNDER_THRESHOLD = 8  # 8 * 80ms = 640ms < 700ms


def _text_token(text: str, t: float) -> AsrToken:
    return AsrToken(text=text, is_silence=False, timestamp=t)


def _silence_token(t: float) -> AsrToken:
    return AsrToken(text="", is_silence=True, timestamp=t)


def test_silence_only_never_emits() -> None:
    seg = Segmenter()
    result = None
    for i in range(50):
        result = seg.feed(_silence_token(i * FRAME_DURATION_S))
    assert result is None


def test_text_without_punctuation_and_short_silence_emits_nothing() -> None:
    seg = Segmenter()
    assert seg.feed(_text_token("Bonjour", 0.0)) is None
    for i in range(SILENCE_FRAMES_UNDER_THRESHOLD):
        assert seg.feed(_silence_token(1.0 + i * FRAME_DURATION_S)) is None


def test_silence_below_threshold_then_above_emits_exactly_at_threshold() -> None:
    seg = Segmenter()
    seg.feed(_text_token("Bonjour", 0.0))
    for i in range(SILENCE_FRAMES_UNDER_THRESHOLD):
        assert seg.feed(_silence_token(i * FRAME_DURATION_S)) is None
    # one more silence frame crosses 700ms (9 * 80ms = 720ms)
    result = seg.feed(_silence_token(SILENCE_FRAMES_UNDER_THRESHOLD * FRAME_DURATION_S))
    assert result is not None
    assert result.text == "Bonjour"
    assert result.final is True


def test_terminal_punctuation_emits_immediately_without_silence() -> None:
    seg = Segmenter()
    seg.feed(_text_token("Bonjour", 0.0))
    result = seg.feed(_text_token(".", 0.1))
    assert result is not None
    assert result.text == "Bonjour."
    assert result.final is True


def test_each_terminal_punctuation_mark_stabilizes() -> None:
    for punct in (".", "!", "?"):
        seg = Segmenter()
        seg.feed(_text_token("Hello", 0.0))
        result = seg.feed(_text_token(punct, 0.1))
        assert result is not None
        assert result.text == f"Hello{punct}"


def test_fragmented_tokens_concatenate_into_one_segment() -> None:
    seg = Segmenter()
    pieces = [" Bon", "jour", " à", " tous", "."]
    result = None
    for i, piece in enumerate(pieces):
        result = seg.feed(_text_token(piece, i * 0.1))
    assert result is not None
    assert result.text == "Bonjour à tous."


def test_multiple_segments_reset_buffer_between_stabilizations() -> None:
    seg = Segmenter()
    first = None
    for tok in [_text_token("Bonjour.", 0.0)]:
        first = seg.feed(tok)
    assert first is not None
    assert first.text == "Bonjour."

    second = None
    for tok in [_text_token("Merci.", 1.0)]:
        second = seg.feed(tok)
    assert second is not None
    assert second.text == "Merci."


def test_flush_stabilizes_pending_buffer() -> None:
    seg = Segmenter()
    seg.feed(_text_token("En cours", 0.0))
    assert seg.flush(1.0) is not None
    result = seg.flush(1.0)
    assert result is None  # buffer already emptied by the first flush


def test_flush_with_empty_buffer_returns_none() -> None:
    seg = Segmenter()
    assert seg.flush(0.0) is None


def test_t_start_set_on_first_non_empty_text_only() -> None:
    seg = Segmenter()
    # An empty-text, non-silence token (defensive edge case: a decoded piece
    # that renders to "") must not stamp a stale t_start before real text
    # arrives.
    seg.feed(AsrToken(text="", is_silence=False, timestamp=0.0))
    seg.feed(_text_token("Bonjour.", 5.0))
    result = seg.feed(_silence_token(5.1))
    # already stabilized by the period in the previous call
    assert result is None

    seg2 = Segmenter()
    seg2.feed(AsrToken(text="", is_silence=False, timestamp=0.0))
    seg2.feed(_text_token("Bonjour", 5.0))
    result2 = seg2.flush(5.5)
    assert result2 is not None
    assert result2.t_start == 5.0


def test_on_partial_callback_receives_growing_buffer() -> None:
    seen: list[str] = []
    seg = Segmenter(on_partial=seen.append)
    seg.feed(_text_token("Bon", 0.0))
    seg.feed(_text_token("jour", 0.1))
    assert seen == ["Bon", "Bonjour"]


def test_partial_segments_never_reach_final_output() -> None:
    """Non-stabilized accumulation must not be returned as a TranscriptSegment."""
    seg = Segmenter()
    assert seg.feed(_text_token("Bon", 0.0)) is None
    assert seg.feed(_text_token("jour", 0.1)) is None
    # only the flush (or punctuation/silence) produces a final segment
    result = seg.flush(0.2)
    assert result is not None
    assert result.final is True


def test_detect_lang_french_sentence() -> None:
    text = "Bonjour à tous, merci de rejoindre cette réunion."
    assert detect_lang(text) == "fr"


def test_detect_lang_english_sentence() -> None:
    text = "Good morning, everyone. Thank you for joining this meeting."
    assert detect_lang(text) == "en"


def test_detect_lang_french_without_accents_still_detected_via_stopwords() -> None:
    text = "Nous allons faire le point sur le planning ensemble"
    assert detect_lang(text) == "fr"


def test_detect_lang_english_technical_sentence() -> None:
    text = (
        "We will review the deployment status and validate the data migration schedule"
    )
    assert detect_lang(text) == "en"
