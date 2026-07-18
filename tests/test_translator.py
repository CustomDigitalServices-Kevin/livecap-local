"""Tests for livecap.translator: nominal, timeout, coalescence, health/autostart.

Uses a local mock Ollama HTTP server (http.server.ThreadingHTTPServer on an
OS-assigned port) that fakes /api/tags and /api/chat -- no dependency on a
real Ollama install for these unit tests. The one exception is
test_translate_text_live_ollama_latency, which talks to a REAL local Ollama
instance if one is reachable (skipped otherwise) to satisfy the PRD's
"latence mediane < 500 ms" acceptance criterion end to end.
"""

from __future__ import annotations

import json
import threading
import time
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator

import pytest

from livecap.contract import TranscriptSegment, make_segment_queue, make_ui_queue
from livecap.translator import (
    OllamaUnavailableError,
    TranslatorThread,
    check_ollama_health,
    ensure_ollama_running,
    translate_text,
)


class _MockOllamaServer(ThreadingHTTPServer):
    """Minimal fake of the two Ollama endpoints this module calls.

    Behavior is driven by plain attributes set by each test:
        tags_up: bool -- whether GET /api/tags answers 200 or is refused.
        chat_delay_s: float -- sleep before answering POST /api/chat (timeout test).
        chat_response_fn: Callable[[dict], str] -- builds the "content" field.
        received_chat_requests: list[dict] -- every decoded /api/chat body received.
    """

    tags_up: bool = True
    chat_delay_s: float = 0.0
    chat_response_fn: Callable[[dict], str] = staticmethod(
        lambda body: "mock translation"
    )
    received_chat_requests: list

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.received_chat_requests = []

    def handle_error(self, request: object, client_address: object) -> None:
        # A client that already timed out (test_translator_thread_timeout_*)
        # closes its socket before this server finishes sleeping and writing
        # the delayed response; that BrokenPipeError/ConnectionResetError is
        # expected in that test and not a real server bug, so stay quiet.
        pass


class _MockOllamaHandler(BaseHTTPRequestHandler):
    server: _MockOllamaServer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # silence default request logging during tests

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            if self.server.tags_up:
                self._reply_json(200, {"models": []})
            else:
                self.send_error(503)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.received_chat_requests.append(body)
        if self.server.chat_delay_s:
            time.sleep(self.server.chat_delay_s)
        content = self.server.chat_response_fn(body)
        self._reply_json(
            200,
            {
                "message": {"content": content},
                "eval_count": 5,
                "eval_duration": 50_000_000,
            },
        )

    def _reply_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def ollama_mock() -> Iterator[_MockOllamaServer]:
    server = _MockOllamaServer(("127.0.0.1", 0), _MockOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    server.chat_url = f"http://{host}:{port}/api/chat"  # type: ignore[attr-defined]
    server.tags_url = f"http://{host}:{port}/api/tags"  # type: ignore[attr-defined]
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _segment(text: str, lang: str) -> TranscriptSegment:
    return TranscriptSegment(text=text, lang=lang, t_start=0.0, t_end=1.0, final=True)


# --- translate_text (direct HTTP call) --------------------------------------


def test_translate_text_nominal(ollama_mock: _MockOllamaServer) -> None:
    ollama_mock.chat_response_fn = lambda body: "Bonjour le monde"
    result = translate_text("Hello world", "fr", chat_url=ollama_mock.chat_url)  # type: ignore[attr-defined]
    assert result == "Bonjour le monde"
    assert ollama_mock.received_chat_requests[0]["model"] == "qwen3:4b-instruct"
    assert ollama_mock.received_chat_requests[0]["options"] == {
        "temperature": 0,
        "num_predict": 128,
    }


def test_translate_text_raises_on_timeout(ollama_mock: _MockOllamaServer) -> None:
    ollama_mock.chat_delay_s = 0.3
    with pytest.raises((TimeoutError, OSError)):
        translate_text("Hello", "fr", chat_url=ollama_mock.chat_url, timeout=0.05)  # type: ignore[attr-defined]


# --- TranslatorThread: nominal / timeout fallback / same-language skip ------


def test_translator_thread_translates_and_publishes(
    ollama_mock: _MockOllamaServer,
) -> None:
    ollama_mock.chat_response_fn = lambda body: "Bonjour le monde"
    segment_q = make_segment_queue()
    ui_q = make_ui_queue()
    thread = TranslatorThread(
        segment_q,
        ui_q,
        target_lang="fr",
        chat_url=ollama_mock.chat_url,  # type: ignore[attr-defined]
        tags_url=ollama_mock.tags_url,  # type: ignore[attr-defined]
        poll_timeout=0.05,
    )
    thread.start()
    try:
        seg = _segment("Hello world", "en")
        segment_q.put(seg)
        translated = ui_q.get(timeout=3)
        assert translated.source is seg
        assert translated.text == "Bonjour le monde"
        assert translated.latency_ms > 0
    finally:
        thread.stop()
        thread.join(timeout=2)


def test_translator_thread_timeout_falls_back_to_vo(
    ollama_mock: _MockOllamaServer,
) -> None:
    ollama_mock.chat_delay_s = 0.5
    segment_q = make_segment_queue()
    ui_q = make_ui_queue()
    thread = TranslatorThread(
        segment_q,
        ui_q,
        target_lang="en",
        chat_url=ollama_mock.chat_url,  # type: ignore[attr-defined]
        tags_url=ollama_mock.tags_url,  # type: ignore[attr-defined]
        translate_timeout=0.1,
        poll_timeout=0.05,
    )
    thread.start()
    try:
        seg = _segment("Bonjour", "fr")
        segment_q.put(seg)
        translated = ui_q.get(timeout=3)
        assert translated.source is seg
        assert translated.text == ""
        assert translated.latency_ms == 0.0
    finally:
        thread.stop()
        thread.join(timeout=2)


def test_translator_thread_skips_translation_when_language_matches(
    ollama_mock: _MockOllamaServer,
) -> None:
    segment_q = make_segment_queue()
    ui_q = make_ui_queue()
    thread = TranslatorThread(
        segment_q,
        ui_q,
        target_lang="fr",
        chat_url=ollama_mock.chat_url,  # type: ignore[attr-defined]
        tags_url=ollama_mock.tags_url,  # type: ignore[attr-defined]
        poll_timeout=0.05,
    )
    thread.start()
    try:
        seg = _segment("Bonjour", "fr")
        segment_q.put(seg)
        translated = ui_q.get(timeout=3)
        assert translated.text == ""
        assert translated.latency_ms == 0.0
    finally:
        thread.stop()
        thread.join(timeout=2)
    assert ollama_mock.received_chat_requests == []


def test_translator_thread_rejects_invalid_target_lang(
    ollama_mock: _MockOllamaServer,
) -> None:
    with pytest.raises(ValueError, match="target_lang"):
        TranslatorThread(
            make_segment_queue(),
            make_ui_queue(),
            target_lang="de",
            chat_url=ollama_mock.chat_url,  # type: ignore[attr-defined]
            tags_url=ollama_mock.tags_url,  # type: ignore[attr-defined]
        )


# --- Coalescence: tested against the private method directly ---------------
#
# A real threaded test racing segment_q.put() against the thread's own
# get()/get_nowait() loop would be timing-dependent and flaky. The
# coalescence behavior is fully captured by _drain_backlog_keep_latest, so it
# is exercised directly and deterministically here instead.


def test_drain_backlog_keeps_latest_and_publishes_fallback_for_rest(
    ollama_mock: _MockOllamaServer,
) -> None:
    segment_q = make_segment_queue()
    ui_q = make_ui_queue()
    thread = TranslatorThread(
        segment_q,
        ui_q,
        target_lang="fr",
        chat_url=ollama_mock.chat_url,  # type: ignore[attr-defined]
        tags_url=ollama_mock.tags_url,  # type: ignore[attr-defined]
    )
    seg1 = _segment("one", "en")
    seg2 = _segment("two", "en")
    seg3 = _segment("three", "en")
    segment_q.put(seg2)
    segment_q.put(seg3)

    latest = thread._drain_backlog_keep_latest(seg1)

    assert latest is seg3
    first = ui_q.get_nowait()
    second = ui_q.get_nowait()
    assert first.source is seg1
    assert first.text == ""
    assert second.source is seg2
    assert second.text == ""
    assert ui_q.empty()


def test_drain_backlog_noop_when_queue_empty(ollama_mock: _MockOllamaServer) -> None:
    segment_q = make_segment_queue()
    ui_q = make_ui_queue()
    thread = TranslatorThread(
        segment_q,
        ui_q,
        target_lang="fr",
        chat_url=ollama_mock.chat_url,  # type: ignore[attr-defined]
        tags_url=ollama_mock.tags_url,  # type: ignore[attr-defined]
    )
    seg = _segment("solo", "en")
    latest = thread._drain_backlog_keep_latest(seg)
    assert latest is seg
    assert ui_q.empty()


# --- Health check / auto-start ------------------------------------------


def test_check_ollama_health_true_when_reachable(
    ollama_mock: _MockOllamaServer,
) -> None:
    assert check_ollama_health(ollama_mock.tags_url) is True  # type: ignore[attr-defined]


def test_check_ollama_health_false_when_unreachable() -> None:
    assert check_ollama_health("http://127.0.0.1:1/api/tags", timeout=0.2) is False


def test_ensure_ollama_running_noop_when_already_up(
    ollama_mock: _MockOllamaServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Popen should not be called when Ollama is already healthy"
        )

    monkeypatch.setattr("livecap.translator.subprocess.Popen", _fail_popen)
    ensure_ollama_running(tags_url=ollama_mock.tags_url)  # type: ignore[attr-defined]


def test_ensure_ollama_running_raises_when_ollama_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing_popen(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("ollama not on PATH")

    monkeypatch.setattr("livecap.translator.subprocess.Popen", _missing_popen)
    with pytest.raises(OllamaUnavailableError, match="introuvable"):
        ensure_ollama_running(
            tags_url="http://127.0.0.1:1/api/tags", poll_interval=0.01, max_attempts=1
        )


def test_ensure_ollama_running_recovers_after_autostart(
    ollama_mock: _MockOllamaServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    ollama_mock.tags_up = False

    def _fake_popen(*args: object, **kwargs: object) -> unittest.mock.Mock:
        def _bring_up() -> None:
            time.sleep(0.05)
            ollama_mock.tags_up = True

        threading.Thread(target=_bring_up, daemon=True).start()
        return unittest.mock.Mock()

    monkeypatch.setattr("livecap.translator.subprocess.Popen", _fake_popen)
    ensure_ollama_running(
        tags_url=ollama_mock.tags_url,  # type: ignore[attr-defined]
        poll_interval=0.02,
        max_attempts=20,
    )
    assert ollama_mock.tags_up is True


def test_ensure_ollama_running_raises_after_exhausting_attempts(
    ollama_mock: _MockOllamaServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    ollama_mock.tags_up = False
    monkeypatch.setattr(
        "livecap.translator.subprocess.Popen", lambda *a, **k: unittest.mock.Mock()
    )
    with pytest.raises(OllamaUnavailableError, match="ne repond toujours pas"):
        ensure_ollama_running(
            tags_url=ollama_mock.tags_url,  # type: ignore[attr-defined]
            poll_interval=0.01,
            max_attempts=2,
        )


# --- Live integration (real Ollama) -----------------------------------------


@pytest.mark.skipif(
    not check_ollama_health(), reason="Ollama non demarre localement sur :11434"
)
def test_translate_text_live_ollama_latency() -> None:
    """PRD acceptance: latence mediane < 500 ms pour la traduction (POC 2026-07-18)."""
    # Warmup excluded from the measured stats (first call pays model load).
    translate_text("Hello, this is a warmup.", "fr")
    segments = [
        "Good morning everyone, thank you for joining this meeting.",
        "Can you confirm the data migration schedule before the end of the month?",
        "Let's move on to the next agenda item.",
    ]
    latencies_ms = []
    for text in segments:
        t0 = time.perf_counter()
        result = translate_text(text, "fr")
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        assert result  # non-empty translation
    median_ms = sorted(latencies_ms)[len(latencies_ms) // 2]
    assert median_ms < 500, f"latence mediane {median_ms:.0f} ms >= 500 ms"
