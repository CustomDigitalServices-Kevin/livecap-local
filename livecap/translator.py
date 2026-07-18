"""Ollama translation client for LiveCap Local.

Reuses the POC pattern validated in livecap-poc/bench_translate.py: model
qwen3:4b-instruct via the Ollama HTTP API (urllib stdlib, no third-party HTTP
client), temperature 0, num_predict 128, "subtitle translator" system prompt.
See CONTRACT.md for the TranscriptSegment / TranslatedSegment / queue shapes
this module reads from segment_q and writes to ui_q -- imported from
livecap.contract, never redefined here.

Direction: config.yaml only exposes a single target_lang (see
livecap.config.LiveCapConfig). Each TranscriptSegment already carries its own
detected source language (segment.lang, "fr" or "en"), so the direction is
segment.lang -> target_lang. When segment.lang == target_lang the speaker is
already talking in the target language: no Ollama call is made and the
segment is published with text="" (same convention as a failed/timed-out
translation -- overlay.py falls back to showing the VO line alone either
way).

Coalescence policy: segment_q only ever carries final=True segments per
CONTRACT.md (segmenter.py never lets unstable partials cross this queue), so
there is no partial-vs-final filtering to do here. What this module DOES do
is protect against Ollama falling behind a fast talker: after taking one
segment off segment_q, it drains every segment already queued behind it
non-blockingly. Only the single most recent of that backlog is sent to
Ollama; every older segment in the backlog is still published to ui_q
immediately with text="" (VO-only fallback, latency_ms=0.0) so the
transcript/history is never silently dropped -- only its translation is
skipped while the translator is catching up. This keeps ui_q (and the
overlay) from lagging the live conversation, at the cost of not translating
every single segment during a backlog.

Ollama availability: check_ollama_health() hits GET /api/tags. TranslatorThread
calls ensure_ollama_running() once at thread start; if Ollama is unreachable
it tries a detached `ollama serve` subprocess and polls for a short window
before raising OllamaUnavailableError with an actionable message. Once
running, per-segment failures/timeouts never raise out of the thread -- they
fall back to VO-only publication (see _translate_and_publish).
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from livecap.contract import TranscriptSegment, TranslatedSegment

logger = logging.getLogger(__name__)

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
MODEL = "qwen3:4b-instruct"

TRANSLATE_TIMEOUT_S = 5.0
HEALTH_CHECK_TIMEOUT_S = 2.0
AUTOSTART_POLL_INTERVAL_S = 1.0
AUTOSTART_POLL_ATTEMPTS = 10  # ~10 s to let a freshly spawned `ollama serve` come up
DEFAULT_POLL_TIMEOUT_S = 0.5  # how often the thread checks its stop flag

# Maps the config.yaml target_lang code to the natural-language name Ollama
# expects in the "translate to <language>" instruction (matches the POC).
_TARGET_LANG_NAMES = {"fr": "French", "en": "English"}


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama is unreachable and the auto-start attempt failed."""


def check_ollama_health(
    tags_url: str = OLLAMA_TAGS_URL, timeout: float = HEALTH_CHECK_TIMEOUT_S
) -> bool:
    """Return True if the Ollama HTTP API answers 200 at tags_url within timeout."""
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def ensure_ollama_running(
    tags_url: str = OLLAMA_TAGS_URL,
    poll_interval: float = AUTOSTART_POLL_INTERVAL_S,
    max_attempts: int = AUTOSTART_POLL_ATTEMPTS,
) -> None:
    """Verify Ollama answers; if not, try `ollama serve` detached, then poll.

    No-op if Ollama already answers at tags_url. Raises OllamaUnavailableError
    with an actionable message if Ollama is still unreachable after the
    auto-start attempt (either because `ollama` isn't on PATH, or because it
    didn't come up within poll_interval * max_attempts seconds).
    """
    if check_ollama_health(tags_url):
        return

    logger.warning(
        "Ollama injoignable sur %s, tentative de demarrage automatique...", tags_url
    )
    try:
        creationflags = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0
        )
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except (FileNotFoundError, OSError) as exc:
        raise OllamaUnavailableError(
            f"Ollama introuvable sur {tags_url} et impossible de lancer "
            f"'ollama serve' automatiquement ({exc}). Installer/demarrer Ollama "
            "manuellement (https://ollama.com) puis relancer LiveCap Local."
        ) from exc

    for _ in range(max_attempts):
        time.sleep(poll_interval)
        if check_ollama_health(tags_url):
            logger.info("Ollama demarre avec succes sur %s.", tags_url)
            return

    raise OllamaUnavailableError(
        f"Ollama ne repond toujours pas sur {tags_url} apres "
        f"{max_attempts * poll_interval:.0f} s de tentative de demarrage automatique. "
        "Verifier l'installation (https://ollama.com) et que le modele "
        f"'{MODEL}' est disponible (`ollama pull {MODEL}`)."
    )


def translate_text(
    text: str,
    target_lang: str,
    chat_url: str = OLLAMA_CHAT_URL,
    model: str = MODEL,
    timeout: float = TRANSLATE_TIMEOUT_S,
) -> str:
    """Call Ollama /api/chat once and return the translated text.

    Mirrors livecap-poc/bench_translate.py exactly: temperature 0,
    num_predict 128, "subtitle translator" system prompt, stream=False.
    Raises urllib.error.URLError/TimeoutError/OSError on network failure,
    json.JSONDecodeError/KeyError on a malformed response body -- callers
    (TranslatorThread) are responsible for the VO-only fallback.
    """
    target_name = _TARGET_LANG_NAMES[target_lang]
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a subtitle translator. Translate the user message "
                    f"to {target_name}. Output ONLY the translation, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 128},
    }
    req = urllib.request.Request(
        chat_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body["message"]["content"].strip()


class TranslatorThread(threading.Thread):
    """Consumes segment_q, produces TranslatedSegment onto ui_q.

    See the module docstring for the direction/coalescence/fallback policy.
    Call stop() then join() for a clean shutdown -- the run loop polls
    segment_q with a short timeout so it notices the stop flag promptly
    instead of blocking forever on an empty queue.
    """

    def __init__(
        self,
        segment_q: "queue.Queue[TranscriptSegment]",
        ui_q: "queue.Queue[TranslatedSegment]",
        target_lang: str,
        chat_url: str = OLLAMA_CHAT_URL,
        tags_url: str = OLLAMA_TAGS_URL,
        model: str = MODEL,
        translate_timeout: float = TRANSLATE_TIMEOUT_S,
        poll_timeout: float = DEFAULT_POLL_TIMEOUT_S,
    ) -> None:
        if target_lang not in _TARGET_LANG_NAMES:
            raise ValueError(
                f"target_lang={target_lang!r} invalide, attendu 'fr' ou 'en'"
            )
        super().__init__(name="TranslatorThread", daemon=True)
        self._segment_q = segment_q
        self._ui_q = ui_q
        self._target_lang = target_lang
        self._chat_url = chat_url
        self._tags_url = tags_url
        self._model = model
        self._translate_timeout = translate_timeout
        self._poll_timeout = poll_timeout
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        ensure_ollama_running(tags_url=self._tags_url)
        while not self._stop_event.is_set():
            try:
                segment = self._segment_q.get(timeout=self._poll_timeout)
            except queue.Empty:
                continue
            segment = self._drain_backlog_keep_latest(segment)
            self._translate_and_publish(segment)

    def _drain_backlog_keep_latest(
        self, segment: TranscriptSegment
    ) -> TranscriptSegment:
        """Non-blockingly collapse any backlog down to the most recent segment.

        Every segment discarded this way is still published to ui_q with an
        empty translation (VO-only fallback) so the transcript is not lost --
        only its translation is skipped under load. See module docstring.
        """
        latest = segment
        while True:
            try:
                next_segment = self._segment_q.get_nowait()
            except queue.Empty:
                break
            self._publish(latest, text="", latency_ms=0.0)
            latest = next_segment
        return latest

    def _translate_and_publish(self, segment: TranscriptSegment) -> None:
        if segment.lang == self._target_lang:
            # Already in the target language: nothing to translate, and this
            # is not a failure, so no warning -- just VO-only publication.
            self._publish(segment, text="", latency_ms=0.0)
            return
        t0 = time.perf_counter()
        try:
            translation = translate_text(
                segment.text,
                self._target_lang,
                chat_url=self._chat_url,
                model=self._model,
                timeout=self._translate_timeout,
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
            KeyError,
        ) as exc:
            logger.warning(
                "Traduction echouee/timeout pour %r: %s (fallback VO)",
                segment.text,
                exc,
            )
            self._publish(segment, text="", latency_ms=0.0)
            return
        latency_ms = (time.perf_counter() - t0) * 1000
        self._publish(segment, text=translation, latency_ms=latency_ms)

    def _publish(
        self, segment: TranscriptSegment, *, text: str, latency_ms: float
    ) -> None:
        translated = TranslatedSegment(source=segment, text=text, latency_ms=latency_ms)
        try:
            self._ui_q.put(translated, timeout=self._poll_timeout)
        except queue.Full:
            logger.warning("ui_q pleine, segment traduit perdu: %r", segment.text)
