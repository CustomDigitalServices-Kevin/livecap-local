"""WASAPI double-stream audio capture: system loopback + microphone.

Captures the default (or configured) WASAPI loopback device (system audio,
i.e. Teams interlocutors) and the default (or configured) microphone via
PyAudioWPatch, resamples both to 16 kHz mono float32, mixes them with a
simple level-based anti-echo attenuation, and pushes ``AudioChunk`` (see
``livecap.contract``) onto ``audio_q`` in blocks <= ``BLOCK_MS`` milliseconds.

API surface verified against PyAudioWPatch 0.2.12.8 (installed in this
project's .venv) and https://github.com/s0d3s/PyAudioWPatch (README +
``examples/pawp_record_wasapi_loopback.py`` /
``examples/pawp_another_record_wasapi_loopback.py``), plus the standard
PyAudio callback-mode contract at
https://people.csail.mit.edu/hubert/pyaudio/docs/ (open() params,
stream_callback signature/return tuple, paContinue/paComplete/paAbort).

Design choices (documented here because they are not obvious from the
code alone):

- Sample format: streams are opened with ``paInt16`` (not ``paFloat32``),
  matching the two official PyAudioWPatch loopback examples, which is the
  only format/pattern actually demonstrated for WASAPI loopback streams.
  Conversion to float32 happens in Python right after each callback.
- Resampling: WASAPI loopback devices typically report 48 kHz stereo (the
  Windows shared-mode mix format), sometimes 44100/96000 Hz depending on
  the device. Rather than add ``scipy`` (not in requirements.txt) for a
  polyphase resampler, resampling to 16 kHz uses linear interpolation via
  ``numpy.interp`` over a normalized [0, 1) time axis. This is sufficient
  for ASR speech clarity (Kyutai stt-1b-en_fr does not need hi-fi audio);
  the resulting mild aliasing on the discarded high frequencies is a known
  v1 limitation, documented in README.md (tache 6).
- Anti-echo v1: when the loopback stream's RMS exceeds
  ``DEFAULT_ECHO_RMS_THRESHOLD`` (speaker audio is actually playing), the
  simultaneous mic chunk is attenuated by ``DEFAULT_MIC_ATTENUATION``
  before mixing, to reduce (not eliminate) the ASR re-transcribing audio
  picked up by the mic from the speakers. This is a level-based
  attenuation, not adaptive acoustic echo cancellation (AEC) -- a real AEC
  is out of scope for v1 and noted as a known limitation.
- Threading: each PyAudioWPatch stream runs in callback mode (audio
  callbacks fire on PortAudio's own native thread, not a Python thread we
  manage); callbacks only resample and append to a lock-protected buffer
  (cheap). A single Python "mixer" thread wakes up every ``block_ms`` to
  drain both buffers, mix them, and push the result to ``audio_q``. This
  keeps the audio callbacks fast (no queue contention, no ASR-side work)
  and keeps chunk cadence to a single controllable Python thread that
  ``stop()`` joins cleanly (no zombie thread).
- Backpressure: ``audio_q`` is bounded (see ``livecap.contract``). If
  full, the oldest queued chunk is dropped in favor of the new one --
  for live captioning, a stale chunk is worse than a dropped one.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from livecap.config import DevicesConfig
from livecap.contract import AudioChunk

logger = logging.getLogger(__name__)

TARGET_RATE = 16_000
BLOCK_MS = 80  # <= 80 ms per CONTRACT.md / PRD (Kyutai streaming step budget)
DEFAULT_MIC_ATTENUATION = 0.25
DEFAULT_ECHO_RMS_THRESHOLD = 0.02


class CaptureError(RuntimeError):
    """Raised when the WASAPI loopback (system audio) device cannot be
    resolved or opened. Unlike the microphone, system-audio capture has no
    fallback: it is the primary purpose of this module."""


# --------------------------------------------------------------------------
# Pure signal-processing helpers (unit tested independently of PyAudioWPatch)
# --------------------------------------------------------------------------


def _pcm16_bytes_to_float32(raw: bytes) -> NDArray[np.float32]:
    """Convert little-endian int16 PCM bytes to float32 samples in [-1, 1)."""
    ints = np.frombuffer(raw, dtype=np.int16)
    return (ints.astype(np.float32) / 32768.0).astype(np.float32)


def _downmix_to_mono(
    samples: NDArray[np.float32], channels: int
) -> NDArray[np.float32]:
    """Average interleaved channels down to mono. ``samples`` is 1-D interleaved."""
    if channels <= 1:
        return samples
    n_frames = samples.size // channels
    trimmed = samples[: n_frames * channels]
    if n_frames == 0:
        return np.empty(0, dtype=np.float32)
    return trimmed.reshape(n_frames, channels).mean(axis=1).astype(np.float32)


def resample_mono_to_16k(
    samples: NDArray[np.float32], orig_rate: int
) -> NDArray[np.float32]:
    """Resample mono float32 audio from ``orig_rate`` to ``TARGET_RATE`` (16 kHz).

    Method: linear interpolation (``numpy.interp``) over a normalized time
    axis. See the module docstring for why this was chosen over a
    scipy-based polyphase resampler.
    """
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)
    if orig_rate == TARGET_RATE:
        return samples.astype(np.float32, copy=False)

    n_out = max(1, round(samples.size * TARGET_RATE / orig_rate))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def _rms(samples: NDArray[np.float32]) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _pad_to(arr: NDArray[np.float32], n: int) -> NDArray[np.float32]:
    if arr.size >= n:
        return arr[:n]
    return np.concatenate([arr, np.zeros(n - arr.size, dtype=np.float32)])


def mix_chunks(
    loopback: NDArray[np.float32] | None,
    mic: NDArray[np.float32] | None,
    *,
    mic_attenuation: float = DEFAULT_MIC_ATTENUATION,
    echo_rms_threshold: float = DEFAULT_ECHO_RMS_THRESHOLD,
) -> NDArray[np.float32]:
    """Mix a loopback chunk and a mic chunk (both already 16 kHz mono float32).

    If one side is ``None`` or empty, the other side is returned alone (or
    silence if both are absent). The shorter side is zero-padded to the
    longer side's length so no audio from the longer stream is lost. The
    anti-echo attenuation (see module docstring) is applied to the mic
    side whenever the loopback RMS exceeds ``echo_rms_threshold``. Output
    is clipped to [-1, 1] to avoid clipping artifacts from summing two
    full-scale signals.
    """
    if loopback is None or loopback.size == 0:
        if mic is None or mic.size == 0:
            return np.empty(0, dtype=np.float32)
        return np.clip(mic, -1.0, 1.0).astype(np.float32)
    if mic is None or mic.size == 0:
        return np.clip(loopback, -1.0, 1.0).astype(np.float32)

    n = max(loopback.size, mic.size)
    loop_seg = _pad_to(loopback, n)
    mic_seg = _pad_to(mic, n)

    gain = mic_attenuation if _rms(loop_seg) > echo_rms_threshold else 1.0
    mixed = loop_seg + gain * mic_seg
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Device resolution (pure functions taking an already-constructed PyAudio
# instance, so they are unit-testable with a fake instance)
# --------------------------------------------------------------------------


def _resolve_loopback_device(pa: Any, pyaudio_mod: Any, wanted: str) -> dict[str, Any]:
    """Resolve the WASAPI loopback device to capture system audio from.

    ``wanted == "auto"`` uses the default output device's loopback
    analogue. Otherwise, a case-insensitive substring match is done
    against loopback device names (loopback devices carry a distinct name,
    typically suffixed "[Loopback]", per PyAudioWPatch's documented
    pattern).

    Raises ``CaptureError`` with a precise message if WASAPI is
    unavailable or no matching loopback device is found: capturing system
    audio is a hard requirement here, unlike the mic (see
    ``_resolve_mic_device``).
    """
    if wanted == "auto":
        try:
            return pa.get_default_wasapi_loopback()
        except OSError as exc:
            raise CaptureError(
                "WASAPI indisponible sur ce systeme : impossible de capturer "
                "l'audio systeme (loopback). Verifier le pilote audio."
            ) from exc
        except LookupError as exc:
            raise CaptureError(
                "Peripherique de sortie par defaut sans loopback WASAPI "
                "correspondant. Lancer `python -m pyaudiowpatch` pour lister "
                "les peripheriques disponibles."
            ) from exc

    wanted_lower = wanted.lower()
    for device in pa.get_loopback_device_info_generator():
        if wanted_lower in str(device["name"]).lower():
            return device
    raise CaptureError(
        f"Aucun peripherique loopback WASAPI ne correspond a devices.loopback={wanted!r}. "
        "Lancer `python -m pyaudiowpatch` pour lister les peripheriques disponibles."
    )


def _resolve_mic_device(
    pa: Any, pyaudio_mod: Any, wanted: str
) -> dict[str, Any] | None:
    """Resolve the microphone device, or ``None`` if disabled/absent.

    ``wanted == "none"``: mic explicitly disabled by config, returns
    ``None``. ``wanted == "auto"``: WASAPI default input device, falling
    back to PyAudio's generic default input device if WASAPI reports
    none. Otherwise: case-insensitive substring match among WASAPI input
    devices (excluding loopback devices).

    Unlike the loopback device, a missing/unresolvable mic is NOT a hard
    error: per the PRD, capture must fall back to loopback-only (system
    audio only) with a clear log line -- the caller (``AudioCapture``)
    logs the fallback.
    """
    if wanted == "none":
        return None

    if wanted == "auto":
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio_mod.paWASAPI)
            device_index = wasapi_info.get("defaultInputDevice", -1)
            if device_index is not None and device_index >= 0:
                return pa.get_device_info_by_index(device_index)
        except (OSError, KeyError, AttributeError):
            pass
        try:
            return pa.get_default_input_device_info()
        except OSError:
            return None

    wanted_lower = wanted.lower()
    try:
        candidates = pa.get_device_info_generator_by_host_api(pyaudio_mod.paWASAPI)
    except OSError:
        return None
    for device in candidates:
        if device.get("isLoopbackDevice"):
            continue
        if device["maxInputChannels"] <= 0:
            continue
        if wanted_lower in str(device["name"]).lower():
            return device
    return None


def list_devices(pyaudio_module: Any | None = None) -> list[dict[str, Any]]:
    """Enumerate all audio devices for debugging (choosing config.yaml values).

    Returns plain dicts: index, name, is_loopback, max_input_channels,
    default_sample_rate, host_api. Complements ``python -m pyaudiowpatch``
    (which prints, but doesn't return structured data callers can use).
    """
    pyaudio_mod = pyaudio_module
    if pyaudio_mod is None:
        import pyaudiowpatch as pyaudio_mod  # heavy native import, kept lazy

    devices: list[dict[str, Any]] = []
    with pyaudio_mod.PyAudio() as pa:
        for info in pa.get_device_info_generator():
            devices.append(
                {
                    "index": info["index"],
                    "name": info["name"],
                    "is_loopback": bool(info.get("isLoopbackDevice", False)),
                    "max_input_channels": info["maxInputChannels"],
                    "default_sample_rate": info["defaultSampleRate"],
                    "host_api": info["hostApi"],
                }
            )
    return devices


# --------------------------------------------------------------------------
# AudioCapture: owns the PyAudio streams, the mixer thread, and audio_q
# --------------------------------------------------------------------------


class AudioCapture:
    """Captures WASAPI loopback + microphone and pushes mixed 16 kHz mono
    float32 ``AudioChunk`` onto ``audio_q``.

    Lifecycle: ``start()`` opens PyAudio, the loopback stream, optionally
    the mic stream, and the mixer thread. ``stop()`` tears everything down
    (mixer thread joined, streams stopped+closed, PyAudio terminated) --
    safe to call even if ``start()`` partially failed. Also usable as a
    context manager.
    """

    def __init__(
        self,
        devices: DevicesConfig,
        audio_q: "queue.Queue[AudioChunk]",
        *,
        block_ms: int = BLOCK_MS,
        mic_attenuation: float = DEFAULT_MIC_ATTENUATION,
        echo_rms_threshold: float = DEFAULT_ECHO_RMS_THRESHOLD,
        pyaudio_module: Any | None = None,
    ) -> None:
        self._devices_cfg = devices
        self._audio_q = audio_q
        self._block_ms = block_ms
        self._mic_attenuation = mic_attenuation
        self._echo_rms_threshold = echo_rms_threshold
        self._pyaudio_module = pyaudio_module  # injection point for tests

        self._pa: Any | None = None
        self._loopback_stream: Any | None = None
        self._mic_stream: Any | None = None
        self._mic_enabled = devices.mic != "none"

        self._loop_rate = TARGET_RATE
        self._loop_channels = 1
        self._mic_rate = TARGET_RATE
        self._mic_channels = 1

        self._loop_lock = threading.Lock()
        self._mic_lock = threading.Lock()
        self._loop_buffer: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._mic_buffer: NDArray[np.float32] = np.empty(0, dtype=np.float32)

        self._mixer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

    @property
    def mic_active(self) -> bool:
        """True if the mic stream was resolved and opened (not disabled/absent)."""
        return self._mic_stream is not None

    def start(self) -> None:
        if self._started:
            return

        pyaudio_mod = self._pyaudio_module
        if pyaudio_mod is None:
            import pyaudiowpatch as pyaudio_mod  # heavy native import, kept lazy
        self._pyaudio_module = pyaudio_mod

        self._pa = pyaudio_mod.PyAudio()
        try:
            loop_device = _resolve_loopback_device(
                self._pa, pyaudio_mod, self._devices_cfg.loopback
            )
        except CaptureError:
            self._pa.terminate()
            self._pa = None
            raise

        self._loop_rate = int(loop_device["defaultSampleRate"])
        self._loop_channels = max(1, int(loop_device["maxInputChannels"]))
        loop_frames = max(1, int(self._loop_rate * self._block_ms / 1000))

        self._loopback_stream = self._pa.open(
            format=pyaudio_mod.paInt16,
            channels=self._loop_channels,
            rate=self._loop_rate,
            input=True,
            input_device_index=loop_device["index"],
            frames_per_buffer=loop_frames,
            stream_callback=self._make_loopback_callback(pyaudio_mod),
        )
        logger.info(
            "Loopback capture: %s (index=%s, %d Hz, %d ch)",
            loop_device.get("name", "?"),
            loop_device["index"],
            self._loop_rate,
            self._loop_channels,
        )

        self._mic_stream = None
        if self._mic_enabled:
            mic_device = _resolve_mic_device(
                self._pa, pyaudio_mod, self._devices_cfg.mic
            )
            if mic_device is None:
                logger.warning(
                    "Peripherique micro %r introuvable ou absent : fallback "
                    "loopback seul (audio systeme uniquement, pas de micro).",
                    self._devices_cfg.mic,
                )
            else:
                self._mic_rate = int(mic_device["defaultSampleRate"])
                self._mic_channels = max(1, int(mic_device["maxInputChannels"]))
                mic_frames = max(1, int(self._mic_rate * self._block_ms / 1000))
                self._mic_stream = self._pa.open(
                    format=pyaudio_mod.paInt16,
                    channels=self._mic_channels,
                    rate=self._mic_rate,
                    input=True,
                    input_device_index=mic_device["index"],
                    frames_per_buffer=mic_frames,
                    stream_callback=self._make_mic_callback(pyaudio_mod),
                )
                logger.info(
                    "Micro capture: %s (index=%s, %d Hz, %d ch)",
                    mic_device.get("name", "?"),
                    mic_device["index"],
                    self._mic_rate,
                    self._mic_channels,
                )
        else:
            logger.info("Micro desactive par config (devices.mic=none).")

        self._stop_event.clear()
        self._mixer_thread = threading.Thread(
            target=self._mixer_loop, name="livecap-audio-mixer", daemon=True
        )
        self._mixer_thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        if self._mixer_thread is not None:
            self._mixer_thread.join(timeout=2.0)
            self._mixer_thread = None
        for stream in (self._loopback_stream, self._mic_stream):
            if stream is None:
                continue
            try:
                stream.stop_stream()
            finally:
                stream.close()
        self._loopback_stream = None
        self._mic_stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        self._started = False

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- PortAudio callbacks (run on PortAudio's native audio thread) ----

    def _make_loopback_callback(self, pyaudio_mod: Any):
        def _callback(in_data: bytes, frame_count: int, time_info: dict, status: int):
            mono = _downmix_to_mono(
                _pcm16_bytes_to_float32(in_data), self._loop_channels
            )
            resampled = resample_mono_to_16k(mono, self._loop_rate)
            with self._loop_lock:
                self._loop_buffer = np.concatenate([self._loop_buffer, resampled])
            return (None, pyaudio_mod.paContinue)

        return _callback

    def _make_mic_callback(self, pyaudio_mod: Any):
        def _callback(in_data: bytes, frame_count: int, time_info: dict, status: int):
            mono = _downmix_to_mono(
                _pcm16_bytes_to_float32(in_data), self._mic_channels
            )
            resampled = resample_mono_to_16k(mono, self._mic_rate)
            with self._mic_lock:
                self._mic_buffer = np.concatenate([self._mic_buffer, resampled])
            return (None, pyaudio_mod.paContinue)

        return _callback

    # -- mixer thread ------------------------------------------------------

    def _drain(self, lock: threading.Lock, attr: str) -> NDArray[np.float32]:
        with lock:
            data: NDArray[np.float32] = getattr(self, attr)
            setattr(self, attr, np.empty(0, dtype=np.float32))
        return data

    def _mixer_loop(self) -> None:
        tick = self._block_ms / 1000.0
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            next_tick += tick
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            if self._stop_event.is_set():
                break

            chunk_start = time.monotonic()
            loop_samples = self._drain(self._loop_lock, "_loop_buffer")
            mic_samples = (
                self._drain(self._mic_lock, "_mic_buffer")
                if self._mic_stream is not None
                else None
            )
            if loop_samples.size == 0 and (
                mic_samples is None or mic_samples.size == 0
            ):
                continue

            mixed = mix_chunks(
                loop_samples if loop_samples.size else None,
                mic_samples,
                mic_attenuation=self._mic_attenuation,
                echo_rms_threshold=self._echo_rms_threshold,
            )
            if mixed.size == 0:
                continue
            self._push(AudioChunk(samples=mixed, timestamp=chunk_start))

    def _push(self, chunk: AudioChunk) -> None:
        try:
            self._audio_q.put_nowait(chunk)
        except queue.Full:
            # Drop the oldest queued chunk to stay real-time: a stale chunk
            # is worse than a dropped one for live captioning.
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_q.put_nowait(chunk)
            except queue.Full:
                logger.warning("audio_q pleine et non vidangeable : chunk audio perdu")


if __name__ == "__main__":
    for device in list_devices():
        print(device)
