"""Tests for livecap.capture.

Mocking rationale (documented once here, applies to every test in this
file): PyAudioWPatch binds a native Windows WASAPI library. No real audio
device is available/reliable in a CI or headless environment, and the
real ``pyaudiowpatch.PyAudio()`` constructor talks to actual hardware.
Every test below builds a small ``FakePyAudio``/``FakeStream`` pair that
mimics the exact dict/constant shapes PyAudioWPatch returns (verified
against the installed 0.2.12.8 package and its README/examples, see
livecap/capture.py's module docstring), and injects it via
``AudioCapture(..., pyaudio_module=...)`` or by calling the
``_resolve_*``/``list_devices`` functions directly with a fake instance.
None of these tests import the real ``pyaudiowpatch`` package.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Any

import numpy as np
import pytest

from livecap.capture import (
    AudioCapture,
    CaptureError,
    _downmix_to_mono,
    _pcm16_bytes_to_float32,
    _resolve_loopback_device,
    _resolve_mic_device,
    list_devices,
    mix_chunks,
    resample_mono_to_16k,
)
from livecap.config import DevicesConfig
from livecap.contract import AudioChunk, make_audio_queue

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeStream:
    """Stands in for a PyAudio Stream: records lifecycle calls, exposes the
    registered stream_callback so tests can invoke it manually to simulate
    audio arriving on PortAudio's native thread."""

    def __init__(self, callback: Any, **open_kwargs: Any) -> None:
        self.callback = callback
        self.open_kwargs = open_kwargs
        self.stopped = False
        self.closed = False

    def stop_stream(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakePyAudio:
    """Stands in for pyaudiowpatch.PyAudio(). Devices are injected as plain
    dicts using the same keys the real library returns (name, index,
    isLoopbackDevice, maxInputChannels, defaultSampleRate)."""

    def __init__(
        self,
        loopback_device: dict[str, Any] | None = None,
        mic_device: dict[str, Any] | None = None,
        *,
        loopback_devices: list[dict[str, Any]] | None = None,
        input_devices: list[dict[str, Any]] | None = None,
        all_devices: list[dict[str, Any]] | None = None,
        wasapi_available: bool = True,
    ) -> None:
        self._loopback_device = loopback_device
        self._mic_device = mic_device
        self._loopback_devices = loopback_devices or (
            [loopback_device] if loopback_device else []
        )
        self._input_devices = input_devices or ([mic_device] if mic_device else [])
        self._all_devices = all_devices or (
            self._loopback_devices + self._input_devices
        )
        self._wasapi_available = wasapi_available
        self.streams: list[FakeStream] = []
        self.terminated = False

    # -- lifecycle --
    def terminate(self) -> None:
        self.terminated = True

    def __enter__(self) -> "FakePyAudio":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.terminate()

    # -- device resolution API used by capture.py --
    def get_default_wasapi_loopback(self) -> dict[str, Any]:
        if not self._wasapi_available:
            raise OSError("WASAPI not available")
        if self._loopback_device is None:
            raise LookupError("no default loopback")
        return self._loopback_device

    def get_loopback_device_info_generator(self):
        yield from self._loopback_devices

    def get_host_api_info_by_type(self, _api_type: int) -> dict[str, Any]:
        if not self._wasapi_available:
            raise OSError("WASAPI not available")
        return {
            "defaultInputDevice": self._mic_device["index"] if self._mic_device else -1
        }

    def get_device_info_by_index(self, index: int) -> dict[str, Any]:
        for device in self._all_devices:
            if device["index"] == index:
                return device
        raise OSError(f"no device at index {index}")

    def get_default_input_device_info(self) -> dict[str, Any]:
        if self._mic_device is None:
            raise OSError("no default input device")
        return self._mic_device

    def get_device_info_generator_by_host_api(self, _api_type: int):
        yield from self._input_devices + self._loopback_devices

    def get_device_info_generator(self):
        yield from self._all_devices

    # -- stream API --
    def open(self, **kwargs: Any) -> FakeStream:
        stream = FakeStream(kwargs["stream_callback"], **kwargs)
        self.streams.append(stream)
        return stream


class FakePyAudioModule:
    """Stands in for the `pyaudiowpatch` module: constants + PyAudio() factory."""

    paInt16 = 8
    paFloat32 = 1
    paContinue = 0
    paWASAPI = 13

    def __init__(self, pa_instance: FakePyAudio) -> None:
        self._pa_instance = pa_instance

    def PyAudio(self) -> FakePyAudio:
        return self._pa_instance


LOOPBACK_DEVICE = {
    "index": 0,
    "name": "Speakers (Realtek) [Loopback]",
    "isLoopbackDevice": True,
    "maxInputChannels": 2,
    "defaultSampleRate": 48000.0,
    "hostApi": 13,
}
MIC_DEVICE = {
    "index": 1,
    "name": "Microphone Array (Realtek)",
    "isLoopbackDevice": False,
    "maxInputChannels": 1,
    "defaultSampleRate": 48000.0,
    "hostApi": 13,
}


def int16_bytes(values: list[int]) -> bytes:
    return np.array(values, dtype=np.int16).tobytes()


# --------------------------------------------------------------------------
# _pcm16_bytes_to_float32
# --------------------------------------------------------------------------


def test_pcm16_to_float32_zero_is_zero() -> None:
    out = _pcm16_bytes_to_float32(int16_bytes([0, 0, 0]))
    assert out.dtype == np.float32
    assert np.allclose(out, 0.0)


def test_pcm16_to_float32_extremes() -> None:
    out = _pcm16_bytes_to_float32(int16_bytes([32767, -32768]))
    assert out[0] == pytest.approx(32767 / 32768.0, abs=1e-6)
    assert out[1] == pytest.approx(-1.0, abs=1e-6)


# --------------------------------------------------------------------------
# _downmix_to_mono
# --------------------------------------------------------------------------


def test_downmix_mono_passthrough() -> None:
    samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = _downmix_to_mono(samples, channels=1)
    assert np.array_equal(out, samples)


def test_downmix_stereo_averages_channels() -> None:
    # interleaved L,R,L,R: (1.0,-1.0) and (0.0, 0.5)
    samples = np.array([1.0, -1.0, 0.0, 0.5], dtype=np.float32)
    out = _downmix_to_mono(samples, channels=2)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.25)


def test_downmix_drops_incomplete_trailing_frame() -> None:
    # 5 samples, 2 channels -> only 2 full frames, trailing sample dropped
    samples = np.array([1.0, 1.0, 2.0, 2.0, 9.0], dtype=np.float32)
    out = _downmix_to_mono(samples, channels=2)
    assert out.shape == (2,)


# --------------------------------------------------------------------------
# resample_mono_to_16k
# --------------------------------------------------------------------------


def test_resample_identity_when_already_target_rate() -> None:
    samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    out = resample_mono_to_16k(samples, 16_000)
    assert np.array_equal(out, samples)


def test_resample_empty_input_returns_empty() -> None:
    out = resample_mono_to_16k(np.empty(0, dtype=np.float32), 48_000)
    assert out.size == 0
    assert out.dtype == np.float32


def test_resample_48k_to_16k_downsamples_by_three() -> None:
    # 48 kHz for 10ms = 480 samples -> 16 kHz for 10ms = 160 samples.
    n_in = 480
    ramp = np.linspace(0.0, 1.0, num=n_in, endpoint=False).astype(np.float32)
    out = resample_mono_to_16k(ramp, 48_000)
    assert out.size == 160
    # Linear interpolation of a ramp stays a ramp: endpoints preserved.
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[-1] == pytest.approx(ramp[-1], abs=1e-2)
    assert np.all(np.diff(out) >= -1e-6)  # still monotonic non-decreasing


def test_resample_preserves_dtype() -> None:
    ramp = np.linspace(0.0, 1.0, num=100, endpoint=False).astype(np.float32)
    out = resample_mono_to_16k(ramp, 44_100)
    assert out.dtype == np.float32


# --------------------------------------------------------------------------
# mix_chunks
# --------------------------------------------------------------------------


def test_mix_chunks_both_none_returns_empty() -> None:
    out = mix_chunks(None, None)
    assert out.size == 0


def test_mix_chunks_loopback_only() -> None:
    loop = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = mix_chunks(loop, None)
    assert np.allclose(out, loop)


def test_mix_chunks_mic_only() -> None:
    mic = np.array([0.4, -0.4], dtype=np.float32)
    out = mix_chunks(None, mic)
    assert np.allclose(out, mic)


def test_mix_chunks_silence_total() -> None:
    silence = np.zeros(160, dtype=np.float32)
    out = mix_chunks(silence, silence)
    assert out.shape == (160,)
    assert np.allclose(out, 0.0)


def test_mix_chunks_no_attenuation_when_loopback_quiet() -> None:
    quiet_loop = np.full(
        10, 0.001, dtype=np.float32
    )  # RMS well below default threshold
    mic = np.full(10, 0.1, dtype=np.float32)
    out = mix_chunks(quiet_loop, mic, echo_rms_threshold=0.02)
    expected = np.clip(quiet_loop + 1.0 * mic, -1.0, 1.0)
    assert np.allclose(out, expected)


def test_mix_chunks_attenuates_mic_when_loopback_loud() -> None:
    loud_loop = np.full(10, 0.5, dtype=np.float32)  # RMS well above default threshold
    mic = np.full(10, 0.1, dtype=np.float32)
    out = mix_chunks(loud_loop, mic, mic_attenuation=0.25, echo_rms_threshold=0.02)
    expected = np.clip(loud_loop + 0.25 * mic, -1.0, 1.0)
    assert np.allclose(out, expected)


def test_mix_chunks_pads_shorter_side_with_silence() -> None:
    loop = np.full(5, 0.1, dtype=np.float32)
    mic = np.full(2, 0.1, dtype=np.float32)
    out = mix_chunks(loop, mic, echo_rms_threshold=1.0)  # force gain=1.0 (quiet branch)
    assert out.shape == (5,)
    # Tail samples (index 2..4) only have loopback contribution, no mic padding leak.
    assert out[2:].tolist() == pytest.approx(loop[2:].tolist())


def test_mix_chunks_clips_to_valid_range() -> None:
    loop = np.full(4, 0.9, dtype=np.float32)
    mic = np.full(4, 0.9, dtype=np.float32)
    out = mix_chunks(loop, mic, mic_attenuation=1.0, echo_rms_threshold=1.0)
    assert np.all(out <= 1.0)
    assert np.all(out >= -1.0)


# --------------------------------------------------------------------------
# _resolve_loopback_device
# --------------------------------------------------------------------------


def test_resolve_loopback_auto_success() -> None:
    pa = FakePyAudio(loopback_device=LOOPBACK_DEVICE)
    pyaudio_mod = FakePyAudioModule(pa)
    device = _resolve_loopback_device(pa, pyaudio_mod, "auto")
    assert device is LOOPBACK_DEVICE


def test_resolve_loopback_auto_wasapi_unavailable_raises() -> None:
    pa = FakePyAudio(loopback_device=LOOPBACK_DEVICE, wasapi_available=False)
    pyaudio_mod = FakePyAudioModule(pa)
    with pytest.raises(CaptureError, match="WASAPI indisponible"):
        _resolve_loopback_device(pa, pyaudio_mod, "auto")


def test_resolve_loopback_auto_not_found_raises() -> None:
    pa = FakePyAudio(loopback_device=None)
    pyaudio_mod = FakePyAudioModule(pa)
    with pytest.raises(CaptureError, match="loopback WASAPI"):
        _resolve_loopback_device(pa, pyaudio_mod, "auto")


def test_resolve_loopback_named_match_case_insensitive() -> None:
    pa = FakePyAudio(loopback_devices=[LOOPBACK_DEVICE])
    pyaudio_mod = FakePyAudioModule(pa)
    device = _resolve_loopback_device(pa, pyaudio_mod, "realtek")
    assert device is LOOPBACK_DEVICE


def test_resolve_loopback_named_not_found_raises() -> None:
    pa = FakePyAudio(loopback_devices=[LOOPBACK_DEVICE])
    pyaudio_mod = FakePyAudioModule(pa)
    with pytest.raises(CaptureError, match="devices.loopback"):
        _resolve_loopback_device(pa, pyaudio_mod, "nonexistent device")


# --------------------------------------------------------------------------
# _resolve_mic_device
# --------------------------------------------------------------------------


def test_resolve_mic_none_disabled_returns_none() -> None:
    pa = FakePyAudio(mic_device=MIC_DEVICE)
    pyaudio_mod = FakePyAudioModule(pa)
    assert _resolve_mic_device(pa, pyaudio_mod, "none") is None


def test_resolve_mic_auto_uses_wasapi_default_input() -> None:
    pa = FakePyAudio(mic_device=MIC_DEVICE)
    pyaudio_mod = FakePyAudioModule(pa)
    device = _resolve_mic_device(pa, pyaudio_mod, "auto")
    assert device is MIC_DEVICE


def test_resolve_mic_auto_falls_back_to_generic_default_when_wasapi_missing() -> None:
    pa = FakePyAudio(mic_device=MIC_DEVICE, wasapi_available=False)
    pyaudio_mod = FakePyAudioModule(pa)
    device = _resolve_mic_device(pa, pyaudio_mod, "auto")
    assert device is MIC_DEVICE  # via get_default_input_device_info fallback


def test_resolve_mic_auto_absent_returns_none() -> None:
    pa = FakePyAudio(mic_device=None, wasapi_available=False)
    pyaudio_mod = FakePyAudioModule(pa)
    assert _resolve_mic_device(pa, pyaudio_mod, "auto") is None


def test_resolve_mic_named_match() -> None:
    pa = FakePyAudio(input_devices=[MIC_DEVICE])
    pyaudio_mod = FakePyAudioModule(pa)
    device = _resolve_mic_device(pa, pyaudio_mod, "microphone array")
    assert device is MIC_DEVICE


def test_resolve_mic_named_not_found_returns_none_not_raises() -> None:
    pa = FakePyAudio(input_devices=[MIC_DEVICE])
    pyaudio_mod = FakePyAudioModule(pa)
    assert _resolve_mic_device(pa, pyaudio_mod, "nonexistent mic") is None


def test_resolve_mic_named_excludes_loopback_devices() -> None:
    pa = FakePyAudio(input_devices=[LOOPBACK_DEVICE])
    pyaudio_mod = FakePyAudioModule(pa)
    # LOOPBACK_DEVICE's name would otherwise substring-match "speakers"
    assert _resolve_mic_device(pa, pyaudio_mod, "speakers") is None


# --------------------------------------------------------------------------
# list_devices
# --------------------------------------------------------------------------


def test_list_devices_returns_expected_shape() -> None:
    pa = FakePyAudio(all_devices=[LOOPBACK_DEVICE, MIC_DEVICE])
    pyaudio_mod = FakePyAudioModule(pa)
    devices = list_devices(pyaudio_module=pyaudio_mod)
    assert len(devices) == 2
    assert devices[0] == {
        "index": 0,
        "name": "Speakers (Realtek) [Loopback]",
        "is_loopback": True,
        "max_input_channels": 2,
        "default_sample_rate": 48000.0,
        "host_api": 13,
    }
    assert devices[1]["is_loopback"] is False
    assert pa.terminated is True  # context manager closed it


# --------------------------------------------------------------------------
# AudioCapture integration (fake PyAudio, real threading/queue)
# --------------------------------------------------------------------------


def _sine_int16(n: int, amplitude: float = 0.5) -> bytes:
    t = np.arange(n)
    samples = (amplitude * np.sin(2 * np.pi * 0.05 * t)).astype(np.float32)
    return (samples * 32767).astype(np.int16).tobytes()


def _silence_int16(n: int) -> bytes:
    return np.zeros(n, dtype=np.int16).tobytes()


def test_audio_capture_start_opens_loopback_and_mic_streams() -> None:
    pa = FakePyAudio(loopback_device=LOOPBACK_DEVICE, mic_device=MIC_DEVICE)
    pyaudio_mod = FakePyAudioModule(pa)
    audio_q = make_audio_queue()
    devices_cfg = DevicesConfig(loopback="auto", mic="auto")
    capture = AudioCapture(
        devices_cfg, audio_q, block_ms=10, pyaudio_module=pyaudio_mod
    )

    capture.start()
    try:
        assert len(pa.streams) == 2
        assert capture.mic_active is True
    finally:
        capture.stop()

    assert pa.streams[0].stopped and pa.streams[0].closed
    assert pa.streams[1].stopped and pa.streams[1].closed
    assert pa.terminated is True
    assert capture._mixer_thread is None


def test_audio_capture_mic_none_config_opens_loopback_only() -> None:
    pa = FakePyAudio(loopback_device=LOOPBACK_DEVICE, mic_device=MIC_DEVICE)
    pyaudio_mod = FakePyAudioModule(pa)
    audio_q = make_audio_queue()
    devices_cfg = DevicesConfig(loopback="auto", mic="none")
    capture = AudioCapture(
        devices_cfg, audio_q, block_ms=10, pyaudio_module=pyaudio_mod
    )

    capture.start()
    try:
        assert len(pa.streams) == 1
        assert capture.mic_active is False
    finally:
        capture.stop()


def test_audio_capture_mic_absent_falls_back_to_loopback_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pa = FakePyAudio(
        loopback_device=LOOPBACK_DEVICE, mic_device=None, wasapi_available=True
    )
    pyaudio_mod = FakePyAudioModule(pa)
    audio_q = make_audio_queue()
    devices_cfg = DevicesConfig(loopback="auto", mic="auto")
    capture = AudioCapture(
        devices_cfg, audio_q, block_ms=10, pyaudio_module=pyaudio_mod
    )

    with caplog.at_level(logging.WARNING):
        capture.start()
    try:
        assert len(pa.streams) == 1
        assert capture.mic_active is False
        assert any("fallback" in record.message for record in caplog.records)
    finally:
        capture.stop()


def test_audio_capture_loopback_absent_raises_and_cleans_up() -> None:
    pa = FakePyAudio(loopback_device=None)
    pyaudio_mod = FakePyAudioModule(pa)
    audio_q = make_audio_queue()
    devices_cfg = DevicesConfig(loopback="auto", mic="none")
    capture = AudioCapture(
        devices_cfg, audio_q, block_ms=10, pyaudio_module=pyaudio_mod
    )

    with pytest.raises(CaptureError):
        capture.start()
    assert pa.terminated is True
    assert capture._started is False


def test_audio_capture_pushes_mixed_chunks_with_monotonic_timestamps() -> None:
    # Both devices report 16 kHz so the resampling path is a no-op and the
    # test only exercises mixing/threading/queue wiring (resampling itself
    # is covered separately above).
    loopback_16k = dict(LOOPBACK_DEVICE, defaultSampleRate=16000.0, maxInputChannels=1)
    mic_16k = dict(MIC_DEVICE, defaultSampleRate=16000.0, maxInputChannels=1)
    pa = FakePyAudio(loopback_device=loopback_16k, mic_device=mic_16k)
    pyaudio_mod = FakePyAudioModule(pa)
    audio_q = make_audio_queue()
    devices_cfg = DevicesConfig(loopback="auto", mic="auto")
    capture = AudioCapture(
        devices_cfg, audio_q, block_ms=10, pyaudio_module=pyaudio_mod
    )

    capture.start()
    try:
        loop_stream, mic_stream = pa.streams
        for _ in range(3):
            loop_stream.callback(_sine_int16(160), 160, {}, 0)
            mic_stream.callback(_silence_int16(160), 160, {}, 0)
            time.sleep(0.03)
    finally:
        capture.stop()

    chunks: list[AudioChunk] = []
    while True:
        try:
            chunks.append(audio_q.get_nowait())
        except queue.Empty:
            break

    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, AudioChunk)
        assert chunk.samples.dtype == np.float32
    timestamps = [c.timestamp for c in chunks]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)  # strictly increasing, no duplicates


def test_audio_capture_silence_only_still_produces_zero_chunks() -> None:
    loopback_16k = dict(LOOPBACK_DEVICE, defaultSampleRate=16000.0, maxInputChannels=1)
    pa = FakePyAudio(loopback_device=loopback_16k, mic_device=None)
    pyaudio_mod = FakePyAudioModule(pa)
    audio_q = make_audio_queue()
    devices_cfg = DevicesConfig(loopback="auto", mic="none")
    capture = AudioCapture(
        devices_cfg, audio_q, block_ms=10, pyaudio_module=pyaudio_mod
    )

    capture.start()
    try:
        (loop_stream,) = pa.streams
        loop_stream.callback(_silence_int16(160), 160, {}, 0)
        time.sleep(0.03)
    finally:
        capture.stop()

    chunk = audio_q.get_nowait()
    assert np.allclose(chunk.samples, 0.0)


def test_audio_capture_double_start_is_idempotent() -> None:
    pa = FakePyAudio(loopback_device=LOOPBACK_DEVICE, mic_device=None)
    pyaudio_mod = FakePyAudioModule(pa)
    audio_q = make_audio_queue()
    devices_cfg = DevicesConfig(loopback="auto", mic="none")
    capture = AudioCapture(
        devices_cfg, audio_q, block_ms=10, pyaudio_module=pyaudio_mod
    )

    capture.start()
    capture.start()  # second call must be a no-op, not open a second stream
    try:
        assert len(pa.streams) == 1
    finally:
        capture.stop()


def test_audio_capture_double_stop_is_safe() -> None:
    pa = FakePyAudio(loopback_device=LOOPBACK_DEVICE, mic_device=None)
    pyaudio_mod = FakePyAudioModule(pa)
    audio_q = make_audio_queue()
    devices_cfg = DevicesConfig(loopback="auto", mic="none")
    capture = AudioCapture(
        devices_cfg, audio_q, block_ms=10, pyaudio_module=pyaudio_mod
    )

    capture.start()
    capture.stop()
    capture.stop()  # must not raise


def test_audio_capture_context_manager() -> None:
    pa = FakePyAudio(loopback_device=LOOPBACK_DEVICE, mic_device=None)
    pyaudio_mod = FakePyAudioModule(pa)
    audio_q = make_audio_queue()
    devices_cfg = DevicesConfig(loopback="auto", mic="none")

    with AudioCapture(
        devices_cfg, audio_q, block_ms=10, pyaudio_module=pyaudio_mod
    ) as capture:
        assert capture.mic_active is False
    assert pa.terminated is True
