"""The hardware boundary: the mock source works, the ESP32 one is honestly absent."""

import numpy as np
import pytest

from heimdall.audio.config import AudioConfig
from heimdall.audio.sources import (
    AudioSource,
    AudioSourceError,
    ESP32AudioSource,
    SyntheticAudioSource,
)


def test_synthetic_source_is_an_audio_source():
    assert issubclass(SyntheticAudioSource, AudioSource)


def test_esp32_source_is_an_audio_source():
    """The future hardware source must satisfy the same interface."""
    assert issubclass(ESP32AudioSource, AudioSource)


def test_esp32_source_refuses_to_pretend_it_works():
    with pytest.raises(NotImplementedError) as excinfo:
        ESP32AudioSource()
    assert "not implemented" in str(excinfo.value).lower()


def test_source_reports_configured_format():
    source = SyntheticAudioSource(sample_rate=16000, num_channels=2, frame_size=512)
    assert source.sample_rate == 16000
    assert source.num_channels == 2
    assert source.frame_size == 512


def test_frames_have_the_configured_shape():
    source = SyntheticAudioSource(48000, 2, 1024, max_frames=3)
    source.start()
    frame = source.read_frame()
    source.stop()
    assert frame.samples.shape == (1024, 2)
    assert frame.sample_rate == 48000


def test_four_channels_are_supported_without_changes():
    """Phase L readiness: nothing in the source assumes two channels."""
    source = SyntheticAudioSource(48000, 4, 256, max_frames=2)
    source.start()
    frame = source.read_frame()
    source.stop()
    assert frame.num_channels == 4


def test_frame_indices_and_timestamps_increase():
    source = SyntheticAudioSource(48000, 2, 512, max_frames=5)
    source.start()
    frames = [source.read_frame() for _ in range(5)]
    source.stop()

    assert [f.frame_index for f in frames] == [0, 1, 2, 3, 4]
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[1] == pytest.approx(512 / 48000)


def test_max_frames_exhausts_the_source():
    source = SyntheticAudioSource(48000, 2, 256, max_frames=2)
    source.start()
    assert source.read_frame() is not None
    assert source.read_frame() is not None
    assert source.read_frame() is None
    source.stop()


def test_reading_before_start_raises():
    source = SyntheticAudioSource(48000, 2, 256)
    with pytest.raises(AudioSourceError):
        source.read_frame()


def test_source_is_deterministic():
    def capture():
        source = SyntheticAudioSource(48000, 2, 256, max_frames=3, seed=42)
        source.start()
        frames = [source.read_frame().samples.copy() for _ in range(3)]
        source.stop()
        return frames

    for a, b in zip(capture(), capture()):
        assert np.array_equal(a, b)


def test_silence_frames_are_quiet_and_burst_frames_are_not():
    source = SyntheticAudioSource(
        48000, 2, 512, burst_frames=2, silence_frames=2, noise_amplitude=0.0, max_frames=4
    )
    source.start()
    levels = [float(np.sqrt(np.mean(source.read_frame().samples ** 2))) for _ in range(4)]
    source.stop()
    assert levels[0] > 0.01 and levels[1] > 0.01
    assert levels[2] < 1e-6 and levels[3] < 1e-6


def test_from_buffer_replays_exact_samples():
    buffer = np.arange(2048, dtype=np.float32).reshape(1024, 2) / 2048.0
    source = SyntheticAudioSource.from_buffer(buffer, sample_rate=48000, frame_size=256)
    source.start()
    replayed = np.vstack([source.read_frame().samples for _ in range(4)])
    source.stop()
    assert np.allclose(replayed, buffer)


def test_from_buffer_pads_a_short_final_frame():
    buffer = np.ones((300, 2), dtype=np.float32)
    source = SyntheticAudioSource.from_buffer(buffer, 48000, frame_size=256)
    source.start()
    source.read_frame()
    tail = source.read_frame()
    source.stop()
    assert tail.num_samples == 256
    assert np.all(tail.samples[44:] == 0.0)


def test_from_buffer_rejects_a_transposed_buffer():
    with pytest.raises(ValueError):
        SyntheticAudioSource.from_buffer(np.zeros((2, 1024), dtype=np.float32), 48000)


def test_from_config_uses_the_yaml_values():
    config = AudioConfig(sample_rate=16000, num_channels=2, frame_size=320)
    source = SyntheticAudioSource.from_config(config)
    assert (source.sample_rate, source.num_channels, source.frame_size) == (16000, 2, 320)


def test_context_manager_starts_and_stops():
    with SyntheticAudioSource(48000, 2, 128, max_frames=1) as source:
        assert source.is_running
        assert source.read_frame() is not None
    assert not source.is_running
