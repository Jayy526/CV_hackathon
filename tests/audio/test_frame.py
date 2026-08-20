"""Frame representation: shape, framing metadata, channel access."""

import numpy as np
import pytest

from heimdall.audio.frame import AudioFrame


def make_frame(num_samples=1024, num_channels=2, sample_rate=48000, index=0):
    samples = np.zeros((num_samples, num_channels), dtype=np.float32)
    return AudioFrame(
        samples=samples,
        timestamp=index * num_samples / sample_rate,
        frame_index=index,
        sample_rate=sample_rate,
    )


def test_frame_reports_its_shape():
    frame = make_frame(1024, 2)
    assert frame.num_samples == 1024
    assert frame.num_channels == 2


def test_frame_duration_matches_sample_rate():
    frame = make_frame(48000, 2, sample_rate=48000)
    assert frame.duration == pytest.approx(1.0)
    assert frame.end_timestamp == pytest.approx(frame.timestamp + 1.0)


def test_frame_size_is_configurable():
    for size in (256, 512, 1024, 2048):
        assert make_frame(size).num_samples == size


def test_channel_access_returns_1d():
    frame = make_frame(64, 2)
    assert frame.channel(0).shape == (64,)
    assert frame.channel(1).shape == (64,)


def test_channel_index_out_of_range_raises():
    frame = make_frame(64, 2)
    with pytest.raises(IndexError):
        frame.channel(2)
    with pytest.raises(IndexError):
        frame.channel(-1)


def test_one_dimensional_samples_rejected():
    with pytest.raises(ValueError):
        AudioFrame(np.zeros(1024, dtype=np.float32), 0.0, 0, 48000)


def test_non_positive_sample_rate_rejected():
    with pytest.raises(ValueError):
        AudioFrame(np.zeros((16, 2), dtype=np.float32), 0.0, 0, 0)


def test_four_channel_frame_is_supported():
    """The frame type must not assume two channels (Phase L readiness)."""
    frame = make_frame(512, 4)
    assert frame.num_channels == 4
    assert frame.channel(3).shape == (512,)


def test_with_samples_preserves_timing():
    frame = make_frame(64, 2, index=7)
    replaced = frame.with_samples(np.ones((64, 2), dtype=np.float32))
    assert replaced.frame_index == frame.frame_index
    assert replaced.timestamp == frame.timestamp
    assert replaced.sample_rate == frame.sample_rate
    assert np.all(replaced.samples == 1.0)
