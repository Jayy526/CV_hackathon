"""Receiver: framing, format reporting, non-blocking behaviour, no-audio case."""

import numpy as np

from heimdall.audio.frame import AudioFrame
from heimdall.audio.receiver import AudioReceiver
from heimdall.audio.sources import AudioSource, SyntheticAudioSource


class NoAudioSource(AudioSource):
    """A source that opens fine and then produces nothing at all."""

    def start(self):
        self._running = True

    def read_frame(self):
        return None

    def stop(self):
        self._running = False


class FailingSource(AudioSource):
    """A source whose device dies mid-stream."""

    def start(self):
        self._running = True

    def read_frame(self):
        raise OSError("device disconnected")

    def stop(self):
        self._running = False


def test_receiver_reports_the_source_format():
    receiver = AudioReceiver(SyntheticAudioSource(16000, 2, 320))
    assert receiver.sample_rate == 16000
    assert receiver.num_channels == 2
    assert receiver.frame_size == 320


def test_receiver_yields_frames_of_the_configured_size():
    with AudioReceiver(SyntheticAudioSource(48000, 2, 1024, max_frames=4)) as receiver:
        frame = receiver.read_frame(timeout=2.0)
    assert isinstance(frame, AudioFrame)
    assert frame.samples.shape == (1024, 2)


def test_frames_arrive_in_order():
    with AudioReceiver(SyntheticAudioSource(48000, 2, 256, max_frames=6)) as receiver:
        frames = receiver.read_frames(6, timeout=2.0)
    assert [f.frame_index for f in frames] == [0, 1, 2, 3, 4, 5]


def test_read_frames_stops_early_at_end_of_stream():
    with AudioReceiver(SyntheticAudioSource(48000, 2, 256, max_frames=3)) as receiver:
        frames = receiver.read_frames(10, timeout=1.0)
    assert len(frames) == 3


def test_no_audio_condition_returns_none_without_hanging():
    """No ESP32, no samples: read_frame must time out cleanly, not block forever."""
    receiver = AudioReceiver(NoAudioSource(48000, 2, 1024))
    receiver.start()
    assert receiver.read_frame(timeout=0.2) is None
    assert receiver.is_exhausted
    receiver.stop()


def test_source_failure_is_surfaced_not_swallowed():
    receiver = AudioReceiver(FailingSource(48000, 2, 1024))
    receiver.start()
    error = None
    for _ in range(20):
        try:
            receiver.read_frame(timeout=0.1)
        except OSError as exc:
            error = exc
            break
    receiver.stop()
    assert error is not None


def test_slow_consumer_drops_frames_instead_of_blocking():
    """A backed-up consumer must never stall acquisition."""
    receiver = AudioReceiver(SyntheticAudioSource(48000, 2, 64, max_frames=500), queue_size=4)
    receiver.start()
    while not receiver._exhausted.is_set():  # noqa: SLF001 - deliberate white-box wait
        pass
    frames = receiver.read_frames(500, timeout=0.2)
    receiver.stop()

    assert receiver.stats.frames_received == 500
    assert receiver.stats.frames_dropped > 0
    assert len(frames) <= 5


def test_stats_track_received_frames():
    with AudioReceiver(SyntheticAudioSource(48000, 2, 256, max_frames=4)) as receiver:
        receiver.read_frames(4, timeout=2.0)
        assert receiver.stats.frames_received == 4
        assert receiver.stats.last_timestamp is not None


def test_stop_is_idempotent():
    receiver = AudioReceiver(SyntheticAudioSource(48000, 2, 128, max_frames=2))
    receiver.start()
    receiver.stop()
    receiver.stop()
    assert not receiver.is_running


def test_synthetic_helper_builds_a_working_receiver():
    with AudioReceiver.synthetic(max_frames=2) as receiver:
        frame = receiver.read_frame(timeout=2.0)
    assert frame is not None
    assert np.isfinite(frame.samples).all()
