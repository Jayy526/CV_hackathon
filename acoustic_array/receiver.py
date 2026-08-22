"""Phase D: the laptop-side audio receiver.

Pulls frames off an AudioSource on a background thread and hands them to the
rest of the system through a bounded queue, so a slow consumer never blocks
acquisition (it drops the oldest frames instead and counts the drops).

Knows nothing about GCC-PHAT, DOA or seats.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from acoustic_array.config import AudioConfig, load_audio_config
from acoustic_array.frame import AudioFrame
from acoustic_array.sources import AudioSource, SyntheticAudioSource


@dataclass
class ReceiverStats:
    frames_received: int = 0
    frames_dropped: int = 0
    last_timestamp: float | None = None


class AudioReceiver:
    """Non-blocking frame reader over an arbitrary AudioSource.

        receiver = AudioReceiver(SyntheticAudioSource())
        receiver.start()
        frame = receiver.read_frame(timeout=1.0)
        receiver.stop()
    """

    def __init__(self, source: AudioSource, queue_size: int = 32) -> None:
        self.source = source
        self._queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._exhausted = threading.Event()
        self._error: BaseException | None = None
        self.stats = ReceiverStats()

    @classmethod
    def synthetic(cls, config: AudioConfig | None = None, **kwargs: object) -> "AudioReceiver":
        """Convenience constructor for hardware-free development."""
        config = config or load_audio_config()
        return cls(SyntheticAudioSource.from_config(config, **kwargs))

    @property
    def sample_rate(self) -> int:
        return self.source.sample_rate

    @property
    def num_channels(self) -> int:
        return self.source.num_channels

    @property
    def frame_size(self) -> int:
        return self.source.frame_size

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_exhausted(self) -> bool:
        """True when the source ran out AND the queue has been drained."""
        return self._exhausted.is_set() and self._queue.empty()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._exhausted.clear()
        self._error = None
        self.source.start()
        self._thread = threading.Thread(
            target=self._run, name="heimdall-audio-receiver", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self.source.read_frame()
                if frame is None:
                    self._exhausted.set()
                    return
                self.stats.frames_received += 1
                self.stats.last_timestamp = frame.timestamp
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    # Consumer is behind: drop the oldest frame, keep the newest.
                    try:
                        self._queue.get_nowait()
                        self.stats.frames_dropped += 1
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put_nowait(frame)
                    except queue.Full:
                        self.stats.frames_dropped += 1
        except BaseException as exc:  # noqa: BLE001 - surfaced on read_frame()
            self._error = exc
            self._exhausted.set()

    def read_frame(self, timeout: float | None = 1.0) -> AudioFrame | None:
        """Return the next frame, or None on timeout / end of stream."""
        if self._error is not None:
            raise self._error
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            if self._error is not None:
                raise self._error
            return None

    def read_frames(self, count: int, timeout: float | None = 1.0) -> list[AudioFrame]:
        """Read up to `count` frames, stopping early at end of stream."""
        frames: list[AudioFrame] = []
        for _ in range(count):
            frame = self.read_frame(timeout=timeout)
            if frame is None:
                break
            frames.append(frame)
        return frames

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.source.stop()

    def __enter__(self) -> "AudioReceiver":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
