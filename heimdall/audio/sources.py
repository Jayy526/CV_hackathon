"""The hardware boundary.

    AudioSource                (abstract)
    |-- SyntheticAudioSource   (implemented - deterministic, no hardware)
    |-- ESP32AudioSource       (NOT IMPLEMENTED - awaiting exact hardware)

Everything downstream of this file depends on AudioSource only. Adding the
ESP32 later means adding one subclass here and changing nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from heimdall.audio import synthetic
from heimdall.audio.config import AudioConfig, load_audio_config
from heimdall.audio.frame import AudioFrame


class AudioSourceError(RuntimeError):
    """Raised when a source cannot be opened or has failed irrecoverably."""


class AudioSource(ABC):
    """A producer of fixed-size multi-channel AudioFrames."""

    def __init__(self, sample_rate: int, num_channels: int, frame_size: int) -> None:
        self._sample_rate = int(sample_rate)
        self._num_channels = int(num_channels)
        self._frame_size = int(frame_size)
        self._running = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def frame_size(self) -> int:
        return self._frame_size

    @property
    def is_running(self) -> bool:
        return self._running

    @abstractmethod
    def start(self) -> None:
        """Open the underlying device/generator."""

    @abstractmethod
    def read_frame(self) -> AudioFrame | None:
        """Return the next frame, or None if the source is exhausted."""

    @abstractmethod
    def stop(self) -> None:
        """Release the underlying device."""

    def __enter__(self) -> "AudioSource":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


class SyntheticAudioSource(AudioSource):
    """Deterministic mock source: no ESP32, no COM port, no sound card.

    By default it produces a repeating pattern of silence and speech-like bursts
    arriving from `angle_degrees`, so the full pipeline (GCC-PHAT -> DOA -> seat
    mapping -> events) can be exercised end to end in tests.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        num_channels: int = 2,
        frame_size: int = 1024,
        *,
        angle_degrees: float = 0.0,
        mic_spacing_m: float = 0.3,
        noise_amplitude: float = 0.002,
        burst_frames: int = 8,
        silence_frames: int = 8,
        max_frames: int | None = None,
        seed: int = 0,
        buffer: np.ndarray | None = None,
    ) -> None:
        super().__init__(sample_rate, num_channels, frame_size)
        self.angle_degrees = angle_degrees
        self.mic_spacing_m = mic_spacing_m
        self.noise_amplitude = noise_amplitude
        self.burst_frames = max(int(burst_frames), 0)
        self.silence_frames = max(int(silence_frames), 0)
        self.max_frames = max_frames
        self.seed = seed
        self._buffer = buffer
        self._frame_index = 0

    @classmethod
    def from_buffer(
        cls,
        buffer: np.ndarray,
        sample_rate: int,
        frame_size: int = 1024,
    ) -> "SyntheticAudioSource":
        """Replay a fixed (num_samples, num_channels) array frame by frame."""
        buffer = np.atleast_2d(np.asarray(buffer, dtype=np.float32))
        if buffer.shape[0] < buffer.shape[1]:
            raise ValueError(
                "buffer must be shaped (num_samples, num_channels); "
                f"got {buffer.shape} which looks transposed"
            )
        return cls(
            sample_rate=sample_rate,
            num_channels=buffer.shape[1],
            frame_size=frame_size,
            buffer=buffer,
        )

    @classmethod
    def from_config(cls, config: AudioConfig | None = None, **kwargs: object) -> "SyntheticAudioSource":
        config = config or load_audio_config()
        return cls(
            sample_rate=config.sample_rate,
            num_channels=config.num_channels,
            frame_size=config.frame_size,
            **kwargs,  # type: ignore[arg-type]
        )

    def start(self) -> None:
        self._running = True
        self._frame_index = 0

    def stop(self) -> None:
        self._running = False

    def _delays(self) -> np.ndarray:
        """Per-channel delays in samples for a uniform linear array."""
        tdoa = synthetic.tdoa_for_angle(
            self.angle_degrees, self.mic_spacing_m, self.sample_rate
        )
        # tdoa_for_angle gives the delay of channel 0 relative to channel 1, so
        # channel c is delayed by -c * tdoa for a uniform linear array.
        return -np.arange(self.num_channels, dtype=np.float64) * tdoa

    def _generate_frame(self, index: int) -> np.ndarray:
        period = self.burst_frames + self.silence_frames
        # silence_frames=0 means a continuous sound; burst_frames=0 means silence.
        in_burst = period > 0 and (index % period) < self.burst_frames

        if in_burst:
            source = synthetic.speech_like(
                self.frame_size, self.sample_rate, seed=self.seed + index, amplitude=0.3
            )
        else:
            source = np.zeros(self.frame_size)

        return synthetic.simulate_array_signals(
            source,
            self._delays(),
            noise_amplitude=self.noise_amplitude,
            seed=self.seed + index,
        )

    def read_frame(self) -> AudioFrame | None:
        if not self._running:
            raise AudioSourceError("read_frame() called before start()")

        index = self._frame_index
        if self.max_frames is not None and index >= self.max_frames:
            return None

        if self._buffer is not None:
            begin = index * self.frame_size
            if begin >= self._buffer.shape[0]:
                return None
            chunk = self._buffer[begin : begin + self.frame_size]
            if chunk.shape[0] < self.frame_size:
                pad = np.zeros(
                    (self.frame_size - chunk.shape[0], chunk.shape[1]), dtype=np.float32
                )
                chunk = np.vstack([chunk, pad])
            samples = chunk.astype(np.float32)
        else:
            samples = self._generate_frame(index)

        self._frame_index += 1
        return AudioFrame(
            samples=samples,
            timestamp=index * self.frame_size / self.sample_rate,
            frame_index=index,
            sample_rate=self.sample_rate,
        )


class ESP32AudioSource(AudioSource):
    """Placeholder for the real hardware source.

    Deliberately NOT implemented. The exact ESP32-S3 board and INMP441 module
    have not been identified yet, so the serial framing, sample width and channel
    interleaving are unknown. Implementing it now would mean guessing the
    hardware, which is worse than not having it.

    When the hardware is known, implement start()/read_frame()/stop() here and
    the rest of the pipeline needs no changes.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "ESP32AudioSource is not implemented yet: the exact ESP32-S3 board and "
            "INMP441 wiring have not been provided. Use SyntheticAudioSource for "
            "hardware-independent development."
        )

    def start(self) -> None:  # pragma: no cover - unreachable
        raise NotImplementedError

    def read_frame(self) -> AudioFrame | None:  # pragma: no cover - unreachable
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - unreachable
        raise NotImplementedError
