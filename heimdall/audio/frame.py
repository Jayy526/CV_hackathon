"""AudioFrame: the unit of audio that flows through the whole pipeline.

Every stage (GCC-PHAT, DOA, event detection) consumes AudioFrame objects and
nothing else, so the pipeline never learns where the audio came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    """A fixed-size block of multi-channel audio.

    Attributes:
        samples: float32 array of shape (num_samples, num_channels), nominally
            in [-1.0, 1.0]. Channel order is channel_0, channel_1, ...
        timestamp: seconds, monotonic, referring to the FIRST sample in the frame.
        frame_index: monotonically increasing counter from the source.
        sample_rate: Hz.
    """

    samples: np.ndarray
    timestamp: float
    frame_index: int
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError(
                f"samples must be 2-D (num_samples, num_channels), got shape {self.samples.shape}"
            )
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def num_samples(self) -> int:
        return self.samples.shape[0]

    @property
    def num_channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate

    @property
    def end_timestamp(self) -> float:
        return self.timestamp + self.duration

    def channel(self, index: int) -> np.ndarray:
        """Return channel `index` as a 1-D array."""
        if not 0 <= index < self.num_channels:
            raise IndexError(
                f"channel {index} out of range for {self.num_channels}-channel frame"
            )
        return self.samples[:, index]

    def with_samples(self, samples: np.ndarray) -> "AudioFrame":
        """Return a copy carrying different samples but the same timing metadata."""
        return AudioFrame(
            samples=samples,
            timestamp=self.timestamp,
            frame_index=self.frame_index,
            sample_rate=self.sample_rate,
        )
