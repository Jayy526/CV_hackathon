"""Phase E: frame analysis and WAV I/O.

WAV reading/writing uses the standard library `wave` module on purpose: no
sound card, no PortAudio, no optional dependency, so the tests run anywhere.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from heimdall.audio.frame import AudioFrame

EPS = 1e-12


def rms(x: np.ndarray) -> float:
    """Root-mean-square of any-shaped audio. Returns 0.0 for empty input."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def peak(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def channel_rms(frame: AudioFrame) -> np.ndarray:
    """Per-channel RMS, shape (num_channels,)."""
    if frame.num_samples == 0:
        return np.zeros(frame.num_channels)
    return np.sqrt(np.mean(np.square(frame.samples.astype(np.float64)), axis=0))


def channel_peak(frame: AudioFrame) -> np.ndarray:
    if frame.num_samples == 0:
        return np.zeros(frame.num_channels)
    return np.max(np.abs(frame.samples.astype(np.float64)), axis=0)


def db_fs(value: float) -> float:
    """Convert a linear amplitude (1.0 = full scale) to dBFS."""
    return float(20.0 * np.log10(max(float(value), EPS)))


def zero_crossing_rate(x: np.ndarray) -> float:
    """Fraction of adjacent sample pairs that change sign."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return 0.0
    return float(np.mean(np.diff(np.signbit(x)) != 0))


def spectral_flatness(x: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of the power spectrum.

    Near 1.0 for noise-like signals, near 0.0 for tonal/harmonic signals.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    spectrum = spectrum[1:]  # drop DC
    if spectrum.size == 0 or not np.any(spectrum > 0):
        return 0.0
    geometric = np.exp(np.mean(np.log(spectrum + EPS)))
    arithmetic = np.mean(spectrum) + EPS
    return float(np.clip(geometric / arithmetic, 0.0, 1.0))


def band_energy_ratio(
    x: np.ndarray,
    sample_rate: int,
    low_hz: float = 300.0,
    high_hz: float = 3400.0,
) -> float:
    """Fraction of total spectral energy inside [low_hz, high_hz].

    The default band is the speech band, so this is a cheap speech-likeness cue.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    total = float(np.sum(spectrum)) + EPS
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(spectrum[mask]) / total)


@dataclass(frozen=True)
class FrameStats:
    """Everything the diagnostics tool needs about one frame."""

    frame_index: int
    timestamp: float
    rms: float
    peak: float
    channel_rms: tuple[float, ...]
    channel_peak: tuple[float, ...]
    zero_crossing_rate: float
    spectral_flatness: float
    speech_band_ratio: float

    @property
    def rms_dbfs(self) -> float:
        return db_fs(self.rms)


def analyse_frame(frame: AudioFrame) -> FrameStats:
    mono = frame.samples.mean(axis=1) if frame.num_channels > 1 else frame.channel(0)
    return FrameStats(
        frame_index=frame.frame_index,
        timestamp=frame.timestamp,
        rms=rms(frame.samples),
        peak=peak(frame.samples),
        channel_rms=tuple(float(v) for v in channel_rms(frame)),
        channel_peak=tuple(float(v) for v in channel_peak(frame)),
        zero_crossing_rate=zero_crossing_rate(mono),
        spectral_flatness=spectral_flatness(mono),
        speech_band_ratio=band_energy_ratio(mono, frame.sample_rate),
    )


def spectrogram(
    x: np.ndarray,
    sample_rate: int,
    window_size: int = 512,
    hop: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude spectrogram. Returns (freqs, times, magnitude[freq, time])."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < window_size:
        x = np.pad(x, (0, window_size - x.size))

    window = np.hanning(window_size)
    starts = range(0, x.size - window_size + 1, hop)
    columns = [np.abs(np.fft.rfft(x[s : s + window_size] * window)) for s in starts]

    magnitude = np.stack(columns, axis=1) if columns else np.zeros((window_size // 2 + 1, 0))
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    times = np.array([s / sample_rate for s in starts])
    return freqs, times, magnitude


def concatenate_frames(frames: list[AudioFrame]) -> np.ndarray:
    """Stack frames into one (num_samples, num_channels) array."""
    if not frames:
        return np.zeros((0, 0), dtype=np.float32)
    channels = frames[0].num_channels
    if any(f.num_channels != channels for f in frames):
        raise ValueError("cannot concatenate frames with differing channel counts")
    return np.vstack([f.samples for f in frames]).astype(np.float32)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write float samples in [-1, 1] as a 16-bit PCM WAV.

    Channel 0 becomes the left/first WAV channel, channel 1 the second, so the
    file can be inspected directly in any audio editor.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    if data.ndim != 2:
        raise ValueError(f"samples must be 2-D, got shape {data.shape}")

    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(pcm.shape[1])
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    return path


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV back as (num_samples, num_channels) float32."""
    with wave.open(str(path), "rb") as fh:
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        sample_rate = fh.getframerate()
        raw = fh.readframes(fh.getnframes())

    if width != 2:
        raise ValueError(f"only 16-bit PCM WAV is supported, got {width * 8}-bit")

    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
    return (pcm.astype(np.float32) / 32767.0), sample_rate
