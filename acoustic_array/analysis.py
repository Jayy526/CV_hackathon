"""Phase E: frame analysis and WAV I/O.

WAV reading/writing uses the standard library `wave` module on purpose: no
sound card, no PortAudio, no optional dependency, so the tests run anywhere.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from acoustic_array.frame import AudioFrame

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


# --- transient detection -----------------------------------------------------
#
# WHY THIS EXISTS. A clap's direct sound is ~2 ms. Selecting frames by an
# absolute RMS threshold over a 12.8 s recording admits the whole recording
# whenever the room tone sits above that threshold - and then a steady source
# (a fan, a PC, the board's own regulator) dominates every estimate. Measured:
# with a steady source at -25 deg, RMS-gated selection returned -25.3 deg for
# claps at -60, -30, 0, +30 and +60. It was measuring the fan, not the claps.
#
# Onsets, not levels. And a floor measured from the recording, not a constant.


def noise_floor_rms(samples: np.ndarray, window: int = 256,
                    quantile: float = 0.2) -> float:
    """The room's own level, taken from the quietest part of the recording.

    An absolute threshold cannot know whether 0.01 is silence or a running fan.
    This does, because it asks the recording.
    """
    mono = np.asarray(samples, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    usable = (len(mono) // window) * window
    if usable == 0:
        return float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    blocks = mono[:usable].reshape(-1, window)
    levels = np.sqrt(np.mean(blocks ** 2, axis=1))
    return float(np.quantile(levels, quantile))


def find_onsets(
    samples: np.ndarray,
    sample_rate: int,
    *,
    # 10 dB above the measured floor. Tuned, not guessed: at 8/10/12 dB the
    # false-positive count on pure room tone is zero across five runs, and 10
    # still catches claps that 12 misses in a noisy room.
    threshold_db: float = 10.0,
    min_gap_seconds: float = 0.25,
    window: int = 64,
    lookback: int = 512,
    attack_fraction: float = 0.2,
) -> list[int]:
    """Sample indices where a transient starts, relative to the noise floor.

    `threshold_db` is above the measured floor, so a loud room raises the bar
    instead of flooding the result. Each index is walked back from the envelope
    peak to the START of the attack: by the time the envelope peaks the first
    reflections have usually already arrived, and correlating those measures
    the room rather than the source.
    """
    mono = np.asarray(samples, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    if mono.size < window * 2:
        return []

    floor = noise_floor_rms(mono, window=max(window, 256))
    threshold = max(floor * (10.0 ** (threshold_db / 20.0)), 1e-9)

    usable = (mono.size // window) * window
    # RMS per window, not peak: the floor is an RMS, and max-of-N on Gaussian
    # noise runs ~2.7x its RMS, which would silently eat 8.6 dB of the margin
    # and let noise peaks register as transients.
    envelope = np.sqrt(np.mean(
        mono[:usable].reshape(-1, window) ** 2, axis=1))
    min_gap = int(min_gap_seconds * sample_rate)

    onsets: list[int] = []
    for index, level in enumerate(envelope):
        if level < threshold:
            continue
        centre = index * window
        if onsets and centre - onsets[-1] < min_gap:
            continue
        peak_index = int(np.argmax(np.abs(mono[centre:centre + window])) + centre)
        # Walk back over the CONTIGUOUS run containing the peak, on a short
        # envelope. Taking the first sample anywhere in the lookback that
        # crosses a fraction of the peak lets an ordinary noise excursion,
        # hundreds of samples early, become the "onset" - and the correlation
        # window then fills with room instead of transient. Measured: that
        # dragged a +40 deg clap to +7 deg with a fan at -25 deg.
        peak_index = _attack_start(mono, peak_index, lookback, floor,
                                   attack_fraction)
        if onsets and peak_index - onsets[-1] < min_gap:
            continue
        onsets.append(peak_index)
    return onsets


def _attack_start(mono: np.ndarray, peak_index: int, lookback: int,
                  floor: float, attack_fraction: float,
                  envelope_window: int = 16) -> int:
    """First sample of the contiguous transient that peaks at `peak_index`.

    Walks back on a short RMS envelope and stops the moment the signal drops
    back into the room, so the returned index is where the transient actually
    begins rather than the first noise blip in the lookback.
    """
    begin = max(0, peak_index - lookback)
    span = mono[begin:peak_index + envelope_window]
    if span.size < envelope_window * 2:
        return peak_index

    usable = (span.size // envelope_window) * envelope_window
    env = np.sqrt(np.mean(
        span[:usable].reshape(-1, envelope_window) ** 2, axis=1))
    if env.size == 0:
        return peak_index

    gate = max(attack_fraction * float(env.max()), 3.0 * floor)
    index = int(np.argmax(env))
    while index > 0 and env[index - 1] >= gate:
        index -= 1
    return begin + index * envelope_window
