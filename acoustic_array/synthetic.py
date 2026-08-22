"""Deterministic synthetic audio generation.

Used by the mock audio source, by the unit tests and by the calibration tool so
that the entire pipeline can be exercised without an ESP32, a microphone, a COM
port or a sound card. Every function here is deterministic given its seed.
"""

from __future__ import annotations

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s at ~20 C


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def white_noise(num_samples: int, seed: int = 0, amplitude: float = 0.1) -> np.ndarray:
    return (amplitude * _rng(seed).standard_normal(num_samples)).astype(np.float64)


def tone(
    num_samples: int,
    sample_rate: int,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    phase: float = 0.0,
) -> np.ndarray:
    t = np.arange(num_samples) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def click(
    num_samples: int,
    position: int | None = None,
    amplitude: float = 1.0,
    width: int = 6,
    seed: int = 11,
) -> np.ndarray:
    """A short broadband transient - the synthetic equivalent of a hand clap.

    A Gaussian-enveloped burst of noise. Broadband on purpose: a narrowband
    "click" would have an oscillating autocorrelation and an ambiguous delay,
    which is not what a real clap does.
    """
    position = num_samples // 2 if position is None else position
    idx = np.arange(num_samples)
    envelope = np.exp(-0.5 * ((idx - position) / max(width, 1)) ** 2)
    burst = envelope * _rng(seed).standard_normal(num_samples)
    peak = np.max(np.abs(burst))
    if peak > 0:
        burst = burst / peak
    return amplitude * burst


def speech_like(
    num_samples: int,
    sample_rate: int,
    seed: int = 0,
    amplitude: float = 0.3,
    f0: float = 130.0,
) -> np.ndarray:
    """Voiced-speech-like signal: a harmonic stack shaped into the 300-3400 Hz
    band with a slow amplitude envelope. Not real speech - just something with
    speech-like spectral and temporal structure for detector tests."""
    rng = _rng(seed)
    t = np.arange(num_samples) / sample_rate
    signal = np.zeros(num_samples)

    for harmonic in range(1, 26):
        freq = f0 * harmonic
        if freq >= sample_rate / 2:
            break
        # Formant-ish weighting: emphasise the 300-3400 Hz telephone band.
        weight = np.exp(-0.5 * ((freq - 800.0) / 700.0) ** 2) + 0.4 * np.exp(
            -0.5 * ((freq - 2200.0) / 900.0) ** 2
        )
        signal += weight * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))

    # Syllabic envelope at ~4 Hz.
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 2 * np.pi))
    signal *= envelope

    # Real voiced speech is not a pure harmonic stack: aspiration and fricative
    # energy fill in between the harmonics. Without it the spectrum is sparse in
    # a way no microphone ever sees.
    aspiration = rng.standard_normal(num_samples)
    signal = signal / (np.max(np.abs(signal)) + 1e-12)
    signal = signal + 0.12 * aspiration * envelope

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return amplitude * signal


def delay_signal(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay `x` by `delay_samples` (may be fractional and/or negative).

    Implemented as a frequency-domain phase shift with zero padding so there is
    no circular wrap-around. A positive delay shifts the signal later in time.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x.copy()

    pad = int(np.ceil(abs(delay_samples))) + 64
    padded = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    total = padded.size

    spectrum = np.fft.rfft(padded)
    freqs = np.fft.rfftfreq(total, d=1.0)  # cycles per sample
    shifted = spectrum * np.exp(-2j * np.pi * freqs * delay_samples)
    out = np.fft.irfft(shifted, total)
    return out[pad : pad + n]


def simulate_array_signals(
    source: np.ndarray,
    delays_samples: list[float] | np.ndarray,
    noise_amplitude: float = 0.0,
    seed: int = 0,
    gains: list[float] | None = None,
) -> np.ndarray:
    """Build a (num_samples, num_channels) array from one source signal.

    Each channel is the source delayed by delays_samples[c], optionally scaled
    and with independent noise added. This is the far-field model used by the
    tests: a single plane wave hitting the array.
    """
    source = np.asarray(source, dtype=np.float64)
    delays = np.asarray(delays_samples, dtype=np.float64)
    num_channels = delays.size
    gains = [1.0] * num_channels if gains is None else gains

    channels = []
    for c in range(num_channels):
        chan = gains[c] * delay_signal(source, float(delays[c]))
        if noise_amplitude > 0:
            chan = chan + white_noise(source.size, seed=seed + 1000 + c, amplitude=noise_amplitude)
        channels.append(chan)

    return np.stack(channels, axis=1).astype(np.float32)


def tdoa_for_angle(
    angle_degrees: float,
    mic_spacing_m: float,
    sample_rate: int,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> float:
    """Far-field TDOA in samples for a plane wave arriving at `angle_degrees`.

    Angle convention: 0 degrees is broadside (perpendicular to the mic axis).
    Positive angles steer toward channel 0, so channel 0 receives the wavefront
    EARLIER and the returned delay of channel 0 relative to channel 1 is negative.
    """
    tau_seconds = mic_spacing_m * np.sin(np.radians(angle_degrees)) / speed_of_sound
    return -tau_seconds * sample_rate
