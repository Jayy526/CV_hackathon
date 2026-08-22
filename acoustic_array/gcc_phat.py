"""Phase F: GCC-PHAT time-delay estimation.

Generalised Cross Correlation with Phase Transform. Whitening the cross-spectrum
makes the correlation peak sharp and largely independent of the source spectrum,
which is what makes it work on speech in a reverberant room.

Nothing here hard-codes a microphone spacing or a sample rate: the physically
possible delay range is derived from the geometry the caller passes in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len

SPEED_OF_SOUND = 343.0  # m/s at ~20 C
EPS = 1e-12


class GccPhatError(ValueError):
    """Raised for structurally invalid input (wrong shape, mismatched lengths)."""


@dataclass(frozen=True)
class TdoaResult:
    """Estimated time difference of arrival between two channels.

    Sign convention: a POSITIVE delay means `signal` arrived LATER than
    `reference`, i.e. signal(t) ~= reference(t - delay).

    `valid` is False when the estimate is meaningless (silent or degenerate
    input). Callers must check it rather than trusting the numbers.
    """

    delay_samples: float
    delay_seconds: float
    correlation: float
    confidence: float
    sample_rate: int
    max_delay_samples: float
    valid: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "delay_samples": self.delay_samples,
            "delay_seconds": self.delay_seconds,
            "correlation": self.correlation,
            "confidence": self.confidence,
            "valid": self.valid,
            "reason": self.reason,
        }


def max_delay_seconds(mic_spacing_m: float, speed_of_sound: float = SPEED_OF_SOUND) -> float:
    """Largest physically possible TDOA for a mic pair, in seconds.

    Reached when the source lies on the line through both microphones.
    """
    if mic_spacing_m <= 0:
        raise GccPhatError("mic_spacing_m must be positive, got %r" % (mic_spacing_m,))
    if speed_of_sound <= 0:
        raise GccPhatError("speed_of_sound must be positive, got %r" % (speed_of_sound,))
    return float(mic_spacing_m) / float(speed_of_sound)


def max_delay_samples(
    mic_spacing_m: float,
    sample_rate: int,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> float:
    """Largest physically possible TDOA for a mic pair, in samples."""
    return max_delay_seconds(mic_spacing_m, speed_of_sound) * sample_rate


def _peak_to_sidelobe(window: np.ndarray, peak_index: int, guard: int = 16) -> float:
    """Confidence as a peak-to-sidelobe ratio, in [0, 1].

    Walks out from the main peak until the correlation drops below half its
    height, excludes that main lobe plus a guard band, and compares the peak
    against the tallest thing remaining. A dominant single peak scores near 1;
    a correlation with rival peaks (uncorrelated channels, or a periodic source
    whose delay is genuinely ambiguous) scores near 0.
    """
    amplitude = np.abs(window)
    peak = float(amplitude[peak_index])
    if peak <= EPS:
        return 0.0

    half = 0.5 * peak
    low = peak_index
    while low > 0 and amplitude[low - 1] < amplitude[low] and amplitude[low - 1] > half:
        low -= 1
    high = peak_index
    last = amplitude.size - 1
    while high < last and amplitude[high + 1] < amplitude[high] and amplitude[high + 1] > half:
        high += 1

    low = max(0, low - guard)
    high = min(last, high + guard)

    outside = np.concatenate([amplitude[:low], amplitude[high + 1 :]])
    if outside.size == 0:
        return 1.0
    return float(np.clip(1.0 - float(np.max(outside)) / peak, 0.0, 1.0))


def _validate(signal: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()

    if signal.size == 0 or reference.size == 0:
        raise GccPhatError("signals must be non-empty")
    if signal.size != reference.size:
        raise GccPhatError(
            "signals must be the same length, got %d and %d" % (signal.size, reference.size)
        )
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(reference)):
        raise GccPhatError("signals must not contain NaN or inf")
    return signal, reference


def gcc_phat(
    signal: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    *,
    mic_spacing_m: float | None = None,
    max_tau: float | None = None,
    speed_of_sound: float = SPEED_OF_SOUND,
    interp: int = 16,
    silence_threshold: float = 1e-6,
    regularization: float = 0.01,
    # Below this, sound carries no usable direction for a 13.5 cm array and
    # in practice carries most of the room's energy. See the band-limit
    # comment below for the measurement that set it.
    min_frequency_hz: float = 300.0,
    max_frequency_hz: float | None = None,
) -> TdoaResult:
    """Estimate the delay of `signal` relative to `reference`.

    The search window is limited to what is physically possible, taken from
    `max_tau` (seconds) if given, otherwise derived from `mic_spacing_m`. If
    neither is given the full correlation range is searched, which will happily
    return physically impossible delays - so pass one of them.

    `regularization` sets the coherent-energy floor, as a fraction of the
    strongest bin, below which a frequency bin stops contributing. It is what
    keeps empty bins from dominating the whitened cross-spectrum.

    Known limitation: a continuously periodic source (a pure tone, a hum) has a
    genuinely ambiguous delay, since shifting by a whole period is
    indistinguishable from not shifting at all. Such inputs get a low confidence
    rather than a wrong answer presented as certain, but they are not reliable.
    """
    signal, reference = _validate(signal, reference)

    if sample_rate <= 0:
        raise GccPhatError("sample_rate must be positive, got %r" % (sample_rate,))
    if interp < 1:
        raise GccPhatError("interp must be >= 1, got %r" % (interp,))

    if max_tau is None and mic_spacing_m is not None:
        max_tau = max_delay_seconds(mic_spacing_m, speed_of_sound)
    if max_tau is not None and max_tau <= 0:
        raise GccPhatError("max_tau must be positive, got %r" % (max_tau,))

    limit_samples = float(max_tau * sample_rate) if max_tau is not None else float("inf")

    # Degenerate input: silence or a constant. Fail gracefully, do not guess.
    sig_rms = float(np.sqrt(np.mean(signal**2)))
    ref_rms = float(np.sqrt(np.mean(reference**2)))
    if sig_rms < silence_threshold or ref_rms < silence_threshold:
        return TdoaResult(
            delay_samples=0.0,
            delay_seconds=0.0,
            correlation=0.0,
            confidence=0.0,
            sample_rate=sample_rate,
            max_delay_samples=limit_samples,
            valid=False,
            reason="signal below silence threshold",
        )

    n = signal.size + reference.size
    nfft = next_fast_len(n)

    spec_sig = np.fft.rfft(signal, nfft)
    spec_ref = np.fft.rfft(reference, nfft)
    cross = spec_sig * np.conj(spec_ref)

    magnitude = np.abs(cross)
    if not np.any(magnitude > EPS):
        return TdoaResult(
            delay_samples=0.0,
            delay_seconds=0.0,
            correlation=0.0,
            confidence=0.0,
            sample_rate=sample_rate,
            max_delay_samples=limit_samples,
            valid=False,
            reason="empty cross-spectrum",
        )

    # PHAT weighting: keep the phase, discard the magnitude. Applied alone this
    # divides every bin by its own magnitude, so bins containing nothing but
    # numerical noise become full-amplitude garbage - which wrecks the estimate
    # for narrowband or harmonic sources. The mask below suppresses those bins.
    whitened = cross / (magnitude + EPS)

    # Coherent-energy mask: a bin only contributes if BOTH channels have energy
    # there. Bins where either channel is empty carry no delay information.
    coherent_energy = np.sqrt(np.abs(spec_sig) * np.abs(spec_ref))
    floor = float(regularization) * float(np.max(coherent_energy))
    mask = coherent_energy / (coherent_energy + floor + EPS)

    # BAND LIMIT. Measured on the real array, in a quiet room: 93.6% of all
    # energy sat below 200 Hz, at 0.82 coherence - and its correlation was FLAT
    # across every physically possible lag (-7..+7 all read +0.29). At 100 Hz
    # one cycle is 160 samples, so a 7-sample shift barely changes anything:
    # that rumble carries no direction, it only drowns the band that does.
    #
    # Removing it turned a flat line into a clean peak of 0.73 at lag +1 on the
    # same hardware. This is not a cosmetic filter; without it the estimator is
    # measuring mains hum and HVAC.
    if min_frequency_hz > 0.0:
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        mask = np.where(freqs < float(min_frequency_hz), 0.0, mask)
    if max_frequency_hz:
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        mask = np.where(freqs > float(max_frequency_hz), 0.0, mask)

    weighted = whitened * mask

    # Zero-padding in the frequency domain interpolates the correlation, giving
    # sub-sample delay resolution.
    cc = np.fft.irfft(weighted, nfft * interp)

    max_shift = int(interp * nfft / 2)
    if np.isfinite(limit_samples):
        max_shift = min(max_shift, max(int(np.ceil(limit_samples * interp)), 1))

    window = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    peak_index = int(np.argmax(np.abs(window)))
    peak_value = float(np.abs(window[peak_index]))

    shift = peak_index - max_shift
    delay_samples = shift / float(interp)
    delay_seconds = delay_samples / sample_rate

    confidence = _peak_to_sidelobe(window, peak_index, guard=interp)

    return TdoaResult(
        delay_samples=float(delay_samples),
        delay_seconds=float(delay_seconds),
        correlation=peak_value,
        confidence=confidence,
        sample_rate=sample_rate,
        max_delay_samples=limit_samples,
        valid=True,
        reason="",
    )


def gcc_phat_frame(
    frame,  # AudioFrame - untyped to avoid a circular import
    channel: int = 0,
    reference_channel: int = 1,
    **kwargs: object,
) -> TdoaResult:
    """Convenience wrapper: run GCC-PHAT on two channels of an AudioFrame."""
    return gcc_phat(
        frame.channel(channel),
        frame.channel(reference_channel),
        frame.sample_rate,
        **kwargs,  # type: ignore[arg-type]
    )
