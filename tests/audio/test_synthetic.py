"""Synthetic audio generation: determinism and correct delays.

The whole hardware-free test suite rests on these, so they are checked
independently of GCC-PHAT using plain cross-correlation.
"""

import numpy as np
import pytest

from heimdall.audio import synthetic


def argmax_lag(signal, reference):
    """Delay of `signal` relative to `reference`, by direct cross-correlation."""
    correlation = np.correlate(signal - signal.mean(), reference - reference.mean(), mode="full")
    lags = np.arange(-reference.size + 1, signal.size)
    return int(lags[np.argmax(np.abs(correlation))])


def test_generators_are_deterministic():
    assert np.array_equal(synthetic.white_noise(256, seed=1), synthetic.white_noise(256, seed=1))
    assert np.array_equal(
        synthetic.speech_like(256, 48000, seed=1), synthetic.speech_like(256, 48000, seed=1)
    )
    assert np.array_equal(synthetic.click(256, seed=3), synthetic.click(256, seed=3))


def test_different_seeds_give_different_signals():
    assert not np.array_equal(synthetic.white_noise(256, seed=1), synthetic.white_noise(256, seed=2))


def test_tone_has_the_requested_frequency():
    sample_rate, freq = 48000, 1000.0
    x = synthetic.tone(4096, sample_rate, frequency=freq)
    spectrum = np.abs(np.fft.rfft(x))
    peak_hz = np.fft.rfftfreq(x.size, 1 / sample_rate)[np.argmax(spectrum)]
    assert peak_hz == pytest.approx(freq, abs=sample_rate / x.size)


def test_delay_signal_shifts_by_the_requested_amount():
    x = synthetic.white_noise(2048, seed=4, amplitude=0.5)
    for delay in (0, 5, 12, 40):
        assert argmax_lag(synthetic.delay_signal(x, delay), x) == delay


def test_delay_signal_handles_negative_delays():
    x = synthetic.white_noise(2048, seed=4, amplitude=0.5)
    for delay in (-3, -11, -25):
        assert argmax_lag(synthetic.delay_signal(x, delay), x) == delay


def test_delay_signal_does_not_wrap_around():
    """A zero-padding implementation must not fold energy back to the start."""
    x = synthetic.click(1024, position=512, seed=2)
    delayed = synthetic.delay_signal(x, 200)
    assert np.max(np.abs(delayed[:100])) < 1e-6


def test_zero_delay_is_a_no_op():
    x = synthetic.white_noise(512, seed=9)
    assert np.allclose(synthetic.delay_signal(x, 0.0), x, atol=1e-9)


def test_click_is_broadband():
    """A narrowband click would have an ambiguous delay - see gcc_phat.py."""
    x = synthetic.click(4096, seed=7)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    # Energy must be spread, not concentrated in a few bins.
    fraction_in_top_bins = np.sort(spectrum)[-50:].sum() / spectrum.sum()
    assert fraction_in_top_bins < 0.5


def test_simulate_array_signals_shape_and_channel_count():
    source = synthetic.white_noise(1024, seed=1)
    signals = synthetic.simulate_array_signals(source, [0.0, 5.0, -3.0, 8.0])
    assert signals.shape == (1024, 4)
    assert signals.dtype == np.float32


def test_simulate_array_signals_applies_per_channel_delays():
    source = synthetic.white_noise(2048, seed=2, amplitude=0.5)
    signals = synthetic.simulate_array_signals(source, [0.0, 7.0])
    assert argmax_lag(signals[:, 1].astype(np.float64), source) == 7


def test_tdoa_for_angle_is_zero_at_broadside():
    assert synthetic.tdoa_for_angle(0.0, 0.3, 48000) == pytest.approx(0.0, abs=1e-12)


def test_tdoa_for_angle_is_antisymmetric():
    a = synthetic.tdoa_for_angle(35.0, 0.3, 48000)
    b = synthetic.tdoa_for_angle(-35.0, 0.3, 48000)
    assert a == pytest.approx(-b)


def test_tdoa_for_angle_never_exceeds_the_physical_limit():
    spacing, sample_rate = 0.3, 48000
    limit = spacing / synthetic.SPEED_OF_SOUND * sample_rate
    for angle in range(-90, 91, 5):
        assert abs(synthetic.tdoa_for_angle(angle, spacing, sample_rate)) <= limit + 1e-9


def test_positive_angle_reaches_channel_zero_first():
    """Sign convention check - the rest of the pipeline depends on it."""
    tdoa = synthetic.tdoa_for_angle(45.0, 0.3, 48000)
    assert tdoa < 0  # channel 0 delayed relative to channel 1 by a negative amount
