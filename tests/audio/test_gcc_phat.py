"""GCC-PHAT against synthetic signals with delays we chose ourselves.

Nothing here depends on a microphone, so a failure means the algorithm is wrong,
not that the room was noisy.
"""

import numpy as np
import pytest

from heimdall.audio import synthetic
from heimdall.audio.frame import AudioFrame
from heimdall.audio.gcc_phat import (
    GccPhatError,
    gcc_phat,
    gcc_phat_frame,
    max_delay_samples,
    max_delay_seconds,
)

SAMPLE_RATE = 48000
SPACING = 0.30
LENGTH = 8192


@pytest.fixture
def source():
    return synthetic.speech_like(LENGTH, SAMPLE_RATE, seed=1)


def estimate(signal, reference, **kwargs):
    kwargs.setdefault("mic_spacing_m", SPACING)
    return gcc_phat(signal, reference, SAMPLE_RATE, **kwargs)


# --- the physically possible delay window -----------------------------------

def test_max_delay_matches_spacing_over_speed_of_sound():
    assert max_delay_seconds(0.343) == pytest.approx(0.001)
    assert max_delay_samples(0.343, 48000) == pytest.approx(48.0)


def test_max_delay_scales_with_spacing():
    assert max_delay_seconds(0.6) == pytest.approx(2 * max_delay_seconds(0.3))


def test_max_delay_rejects_impossible_geometry():
    with pytest.raises(GccPhatError):
        max_delay_seconds(0.0)
    with pytest.raises(GccPhatError):
        max_delay_seconds(-0.3)
    with pytest.raises(GccPhatError):
        max_delay_seconds(0.3, speed_of_sound=0.0)


def test_search_window_is_limited_to_physical_delays():
    """A delay larger than the spacing allows must not be returned."""
    source = synthetic.click(LENGTH, position=2000, seed=3)
    impossible = synthetic.delay_signal(source, 500.0)
    result = estimate(impossible, source)
    assert abs(result.delay_samples) <= max_delay_samples(SPACING, SAMPLE_RATE) + 1


# --- known delays ------------------------------------------------------------

def test_identical_signals_give_zero_delay(source):
    result = estimate(source, source)
    assert result.valid
    assert result.delay_samples == pytest.approx(0.0, abs=0.1)
    assert result.delay_seconds == pytest.approx(0.0, abs=1e-6)
    assert result.confidence > 0.5


def test_known_positive_delay_is_recovered(source):
    result = estimate(synthetic.delay_signal(source, 10.0), source)
    assert result.valid
    assert result.delay_samples == pytest.approx(10.0, abs=0.25)


def test_known_negative_delay_is_recovered(source):
    result = estimate(synthetic.delay_signal(source, -7.0), source)
    assert result.valid
    assert result.delay_samples == pytest.approx(-7.0, abs=0.25)


@pytest.mark.parametrize("delay", [-30.0, -12.5, -1.0, 0.0, 1.0, 3.5, 12.5, 30.0])
def test_a_range_of_delays_is_recovered(source, delay):
    result = estimate(synthetic.delay_signal(source, delay), source)
    assert result.delay_samples == pytest.approx(delay, abs=0.3)


def test_sub_sample_delays_are_resolved(source):
    """Interpolation must beat one-sample granularity."""
    result = estimate(synthetic.delay_signal(source, 4.25), source)
    assert result.delay_samples == pytest.approx(4.25, abs=0.15)


def test_delay_seconds_matches_delay_samples(source):
    result = estimate(synthetic.delay_signal(source, 10.0), source)
    assert result.delay_seconds == pytest.approx(result.delay_samples / SAMPLE_RATE)


def test_delay_sign_convention(source):
    """Positive delay means `signal` arrived later than `reference`."""
    late = synthetic.delay_signal(source, 8.0)
    assert estimate(late, source).delay_samples > 0
    assert estimate(source, late).delay_samples < 0


def test_sample_rate_is_configurable(source):
    """The same physical delay is a different number of samples at 16 kHz."""
    at_16k = gcc_phat(
        synthetic.delay_signal(synthetic.speech_like(LENGTH, 16000, seed=1), 5.0),
        synthetic.speech_like(LENGTH, 16000, seed=1),
        16000,
        mic_spacing_m=SPACING,
    )
    assert at_16k.delay_samples == pytest.approx(5.0, abs=0.3)
    assert at_16k.delay_seconds == pytest.approx(5.0 / 16000, rel=0.1)


def test_broadband_transients_are_recovered():
    clap = synthetic.click(LENGTH, position=3000, seed=5)
    result = estimate(synthetic.delay_signal(clap, 15.0), clap)
    assert result.delay_samples == pytest.approx(15.0, abs=0.25)
    assert result.confidence > 0.5


# --- noise -------------------------------------------------------------------

@pytest.mark.parametrize("noise_amplitude", [0.005, 0.02, 0.05])
def test_noisy_signals_still_recover_the_delay(source, noise_amplitude):
    signal = synthetic.delay_signal(source, 10.0) + synthetic.white_noise(
        LENGTH, seed=11, amplitude=noise_amplitude
    )
    reference = source + synthetic.white_noise(LENGTH, seed=12, amplitude=noise_amplitude)
    result = estimate(signal, reference)
    assert result.valid
    assert result.delay_samples == pytest.approx(10.0, abs=1.0)


def test_confidence_falls_as_noise_rises(source):
    confidences = []
    for amplitude in (0.001, 0.02, 0.1):
        signal = synthetic.delay_signal(source, 10.0) + synthetic.white_noise(
            LENGTH, seed=11, amplitude=amplitude
        )
        reference = source + synthetic.white_noise(LENGTH, seed=12, amplitude=amplitude)
        confidences.append(estimate(signal, reference).confidence)
    assert confidences[0] > confidences[1] > confidences[2]


def test_uncorrelated_channels_get_low_confidence(source):
    unrelated = synthetic.white_noise(LENGTH, seed=99, amplitude=0.3)
    result = estimate(unrelated, source)
    assert result.confidence < 0.3


def test_clean_signals_get_high_confidence(source):
    assert estimate(synthetic.delay_signal(source, 6.0), source).confidence > 0.5


# --- graceful failure --------------------------------------------------------

def test_silence_is_reported_invalid_not_guessed(source):
    result = estimate(np.zeros(LENGTH), source)
    assert not result.valid
    assert result.confidence == 0.0
    assert result.delay_samples == 0.0
    assert "silence" in result.reason


def test_both_channels_silent_is_invalid():
    result = estimate(np.zeros(LENGTH), np.zeros(LENGTH))
    assert not result.valid


def test_mismatched_lengths_raise():
    with pytest.raises(GccPhatError):
        gcc_phat(np.zeros(100), np.zeros(200), SAMPLE_RATE, mic_spacing_m=SPACING)


def test_empty_signals_raise():
    with pytest.raises(GccPhatError):
        gcc_phat(np.array([]), np.array([]), SAMPLE_RATE, mic_spacing_m=SPACING)


def test_nan_and_inf_raise(source):
    broken = source.copy()
    broken[10] = np.nan
    with pytest.raises(GccPhatError):
        estimate(broken, source)

    broken[10] = np.inf
    with pytest.raises(GccPhatError):
        estimate(broken, source)


def test_invalid_parameters_raise(source):
    with pytest.raises(GccPhatError):
        gcc_phat(source, source, 0, mic_spacing_m=SPACING)
    with pytest.raises(GccPhatError):
        gcc_phat(source, source, SAMPLE_RATE, mic_spacing_m=SPACING, interp=0)
    with pytest.raises(GccPhatError):
        gcc_phat(source, source, SAMPLE_RATE, max_tau=-1.0)


def test_result_is_json_friendly(source):
    payload = estimate(synthetic.delay_signal(source, 5.0), source).as_dict()
    assert set(payload) >= {"delay_samples", "delay_seconds", "confidence", "valid"}


# --- frame helper ------------------------------------------------------------

def test_gcc_phat_frame_uses_two_channels(source):
    samples = synthetic.simulate_array_signals(source, [9.0, 0.0])
    frame = AudioFrame(samples, 0.0, 0, SAMPLE_RATE)
    result = gcc_phat_frame(frame, 0, 1, mic_spacing_m=SPACING)
    assert result.delay_samples == pytest.approx(9.0, abs=0.3)
