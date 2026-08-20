"""DOA against synthetic sources at angles we chose ourselves.

Also pins down what a two-microphone array is NOT allowed to claim.
"""

import numpy as np
import pytest

from heimdall.audio import synthetic
from heimdall.audio.doa import (
    DoaError,
    angle_from_tdoa,
    angular_resolution_degrees,
    estimate_doa,
    estimate_doa_from_config,
)
from heimdall.audio.frame import AudioFrame
from heimdall.audio.geometry import Microphone, MicrophoneArray, load_classroom_config

SAMPLE_RATE = 48000
SPACING = 0.30
LENGTH = 8192


@pytest.fixture
def array():
    """Channel 0 at negative x, so positive angles are toward -x."""
    return MicrophoneArray(
        microphones=(
            Microphone("mic_1", -SPACING / 2, 0.0),
            Microphone("mic_2", SPACING / 2, 0.0),
        ),
        orientation_degrees=90.0,
    )


def frame_at_angle(angle_degrees, noise_amplitude=0.002, seed=3, spacing=SPACING):
    """Build a two-channel frame containing a source at a known bearing."""
    source = synthetic.speech_like(LENGTH, SAMPLE_RATE, seed=2)
    tdoa = synthetic.tdoa_for_angle(angle_degrees, spacing, SAMPLE_RATE)
    samples = synthetic.simulate_array_signals(
        source, [0.0, -tdoa], noise_amplitude=noise_amplitude, seed=seed
    )
    return AudioFrame(samples, 0.0, 0, SAMPLE_RATE)


# --- the TDOA-to-angle mapping ----------------------------------------------

def test_zero_tdoa_is_broadside():
    assert angle_from_tdoa(0.0, SPACING) == pytest.approx(0.0)


def test_maximum_tdoa_is_end_fire():
    limit = SPACING / 343.0
    assert angle_from_tdoa(-limit, SPACING) == pytest.approx(90.0)
    assert angle_from_tdoa(limit, SPACING) == pytest.approx(-90.0)


def test_impossible_tdoa_returns_none():
    """More delay than the spacing allows is not an angle - it is a mistake."""
    assert angle_from_tdoa(-2.0 * SPACING / 343.0, SPACING) is None


def test_angle_from_tdoa_rejects_bad_spacing():
    with pytest.raises(DoaError):
        angle_from_tdoa(0.0, 0.0)


def test_resolution_degrades_toward_the_array_axis():
    """A linear array is far less certain about sources near its own axis."""
    broadside = angular_resolution_degrees(0.0, SPACING, SAMPLE_RATE)
    oblique = angular_resolution_degrees(60.0, SPACING, SAMPLE_RATE)
    endfire = angular_resolution_degrees(88.0, SPACING, SAMPLE_RATE)
    assert broadside < oblique < endfire


def test_wider_spacing_improves_resolution():
    narrow = angular_resolution_degrees(0.0, 0.10, SAMPLE_RATE)
    wide = angular_resolution_degrees(0.0, 0.50, SAMPLE_RATE)
    assert wide < narrow


def test_higher_sample_rate_improves_resolution():
    at_16k = angular_resolution_degrees(0.0, SPACING, 16000)
    at_48k = angular_resolution_degrees(0.0, SPACING, 48000)
    assert at_48k < at_16k


# --- known source angles -----------------------------------------------------

@pytest.mark.parametrize("angle", [0, 10, 25, 45, 60, 75, -15, -35, -70])
def test_known_angles_are_recovered(array, angle):
    result = estimate_doa(frame_at_angle(angle), array)
    assert result.valid
    assert result.angle_degrees == pytest.approx(angle, abs=1.5)


def test_recovered_angle_is_accurate_near_broadside(array):
    result = estimate_doa(frame_at_angle(20.0, noise_amplitude=0.0), array)
    assert result.angle_degrees == pytest.approx(20.0, abs=0.3)


def test_sign_convention_matches_geometry(array):
    """A positive angle must point at the same side the geometry calls positive."""
    result = estimate_doa(frame_at_angle(40.0), array)
    seat_side = array.bearing_to((-5.0, 5.0))
    assert result.angle_degrees > 0 and seat_side > 0


def test_raw_position_array_is_accepted():
    positions = np.array([[-SPACING / 2, 0.0], [SPACING / 2, 0.0]])
    result = estimate_doa(frame_at_angle(30.0), positions)
    assert result.angle_degrees == pytest.approx(30.0, abs=1.5)


def test_microphone_spacing_is_not_hard_coded():
    """Doubling the spacing halves the sine, so the same TDOA is a new angle."""
    wide = MicrophoneArray(
        microphones=(Microphone("mic_1", -0.30, 0.0), Microphone("mic_2", 0.30, 0.0))
    )
    result = estimate_doa(frame_at_angle(30.0, spacing=0.60), wide)
    assert result.angle_degrees == pytest.approx(30.0, abs=1.5)


def test_speed_of_sound_is_configurable(array):
    frame = frame_at_angle(30.0)
    warm = estimate_doa(frame, array, speed_of_sound=350.0)
    cold = estimate_doa(frame, array, speed_of_sound=330.0)
    assert warm.angle_degrees != cold.angle_degrees


def test_tdoa_is_reported_alongside_the_angle(array):
    result = estimate_doa(frame_at_angle(45.0), array)
    assert result.tdoa_seconds != 0.0
    assert result.tdoa_samples == pytest.approx(result.tdoa_seconds * SAMPLE_RATE, rel=1e-6)


def test_confidence_falls_with_noise(array):
    clean = estimate_doa(frame_at_angle(30.0, noise_amplitude=0.001), array)
    noisy = estimate_doa(frame_at_angle(30.0, noise_amplitude=0.1), array)
    assert clean.confidence > noisy.confidence


# --- what two microphones may not claim --------------------------------------

def test_linear_array_reports_no_position(array):
    """No range information exists, so `position` must stay None."""
    assert estimate_doa(frame_at_angle(30.0), array).position is None


def test_linear_array_reports_front_back_ambiguity(array):
    result = estimate_doa(frame_at_angle(30.0), array)
    assert result.ambiguous
    assert result.alternative_angle_degrees == pytest.approx(150.0, abs=2.0)


def test_ambiguous_alternative_mirrors_negative_angles(array):
    result = estimate_doa(frame_at_angle(-30.0), array)
    assert result.alternative_angle_degrees == pytest.approx(-150.0, abs=2.0)


def test_angular_resolution_is_reported(array):
    result = estimate_doa(frame_at_angle(0.0), array)
    assert result.angular_resolution_degrees is not None
    assert result.angular_resolution_degrees > 0


# --- failure modes -----------------------------------------------------------

def test_silence_yields_no_angle(array):
    silent = AudioFrame(np.zeros((LENGTH, 2), dtype=np.float32), 0.0, 0, SAMPLE_RATE)
    result = estimate_doa(silent, array)
    assert not result.valid
    assert result.angle_degrees is None
    assert result.confidence == 0.0


def test_mono_frame_is_rejected(array):
    mono = AudioFrame(np.zeros((LENGTH, 1), dtype=np.float32), 0.0, 0, SAMPLE_RATE)
    result = estimate_doa(mono, array)
    assert not result.valid
    assert "2 channels" in result.reason


def test_channel_count_mismatch_is_rejected(array):
    four_channel = AudioFrame(np.zeros((LENGTH, 4), dtype=np.float32), 0.0, 0, SAMPLE_RATE)
    result = estimate_doa(four_channel, array)
    assert not result.valid
    assert "channels" in result.reason


def test_low_confidence_can_be_rejected_by_threshold(array):
    """Uncorrelated channels must not be turned into a direction."""
    left = synthetic.white_noise(LENGTH, seed=21, amplitude=0.3)
    right = synthetic.white_noise(LENGTH, seed=22, amplitude=0.3)
    samples = np.stack([left, right], axis=1).astype(np.float32)
    frame = AudioFrame(samples, 0.0, 0, SAMPLE_RATE)

    result = estimate_doa(frame, array, min_confidence=0.5)
    assert not result.valid
    assert "below threshold" in result.reason


def test_malformed_positions_raise(array):
    with pytest.raises(DoaError):
        estimate_doa(frame_at_angle(0.0), np.zeros((3, 3)))


def test_result_is_json_friendly(array):
    payload = estimate_doa(frame_at_angle(15.0), array).as_dict()
    assert set(payload) >= {"angle_degrees", "confidence", "tdoa", "position"}


def test_estimate_from_classroom_config_uses_its_array():
    room = load_classroom_config()
    frame = frame_at_angle(25.0, spacing=room.array.spacing)
    result = estimate_doa_from_config(frame, room)
    assert result.angle_degrees == pytest.approx(25.0, abs=1.5)
