"""Phase G: direction of arrival.

WHAT TWO MICROPHONES CAN AND CANNOT DO
--------------------------------------
Two microphones measure exactly one number: the time difference of arrival
between them. One number buys one angle. That gives:

  * a BEARING relative to the array - useful, real, testable;
  * NO range. The array cannot tell a whisper 1 m away from a shout 6 m away
    along the same line;
  * NO front/back discrimination. A source in front and its mirror image behind
    the array produce an identical TDOA. Config `orientation_degrees` resolves
    this by asserting the students are all on one side;
  * degrading resolution toward the array axis. Near +-90 degrees the mapping
    from TDOA to angle flattens out, so the same timing error becomes a much
    larger angular error. `angular_resolution_degrees` reports this honestly.

This is a 1-D bearing estimator. It is NOT 2-D classroom localization, and
`position` is None for a linear array by design. Four non-collinear microphones
would give a genuine 2-D fix; the `position` field and the Localizer split below
exist so that upgrade drops in without changing any caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from acoustic_array.frame import AudioFrame
from acoustic_array.gcc_phat import TdoaResult, gcc_phat
from acoustic_array.geometry import Microphone, MicrophoneArray

SPEED_OF_SOUND = 343.0


class DoaError(ValueError):
    """Raised for input the geometry cannot support."""


@dataclass(frozen=True)
class DoaResult:
    """A direction estimate, with an explicit account of what it does not know.

    Callers must check `valid` before using `angle_degrees`. `confidence` is in
    [0, 1] and is inherited from the underlying correlation quality.
    """

    angle_degrees: float | None
    confidence: float
    tdoa_seconds: float
    tdoa_samples: float
    valid: bool
    ambiguous: bool = False
    alternative_angle_degrees: float | None = None
    position: tuple[float, float] | None = None
    angular_resolution_degrees: float | None = None
    num_channels: int = 2
    reason: str = ""
    tdoa_result: TdoaResult | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "angle_degrees": self.angle_degrees,
            "confidence": self.confidence,
            "tdoa": self.tdoa_seconds,
            "tdoa_samples": self.tdoa_samples,
            "valid": self.valid,
            "ambiguous": self.ambiguous,
            "alternative_angle_degrees": self.alternative_angle_degrees,
            "position": self.position,
            "angular_resolution_degrees": self.angular_resolution_degrees,
            "reason": self.reason,
        }


def _invalid(reason: str, tdoa: TdoaResult | None = None, channels: int = 2) -> DoaResult:
    return DoaResult(
        angle_degrees=None,
        confidence=0.0,
        tdoa_seconds=tdoa.delay_seconds if tdoa else 0.0,
        tdoa_samples=tdoa.delay_samples if tdoa else 0.0,
        valid=False,
        num_channels=channels,
        reason=reason,
        tdoa_result=tdoa,
    )


def angle_from_tdoa(
    tdoa_seconds: float,
    mic_spacing_m: float,
    speed_of_sound: float = SPEED_OF_SOUND,
    tolerance: float = 0.0,
) -> float | None:
    """Far-field bearing, in degrees, from the TDOA of channel 0 vs channel 1.

    Sign convention: channel 0 hearing the sound EARLIER (a negative delay of
    channel 0 relative to channel 1) means the source is on the channel-0 side,
    which is a POSITIVE angle. See the convention block in geometry.py.

    A source at end-fire (+-90 degrees) produces exactly the maximum possible
    TDOA, so ordinary estimation error pushes the implied sine slightly past 1.
    `tolerance` is how much overshoot still counts as end-fire rather than as a
    broken measurement; callers should size it to one sample of timing error.
    Beyond that the TDOA is not physically achievable and None is returned.
    """
    if mic_spacing_m <= 0:
        raise DoaError("mic_spacing_m must be positive, got %r" % (mic_spacing_m,))

    sine = -tdoa_seconds * speed_of_sound / mic_spacing_m
    if abs(sine) > 1.0 + max(tolerance, 1e-9):
        return None
    return float(np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0))))


def angular_resolution_degrees(
    angle_degrees: float,
    mic_spacing_m: float,
    sample_rate: int,
    speed_of_sound: float = SPEED_OF_SOUND,
    timing_error_samples: float = 0.5,
) -> float:
    """Angular error produced by a timing error of `timing_error_samples`.

    This is the honest resolution limit of the array at this bearing. It blows
    up toward +-90 degrees, which is exactly the point: a linear array is much
    less certain about sources near its own axis than about sources in front.
    """
    delta_tau = timing_error_samples / float(sample_rate)
    sine = np.sin(np.radians(angle_degrees))
    cosine = np.cos(np.radians(angle_degrees))
    if abs(cosine) < 1e-6:
        return 90.0
    d_sine = delta_tau * speed_of_sound / mic_spacing_m
    return float(np.degrees(abs(d_sine / cosine)))


def _as_array(microphone_positions) -> MicrophoneArray:
    """Accept a MicrophoneArray or a raw (N, 2) array of positions."""
    if isinstance(microphone_positions, MicrophoneArray):
        return microphone_positions

    positions = np.atleast_2d(np.asarray(microphone_positions, dtype=np.float64))
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise DoaError(
            "microphone_positions must be (N, 2) room coordinates, got shape %r"
            % (positions.shape,)
        )
    mics = tuple(
        Microphone(id="mic_%d" % (i + 1), x=float(p[0]), y=float(p[1]))
        for i, p in enumerate(positions)
    )
    return MicrophoneArray(microphones=mics)


def estimate_doa(
    audio_frame: AudioFrame,
    microphone_positions,
    sample_rate: int | None = None,
    *,
    speed_of_sound: float = SPEED_OF_SOUND,
    min_confidence: float = 0.0,
    **gcc_kwargs: object,
) -> DoaResult:
    """Estimate the bearing of the dominant sound source in `audio_frame`.

    `microphone_positions` is a MicrophoneArray, or an (N, 2) array of room
    coordinates in channel order. `sample_rate` defaults to the frame's own.
    """
    array = _as_array(microphone_positions)
    sample_rate = int(sample_rate or audio_frame.sample_rate)

    if audio_frame.num_channels < 2:
        return _invalid(
            "need at least 2 channels, got %d" % audio_frame.num_channels,
            channels=audio_frame.num_channels,
        )
    if array.num_channels != audio_frame.num_channels:
        return _invalid(
            "geometry describes %d microphones but the frame has %d channels"
            % (array.num_channels, audio_frame.num_channels),
            channels=audio_frame.num_channels,
        )

    spacing = array.spacing
    tdoa = gcc_phat(
        audio_frame.channel(0),
        audio_frame.channel(1),
        sample_rate,
        mic_spacing_m=spacing,
        speed_of_sound=speed_of_sound,
        **gcc_kwargs,  # type: ignore[arg-type]
    )

    if not tdoa.valid:
        return _invalid(tdoa.reason or "invalid TDOA", tdoa, audio_frame.num_channels)
    if tdoa.confidence < min_confidence:
        return _invalid(
            "confidence %.3f below threshold %.3f" % (tdoa.confidence, min_confidence),
            tdoa,
            audio_frame.num_channels,
        )

    # One sample of timing error, expressed as a change in sin(angle). This is
    # the array's own precision limit, so overshoot within it is end-fire.
    endfire_tolerance = speed_of_sound / (spacing * sample_rate)
    angle = angle_from_tdoa(
        tdoa.delay_seconds, spacing, speed_of_sound, tolerance=endfire_tolerance
    )
    if angle is None:
        return _invalid(
            "TDOA %.4f ms exceeds what a %.3f m spacing allows"
            % (tdoa.delay_seconds * 1000.0, spacing),
            tdoa,
            audio_frame.num_channels,
        )

    resolution = angular_resolution_degrees(angle, spacing, sample_rate, speed_of_sound)

    # A linear array yields a cone of possible directions; in 2-D that collapses
    # to the estimate and its mirror image behind the array. No range, so no
    # position. Both facts are reported rather than papered over.
    linear = array.is_linear
    return DoaResult(
        angle_degrees=angle,
        confidence=tdoa.confidence,
        tdoa_seconds=tdoa.delay_seconds,
        tdoa_samples=tdoa.delay_samples,
        valid=True,
        ambiguous=linear,
        alternative_angle_degrees=(180.0 - angle if angle >= 0 else -180.0 - angle)
        if linear
        else None,
        position=None,
        angular_resolution_degrees=resolution,
        num_channels=audio_frame.num_channels,
        tdoa_result=tdoa,
    )


def estimate_doa_from_config(
    audio_frame: AudioFrame,
    classroom,  # ClassroomConfig - untyped to keep the import direction one-way
    **kwargs: object,
) -> DoaResult:
    """Run estimate_doa using the array and sound speed from a classroom config."""
    return estimate_doa(
        audio_frame,
        classroom.array,
        speed_of_sound=classroom.speed_of_sound,
        **kwargs,  # type: ignore[arg-type]
    )
