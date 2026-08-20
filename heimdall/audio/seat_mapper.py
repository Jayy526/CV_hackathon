"""Phase I: turn a localization estimate into a plausible seat.

Two strategies live behind one interface:

  * BEARING mode (what two microphones give us). We know a direction and
    nothing about range, so every seat along that line of sight is equally
    consistent with the measurement. We report the best-matching seat AND the
    other seats the measurement cannot rule out. Calling this "the seat the
    sound came from" would be a lie; it is "the seat whose direction best
    matches", which is a much weaker claim.

  * POSITION mode (what four non-collinear microphones would give us). A real
    2-D fix, so nearest-seat-by-distance is meaningful.

map_audio_to_seat() picks the mode from what the DoaResult actually contains.
Replacing the 2-mic localizer with a 4-mic one therefore changes nothing here
and nothing in any caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from heimdall.audio.doa import DoaResult
from heimdall.audio.geometry import ClassroomConfig, Seat

# Below this confidence we refuse to name a seat at all. Tuned against the
# synthetic tests: clean signals score ~0.8, heavy noise ~0.3, uncorrelated
# channels ~0.1. Raise it once real-hardware calibration says what is realistic.
DEFAULT_MIN_CONFIDENCE = 0.30


@dataclass(frozen=True)
class SeatCandidate:
    seat_id: str
    angular_error_degrees: float
    distance_from_array: float
    score: float


@dataclass(frozen=True)
class SeatMatch:
    """The result of mapping a localization onto the seat map.

    `seat_id` is None whenever we refuse to guess: low confidence, an invalid
    localization, no seats configured, or nothing plausibly in range.
    """

    seat_id: str | None
    confidence: float
    distance: float | None
    angular_error_degrees: float | None
    mode: str
    ambiguous: bool
    candidates: tuple[SeatCandidate, ...] = ()
    reason: str = ""
    localization: DoaResult | None = field(default=None, repr=False)

    @property
    def matched(self) -> bool:
        return self.seat_id is not None

    def as_dict(self) -> dict:
        return {
            "seat_id": self.seat_id,
            "confidence": self.confidence,
            "distance": self.distance,
            "angular_error_degrees": self.angular_error_degrees,
            "mode": self.mode,
            "ambiguous": self.ambiguous,
            "candidates": [c.seat_id for c in self.candidates],
            "reason": self.reason,
        }


def _no_match(reason: str, confidence: float, mode: str, localization: DoaResult | None) -> SeatMatch:
    return SeatMatch(
        seat_id=None,
        confidence=confidence,
        distance=None,
        angular_error_degrees=None,
        mode=mode,
        ambiguous=False,
        reason=reason,
        localization=localization,
    )


def _visible_seats(classroom: ClassroomConfig) -> list[Seat]:
    """Seats on the classroom side of the array.

    A two-microphone array cannot tell front from back, but the room can: seats
    behind the array do not exist, so the mirror-image solution is discarded
    here rather than being presented as a real alternative.
    """
    return [s for s in classroom.seats if classroom.array.is_in_front(s.position)]


def map_bearing_to_seat(
    localization: DoaResult,
    classroom: ClassroomConfig,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_angular_error_degrees: float | None = None,
) -> SeatMatch:
    """Bearing-only mapping: match on direction, admit range is unknown.

    `max_angular_error_degrees` defaults to the array's own resolution at the
    estimated bearing, floored at 5 degrees - there is no point demanding a
    tighter match than the hardware can deliver.
    """
    seats = _visible_seats(classroom)
    if not seats:
        return _no_match("no seats in front of the array", localization.confidence, "bearing", localization)

    angle = float(localization.angle_degrees)  # checked by the caller
    tolerance = max_angular_error_degrees
    if tolerance is None:
        resolution = localization.angular_resolution_degrees or 0.0
        tolerance = max(3.0 * resolution, 5.0)

    candidates: list[SeatCandidate] = []
    for seat in seats:
        bearing = classroom.array.bearing_to(seat.position)
        error = abs(bearing - angle)
        if error <= tolerance:
            candidates.append(
                SeatCandidate(
                    seat_id=seat.id,
                    angular_error_degrees=error,
                    distance_from_array=classroom.array.distance_to(seat.position),
                    score=float(np.exp(-0.5 * (error / max(tolerance, 1e-6)) ** 2)),
                )
            )

    if not candidates:
        nearest = min(
            seats, key=lambda s: abs(classroom.array.bearing_to(s.position) - angle)
        )
        nearest_error = abs(classroom.array.bearing_to(nearest.position) - angle)
        return _no_match(
            "no seat within %.1f degrees of bearing %.1f (nearest is %s at %.1f degrees)"
            % (tolerance, angle, nearest.id, nearest_error),
            localization.confidence,
            "bearing",
            localization,
        )

    candidates.sort(key=lambda c: (c.angular_error_degrees, c.distance_from_array))
    best = candidates[0]

    # Every candidate lies near the same line of sight. With no range
    # information we cannot separate them, and saying so is the whole point.
    ambiguous = len(candidates) > 1

    return SeatMatch(
        seat_id=best.seat_id,
        confidence=localization.confidence,
        distance=best.distance_from_array,
        angular_error_degrees=best.angular_error_degrees,
        mode="bearing",
        ambiguous=ambiguous,
        candidates=tuple(candidates),
        reason="%d seats share this bearing; range is unknown with a linear array"
        % len(candidates)
        if ambiguous
        else "",
        localization=localization,
    )


def map_position_to_seat(
    localization: DoaResult,
    classroom: ClassroomConfig,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_distance_m: float = 1.5,
) -> SeatMatch:
    """Position mapping: nearest seat by Euclidean distance.

    Only reachable once a localizer actually produces a position, which a
    two-microphone array never does.
    """
    position = np.asarray(localization.position, dtype=np.float64)

    if not classroom.contains(position):
        return _no_match(
            "estimated position (%.2f, %.2f) is outside the %.1f x %.1f m classroom"
            % (position[0], position[1], classroom.width, classroom.height),
            localization.confidence,
            "position",
            localization,
        )
    if not classroom.seats:
        return _no_match("classroom has no seats configured", localization.confidence, "position", localization)

    distances = [
        (seat, float(np.linalg.norm(seat.position - position))) for seat in classroom.seats
    ]
    distances.sort(key=lambda pair: pair[1])
    nearest, distance = distances[0]

    if distance > max_distance_m:
        return _no_match(
            "nearest seat %s is %.2f m away, beyond the %.2f m limit"
            % (nearest.id, distance, max_distance_m),
            localization.confidence,
            "position",
            localization,
        )

    runner_up = distances[1][1] if len(distances) > 1 else float("inf")
    candidates = tuple(
        SeatCandidate(
            seat_id=seat.id,
            angular_error_degrees=abs(
                classroom.array.bearing_to(seat.position)
                - (localization.angle_degrees or classroom.array.bearing_to(position))
            ),
            distance_from_array=classroom.array.distance_to(seat.position),
            score=float(np.exp(-0.5 * (dist / max(max_distance_m, 1e-6)) ** 2)),
        )
        for seat, dist in distances
        if dist <= max_distance_m
    )

    return SeatMatch(
        seat_id=nearest.id,
        confidence=localization.confidence,
        distance=distance,
        angular_error_degrees=None,
        mode="position",
        # Sitting between two seats: the runner-up is nearly as close.
        ambiguous=bool(runner_up - distance < 0.25 * max(distance, 1e-6)),
        candidates=candidates,
        localization=localization,
    )


def map_audio_to_seat(
    localization: DoaResult,
    classroom_config: ClassroomConfig,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    **kwargs: object,
) -> SeatMatch:
    """Map a localization onto a seat, refusing to guess when it should.

    Dispatches to position mode when the localizer produced a real 2-D fix and
    to bearing mode otherwise, so a future four-microphone localizer needs no
    change here.
    """
    if localization is None:
        return _no_match("no localization supplied", 0.0, "none", None)
    if not localization.valid:
        return _no_match(
            localization.reason or "localization is not valid", 0.0, "none", localization
        )
    if localization.confidence < min_confidence:
        return _no_match(
            "confidence %.3f below threshold %.3f"
            % (localization.confidence, min_confidence),
            localization.confidence,
            "none",
            localization,
        )

    if localization.position is not None:
        return map_position_to_seat(
            localization, classroom_config, min_confidence=min_confidence, **kwargs  # type: ignore[arg-type]
        )

    if localization.angle_degrees is None:
        return _no_match("localization has neither a position nor an angle", localization.confidence, "none", localization)

    return map_bearing_to_seat(
        localization, classroom_config, min_confidence=min_confidence, **kwargs  # type: ignore[arg-type]
    )
