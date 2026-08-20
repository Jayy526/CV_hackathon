"""Seat mapping, including the cases where it must refuse to name a seat."""

import pytest

from heimdall.audio.doa import DoaResult
from heimdall.audio.geometry import (
    ClassroomConfig,
    Microphone,
    MicrophoneArray,
    Seat,
    load_classroom_config,
)
from heimdall.audio.seat_mapper import (
    DEFAULT_MIN_CONFIDENCE,
    map_audio_to_seat,
    map_bearing_to_seat,
)


@pytest.fixture
def room():
    return load_classroom_config()


@pytest.fixture
def single_row_room():
    """One row of seats. In the full grid many rows share a bearing, which is a
    real property of a linear array - here we isolate the between-seats case."""
    array = MicrophoneArray(
        microphones=(Microphone("mic_1", 3.85, 0.0), Microphone("mic_2", 4.15, 0.0)),
        orientation_degrees=90.0,
    )
    seats = tuple(
        Seat(id="A%d" % (i + 1), x=1.75 + 0.9 * i, y=4.0, row="A", column=i + 1)
        for i in range(6)
    )
    return ClassroomConfig(width=8.0, height=10.0, array=array, seats=seats)


def bearing_result(angle, confidence=0.8, resolution=0.7):
    """A DoaResult as a two-microphone array would produce it: bearing, no position."""
    return DoaResult(
        angle_degrees=angle,
        confidence=confidence,
        tdoa_seconds=0.0,
        tdoa_samples=0.0,
        valid=True,
        ambiguous=True,
        alternative_angle_degrees=180.0 - angle,
        position=None,
        angular_resolution_degrees=resolution,
        num_channels=2,
    )


def position_result(x, y, confidence=0.8):
    """A DoaResult as a future four-microphone array would produce it."""
    return DoaResult(
        angle_degrees=None,
        confidence=confidence,
        tdoa_seconds=0.0,
        tdoa_samples=0.0,
        valid=True,
        ambiguous=False,
        position=(x, y),
        angular_resolution_degrees=0.7,
        num_channels=4,
    )


# --- a source clearly on one seat -------------------------------------------

@pytest.mark.parametrize("seat_id", ["A1", "A6", "C3", "E1", "E6"])
def test_seat_on_a_distinct_bearing_is_matched(room, seat_id):
    angle = room.array.bearing_to(room.seat(seat_id).position)
    match = map_audio_to_seat(bearing_result(angle), room)
    assert match.matched
    assert match.seat_id == seat_id
    assert match.angular_error_degrees < 1.0


def test_match_reports_distance_and_mode(room):
    angle = room.array.bearing_to(room.seat("A1").position)
    match = map_audio_to_seat(bearing_result(angle), room)
    assert match.mode == "bearing"
    assert match.distance == pytest.approx(room.array.distance_to(room.seat("A1").position))


def test_confidence_is_carried_through(room):
    angle = room.array.bearing_to(room.seat("A1").position)
    match = map_audio_to_seat(bearing_result(angle, confidence=0.62), room)
    assert match.confidence == pytest.approx(0.62)


# --- the honesty requirements ------------------------------------------------

def test_seats_sharing_a_bearing_are_reported_as_ambiguous(room):
    """Two mics give no range, so seats down the same line cannot be separated."""
    angle = room.array.bearing_to(room.seat("B4").position)
    match = map_audio_to_seat(bearing_result(angle), room)
    assert match.ambiguous
    assert len(match.candidates) > 1
    assert "range is unknown" in match.reason


def test_the_true_seat_ranks_first_among_candidates(room):
    angle = room.array.bearing_to(room.seat("C4").position)
    match = map_audio_to_seat(bearing_result(angle), room)
    assert match.candidates[0].seat_id == "C4"


def test_seats_behind_the_array_are_never_matched(room):
    """The mirror-image solution is discarded because those seats do not exist."""
    behind = ClassroomConfig(
        width=8.0,
        height=10.0,
        array=room.array,
        seats=(Seat("Z1", 4.0, -3.0),),
    )
    match = map_audio_to_seat(bearing_result(0.0), behind)
    assert not match.matched
    assert "in front" in match.reason


# --- a source between two seats ----------------------------------------------

def test_source_between_two_seats_picks_one_and_flags_the_other(single_row_room):
    room = single_row_room
    left = room.array.bearing_to(room.seat("A3").position)
    right = room.array.bearing_to(room.seat("A4").position)
    midpoint = (left + right) / 2.0
    separation = abs(right - left)

    match = map_bearing_to_seat(
        bearing_result(midpoint), room, max_angular_error_degrees=separation
    )
    assert match.matched
    assert match.seat_id in {"A3", "A4"}
    assert match.ambiguous
    assert {"A3", "A4"} <= {c.seat_id for c in match.candidates}


def test_between_seats_error_is_about_half_the_seat_separation(single_row_room):
    room = single_row_room
    left = room.array.bearing_to(room.seat("A3").position)
    right = room.array.bearing_to(room.seat("A4").position)
    separation = abs(right - left)

    match = map_bearing_to_seat(
        bearing_result((left + right) / 2.0), room, max_angular_error_degrees=separation
    )
    assert match.angular_error_degrees == pytest.approx(separation / 2.0, rel=0.05)


def test_a_full_grid_aliases_rows_onto_the_same_bearing(room):
    """Not a defect: seats in different rows genuinely share a line of sight,
    and a bearing-only estimate cannot separate them."""
    angle = room.array.bearing_to(room.seat("A4").position)
    match = map_audio_to_seat(bearing_result(angle), room)
    rows = {c.seat_id[0] for c in match.candidates}
    assert len(rows) > 1
    assert match.ambiguous


# --- a source outside the classroom ------------------------------------------

def test_bearing_with_no_seat_near_it_matches_nothing(room):
    """A sound from the corridor is at a bearing no seat occupies."""
    match = map_audio_to_seat(bearing_result(89.0), room, max_angular_error_degrees=3.0)
    assert not match.matched
    assert match.seat_id is None
    assert "no seat within" in match.reason


def test_position_outside_the_room_is_rejected(room):
    match = map_audio_to_seat(position_result(50.0, 50.0), room)
    assert not match.matched
    assert "outside" in match.reason


# --- low confidence and invalid input ----------------------------------------

def test_low_confidence_refuses_to_name_a_seat(room):
    angle = room.array.bearing_to(room.seat("A1").position)
    match = map_audio_to_seat(bearing_result(angle, confidence=0.05), room)
    assert not match.matched
    assert "below threshold" in match.reason


def test_confidence_threshold_is_configurable(room):
    angle = room.array.bearing_to(room.seat("A1").position)
    result = bearing_result(angle, confidence=0.4)
    assert map_audio_to_seat(result, room, min_confidence=0.2).matched
    assert not map_audio_to_seat(result, room, min_confidence=0.9).matched


def test_default_threshold_is_above_the_uncorrelated_noise_floor():
    """Uncorrelated channels score around 0.1 in the GCC-PHAT tests."""
    assert DEFAULT_MIN_CONFIDENCE > 0.15


def test_invalid_localization_is_rejected(room):
    invalid = DoaResult(
        angle_degrees=None,
        confidence=0.0,
        tdoa_seconds=0.0,
        tdoa_samples=0.0,
        valid=False,
        reason="signal below silence threshold",
    )
    match = map_audio_to_seat(invalid, room)
    assert not match.matched
    assert "silence" in match.reason


def test_missing_localization_is_handled(room):
    match = map_audio_to_seat(None, room)
    assert not match.matched


def test_valid_but_directionless_localization_is_rejected(room):
    odd = DoaResult(
        angle_degrees=None,
        confidence=0.9,
        tdoa_seconds=0.0,
        tdoa_samples=0.0,
        valid=True,
        position=None,
    )
    match = map_audio_to_seat(odd, room)
    assert not match.matched
    assert "neither a position nor an angle" in match.reason


def test_classroom_with_no_seats_matches_nothing(room):
    empty = ClassroomConfig(width=8.0, height=10.0, array=room.array, seats=())
    assert not map_audio_to_seat(bearing_result(0.0), empty).matched


# --- the four-microphone upgrade path ----------------------------------------

def test_a_position_fix_switches_to_nearest_seat_without_api_changes(room):
    """The same call, the same return type - only the localizer changed."""
    seat = room.seat("C3")
    match = map_audio_to_seat(position_result(seat.x + 0.05, seat.y + 0.05), room)
    assert match.mode == "position"
    assert match.seat_id == "C3"
    assert match.distance == pytest.approx(0.0707, abs=0.01)


def test_position_between_two_seats_is_flagged_ambiguous(room):
    a, b = room.seat("C3"), room.seat("C4")
    midpoint = ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
    match = map_audio_to_seat(position_result(*midpoint), room)
    assert match.matched
    assert match.ambiguous


def test_position_far_from_any_seat_matches_nothing(room):
    match = map_audio_to_seat(position_result(0.2, 9.5), room, max_distance_m=0.5)
    assert not match.matched
    assert "beyond" in match.reason


def test_match_is_json_friendly(room):
    angle = room.array.bearing_to(room.seat("A1").position)
    payload = map_audio_to_seat(bearing_result(angle), room).as_dict()
    assert set(payload) >= {"seat_id", "confidence", "distance", "mode", "ambiguous"}


# --- tolerance ---------------------------------------------------------------

def test_tolerance_defaults_to_the_array_resolution(room):
    """A coarse array must accept a looser match than a precise one."""
    # Tolerance is max(3 * resolution, 5 degrees), so the offset must clear the floor.
    angle = room.array.bearing_to(room.seat("A1").position) + 7.0
    coarse = map_bearing_to_seat(bearing_result(angle, resolution=8.0), room)
    precise = map_bearing_to_seat(bearing_result(angle, resolution=0.2), room)
    assert coarse.matched
    assert not precise.matched
