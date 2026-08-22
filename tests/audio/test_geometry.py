"""Classroom geometry: configurable rooms, arrays, seats, and the angle convention."""

import textwrap

import numpy as np
import pytest

from heimdall.audio.gcc_phat import max_delay_samples
from heimdall.audio.geometry import (
    ClassroomConfig,
    GeometryError,
    Microphone,
    MicrophoneArray,
    Seat,
    load_classroom_config,
)


def linear_array(spacing=0.30, orientation=90.0, num=2):
    """A `num`-microphone array centred on the origin, along the x axis."""
    offsets = (np.arange(num) - (num - 1) / 2) * spacing
    # mic_1 (channel 0) sits at the most negative x.
    mics = tuple(
        Microphone(id="mic_%d" % (i + 1), x=float(offset), y=0.0)
        for i, offset in enumerate(offsets)
    )
    return MicrophoneArray(microphones=mics, orientation_degrees=orientation)


def write_config(tmp_path, body):
    path = tmp_path / "classroom.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


# --- the shipped example config ---------------------------------------------

def test_example_config_loads():
    room = load_classroom_config()
    assert room.width > 0 and room.height > 0
    assert room.array.num_channels == 2
    assert room.num_seats == 30


def test_config_spacing_is_the_measured_hardware_spacing():
    """The built array measures 0.135 m centre to centre. Every TDOA search
    window in the system is derived from this number, so it must be the real
    one, not the pre-hardware placeholder."""
    room = load_classroom_config()
    assert room.array.spacing == pytest.approx(0.135, abs=1e-6)


def test_measured_spacing_sets_the_physical_delay_limit():
    """0.135 m allows at most ~18.9 samples of TDOA at 48 kHz. The diagnostic
    sketch searched +-28, which is why it could return unphysical lags."""
    room = load_classroom_config()
    limit = max_delay_samples(room.array.spacing, 48000, room.speed_of_sound)
    assert limit == pytest.approx(0.135 / 343.0 * 48000, rel=1e-6)
    assert 18.0 < limit < 19.0


def test_channel_zero_side_is_unchanged_by_the_spacing_update():
    """Narrowing the array must not flip which side channel 0 faces: the axis
    points toward channel 0 and defines the sign of every bearing."""
    room = load_classroom_config()
    assert room.array.microphones[0].id == "mic_1"
    assert room.array.microphones[0].x < room.array.microphones[1].x
    assert np.allclose(room.array.axis, [-1.0, 0.0])


def test_example_config_seat_grid_is_labelled():
    room = load_classroom_config()
    assert {"A1", "A6", "E1", "E6"} <= {s.id for s in room.seats}
    assert room.seat("B4").row == "B"
    assert room.seat("B4").column == 4


def test_example_seats_are_inside_the_room():
    room = load_classroom_config()
    for seat in room.seats:
        assert room.contains(seat.position), seat.id


# --- the room is configurable, never hard-coded -----------------------------

def test_room_dimensions_come_from_the_file(tmp_path):
    path = write_config(
        tmp_path,
        """
        classroom: {width: 12.0, height: 15.0, name: hall}
        microphones:
          - {id: m1, x: 5.0, y: 0.0}
          - {id: m2, x: 7.0, y: 0.0}
        """,
    )
    room = load_classroom_config(path)
    assert (room.width, room.height, room.name) == (12.0, 15.0, "hall")
    assert room.array.spacing == pytest.approx(2.0)


def test_seats_can_be_listed_explicitly(tmp_path):
    path = write_config(
        tmp_path,
        """
        classroom: {width: 6.0, height: 6.0}
        microphones:
          - {id: m1, x: 2.8, y: 0.0}
          - {id: m2, x: 3.2, y: 0.0}
        seats:
          - {id: X1, x: 1.0, y: 2.0, row: X, column: 1}
          - {id: X2, x: 2.0, y: 2.0, row: X, column: 2}
        """,
    )
    room = load_classroom_config(path)
    assert room.num_seats == 2
    assert room.seat("X2").x == pytest.approx(2.0)


def test_seat_grid_spacing_is_configurable(tmp_path):
    path = write_config(
        tmp_path,
        """
        classroom: {width: 20.0, height: 20.0}
        microphones:
          - {id: m1, x: 9.5, y: 0.0}
          - {id: m2, x: 10.5, y: 0.0}
        seats:
          grid: {rows: 2, columns: 3, row_spacing: 2.0, column_spacing: 1.5,
                 origin_x: 4.0, origin_y: 3.0}
        """,
    )
    room = load_classroom_config(path)
    assert room.num_seats == 6
    assert room.seat("A1").position.tolist() == [4.0, 3.0]
    assert room.seat("A3").x == pytest.approx(7.0)
    assert room.seat("B1").y == pytest.approx(5.0)


def test_four_microphones_load_without_code_changes(tmp_path):
    """Phase L readiness: the config format already carries four channels."""
    path = write_config(
        tmp_path,
        """
        classroom: {width: 8.0, height: 8.0}
        microphones:
          - {id: m1, x: 3.7, y: 0.0}
          - {id: m2, x: 4.3, y: 0.0}
          - {id: m3, x: 3.7, y: 0.6}
          - {id: m4, x: 4.3, y: 0.6}
        """,
    )
    room = load_classroom_config(path)
    assert room.array.num_channels == 4
    assert not room.array.is_linear  # a square is a genuine 2-D aperture


def test_speed_of_sound_is_configurable(tmp_path):
    path = write_config(
        tmp_path,
        """
        classroom: {width: 5.0, height: 5.0, speed_of_sound: 350.0}
        microphones:
          - {id: m1, x: 2.4, y: 0.0}
          - {id: m2, x: 2.6, y: 0.0}
        """,
    )
    assert load_classroom_config(path).speed_of_sound == 350.0


def test_config_without_microphones_is_rejected(tmp_path):
    path = write_config(tmp_path, "classroom: {width: 5.0, height: 5.0}\n")
    with pytest.raises(GeometryError):
        load_classroom_config(path)


# --- array geometry ----------------------------------------------------------

def test_array_needs_at_least_two_microphones():
    with pytest.raises(GeometryError):
        MicrophoneArray(microphones=(Microphone("m1", 0.0, 0.0),))


def test_coincident_microphones_are_rejected():
    with pytest.raises(GeometryError):
        MicrophoneArray(
            microphones=(Microphone("m1", 1.0, 1.0), Microphone("m2", 1.0, 1.0))
        )


def test_duplicate_microphone_ids_are_rejected():
    with pytest.raises(GeometryError):
        MicrophoneArray(
            microphones=(Microphone("m1", 0.0, 0.0), Microphone("m1", 0.3, 0.0))
        )


def test_spacing_and_aperture():
    array = linear_array(spacing=0.4, num=2)
    assert array.spacing == pytest.approx(0.4)
    assert array.aperture == pytest.approx(0.4)


def test_two_and_three_collinear_microphones_are_linear():
    assert linear_array(num=2).is_linear
    assert linear_array(num=3).is_linear


def test_a_square_array_is_not_linear():
    array = MicrophoneArray(
        microphones=(
            Microphone("m1", 0.0, 0.0),
            Microphone("m2", 0.3, 0.0),
            Microphone("m3", 0.0, 0.3),
            Microphone("m4", 0.3, 0.3),
        )
    )
    assert not array.is_linear


def test_broadside_is_perpendicular_to_the_axis():
    array = linear_array()
    assert float(np.dot(array.axis, array.broadside)) == pytest.approx(0.0, abs=1e-12)


def test_orientation_selects_which_side_faces_the_room():
    facing_forward = linear_array(orientation=90.0)
    facing_backward = linear_array(orientation=270.0)
    assert facing_forward.broadside[1] > 0
    assert facing_backward.broadside[1] < 0


# --- the angle convention ----------------------------------------------------

def test_broadside_point_is_zero_degrees():
    array = linear_array()
    assert array.bearing_to((0.0, 5.0)) == pytest.approx(0.0)


def test_channel_zero_side_is_positive():
    """mic_1 is channel 0 and sits at negative x, so -x is the positive side."""
    array = linear_array()
    assert array.axis[0] < 0
    assert array.bearing_to((-5.0, 5.0)) > 0
    assert array.bearing_to((5.0, 5.0)) < 0


def test_bearing_is_antisymmetric():
    array = linear_array()
    assert array.bearing_to((-3.0, 4.0)) == pytest.approx(-array.bearing_to((3.0, 4.0)))


def test_bearing_of_a_45_degree_point():
    array = linear_array()
    assert array.bearing_to((-4.0, 4.0)) == pytest.approx(45.0)


def test_is_in_front_matches_orientation():
    array = linear_array(orientation=90.0)
    assert array.is_in_front((0.0, 3.0))
    assert not array.is_in_front((0.0, -3.0))


def test_distance_to_is_euclidean_from_the_centroid():
    array = linear_array()
    assert array.distance_to((0.0, 4.0)) == pytest.approx(4.0)
    assert array.distance_to((3.0, 4.0)) == pytest.approx(5.0)


# --- classroom validation ----------------------------------------------------

def test_negative_room_dimensions_are_rejected():
    with pytest.raises(GeometryError):
        ClassroomConfig(width=-1.0, height=5.0, array=linear_array())


def test_duplicate_seat_ids_are_rejected():
    with pytest.raises(GeometryError):
        ClassroomConfig(
            width=5.0,
            height=5.0,
            array=linear_array(),
            seats=(Seat("A1", 1.0, 1.0), Seat("A1", 2.0, 2.0)),
        )


def test_contains_checks_the_room_bounds():
    room = ClassroomConfig(width=8.0, height=10.0, array=linear_array())
    assert room.contains((4.0, 5.0))
    assert room.contains((0.0, 0.0))
    assert not room.contains((-0.1, 5.0))
    assert not room.contains((4.0, 10.5))


def test_unknown_seat_id_raises():
    room = load_classroom_config()
    with pytest.raises(KeyError):
        room.seat("Z99")


def test_seat_bearings_cover_every_seat():
    room = load_classroom_config()
    bearings = room.seat_bearings()
    assert len(bearings) == room.num_seats
    assert all(-90.0 <= b <= 90.0 for b in bearings.values())
