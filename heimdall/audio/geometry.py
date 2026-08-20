"""Phase H: classroom and microphone-array geometry.

Everything about the physical room lives in config/classroom.yaml and is loaded
into these types. No algorithm anywhere else may hard-code a room size, a
microphone position or a seat position.

ANGLE CONVENTION (used identically by doa.py and seat_mapper.py):

    The array axis is the line through the microphones. Broadside is
    perpendicular to it, pointing into the room.

        0 degrees   = broadside, straight out in front of the array
        +90 degrees = along the axis, toward channel 0
        -90 degrees = along the axis, toward the last channel

    Angles are therefore in [-90, +90]. A two-microphone array physically
    cannot tell front from back, so `orientation_degrees` in the config says
    which side of the array the classroom is on.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CLASSROOM_PATH = Path(__file__).resolve().parents[2] / "config" / "classroom.yaml"


class GeometryError(ValueError):
    """Raised for a physically impossible or malformed geometry."""


@dataclass(frozen=True)
class Microphone:
    id: str
    x: float
    y: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass(frozen=True)
class Seat:
    id: str
    x: float
    y: float
    row: str | None = None
    column: int | None = None

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass(frozen=True)
class MicrophoneArray:
    """A set of microphones at known room coordinates, in channel order.

    microphones[0] is channel_0, microphones[1] is channel_1, and so on. The
    order matters: it defines the sign of every TDOA in the system.
    """

    microphones: tuple[Microphone, ...]
    orientation_degrees: float = 90.0

    def __post_init__(self) -> None:
        if len(self.microphones) < 2:
            raise GeometryError(
                "an array needs at least 2 microphones, got %d" % len(self.microphones)
            )
        ids = [m.id for m in self.microphones]
        if len(set(ids)) != len(ids):
            raise GeometryError("microphone ids must be unique, got %r" % (ids,))
        if self.spacing <= 0:
            raise GeometryError("microphones 0 and 1 are at the same position")

    @property
    def num_channels(self) -> int:
        return len(self.microphones)

    @property
    def positions(self) -> np.ndarray:
        """(num_channels, 2) array of room coordinates, in channel order."""
        return np.array([m.position for m in self.microphones], dtype=np.float64)

    @property
    def centroid(self) -> np.ndarray:
        return self.positions.mean(axis=0)

    @property
    def spacing(self) -> float:
        """Distance between channel 0 and channel 1, in metres."""
        return float(
            np.linalg.norm(self.microphones[0].position - self.microphones[1].position)
        )

    @property
    def aperture(self) -> float:
        """Largest distance between any two microphones - this sets the resolution."""
        positions = self.positions
        diffs = positions[:, None, :] - positions[None, :, :]
        return float(np.max(np.linalg.norm(diffs, axis=-1)))

    @property
    def is_linear(self) -> bool:
        """True when every microphone lies on one straight line.

        A linear array can only measure a bearing, never a 2D position.
        """
        positions = self.positions
        if positions.shape[0] <= 2:
            return True
        centred = positions - positions.mean(axis=0)
        singular = np.linalg.svd(centred, compute_uv=False)
        return bool(singular[1] < 1e-9 * max(singular[0], 1e-12))

    @property
    def axis(self) -> np.ndarray:
        """Unit vector along the array, pointing toward channel 0."""
        vector = self.microphones[0].position - self.microphones[-1].position
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise GeometryError("first and last microphone are at the same position")
        return vector / norm

    @property
    def broadside(self) -> np.ndarray:
        """Unit normal to the array axis, pointing into the classroom.

        There are two normals; `orientation_degrees` selects the one that faces
        the students.
        """
        axis = self.axis
        candidate = np.array([-axis[1], axis[0]], dtype=np.float64)
        wanted = np.array(
            [
                np.cos(np.radians(self.orientation_degrees)),
                np.sin(np.radians(self.orientation_degrees)),
            ]
        )
        return candidate if float(np.dot(candidate, wanted)) >= 0 else -candidate

    def bearing_to(self, point: np.ndarray | tuple[float, float]) -> float:
        """Angle in degrees of `point` seen from the array, in the convention
        documented at the top of this module."""
        vector = np.asarray(point, dtype=np.float64) - self.centroid
        along = float(np.dot(vector, self.axis))
        ahead = float(np.dot(vector, self.broadside))
        return float(np.degrees(np.arctan2(along, ahead)))

    def is_in_front(self, point: np.ndarray | tuple[float, float]) -> bool:
        """True when `point` is on the classroom side of the array."""
        vector = np.asarray(point, dtype=np.float64) - self.centroid
        return float(np.dot(vector, self.broadside)) >= 0.0

    def distance_to(self, point: np.ndarray | tuple[float, float]) -> float:
        return float(np.linalg.norm(np.asarray(point, dtype=np.float64) - self.centroid))


@dataclass(frozen=True)
class ClassroomConfig:
    """The room, the array in it, and the seats."""

    width: float
    height: float
    array: MicrophoneArray
    seats: tuple[Seat, ...] = ()
    name: str = "classroom"
    speed_of_sound: float = 343.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise GeometryError(
                "classroom dimensions must be positive, got %r x %r" % (self.width, self.height)
            )
        ids = [s.id for s in self.seats]
        if len(set(ids)) != len(ids):
            raise GeometryError("seat ids must be unique")

    @property
    def num_seats(self) -> int:
        return len(self.seats)

    def seat(self, seat_id: str) -> Seat:
        for seat in self.seats:
            if seat.id == seat_id:
                return seat
        raise KeyError("no seat with id %r" % (seat_id,))

    def contains(self, point: np.ndarray | tuple[float, float]) -> bool:
        x, y = float(point[0]), float(point[1])
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height

    def seat_bearings(self) -> dict[str, float]:
        """Bearing of every seat as seen from the microphone array."""
        return {seat.id: self.array.bearing_to(seat.position) for seat in self.seats}


def _build_grid_seats(spec: dict) -> list[Seat]:
    """Generate a rectangular seat grid from rows/columns/spacing."""
    rows = int(spec["rows"])
    columns = int(spec["columns"])
    if rows <= 0 or columns <= 0:
        raise GeometryError("rows and columns must be positive")
    if rows > len(string.ascii_uppercase):
        raise GeometryError("grid supports at most 26 rows; list seats explicitly instead")

    row_spacing = float(spec.get("row_spacing", 0.9))
    column_spacing = float(spec.get("column_spacing", 0.6))
    origin_x = float(spec.get("origin_x", 0.0))
    origin_y = float(spec.get("origin_y", 0.0))

    seats: list[Seat] = []
    for r in range(rows):
        letter = string.ascii_uppercase[r]
        for c in range(columns):
            seats.append(
                Seat(
                    id="%s%d" % (letter, c + 1),
                    x=origin_x + c * column_spacing,
                    y=origin_y + r * row_spacing,
                    row=letter,
                    column=c + 1,
                )
            )
    return seats


def load_classroom_config(path: str | Path | None = None) -> ClassroomConfig:
    """Load config/classroom.yaml (or an explicit path)."""
    path = Path(path) if path is not None else DEFAULT_CLASSROOM_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    room = raw.get("classroom", {})
    mics_raw = raw.get("microphones", [])
    if not mics_raw:
        raise GeometryError("config %s defines no microphones" % path)

    microphones = tuple(
        Microphone(id=str(m["id"]), x=float(m["x"]), y=float(m["y"])) for m in mics_raw
    )
    array = MicrophoneArray(
        microphones=microphones,
        orientation_degrees=float(raw.get("array", {}).get("orientation_degrees", 90.0)),
    )

    seats_raw = raw.get("seats")
    if isinstance(seats_raw, dict) and "grid" in seats_raw:
        seats = tuple(_build_grid_seats(seats_raw["grid"]))
    elif isinstance(seats_raw, list):
        seats = tuple(
            Seat(
                id=str(s["id"]),
                x=float(s["x"]),
                y=float(s["y"]),
                row=str(s["row"]) if s.get("row") is not None else None,
                column=int(s["column"]) if s.get("column") is not None else None,
            )
            for s in seats_raw
        )
    else:
        seats = ()

    return ClassroomConfig(
        width=float(room.get("width", 8.0)),
        height=float(room.get("height", 10.0)),
        array=array,
        seats=seats,
        name=str(room.get("name", "classroom")),
        speed_of_sound=float(room.get("speed_of_sound", 343.0)),
    )
