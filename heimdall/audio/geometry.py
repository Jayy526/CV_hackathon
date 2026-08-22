"""Classroom geometry: the room, its seats, and the array's place in it.

PARKED. The project direction moved to a standalone acoustic direction sensor
(see acoustic_array/). Nothing here is wrong and nothing here is deleted, but
it is not on the current path and should not be extended.

The array geometry itself now lives in acoustic_array.geometry, which knows
nothing about rooms. This module re-exports Microphone and MicrophoneArray so
existing callers keep working unchanged.

ANGLE CONVENTION: unchanged, and defined in acoustic_array.geometry.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from acoustic_array.geometry import (  # noqa: F401  - re-exported for callers
    GeometryError,
    Microphone,
    MicrophoneArray,
)

DEFAULT_CLASSROOM_PATH = Path(__file__).resolve().parents[2] / "config" / "classroom.yaml"


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
