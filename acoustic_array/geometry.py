"""Microphone-array geometry. The array, and nothing about any room.

This is the whole geometric world the acoustic sensor knows: where the
microphones are relative to each other, and which way is forward. There are no
seats here, no room dimensions and no classroom.yaml - those belong to a layer
built ON this one, not inside it.

ANGLE CONVENTION - unchanged from the original section 5, and load-bearing:

    The array axis is the line through the microphones. Broadside is
    perpendicular to it, pointing away from the array's front face.

        0 degrees   = broadside, straight out in front of the array
        +90 degrees = along the axis, toward CHANNEL 0
        -90 degrees = along the axis, toward the last channel

    Angles are therefore in [-90, +90]. `microphones[0]` IS channel 0: the list
    order defines the sign of every TDOA in the system, so do not reorder it.

    A two-microphone array physically cannot tell front from back. Nothing here
    resolves that; a caller with another sensor (a camera, say) must.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# Optional. The package works with no config file at all - see default_array().
DEFAULT_ARRAY_PATH = Path(__file__).resolve().parents[1] / "config" / "array.yaml"

# The built array: 2x INMP441, measured centre to centre.
DEFAULT_SPACING_M = 0.135
DEFAULT_SPEED_OF_SOUND = 343.0


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


def linear_array(num_channels: int = 2, spacing: float = DEFAULT_SPACING_M
                 ) -> MicrophoneArray:
    """A uniform linear array centred on the origin, channel 0 at lower x.

    Which microphone sits at lower x sets the sign of every bearing, so this
    mirrors the physical build: mic 1 (channel 0) on the low-x side.
    """
    if num_channels < 2:
        raise GeometryError(
            "need at least 2 microphones to measure a direction, got %r" % (num_channels,))
    if spacing <= 0:
        raise GeometryError("spacing must be positive, got %r" % (spacing,))
    span = (num_channels - 1) * spacing
    return MicrophoneArray(tuple(
        Microphone(id="mic_%d" % (i + 1), x=-span / 2.0 + i * spacing, y=0.0)
        for i in range(num_channels)
    ))


def default_array() -> MicrophoneArray:
    """The as-built array, with no configuration file required."""
    return linear_array(2, DEFAULT_SPACING_M)


def load_array_config(path: str | Path | None = None) -> MicrophoneArray:
    """Load an array from a small YAML file, or fall back to default_array().

    Deliberately tolerant of the file being absent: this package must be usable
    by someone who has the hardware and none of this repo's room configuration.
    """
    path = Path(path) if path is not None else DEFAULT_ARRAY_PATH
    if not path.exists():
        return default_array()
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    mics = (raw.get("array", {}) or {}).get("microphones")
    if not mics:
        return default_array()
    return MicrophoneArray(tuple(
        Microphone(id=str(m.get("id", "mic_%d" % (i + 1))),
                   x=float(m["x"]), y=float(m["y"]))
        for i, m in enumerate(mics)
    ))


def speed_of_sound_from(path: str | Path | None = None) -> float:
    path = Path(path) if path is not None else DEFAULT_ARRAY_PATH
    if not path.exists():
        return DEFAULT_SPEED_OF_SOUND
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return float((raw.get("array", {}) or {}).get("speed_of_sound",
                                                  DEFAULT_SPEED_OF_SOUND))
