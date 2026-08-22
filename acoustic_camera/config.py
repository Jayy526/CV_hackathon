"""Camera configuration, and the array-to-camera relationship.

Nothing here is guessed. Webcam horizontal fields of view range from about 55
to 90 degrees, and the projection is a tangent map, so a wrong FOV is not a
uniform scaling error - it is right at the centre and increasingly wrong toward
the edges, which is the hardest kind of wrong to notice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CAMERA_PATH = Path(__file__).resolve().parents[1] / "config" / "camera.yaml"

# The array's own angular resolution at 0.135 m spacing and 16 kHz, at
# broadside. Parallax below this is lost in the measurement noise; above it,
# parallax is the dominant error.
ARRAY_RESOLUTION_DEGREES = 4.55


class CameraConfigError(ValueError):
    """Raised for a camera configuration that cannot describe a real camera."""


@dataclass(frozen=True)
class CameraConfig:
    """A camera, and where it sits relative to the microphone array."""

    index: int = 0
    width: int = 1280
    height: int = 720
    horizontal_fov_degrees: float = 70.0
    # Angle between the camera's optical axis and the array's 0 deg broadside.
    # Positive = the camera is aimed toward the +90 (channel 0) side.
    azimuth_offset_degrees: float = 0.0
    # Lateral distance between the camera and the ARRAY CENTRE. See
    # parallax_error_degrees: this error is uncorrectable in principle.
    lateral_offset_m: float = 0.0
    vertical_offset_m: float = 0.0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise CameraConfigError("camera index must be >= 0, got %r" % (self.index,))
        if self.width <= 0 or self.height <= 0:
            raise CameraConfigError(
                "capture resolution must be positive, got %rx%r" % (self.width, self.height))
        if not 0.0 < self.horizontal_fov_degrees < 180.0:
            raise CameraConfigError(
                "horizontal_fov_degrees must be in (0, 180), got %r. A real "
                "camera cannot see 180 degrees or more through a rectilinear "
                "lens, and the tangent projection diverges there."
                % (self.horizontal_fov_degrees,))
        if abs(self.azimuth_offset_degrees) >= 90.0:
            raise CameraConfigError(
                "azimuth_offset_degrees must be within +/-90, got %r: beyond "
                "that the camera and the array are not looking at the same "
                "half-space at all" % (self.azimuth_offset_degrees,))
        if self.lateral_offset_m < 0.0:
            raise CameraConfigError(
                "lateral_offset_m is a distance and must be >= 0, got %r"
                % (self.lateral_offset_m,))

    @property
    def centre_x(self) -> float:
        """Pixel column of the optical axis."""
        return self.width / 2.0

    @property
    def half_fov_degrees(self) -> float:
        return self.horizontal_fov_degrees / 2.0

    @property
    def focal_length_px(self) -> float:
        """f_px = (W/2) / tan(HFOV/2). The whole projection rests on this."""
        return self.centre_x / math.tan(math.radians(self.half_fov_degrees))

    @property
    def degrees_per_pixel_at_centre(self) -> float:
        """Only true AT THE CENTRE. Quoted for intuition, never used to project.

        A linear degrees-to-pixels scale is correct at the optical axis and
        increasingly wrong away from it, which is exactly why the projection
        uses tan instead.
        """
        return math.degrees(math.atan(1.0 / self.focal_length_px))


def parallax_error_degrees(lateral_offset_m: float, range_m: float) -> float:
    """Angular disagreement between camera and array for a source at `range_m`.

    atan(offset / range). Needs the range to correct, and a two-microphone
    array has no range, so this is uncorrectable in principle: it can only be
    kept small by mounting the camera at the array centre.
    """
    if range_m <= 0.0:
        raise CameraConfigError("range must be positive, got %r" % (range_m,))
    return math.degrees(math.atan(lateral_offset_m / range_m))


def parallax_dominant_within_m(
    lateral_offset_m: float,
    resolution_degrees: float = ARRAY_RESOLUTION_DEGREES,
) -> float:
    """Range inside which parallax exceeds the array's own resolution.

    0.0 when the camera is at the array centre - there is then no range at
    which parallax matters.
    """
    if lateral_offset_m <= 0.0:
        return 0.0
    return lateral_offset_m / math.tan(math.radians(resolution_degrees))


def parallax_warning(
    config: CameraConfig,
    resolution_degrees: float = ARRAY_RESOLUTION_DEGREES,
) -> str | None:
    """A sentence when the mounting offset is large enough to matter, else None."""
    if config.lateral_offset_m <= 0.0:
        return None
    inside = parallax_dominant_within_m(config.lateral_offset_m, resolution_degrees)
    at_one_metre = parallax_error_degrees(config.lateral_offset_m, 1.0)
    return (
        f"camera is {config.lateral_offset_m * 100:.0f} cm off the array centre: "
        f"parallax is {at_one_metre:.2f} deg at 1 m and exceeds the array's own "
        f"{resolution_degrees:.2f} deg resolution for anything closer than "
        f"{inside:.2f} m. This CANNOT be calibrated out - correcting it needs "
        f"the range, and this array has none. Move the camera toward the array "
        f"centre; azimuth_offset_degrees will not absorb it."
    )


def load_camera_config(path: str | Path | None = None) -> CameraConfig:
    """Load config/camera.yaml, or fall back to the documented defaults."""
    path = Path(path) if path is not None else DEFAULT_CAMERA_PATH
    if not path.exists():
        return CameraConfig()
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    camera = raw.get("camera", {}) or {}
    return CameraConfig(
        index=int(camera.get("index", 0)),
        width=int(camera.get("width", 1280)),
        height=int(camera.get("height", 720)),
        horizontal_fov_degrees=float(camera.get("horizontal_fov_degrees", 70.0)),
        azimuth_offset_degrees=float(camera.get("azimuth_offset_degrees", 0.0)),
        lateral_offset_m=float(camera.get("lateral_offset_m", 0.0)),
        vertical_offset_m=float(camera.get("vertical_offset_m", 0.0)),
    )
