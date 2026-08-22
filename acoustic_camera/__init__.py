"""Relating a microphone array's bearings to a camera's pixels.

The acoustic_array package knows nothing about cameras, and this package knows
nothing about microphones - it takes an angle and returns a column. The
dependency runs one way only.

    from acoustic_camera import load_camera_config, project_band

    config = load_camera_config()
    band = project_band(bearing_degrees=-12.0,
                        angular_resolution_degrees=4.55,
                        confidence=0.8, config=config)
    if band.on_screen:
        draw(band.left_column, band.right_column)
    else:
        say(f"off-frame to the {band.centre.side}")
"""

from acoustic_camera.config import (
    ARRAY_RESOLUTION_DEGREES,
    CameraConfig,
    CameraConfigError,
    load_camera_config,
    parallax_dominant_within_m,
    parallax_error_degrees,
    parallax_warning,
)
from acoustic_camera.projection import (
    LEFT,
    RIGHT,
    ProjectedBand,
    ProjectedBearing,
    project_band,
    project_bearing,
    project_event,
    uncertainty_degrees,
    visible_bearing_range,
)

__all__ = [
    "ARRAY_RESOLUTION_DEGREES",
    "CameraConfig",
    "CameraConfigError",
    "LEFT",
    "RIGHT",
    "ProjectedBand",
    "ProjectedBearing",
    "load_camera_config",
    "parallax_dominant_within_m",
    "parallax_error_degrees",
    "parallax_warning",
    "project_band",
    "project_bearing",
    "project_event",
    "uncertainty_degrees",
    "visible_bearing_range",
]
