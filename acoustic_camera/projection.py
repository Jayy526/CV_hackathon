"""Bearing -> pixel column, and angular uncertainty -> band width.

    f_px = (W/2) / tan(HFOV/2)
    x    = cx + f_px * tan(bearing - azimuth_offset)

TANGENT, NOT A LINEAR SCALE. A linear degrees-to-pixels map is exact at the
optical axis and, by construction, also at the extreme edge - both maps send
+/-HFOV/2 to the frame boundary. It is wrong EVERYWHERE IN BETWEEN, worst in
the mid-field: at a 70 degree FOV and 1280 px wide, the linear approximation is
about 33 px out (2.6% of frame width) near 20 degrees off axis. Agreeing at
both ends is what makes the error easy to miss - checking the centre and the
edge proves nothing.

TWO THINGS THIS MODULE REFUSES TO DO
------------------------------------
1. It never returns inf or NaN. tan diverges at +/-90 degrees, so a relative
   bearing at or beyond +/-90 is handled as off-frame BEFORE any tangent is
   taken. Nothing infinite can reach a renderer through here.

2. It never clamps an off-frame bearing to the frame edge. A clamped bearing
   draws a confident band at the edge of the picture for a sound that is not in
   shot at all - a lie that looks exactly like a correct answer. Off-frame is
   reported as off-frame, with the side it went off.

Band edges are a separate matter: a band whose CENTRE is visible may genuinely
extend past the frame, and clipping the drawn rectangle there is honest. That
case is flagged with `clipped_left` / `clipped_right`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from acoustic_camera.config import CameraConfig

LEFT = "left"
RIGHT = "right"

# Confidence below this is treated as this, so the uncertainty band widens but
# stays finite. A zero-confidence estimate would otherwise be infinitely wide,
# and the renderer is meant to draw nothing at all in that case anyway.
MIN_CONFIDENCE_FOR_WIDTH = 0.05


@dataclass(frozen=True)
class ProjectedBearing:
    """Where a bearing lands, or why it does not land anywhere."""

    bearing_degrees: float
    relative_degrees: float
    column: float | None
    on_screen: bool
    side: str = ""          # LEFT or RIGHT when off-screen, empty when visible
    reason: str = ""

    @property
    def off_frame(self) -> bool:
        return not self.on_screen


@dataclass(frozen=True)
class ProjectedBand:
    """A full-height vertical band: a column, and how wide the doubt is.

    The band is vertical and full-height because two microphones in a line
    measure AZIMUTH ONLY. There is no y position to compute, and a 2-D blob
    would be inventing depth the sensor does not have.
    """

    centre: ProjectedBearing
    sigma_degrees: float
    left_column: float | None
    right_column: float | None
    clipped_left: bool = False
    clipped_right: bool = False

    @property
    def on_screen(self) -> bool:
        return self.centre.on_screen

    @property
    def width_px(self) -> float | None:
        if self.left_column is None or self.right_column is None:
            return None
        return self.right_column - self.left_column

    @property
    def reason(self) -> str:
        return self.centre.reason


def project_bearing(bearing_degrees: float, config: CameraConfig) -> ProjectedBearing:
    """Project one bearing to a pixel column.

    Off-frame is a first-class answer, never a clamped column.
    """
    relative = bearing_degrees - config.azimuth_offset_degrees

    # Guard the tangent singularity BEFORE evaluating it. At exactly +/-90 the
    # ray is parallel to the image plane and meets it nowhere; beyond that it
    # is behind the camera and tan would silently wrap it back into frame.
    if abs(relative) >= 90.0:
        side = RIGHT if relative > 0 else LEFT
        return ProjectedBearing(
            bearing_degrees=bearing_degrees,
            relative_degrees=relative,
            column=None,
            on_screen=False,
            side=side,
            reason=(f"{relative:+.1f} deg from the optical axis is at or beyond "
                    f"90 deg: the ray never meets the image plane"),
        )

    column = config.centre_x + config.focal_length_px * math.tan(math.radians(relative))

    # Equivalent to |relative| > HFOV/2, but expressed in the pixels that will
    # actually be drawn. The last valid column is width - 1.
    if column < 0.0 or column > config.width - 1:
        side = RIGHT if relative > 0 else LEFT
        return ProjectedBearing(
            bearing_degrees=bearing_degrees,
            relative_degrees=relative,
            column=None,
            on_screen=False,
            side=side,
            reason=(f"{relative:+.1f} deg from the optical axis is outside the "
                    f"{config.horizontal_fov_degrees:.0f} deg field of view "
                    f"(+/-{config.half_fov_degrees:.1f} deg): OFF-FRAME to the "
                    f"{side}"),
        )

    return ProjectedBearing(
        bearing_degrees=bearing_degrees,
        relative_degrees=relative,
        column=column,
        on_screen=True,
    )


def uncertainty_degrees(
    angular_resolution_degrees: float,
    confidence: float,
    min_confidence: float = MIN_CONFIDENCE_FOR_WIDTH,
) -> float:
    """Half-width of the uncertainty band, in degrees.

    The array's own resolution is the floor: at perfect confidence the band is
    still that wide, because the geometry cannot do better. Lower confidence
    widens it in proportion, so a doubtful estimate DRAWS as a doubtful
    estimate rather than as a narrow bar of the same size.

    This is an honest heuristic, not a calibrated posterior. It is monotonic in
    confidence and floored by the physics, which is what the display needs; it
    is not a statistical interval and should not be quoted as one.
    """
    if angular_resolution_degrees <= 0.0:
        raise ValueError(
            "angular_resolution_degrees must be positive, got %r"
            % (angular_resolution_degrees,))
    usable = min(max(confidence, min_confidence), 1.0)
    return angular_resolution_degrees / usable


def project_band(
    bearing_degrees: float,
    angular_resolution_degrees: float,
    confidence: float,
    config: CameraConfig,
) -> ProjectedBand:
    """Project a bearing and its uncertainty to a band of pixel columns.

    THE TWO EDGES ARE PROJECTED SEPARATELY. The same angular uncertainty covers
    a different number of pixels at the edge of the frame than at the centre -
    the tangent map stretches - so computing one width at the centre and
    reusing it would understate the doubt exactly where it is largest.
    """
    sigma = uncertainty_degrees(angular_resolution_degrees, confidence)
    centre = project_bearing(bearing_degrees, config)
    if not centre.on_screen:
        return ProjectedBand(centre=centre, sigma_degrees=sigma,
                             left_column=None, right_column=None)

    lower = project_bearing(bearing_degrees - sigma, config)
    upper = project_bearing(bearing_degrees + sigma, config)

    # An edge falling outside the frame is genuine: the band really does extend
    # past what the camera can see. Clip it for drawing and SAY that it was
    # clipped, which is a different claim from "the source is at the edge".
    clipped_left = lower.column is None
    clipped_right = upper.column is None
    left = 0.0 if clipped_left else lower.column
    right = float(config.width - 1) if clipped_right else upper.column

    if left > right:  # pragma: no cover - defensive; sigma is always positive
        left, right = right, left
    return ProjectedBand(
        centre=centre,
        sigma_degrees=sigma,
        left_column=left,
        right_column=right,
        clipped_left=clipped_left,
        clipped_right=clipped_right,
    )


def project_event(event, config: CameraConfig) -> ProjectedBand | None:
    """Project an AcousticEvent, or None when the sensor declined to answer.

    None means DRAW NOTHING and show `event.reason`. It does not mean "off
    screen" and it must not become a band at the edge.
    """
    direction = getattr(event, "direction_degrees", None)
    if direction is None:
        return None
    resolution = getattr(event, "angular_resolution_degrees", None)
    if resolution is None:
        return None
    confidence = getattr(event, "localization_confidence", None)
    if confidence is None:
        confidence = getattr(event, "confidence", 0.0)
    return project_band(float(direction), float(resolution), float(confidence), config)


def visible_bearing_range(config: CameraConfig) -> tuple[float, float]:
    """The bearings the camera can actually show, in array coordinates.

    Useful for saying on screen how much of the array's +/-90 degree world is
    outside the picture.
    """
    return (config.azimuth_offset_degrees - config.half_fov_degrees,
            config.azimuth_offset_degrees + config.half_fov_degrees)
