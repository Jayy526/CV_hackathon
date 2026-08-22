"""Compositing bearings onto video frames. Pure numpy - no OpenCV in here.

cv2 is needed to open a camera, show a window and write an mp4. None of that is
needed to decide WHAT to draw, so none of it is imported here: the whole
overlay is testable headlessly, which is the only reason the honesty rules
below can be pinned by tests at all.

WHAT MAY BE DRAWN, AND WHY
--------------------------
Two microphones in a line measure AZIMUTH ONLY. There is no range and no
elevation, so:

  * A located sound is a FULL-HEIGHT VERTICAL BAND. Never a blob at some
    (x, y). A blob would invent depth the sensor cannot measure, and it is the
    easiest way to turn this demo into a lie.
  * The band's WIDTH is the uncertainty, from bearing-sigma and bearing+sigma
    projected SEPARATELY (acoustic_camera.projection). It widens as confidence
    falls, and it is wider in pixels at the frame edge than at the centre.
  * Its INTENSITY is event energy, decaying over a window so recent sound is
    bright and old sound fades. Concurrent directions accumulate.
  * OFF-FRAME is not a band. It is an edge WEDGE - a different shape entirely -
    because a band pinned to the edge reads as "the sound is over there", when
    the truth is "the sound is outside the picture".
  * A CLIPPED band is different again and legitimate: the sound is in shot, but
    its uncertainty genuinely reaches past the view. Drawn as a band with a
    marked edge, never conflated with off-frame.
  * When the sensor declines, NOTHING is drawn and the reason is displayed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from acoustic_camera.config import CameraConfig
from acoustic_camera.projection import ProjectedBand, project_band

# BGR, because that is what OpenCV will eventually blit.
WHISPER_COLOUR = (255, 0, 255)      # magenta - the claim being demonstrated
SPEECH_COLOUR = (0, 170, 255)       # amber
SOUND_COLOUR = (200, 140, 60)       # steel blue
OFF_FRAME_COLOUR = (70, 70, 235)    # desaturated red: NOT a located sound

EVENT_COLOURS = {
    "POSSIBLE_WHISPER": WHISPER_COLOUR,
    "POSSIBLE_SPEECH": SPEECH_COLOUR,
    "SOUND_DETECTED": SOUND_COLOUR,
}

SYNTHETIC_BANNER = "SYNTHETIC AUDIO - SIMULATED, NOT A LIVE MEASUREMENT"
BEHIND_NOTICE = ("front/back is ambiguous to a linear array: sound from BEHIND "
                 "cannot be shown and is not drawn")


@dataclass(frozen=True)
class OverlayConfig:
    """How the overlay looks and how long it remembers."""

    # Older than this and an echo is gone entirely - a clean cutoff rather than
    # an exponential tail that never quite reaches zero.
    decay_seconds: float = 1.5
    max_alpha: float = 0.55
    centre_line_alpha: float = 0.95
    centre_line_px: int = 2
    # Audio events may be timestamped slightly ahead of the displayed frame;
    # this much lead is tolerated rather than treated as "not yet".
    sync_tolerance_seconds: float = 0.25
    # Energy mapping, in dBFS. Below floor draws nothing; above ceiling is full.
    energy_floor_db: float = -60.0
    energy_ceiling_db: float = -12.0
    off_frame_wedge_px: int = 48
    ring_capacity: int = 256
    # Fraction of the frame budget the overlay may take before the HUD says so.
    # Measured: ~3.1 ms typical and ~10.5 ms with four overlapping bands at
    # 720p, against 33.3 ms at 30 fps. Announced, never silently absorbed.
    overlay_budget_fraction: float = 0.5
    target_fps: float = 30.0


@dataclass(frozen=True)
class Echo:
    """One event, placed on the display timeline.

    `band` is None when the sound is off-frame or when the sensor declined; the
    two are told apart by `off_frame_side` being set or not.
    """

    display_time: float
    event_type: str
    energy: float
    bearing_degrees: float | None
    confidence: float
    band: ProjectedBand | None = None
    off_frame_side: str = ""
    reason: str = ""

    @property
    def is_located(self) -> bool:
        return self.band is not None and self.band.on_screen

    @property
    def is_off_frame(self) -> bool:
        return self.off_frame_side != ""

    @property
    def declined(self) -> bool:
        return self.band is None and not self.off_frame_side


def energy_from_rms(rms: float, config: OverlayConfig) -> float:
    """Map a level to [0, 1] through dBFS, so quiet sounds draw faintly."""
    if rms <= 0.0:
        return 0.0
    db = 20.0 * math.log10(max(rms, 1e-12))
    span = config.energy_ceiling_db - config.energy_floor_db
    return float(min(max((db - config.energy_floor_db) / span, 0.0), 1.0))


def colour_for(event_type: str) -> tuple[int, int, int]:
    return EVENT_COLOURS.get(event_type, SOUND_COLOUR)


def echo_from_event(
    event,
    camera: CameraConfig,
    config: OverlayConfig,
    display_time: float,
) -> Echo:
    """Turn one AcousticEvent into something drawable, or into a refusal.

    Duck-typed on purpose: this module never imports acoustic_array.
    """
    rms_values = getattr(event, "channel_rms", ()) or ()
    rms = float(sum(rms_values) / len(rms_values)) if rms_values else 0.0
    energy = energy_from_rms(rms, config)
    event_type = str(getattr(event, "event_type", "SOUND_DETECTED"))
    bearing = getattr(event, "direction_degrees", None)
    confidence = float(getattr(event, "localization_confidence", None) or 0.0)

    if bearing is None:
        # The sensor declined. Draw nothing; the reason goes to the HUD.
        return Echo(
            display_time=display_time, event_type=event_type, energy=energy,
            bearing_degrees=None, confidence=confidence,
            reason=str(getattr(event, "reason", "") or "no direction reported"),
        )

    resolution = getattr(event, "angular_resolution_degrees", None)
    if resolution is None or resolution <= 0.0:
        return Echo(
            display_time=display_time, event_type=event_type, energy=energy,
            bearing_degrees=float(bearing), confidence=confidence,
            reason="no angular resolution reported, so the uncertainty is unknown",
        )

    band = project_band(float(bearing), float(resolution), confidence, camera)
    if not band.on_screen:
        return Echo(
            display_time=display_time, event_type=event_type, energy=energy,
            bearing_degrees=float(bearing), confidence=confidence,
            off_frame_side=band.centre.side, reason=band.reason,
        )
    return Echo(
        display_time=display_time, event_type=event_type, energy=energy,
        bearing_degrees=float(bearing), confidence=confidence, band=band,
    )


class BearingRing:
    """A short time-stamped history of bearings, matched to the DISPLAYED frame.

    Video and audio arrive at different rates and different latencies, so the
    newest bearing does not belong to the newest video frame. Every echo
    carries the wall-clock time it should appear at, and rendering selects by
    that rather than by "most recent".
    """

    def __init__(self, config: OverlayConfig | None = None) -> None:
        self.config = config or OverlayConfig()
        self._echoes: deque[Echo] = deque(maxlen=self.config.ring_capacity)

    def __len__(self) -> int:
        return len(self._echoes)

    def add(self, echo: Echo) -> None:
        self._echoes.append(echo)

    def clear(self) -> None:
        self._echoes.clear()

    def active(self, at_time: float) -> list[tuple[Echo, float]]:
        """Echoes visible at `at_time`, each with its decay weight in (0, 1].

        Linear decay to exactly zero at `decay_seconds`: an echo older than the
        window is gone, not merely faint.
        """
        window = self.config.decay_seconds
        tolerance = self.config.sync_tolerance_seconds
        out: list[tuple[Echo, float]] = []
        for echo in self._echoes:
            age = at_time - echo.display_time
            if age < -tolerance:
                continue                     # not due yet
            age = max(age, 0.0)
            if age >= window:
                continue
            out.append((echo, 1.0 - age / window))
        return out

    def latest_decline(self, at_time: float) -> str:
        """Reason from the most recent refusal still inside the window."""
        for echo, _ in reversed(self.active(at_time)):
            if echo.declined:
                return echo.reason
        return ""

    def prune(self, at_time: float) -> None:
        window = self.config.decay_seconds
        while self._echoes and at_time - self._echoes[0].display_time >= window:
            self._echoes.popleft()


def composite(
    frame: np.ndarray,
    ring: BearingRing,
    at_time: float,
    config: OverlayConfig | None = None,
) -> np.ndarray:
    """Draw every active echo onto a copy of `frame`. Returns the new frame."""
    config = config or ring.config
    # Stay in uint8 and touch only the columns that actually change. Converting
    # the whole frame to float32 and blending it end to end costs ~16 ms at
    # 720p - half the 30 fps budget - to paint a band a few hundred pixels wide.
    out = frame.copy()
    height, width = out.shape[:2]

    # Bands are full-height, so alpha and colour vary only along x. One column
    # vector per frame, broadcast over rows: this is what keeps the per-frame
    # cost in the tens of microseconds rather than the milliseconds.
    alpha = np.zeros(width, dtype=np.float32)
    weighted_colour = np.zeros((width, 3), dtype=np.float32)

    active = ring.active(at_time)
    for echo, decay in active:
        if not echo.is_located:
            continue
        band = echo.band
        strength = float(decay * echo.energy * config.max_alpha)
        if strength <= 0.0:
            continue
        left = int(max(0, math.floor(band.left_column)))
        right = int(min(width - 1, math.ceil(band.right_column)))
        if right < left:
            continue
        colour = np.asarray(colour_for(echo.event_type), dtype=np.float32)

        # Concurrent directions ACCUMULATE rather than overwrite.
        alpha[left:right + 1] += strength
        weighted_colour[left:right + 1] += colour * strength

        # A thin bright line at the point estimate, inside the doubt band.
        centre = int(round(band.centre.column))
        half = max(config.centre_line_px // 2, 0)
        lo, hi = max(0, centre - half), min(width - 1, centre + half)
        line_strength = float(decay * config.centre_line_alpha)
        alpha[lo:hi + 1] += line_strength
        weighted_colour[lo:hi + 1] += colour * line_strength

    for lo, hi in _runs(alpha > 0.0):
        total = np.clip(alpha[lo:hi], 0.0, 1.0)
        colour = weighted_colour[lo:hi] / np.maximum(alpha[lo:hi], 1e-6)[:, None]
        # Fixed-point Q7 in uint16, fused through `out=`. The obvious float32
        # version costs ~25% more and allocates a full-size temporary at every
        # step; this is the same arithmetic with one buffer and four passes.
        # 128 not 256: 255*128 + 255*128 = 65280 still fits a uint16.
        inverse = ((1.0 - total) * 128.0).astype(np.uint16)
        addend = (colour * total[:, None] * 128.0).astype(np.uint16)
        region = out[:, lo:hi]
        scratch = np.multiply(region, inverse[None, :, None], dtype=np.uint16)
        np.add(scratch, addend[None, :, :], out=scratch)
        np.right_shift(scratch, 7, out=scratch)
        np.copyto(region, scratch, casting="unsafe")

    for echo, decay in active:
        if echo.is_off_frame:
            _draw_off_frame_wedge(out, echo, decay, config)
        elif echo.is_located and (echo.band.clipped_left or echo.band.clipped_right):
            _mark_clipped_edges(out, echo, decay, config)
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, stop) spans where `mask` is True.

    Blending only these keeps two bands at opposite edges from dragging the
    whole width of the frame through the float path.
    """
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _draw_off_frame_wedge(frame, echo: Echo, decay: float,
                          config: OverlayConfig) -> None:
    """A TRIANGULAR edge wedge - deliberately not the shape of a located sound.

    A band pinned to the edge would read as "the sound is over there". This
    reads as "the sound is outside the picture", which is what off-frame means.
    """
    height, width = frame.shape[:2]
    depth = min(config.off_frame_wedge_px, width // 4)
    if depth <= 0:
        return
    strength = float(decay * max(echo.energy, 0.25) * config.max_alpha)
    colour = np.asarray(OFF_FRAME_COLOUR, dtype=np.float32)

    # Widest at the vertical centre, tapering to nothing at top and bottom.
    rows = np.arange(height, dtype=np.float32)
    taper = 1.0 - np.abs(rows - (height - 1) / 2.0) / ((height - 1) / 2.0 + 1e-6)
    extent = np.clip(taper, 0.0, 1.0) * depth

    # Vectorised: a per-row loop over 720 slices is far too slow to run every
    # video frame. The triangle is a boolean mask over a depth-wide strip.
    columns = np.arange(depth, dtype=np.float32)
    if echo.off_frame_side == "left":
        mask = columns[None, :] < extent[:, None]
        strip = frame[:, 0:depth]
    else:
        mask = columns[None, :] >= (depth - extent[:, None])
        strip = frame[:, width - depth:width]

    blended = (strip.astype(np.float32) * (1.0 - strength) + colour * strength)
    np.copyto(strip, np.clip(blended, 0.0, 255.0).astype(frame.dtype),
              where=mask[:, :, None])


def _mark_clipped_edges(frame, echo: Echo, decay: float,
                        config: OverlayConfig) -> None:
    """Hatch the frame edge a clipped band runs off, so it is not read as a wall.

    Different from the off-frame wedge: the SOURCE is in shot here, only its
    uncertainty reaches past the view.
    """
    height, width = frame.shape[:2]
    colour = np.asarray(colour_for(echo.event_type), dtype=np.float32)
    strength = float(decay * config.centre_line_alpha)
    stripe = 6
    rows = np.arange(height)
    hatched = rows[(rows // stripe) % 2 == 0]
    def blend(block):
        return np.clip(block.astype(np.float32) * (1.0 - strength) + colour * strength,
                       0.0, 255.0).astype(frame.dtype)

    if echo.band.clipped_left:
        frame[hatched, 0:stripe] = blend(frame[hatched, 0:stripe])
    if echo.band.clipped_right:
        frame[hatched, width - stripe:width] = blend(frame[hatched, width - stripe:width])


@dataclass
class Hud:
    """Everything that must be readable on screen, as text."""

    banner: str | None = None
    status: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    decline_reason: str = ""

    def all_text(self) -> str:
        parts = [self.banner or ""] + self.status + self.warnings + [self.decline_reason]
        return "\n".join(p for p in parts if p)


def banner_text(source_kind: str) -> str | None:
    """The unmissable warning, or None when the audio is genuinely live.

    Phase 4a is parked, so this app has never been shown to work on real audio.
    Anyone glancing at the screen must be able to tell a simulation from a
    measurement without reading small print.
    """
    return None if source_kind == "hardware" else SYNTHETIC_BANNER


def build_hud(
    *,
    source_kind: str,
    ring: BearingRing,
    at_time: float,
    camera: CameraConfig,
    link_diagnostics: dict | None = None,
    parallax_note: str | None = None,
    video_fps: float = 0.0,
    overlay_ms: float = 0.0,
    av_latency_ms: float | None = None,
    av_offset_ms: float = 0.0,
    video_frames_dropped: int = 0,
) -> Hud:
    """Assemble the on-screen text. Pure data, so tests can read it."""
    active = ring.active(at_time)
    located = [(e, w) for e, w in active if e.is_located]
    newest = max(located, key=lambda pair: pair[0].display_time, default=None)

    status: list[str] = []
    if newest is not None:
        echo = newest[0]
        status.append(f"{echo.event_type}  bearing {echo.bearing_degrees:+.1f} deg  "
                      f"confidence {echo.confidence:.2f}")
        if echo.band.clipped_left or echo.band.clipped_right:
            status.append("uncertainty band extends beyond the frame edge")
    else:
        off = [e for e, _ in active if e.is_off_frame]
        if off:
            status.append(f"OFF-FRAME to the {off[-1].off_frame_side} - "
                          f"outside the camera's view, not located in shot")
        else:
            status.append("no located sound")

    visible_lo = camera.azimuth_offset_degrees - camera.half_fov_degrees
    visible_hi = camera.azimuth_offset_degrees + camera.half_fov_degrees
    status.append(f"camera shows {visible_lo:+.0f} to {visible_hi:+.0f} deg of the "
                  f"array's -90 to +90 world")
    status.append(BEHIND_NOTICE)

    latency = "n/a" if av_latency_ms is None else f"{av_latency_ms:.0f} ms"
    status.append(f"video {video_fps:.1f} fps  overlay {overlay_ms:.2f} ms/frame  "
                  f"a/v drift {latency}  offset {av_offset_ms:+.0f} ms")

    diagnostics = link_diagnostics or {}
    if diagnostics:
        status.append(
            f"link: {diagnostics.get('packets_dropped_total', 0)} packets dropped "
            f"(header CRC {diagnostics.get('packets_dropped_header_crc', 0)}, "
            f"payload CRC {diagnostics.get('packets_dropped_payload_crc', 0)}), "
            f"{diagnostics.get('frames_abandoned', 0)} frames abandoned")
    if video_frames_dropped:
        status.append(f"video frames dropped: {video_frames_dropped}")

    warnings: list[str] = []
    budget_ms = 1000.0 / ring.config.target_fps
    if overlay_ms > budget_ms * ring.config.overlay_budget_fraction:
        # Say it rather than quietly running slow. Dropping video frames without
        # announcing it would make the demo look smooth while being wrong.
        warnings.append(
            f"OVERLAY COST {overlay_ms:.1f} ms/frame is over "
            f"{100 * ring.config.overlay_budget_fraction:.0f}% of the "
            f"{budget_ms:.1f} ms budget at {ring.config.target_fps:.0f} fps - "
            f"video may not keep up")
    if parallax_note:
        warnings.append(f"PARALLAX: {parallax_note}")
    if diagnostics.get("packets_dropped_total", 0) or diagnostics.get("frames_abandoned", 0):
        warnings.append("LINK DEGRADING: packets are being dropped")

    return Hud(
        banner=banner_text(source_kind),
        status=status,
        warnings=warnings,
        decline_reason=ring.latest_decline(at_time),
    )


def draw_banner(frame: np.ndarray, present: bool,
                colour: tuple[int, int, int] = (0, 0, 220),
                height_px: int = 44) -> np.ndarray:
    """Paint the warning bar. Text is added by the caller, which has cv2.

    A solid bar rather than a caption: it must be impossible to miss at a
    glance, because mistaking a simulation for a measurement is the failure
    this whole app is most likely to cause.
    """
    if not present:
        return frame
    bar = min(height_px, frame.shape[0])
    frame[0:bar] = (frame[0:bar].astype(np.float32) * 0.15
                    + np.asarray(colour, dtype=np.float32) * 0.85).astype(frame.dtype)
    return frame
