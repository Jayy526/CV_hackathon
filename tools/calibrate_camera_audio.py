"""Phase 5: estimate azimuth_offset_degrees by measuring it, not guessing it.

You make a sound at a known angle; this records where the array THINKS it came
from, where that projects to in the picture, and where you say it actually
appears. From several such points it estimates the angle between the camera's
optical axis and the array's 0 degree broadside.

    python tools/calibrate_camera_audio.py
    python tools/calibrate_camera_audio.py --angles -30 -15 0 15 30

NEEDS HARDWARE. With no board or no camera it refuses clearly and exits
non-zero; it does not fall back to synthetic audio, because a calibration
measured against a simulation would be worse than none at all.

It reports "unusable" when the data does not support an estimate. A wrong
azimuth offset skews every overlay in the demo, so a bad calibration silently
accepted is worse than an admitted failure.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from acoustic_array.analysis import channel_rms  # noqa: E402
from acoustic_array.config import load_audio_config  # noqa: E402
from acoustic_array.doa import estimate_doa  # noqa: E402
from acoustic_array.geometry import default_array  # noqa: E402
from acoustic_array.receiver import AudioReceiver  # noqa: E402
from acoustic_array.sources import AudioSourceError  # noqa: E402
from acoustic_camera import (  # noqa: E402
    load_camera_config,
    parallax_warning,
    project_bearing,
)

DEFAULT_ANGLES = [-30.0, -15.0, 0.0, 15.0, 30.0]
RMS_GATE = 0.01
MIN_CONFIDENCE = 0.30

# An estimate is only usable if the individual per-point offsets agree. Wide
# disagreement means the points are measuring something other than a fixed
# mounting angle, and averaging them would manufacture a confident wrong number.
MAX_OFFSET_SPREAD_DEGREES = 8.0
MIN_USABLE_POINTS = 3


@dataclass
class CalibrationPoint:
    known_angle_degrees: float
    measured_bearing_degrees: float | None
    projected_column: float | None
    observed_column: float | None
    localization_confidence: float
    frames_used: int
    frames_rejected: int
    note: str = ""

    @property
    def implied_offset_degrees(self) -> float | None:
        """How far the camera axis sits from the array's zero, from this point.

        The array reported `measured`; the truth is `known`; the difference is
        the mounting angle, provided the sound really was where you said.
        """
        if self.measured_bearing_degrees is None:
            return None
        return self.measured_bearing_degrees - self.known_angle_degrees

    @property
    def column_error_px(self) -> float | None:
        if self.projected_column is None or self.observed_column is None:
            return None
        return self.projected_column - self.observed_column


def measure_bearing(source, array, num_frames, timeout=3.0):
    """Capture frames, keep only loud confident ones, return (bearing, conf, used, rejected)."""
    bearings: list[float] = []
    weights: list[float] = []
    rejected = 0

    with AudioReceiver(source) as receiver:
        for _ in range(num_frames):
            frame = receiver.read_frame(timeout=timeout)
            if frame is None:
                break
            if float(np.max(channel_rms(frame))) < RMS_GATE:
                rejected += 1
                continue
            doa = estimate_doa(frame, array)
            if not doa.valid or doa.angle_degrees is None or doa.confidence < MIN_CONFIDENCE:
                rejected += 1
                continue
            bearings.append(float(doa.angle_degrees))
            weights.append(float(doa.confidence))

    if not bearings:
        return None, 0.0, 0, rejected
    total = sum(weights)
    bearing = sum(b * w for b, w in zip(bearings, weights)) / total
    return bearing, total / len(weights), len(bearings), rejected


def estimate_offset(points: list[CalibrationPoint]) -> tuple[float | None, str, str]:
    """Return (offset_degrees, verdict, explanation). Never a silent average."""
    offsets = [p.implied_offset_degrees for p in points
               if p.implied_offset_degrees is not None]

    if len(offsets) < MIN_USABLE_POINTS:
        return None, "UNUSABLE", (
            f"only {len(offsets)} of {len(points)} points produced a bearing; "
            f"at least {MIN_USABLE_POINTS} are needed. Nothing is estimated from "
            f"this - make the sound louder, or check the array first with "
            f"tools/verify_localization.py."
        )

    spread = statistics.pstdev(offsets) if len(offsets) > 1 else 0.0
    median = statistics.median(offsets)

    # MIRRORING IS CHECKED FIRST. It also produces a huge spread, so testing
    # spread first would report the symptom ("the points disagree") and bury the
    # cause ("the channels are reversed"). Name the cause.
    known = [p.known_angle_degrees for p in points if p.implied_offset_degrees is not None]
    measured = [p.measured_bearing_degrees for p in points
                if p.implied_offset_degrees is not None]
    if len(set(known)) > 1:
        slope = float(np.polyfit(np.asarray(known), np.asarray(measured), 1)[0])
        if slope < 0:
            return None, "MIRRORED", (
                f"measured bearing moves OPPOSITE to the known angle "
                f"(slope {slope:+.2f}). This is not a mounting offset and "
                f"azimuth_offset_degrees cannot fix it: the array's channel "
                f"order is reversed relative to the camera's view. Check which "
                f"microphone is channel 0 and which side of the camera it is on."
            )

    if spread > MAX_OFFSET_SPREAD_DEGREES:
        return None, "UNUSABLE", (
            f"the per-point offsets disagree by {spread:.1f} deg "
            f"(limit {MAX_OFFSET_SPREAD_DEGREES:.0f}): {_fmt(offsets)}.\n"
            f"  A mounting angle is one fixed number; if each point implies a "
            f"different one, the points are not measuring it. Likely causes: "
            f"the known angles were eyeballed rather than measured, the array "
            f"itself is not localizing (run tools/verify_localization.py), or "
            f"the camera is far enough off the array centre that parallax is "
            f"varying with the source distance."
        )

    return median, "OK", (
        f"azimuth_offset_degrees = {median:+.2f} (spread {spread:.2f} deg over "
        f"{len(offsets)} points). Put this in config/camera.yaml."
    )


def _fmt(values) -> str:
    return ", ".join(f"{v:+.1f}" for v in values)


def report(points, offset, verdict, explanation, camera) -> bool:
    print()
    print("--- calibration points ---")
    header = (f"{'known':>8}{'measured':>10}{'implied off':>13}{'projected px':>14}"
              f"{'observed px':>13}{'err px':>9}{'conf':>7}{'frames':>8}")
    print(header)
    print("-" * len(header))
    for point in points:
        def fmt(value, spec=".1f"):
            return "  --" if value is None else format(value, spec)
        print(f"{point.known_angle_degrees:>8.1f}"
              f"{fmt(point.measured_bearing_degrees):>10}"
              f"{fmt(point.implied_offset_degrees):>13}"
              f"{fmt(point.projected_column, '.0f'):>14}"
              f"{fmt(point.observed_column, '.0f'):>13}"
              f"{fmt(point.column_error_px, '.0f'):>9}"
              f"{point.localization_confidence:>7.2f}"
              f"{point.frames_used:>4}/{point.frames_used + point.frames_rejected:<4}")
    print()

    warning = parallax_warning(camera)
    if warning:
        print("--- mounting ---")
        print(f"  {warning}")
        print()

    print("--- verdict ---")
    print(f"  [{verdict}]  {explanation}")
    if verdict != "OK":
        print()
        print("  Nothing has been written. config/camera.yaml is unchanged.")
    print()
    return verdict == "OK"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate the camera/array azimuth offset.")
    parser.add_argument("--port", default=None, help="ESP32 COM port")
    parser.add_argument("--angles", type=float, nargs="*", default=None,
                        help=f"known angles in degrees (default {DEFAULT_ANGLES})")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--no-camera", action="store_true",
                        help="skip the observed-column prompts and estimate from audio only")
    parser.add_argument("--json", default=None)
    return parser


def open_camera(camera):
    """Return an opened cv2.VideoCapture, or raise RuntimeError with a reason."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is not installed, so the camera cannot be opened. "
            "Install it, or pass --no-camera to calibrate from audio alone."
        ) from exc
    capture = cv2.VideoCapture(camera.index)
    if not capture.isOpened():
        raise RuntimeError(
            f"camera index {camera.index} did not open. Is another application "
            f"using it, or is camera.index wrong in config/camera.yaml?")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
    return capture


def main(argv=None, prompt=input) -> int:
    args = build_parser().parse_args(argv)
    audio_config = load_audio_config()
    camera = load_camera_config()
    array = default_array()
    angles = DEFAULT_ANGLES if args.angles is None else args.angles

    print("Camera/array calibration. This needs REAL hardware.")
    print(f"  camera index {camera.index}, {camera.width}x{camera.height}, "
          f"HFOV {camera.horizontal_fov_degrees:.1f} deg")
    print(f"  array {array.num_channels} mics, {array.spacing * 100:.1f} cm apart")
    warning = parallax_warning(camera)
    if warning:
        print(f"  WARNING: {warning}")

    capture = None
    if not args.no_camera:
        try:
            capture = open_camera(camera)
        except RuntimeError as exc:
            print(f"\nCannot open the camera: {exc}", file=sys.stderr)
            return 2

    from acoustic_array.sources import ESP32AudioSource

    points: list[CalibrationPoint] = []
    try:
        for index, angle in enumerate(angles, start=1):
            prompt(f"\n[{index}/{len(angles)}] Stand 1 m from the array centre at "
                   f"{angle:+.0f} deg and clap repeatedly.\n  Press Enter to record...")
            source = ESP32AudioSource(port=args.port, config=audio_config)
            bearing, confidence, used, rejected = measure_bearing(
                source, array, args.frames)

            projected = None
            if bearing is not None:
                result = project_bearing(bearing, camera)
                projected = result.column
                if not result.on_screen:
                    print(f"  note: {result.reason}")

            observed = None
            if capture is not None and bearing is not None:
                raw = prompt("  Pixel column where the sound actually appeared "
                             "(blank to skip): ").strip()
                if raw:
                    try:
                        observed = float(raw)
                    except ValueError:
                        print("  not a number; skipped")

            points.append(CalibrationPoint(
                known_angle_degrees=float(angle),
                measured_bearing_degrees=bearing,
                projected_column=projected,
                observed_column=observed,
                localization_confidence=confidence,
                frames_used=used,
                frames_rejected=rejected,
            ))
            print(f"  -> measured {'n/a' if bearing is None else f'{bearing:+.1f} deg'}, "
                  f"{used} usable frames")
    except AudioSourceError as exc:
        print(f"\nCannot reach the microphone array: {exc}", file=sys.stderr)
        print("This tool needs the ESP32; there is no synthetic fallback, "
              "because a calibration against a simulation is worthless.",
              file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted - nothing measured.", file=sys.stderr)
        return 2
    finally:
        if capture is not None:
            capture.release()

    offset, verdict, explanation = estimate_offset(points)
    ok = report(points, offset, verdict, explanation, camera)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"points": [asdict(p) for p in points],
             "azimuth_offset_degrees": offset, "verdict": verdict},
            indent=2), encoding="utf-8")
        print(f"raw results written to {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
