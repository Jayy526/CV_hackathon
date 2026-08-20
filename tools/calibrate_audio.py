"""Real-world calibration: does this array actually measure direction?

Place a speaker at a known angle, record, compare the estimate to the truth,
and print the error. Run it at several angles and you know whether the hardware
is useful or not.

Right now the "speaker" is synthetic, which validates the maths end to end but
proves nothing about the microphones. The measurement logic is identical either
way: only `source_for_angle` changes when the ESP32 arrives.

    python tools/calibrate_audio.py                          # synthetic sweep
    python tools/calibrate_audio.py --angles 0 30 60 90
    python tools/calibrate_audio.py --noise 0.05             # simulate a noisy room
    python tools/calibrate_audio.py --source esp32           # real hardware, later

Results are never massaged. If the array cannot resolve an angle, the table
says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from heimdall.audio.config import AudioConfig, load_audio_config  # noqa: E402
from heimdall.audio.doa import estimate_doa  # noqa: E402
from heimdall.audio.geometry import ClassroomConfig, load_classroom_config  # noqa: E402
from heimdall.audio.receiver import AudioReceiver  # noqa: E402
from heimdall.audio.sources import AudioSource, SyntheticAudioSource  # noqa: E402

DEFAULT_ANGLES = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]


@dataclass
class CalibrationPoint:
    known_angle_degrees: float
    estimated_angle_degrees: float | None
    error_degrees: float | None
    confidence: float
    expected_resolution_degrees: float | None
    frames_used: int
    frames_rejected: int
    note: str = ""


# --- source factories: the only hardware-aware part of this file -------------

def synthetic_source_factory(audio_config: AudioConfig, classroom: ClassroomConfig, noise: float):
    def make(angle_degrees: float) -> AudioSource:
        return SyntheticAudioSource(
            sample_rate=audio_config.sample_rate,
            num_channels=audio_config.num_channels,
            frame_size=audio_config.frame_size,
            angle_degrees=angle_degrees,
            mic_spacing_m=classroom.array.spacing,
            noise_amplitude=noise,
            burst_frames=1,
            silence_frames=0,
        )

    return make


def esp32_source_factory(audio_config: AudioConfig, classroom: ClassroomConfig, noise: float):
    def make(angle_degrees: float) -> AudioSource:
        from heimdall.audio.sources import ESP32AudioSource

        input(
            "\nPlace the speaker at %.0f degrees from the array, then press Enter..."
            % angle_degrees
        )
        return ESP32AudioSource()

    return make


# --- measurement: identical for synthetic and real audio ---------------------

def measure_angle(
    source: AudioSource,
    classroom: ClassroomConfig,
    num_frames: int,
    min_confidence: float,
) -> tuple[float | None, float, int, int]:
    """Capture `num_frames` and return (angle, confidence, used, rejected).

    Frames whose correlation is untrustworthy are counted and discarded rather
    than averaged in - averaging garbage produces a confident wrong answer.
    """
    angles: list[float] = []
    weights: list[float] = []
    rejected = 0

    with AudioReceiver(source) as receiver:
        for _ in range(num_frames):
            frame = receiver.read_frame(timeout=2.0)
            if frame is None:
                break

            result = estimate_doa(frame, classroom.array)
            if result.valid and result.angle_degrees is not None and result.confidence >= min_confidence:
                angles.append(result.angle_degrees)
                weights.append(result.confidence)
            else:
                rejected += 1

    if not angles:
        return None, 0.0, 0, rejected

    angle_array = np.asarray(angles)
    weight_array = np.asarray(weights)
    estimate = float(np.sum(angle_array * weight_array) / np.sum(weight_array))
    return estimate, float(np.mean(weight_array)), len(angles), rejected


def run_calibration(
    angles,
    source_factory,
    classroom: ClassroomConfig,
    num_frames: int,
    min_confidence: float,
    sample_rate: int,
) -> list[CalibrationPoint]:
    from heimdall.audio.doa import angular_resolution_degrees

    points: list[CalibrationPoint] = []
    for known in angles:
        source = source_factory(known)
        estimate, confidence, used, rejected = measure_angle(
            source, classroom, num_frames, min_confidence
        )

        expected = angular_resolution_degrees(
            known, classroom.array.spacing, sample_rate, classroom.speed_of_sound
        )

        notes = []
        if abs(known) > 85.0:
            notes.append("near the array axis, where a linear array is least reliable")
        if estimate is None:
            notes.append("no usable frames - array could not resolve this direction")
        note = "; ".join(notes)

        points.append(
            CalibrationPoint(
                known_angle_degrees=float(known),
                estimated_angle_degrees=estimate,
                error_degrees=None if estimate is None else float(estimate - known),
                confidence=confidence,
                expected_resolution_degrees=expected,
                frames_used=used,
                frames_rejected=rejected,
                note=note,
            )
        )

    return points


# --- reporting ---------------------------------------------------------------

def print_report(points: list[CalibrationPoint], classroom: ClassroomConfig, sample_rate: int, synthetic: bool):
    print("Heimdall audio - direction calibration")
    print("Array spacing:   %.3f m" % classroom.array.spacing)
    print("Sample rate:     %d Hz" % sample_rate)
    print("Speed of sound:  %.1f m/s" % classroom.speed_of_sound)
    print()

    if synthetic:
        print("SOURCE: SYNTHETIC. These numbers validate the algorithms only.")
        print("They say nothing about whether your microphones work.")
        print()

    print("%-10s %-12s %-10s %-8s %-12s" % ("KNOWN", "ESTIMATED", "ERROR", "CONF", "EXPECTED +-"))
    print("-" * 60)
    for point in points:
        if point.estimated_angle_degrees is None:
            print(
                "%-10.1f %-12s %-10s %-8.2f %-12.2f"
                % (point.known_angle_degrees, "FAILED", "-", point.confidence,
                   point.expected_resolution_degrees or 0.0)
            )
        else:
            print(
                "%-10.1f %-12.2f %-10.2f %-8.2f %-12.2f"
                % (
                    point.known_angle_degrees,
                    point.estimated_angle_degrees,
                    point.error_degrees,
                    point.confidence,
                    point.expected_resolution_degrees or 0.0,
                )
            )
        if point.note:
            print("           note: %s" % point.note)

    errors = [abs(p.error_degrees) for p in points if p.error_degrees is not None]
    failures = [p for p in points if p.estimated_angle_degrees is None]

    print()
    if errors:
        array = np.asarray(errors)
        print("Absolute error:  mean %.2f deg   p95 %.2f deg   max %.2f deg"
              % (float(np.mean(array)), float(np.percentile(array, 95)), float(np.max(array))))
    if failures:
        print("Failed to resolve %d of %d angles: %s"
              % (len(failures), len(points),
                 ", ".join("%.0f" % p.known_angle_degrees for p in failures)))

    print()
    if not errors:
        print("VERDICT: unusable. No angle could be measured.")
    else:
        worst = float(np.max(np.asarray(errors)))
        seat_pitch_degrees = 8.0  # rough: adjacent seats a few metres out
        if worst <= seat_pitch_degrees / 2:
            print("VERDICT: errors are smaller than half a seat spacing. Usable for seat mapping.")
        elif worst <= seat_pitch_degrees * 2:
            print("VERDICT: errors are comparable to seat spacing. Usable for coarse")
            print("         left/right evidence, NOT for naming a specific seat.")
        else:
            print("VERDICT: errors exceed seat spacing. Direction output should not be")
            print("         used to name seats. Widen the array or raise the sample rate.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "esp32"])
    parser.add_argument("--angles", type=float, nargs="+", default=DEFAULT_ANGLES)
    parser.add_argument("--frames", type=int, default=20, help="frames averaged per angle")
    parser.add_argument("--noise", type=float, default=0.005,
                        help="synthetic background noise amplitude")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=Path("calibration/results.json"))
    args = parser.parse_args()

    audio_config = load_audio_config()
    classroom = load_classroom_config()

    for angle in args.angles:
        if not -90.0 <= angle <= 90.0:
            raise SystemExit(
                "angle %.1f is outside [-90, 90]. A linear array cannot represent it."
                % angle
            )

    if args.source == "synthetic":
        factory = synthetic_source_factory(audio_config, classroom, args.noise)
    else:
        factory = esp32_source_factory(audio_config, classroom, args.noise)

    try:
        points = run_calibration(
            args.angles, factory, classroom, args.frames,
            args.min_confidence, audio_config.sample_rate,
        )
    except NotImplementedError as exc:
        print("Cannot calibrate against real hardware yet:\n  %s" % exc)
        return 2

    print_report(points, classroom, audio_config.sample_rate, args.source == "synthetic")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": args.source,
        "sample_rate": audio_config.sample_rate,
        "mic_spacing_m": classroom.array.spacing,
        "speed_of_sound": classroom.speed_of_sound,
        "points": [asdict(p) for p in points],
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nResults written to %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
