"""Per-stage latency benchmark for the audio pipeline.

`AudioModule.performance_report()` already times DOA, detection and seat
mapping, but only for the stages it happens to run. This tool times the whole
chain frame by frame - capture, GCC-PHAT, DOA, detection, seat mapping - and
reports mean/p95/max for each against the real-time budget of one frame.

    python tools/benchmark_audio.py                     # synthetic, 200 frames
    python tools/benchmark_audio.py --frames 500
    python tools/benchmark_audio.py --json benchmarks/audio.json
    python tools/benchmark_audio.py --source esp32      # real hardware, later

What the synthetic source can and cannot tell you: every compute stage below
is the same code that will run on real microphones, so those numbers are real.
"capture" is not. With a synthetic source it measures framing and queue
handover only; the microphone -> I2S -> USB -> laptop path does not exist yet
and cannot be estimated from here. The report says so rather than printing a
number that looks like acquisition latency and is not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heimdall.audio.api import StageTimings  # noqa: E402
from heimdall.audio.config import load_audio_config  # noqa: E402
from heimdall.audio.doa import estimate_doa  # noqa: E402
from heimdall.audio.events import AudioEventDetector  # noqa: E402
from heimdall.audio.gcc_phat import gcc_phat_frame  # noqa: E402
from heimdall.audio.geometry import load_classroom_config  # noqa: E402
from heimdall.audio.receiver import AudioReceiver  # noqa: E402
from heimdall.audio.seat_mapper import DEFAULT_MIN_CONFIDENCE, map_audio_to_seat  # noqa: E402
from heimdall.audio.sources import SyntheticAudioSource  # noqa: E402

# Pipeline order, so the printed table reads like the data flow.
STAGES = ["capture", "gcc_phat", "doa", "detect", "seat_mapping", "frame_total"]

STAGE_NOTES = {
    "capture": "handover from the receiver; NOT acquisition latency on synthetic audio",
    "gcc_phat": "cross-correlation of one channel pair; a breakdown of doa, not extra work",
    "doa": "gcc_phat plus the geometry that turns a TDOA into a bearing",
    "detect": "energy and spectral classification of one frame",
    "seat_mapping": "per completed event, not per frame",
    "frame_total": "capture + doa + detect, i.e. what api.process_frame actually runs",
}


@dataclass
class BenchmarkResult:
    source: str
    sample_rate: int
    frame_size: int
    num_channels: int
    frames_measured: int
    frames_dropped: int
    events_emitted: int
    warmup_frames: int
    frame_duration_ms: float
    stages: dict = field(default_factory=dict)

    @property
    def per_frame_cost_ms(self) -> float:
        """Mean cost of one frame: capture + doa + detect."""
        total = self.stages.get("frame_total")
        return float(total["mean_ms"]) if total else 0.0

    @property
    def realtime_factor(self) -> float:
        """How many times faster than real time. 1.0 means only just keeping up."""
        cost = self.per_frame_cost_ms
        if cost <= 0.0:
            return float("inf")
        return self.frame_duration_ms / cost


def build_source(args, audio_config, classroom):
    """The one place that knows which kind of hardware we are talking to."""
    if args.source == "synthetic":
        return SyntheticAudioSource(
            sample_rate=audio_config.sample_rate,
            num_channels=max(audio_config.num_channels, classroom.array.num_channels),
            frame_size=audio_config.frame_size,
            angle_degrees=args.angle,
            mic_spacing_m=classroom.array.spacing,
            noise_amplitude=args.noise,
            burst_frames=args.burst_frames,
            silence_frames=args.silence_frames,
        )

    if args.source == "esp32":
        from heimdall.audio.sources import ESP32AudioSource

        return ESP32AudioSource(config=audio_config)

    raise SystemExit("unknown source %r" % args.source)


def benchmark(
    source,
    classroom,
    num_frames: int,
    *,
    warmup_frames: int = 5,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    source_name: str = "synthetic",
) -> BenchmarkResult:
    """Time every stage over `num_frames` frames.

    The first `warmup_frames` are run and thrown away. The first FFT of a given
    length pays for plan construction, and that one-off cost would otherwise
    land in the maximum and lift the mean by an amount that has nothing to do
    with steady-state behaviour.
    """
    timings = StageTimings()
    detector = AudioEventDetector()
    array = classroom.array

    measured = 0
    events = 0
    seen = 0

    with AudioReceiver(source) as receiver:
        while seen < num_frames + warmup_frames:
            capture_started = time.perf_counter()
            frame = receiver.read_frame(timeout=2.0)
            capture_elapsed = time.perf_counter() - capture_started
            if frame is None:
                break

            seen += 1

            started = time.perf_counter()
            gcc_phat_frame(frame, mic_spacing_m=array.spacing)
            gcc_elapsed = time.perf_counter() - started

            started = time.perf_counter()
            localization = estimate_doa(frame, array)
            doa_elapsed = time.perf_counter() - started

            started = time.perf_counter()
            completed = detector.process(frame)
            detect_elapsed = time.perf_counter() - started

            # The real pipeline (api.process_frame) runs doa and detect, not a
            # separate gcc_phat pass, so the timing call above must not be
            # charged to the frame. Sum the stages instead of wall time.
            frame_elapsed = capture_elapsed + doa_elapsed + detect_elapsed

            seat_elapsed = None
            if completed:
                events += len(completed)
                started = time.perf_counter()
                for _ in completed:
                    map_audio_to_seat(localization, classroom, min_confidence=min_confidence)
                seat_elapsed = time.perf_counter() - started

            if seen <= warmup_frames:
                continue

            measured += 1
            timings.record("capture", capture_elapsed)
            timings.record("gcc_phat", gcc_elapsed)
            timings.record("doa", doa_elapsed)
            timings.record("detect", detect_elapsed)
            timings.record("frame_total", frame_elapsed)
            if seat_elapsed is not None:
                timings.record("seat_mapping", seat_elapsed)

        dropped = receiver.stats.frames_dropped

    return BenchmarkResult(
        source=source_name,
        sample_rate=source.sample_rate,
        frame_size=source.frame_size,
        num_channels=source.num_channels,
        frames_measured=measured,
        frames_dropped=dropped,
        events_emitted=events,
        warmup_frames=warmup_frames,
        frame_duration_ms=1000.0 * source.frame_size / source.sample_rate,
        stages=timings.report(),
    )


def print_report(result: BenchmarkResult) -> None:
    synthetic = result.source == "synthetic"

    print("Heimdall audio - pipeline latency benchmark")
    print("Source:          %s" % result.source)
    print("Sample rate:     %d Hz" % result.sample_rate)
    print(
        "Frame:           %d samples x %d channels = %.2f ms of audio"
        % (result.frame_size, result.num_channels, result.frame_duration_ms)
    )
    print(
        "Frames measured: %d (%d warm-up frames discarded)"
        % (result.frames_measured, result.warmup_frames)
    )
    print()

    if not result.frames_measured:
        print("No frames captured. Nothing to measure.")
        return

    if synthetic:
        print("SOURCE: SYNTHETIC. The compute stages are real; 'capture' is not.")
        print("Acquisition latency needs hardware and is not estimated here.")
        print()

    print("%-14s %-10s %-10s %-10s %-8s" % ("STAGE", "MEAN ms", "P95 ms", "MAX ms", "N"))
    print("-" * 60)
    for stage in STAGES:
        entry = result.stages.get(stage)
        if entry is None:
            print("%-14s %-10s %-10s %-10s %-8d  (never ran)" % (stage, "-", "-", "-", 0))
            continue
        print(
            "%-14s %-10.3f %-10.3f %-10.3f %-8d"
            % (stage, entry["mean_ms"], entry["p95_ms"], entry["max_ms"], entry["count"])
        )
        print("               %s" % STAGE_NOTES[stage])

    print()
    print("Events emitted:  %d" % result.events_emitted)
    print("Frames dropped:  %d" % result.frames_dropped)
    print()

    cost = result.per_frame_cost_ms
    factor = result.realtime_factor
    print("One frame costs %.2f ms of a %.2f ms budget." % (cost, result.frame_duration_ms))

    if factor >= 5.0:
        print("VERDICT: %.1fx faster than real time. Comfortable headroom." % factor)
    elif factor >= 1.5:
        print("VERDICT: %.1fx faster than real time. Keeps up, but with little" % factor)
        print("         margin for a busier laptop or a fourth microphone.")
    elif factor > 1.0:
        print("VERDICT: %.2fx real time. Marginal - a slow frame will drop audio." % factor)
    else:
        print("VERDICT: SLOWER THAN REAL TIME (%.2fx). The pipeline cannot keep up" % factor)
        print("         and frames will be dropped. Raise the frame size, or profile")
        print("         GCC-PHAT: it is almost always the expensive stage.")

    if result.frames_dropped:
        print()
        if synthetic:
            print("NOTE: %d frames were dropped, which means nothing here. The synthetic"
                  % result.frames_dropped)
            print("      source is unpaced and generates faster than real time, so the")
            print("      queue overflows by construction. On hardware this number is the")
            print("      one that matters: there it means the consumer fell behind.")
        else:
            print("NOTE: the receiver dropped %d frames. That is the consumer falling"
                  % result.frames_dropped)
            print("      behind, and it matters more than any mean below the budget.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "esp32"])
    parser.add_argument("--frames", type=int, default=200, help="frames measured")
    parser.add_argument("--warmup", type=int, default=5,
                        help="frames run and discarded before measuring")
    parser.add_argument("--angle", type=float, default=25.0)
    parser.add_argument("--noise", type=float, default=0.002,
                        help="synthetic background noise amplitude")
    parser.add_argument("--burst-frames", type=int, default=12)
    parser.add_argument("--silence-frames", type=int, default=6)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the raw numbers to this file")
    args = parser.parse_args()

    if args.frames < 1:
        raise SystemExit("--frames must be at least 1")
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative")

    audio_config = load_audio_config()
    classroom = load_classroom_config()

    try:
        source = build_source(args, audio_config, classroom)
    except NotImplementedError as exc:
        print("Cannot benchmark real hardware yet:")
        print("  %s" % exc)
        return 2

    result = benchmark(
        source,
        classroom,
        args.frames,
        warmup_frames=args.warmup,
        min_confidence=args.min_confidence,
        source_name=args.source,
    )
    print_report(result)

    if not result.frames_measured:
        return 1

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        print("\nRaw numbers written to %s" % args.json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
