"""Phase 4a / CONTEXT.md section 15: does the PHYSICAL array localize at all?

Everything in section 8 is synthetic. GCC-PHAT has never seen real air, real
reverberation or real microphone mismatch. This tool measures the basics on
loud, unambiguous sound before any whisper work is attempted.

THE SIGN CONVENTION IS THE POINT OF THIS TEST. Section 5 records that the sign
was verified BY INSPECTION, not by measurement. If real audio next to mic 1
produces a negative bearing, every bearing in the system is mirrored and the
heatmap will be confidently backwards. That result is reported on its own, at
the top, not buried in a table.

Section 5 convention, which this tool checks rather than assumes:

      0 deg  broadside, straight out in front of the array
    +90 deg  along the array axis, toward CHANNEL 0  (mic 1)
    -90 deg  along the array axis, toward the last channel (mic 2)

    python tools/verify_localization.py                        # synthetic self-test
    python tools/verify_localization.py --source esp32

Nothing here is massaged. Every station reports its spread, not its best trial.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from acoustic_array.analysis import find_onsets, noise_floor_rms  # noqa: E402
from acoustic_array.config import AudioConfig, load_audio_config  # noqa: E402
from acoustic_array.doa import estimate_doa  # noqa: E402
from acoustic_array.frame import AudioFrame  # noqa: E402
from acoustic_array.gcc_phat import max_delay_samples  # noqa: E402
from acoustic_array.receiver import AudioReceiver  # noqa: E402
from heimdall.audio.geometry import (  # noqa: E402  - parked layer, array only
    ClassroomConfig,
    load_classroom_config,
)
from acoustic_array.sources import (  # noqa: E402
    AudioSource,
    AudioSourceError,
    SyntheticAudioSource,
)

# Only frames carrying an actual transient are measured. A clap is loud; the
# room between claps is not, and averaging silence into the estimate would
# produce a confident number describing nothing.
# A clap's direct sound is ~2 ms. 256 samples at 16 kHz is 16 ms: long enough
# to correlate, short enough that the first wall reflection has not taken over.
ONSET_WINDOW = 256
# Kept so an old --rms-gate argument still parses. Level gating is exactly what
# made this tool measure the room instead of the claps; it is no longer used.
DEFAULT_RMS_GATE = 0.01
# 0.30 == seat_mapper.DEFAULT_MIN_CONFIDENCE, duplicated rather than imported:
# seat_mapper is parked and belongs to the classroom layer. The number is
# section 7's garbage floor (uncorrelated noise scores 0.12, heavy noise 0.29);
# a test pins the two together so they cannot drift apart.
DEFAULT_MIN_CONFIDENCE = 0.30
DEFAULT_ANGLES = [-60.0, -30.0, 0.0, 30.0, 60.0]

PASS, FAIL, NA = "PASS", "FAIL", "N/A"

# How close an end-fire station must come to +/-90 to count as end-fire. Loose
# on purpose: a hand held "very close to mic 1" is not exactly on the axis, and
# the estimator clamps at +/-90 anyway.
ENDFIRE_TOLERANCE = 35.0
BROADSIDE_TOLERANCE = 15.0
# Mean absolute error across the known-angle sweep. The array's own resolution
# is 4.55 deg at broadside at 16 kHz, so anything under 10 deg in a real room
# is working; beyond that something is wrong, not merely noisy.
ERROR_BUDGET_DEGREES = 10.0

# COHERENCE PRECONDITION. No categorical verdict may be drawn from a station
# that did not actually measure anything.
#
# This exists because the tool once reported MIRRORED from a station whose
# median bearing was -2.3 deg with 52 deg of spread over 82 frames - 0.4
# standard errors from zero. Mirroring flips the SIGN; it does not collapse the
# MAGNITUDE. A mirrored array clapped 5 cm from mic 1 reads -90 deg, not -2.3.
# Reading a sign off noise centred on zero is not a measurement, and acting on
# it would have meant rewiring a correctly-wired array.
#
# A station is coherent only if BOTH hold:
MAX_COHERENT_SPREAD_DEGREES = 25.0   # frame-to-frame sd; above this it is noise
# and, for an end-fire station, the lag must actually approach the geometric
# maximum, since that is what "on the array axis" physically means.
MIN_ENDFIRE_LAG_FRACTION = 0.40


@dataclass(frozen=True)
class Station:
    """One place the operator makes a sound, and what should come back."""

    key: str
    label: str
    instruction: str
    expected_degrees: float | None
    # Only the three geometric stations test the sign; the angle sweep tests
    # accuracy. Kept separate so a wrong sign cannot hide in an error average.
    role: str = "angle"           # "endfire_ch0" | "broadside" | "endfire_ch1" | "angle"


@dataclass(frozen=True)
class Observation:
    lag_samples: float
    tdoa_us: float
    bearing_degrees: float
    confidence: float
    rms_ch0: float
    rms_ch1: float


@dataclass
class StationResult:
    station: Station
    observations: list[Observation] = field(default_factory=list)
    frames_captured: int = 0
    frames_below_gate: int = 0
    frames_low_confidence: int = 0
    onsets_found: int = 0
    noise_floor: float = 0.0
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.observations)

    def _spread(self, values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        return mean, sd

    @property
    def bearing(self) -> tuple[float | None, float | None]:
        return self._spread([o.bearing_degrees for o in self.observations])

    @property
    def lag(self) -> tuple[float | None, float | None]:
        return self._spread([o.lag_samples for o in self.observations])

    @property
    def tdoa(self) -> tuple[float | None, float | None]:
        return self._spread([o.tdoa_us for o in self.observations])

    @property
    def confidence(self) -> tuple[float | None, float | None]:
        return self._spread([o.confidence for o in self.observations])

    @property
    def median_bearing(self) -> float | None:
        if not self.observations:
            return None
        return statistics.median(o.bearing_degrees for o in self.observations)

    @property
    def mean_abs_error(self) -> float | None:
        """Mean |measured - expected|, or None when there is no truth to compare."""
        if not self.observations or self.station.expected_degrees is None:
            return None
        return statistics.fmean(
            abs(o.bearing_degrees - self.station.expected_degrees)
            for o in self.observations
        )


def default_stations(angles: list[float] | None = None) -> list[Station]:
    """The four section 15 tests: three geometric, then the angle sweep."""
    angles = DEFAULT_ANGLES if angles is None else angles
    stations = [
        Station(
            key="near_mic1",
            label="very close to MIC 1 (channel 0)",
            instruction=(
                "Hold your hands ~5 cm from MIC 1, on the array axis, and clap "
                "3-4 times.\n  Section 5 says this must read close to +90 deg. "
                "A NEGATIVE reading means\n  every bearing in the system is "
                "mirrored."
            ),
            expected_degrees=90.0,
            role="endfire_ch0",
        ),
        Station(
            key="midpoint",
            label="midway between the microphones",
            instruction=(
                "Clap 3-4 times directly in front of the MIDPOINT of the two "
                "mics,\n  about 50 cm out, square to the array. Expect ~0 deg."
            ),
            expected_degrees=0.0,
            role="broadside",
        ),
        Station(
            key="near_mic2",
            label="very close to MIC 2 (channel 1)",
            instruction=(
                "Now ~5 cm from MIC 2, on the array axis, clap 3-4 times.\n"
                "  This must read close to -90 deg: the mirror of station 1."
            ),
            expected_degrees=-90.0,
            role="endfire_ch1",
        ),
    ]
    for angle in angles:
        side = "toward MIC 1" if angle > 0 else ("toward MIC 2" if angle < 0 else "straight ahead")
        stations.append(Station(
            key=f"angle_{angle:+.0f}".replace("+", "p").replace("-", "m"),
            label=f"{angle:+.0f} deg, 1 m out",
            instruction=(
                f"Stand 1 m from the array centre at {angle:+.0f} deg ({side}) "
                f"and clap 3-4 times."
            ),
            expected_degrees=float(angle),
        ))
    return stations


# --- source factories: the only hardware-aware part of this file -------------

def synthetic_source_factory(config: AudioConfig, classroom: ClassroomConfig,
                             noise: float = 0.005):
    """Self-test source. Proves the MEASUREMENT is right, not the microphones.

    Worth having: if this tool reported a mirrored sign on synthetic audio of a
    known angle, the fault would be in the tool. Passing here means a
    disagreement on hardware is the hardware or the wiring, not the arithmetic.
    """
    def make(station: Station) -> AudioSource:
        angle = station.expected_degrees or 0.0
        return SyntheticAudioSource(
            sample_rate=config.transport.transmit_sample_rate,
            num_channels=config.num_channels,
            frame_size=config.frame_size,
            angle_degrees=angle,
            mic_spacing_m=classroom.array.spacing,
            noise_amplitude=noise,
            # Transients with quiet between them, so the onset detector has
            # something to detect. A continuous burst has no attack to find and
            # would silently skip the very path this tool now depends on.
            burst_frames=1,
            silence_frames=6,
        )
    return make


def esp32_source_factory(config: AudioConfig, port: str | None):
    def make(station: Station) -> AudioSource:
        from acoustic_array.sources import ESP32AudioSource

        return ESP32AudioSource(port=port, config=config)
    return make


# --- measurement -------------------------------------------------------------

def measure_station(
    source: AudioSource,
    station: Station,
    classroom: ClassroomConfig,
    *,
    num_frames: int = 200,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    timeout: float = 3.0,
    onset_window: int = ONSET_WINDOW,
    frame_size: int = 1024,
    rms_gate: float | None = None,   # accepted and ignored; see the docstring
) -> StationResult:
    """Buffer the recording, find the CLAPS in it, and measure only those.

    THIS IS ONSET-SELECTED, NOT LEVEL-GATED, AND THAT IS THE WHOLE POINT.

    The previous version kept every frame whose RMS cleared an absolute 0.01.
    Over 12.8 s that admits the entire recording whenever the room tone sits
    above the threshold - and a clap's direct sound is only ~2 ms, roughly 4
    frames in 200. Whatever steady source is loudest in the room then wins
    every estimate.

    That is not hypothetical. Reproduced in tests: with a steady source at
    -25 deg, level gating returned -25.3 deg for claps at -60, -30, 0, +30 and
    +60 - the fan's bearing at every station, with `below gate 0`. It matched
    the hardware run exactly. Onset selection recovers all five to within 1 deg.
    """
    result = StationResult(station=station)
    blocks: list[np.ndarray] = []

    with AudioReceiver(source) as receiver:
        for _ in range(num_frames):
            frame = receiver.read_frame(timeout=timeout)
            if frame is None:
                break
            result.frames_captured += 1
            blocks.append(frame.samples)
            sample_rate = frame.sample_rate

    if not blocks:
        result.note = "no audio captured at all"
        return result

    audio = np.vstack(blocks)
    result.noise_floor = noise_floor_rms(audio)
    onsets = find_onsets(audio, sample_rate)
    result.onsets_found = len(onsets)

    # Everything that is not part of a transient is room, not signal.
    frames_in_onsets = max(1, (len(onsets) * onset_window) // frame_size) if onsets else 0
    result.frames_below_gate = max(result.frames_captured - frames_in_onsets, 0)

    for onset in onsets:
        window = audio[onset:onset + onset_window]
        if window.shape[0] < onset_window // 2:
            continue
        levels = np.sqrt(np.mean(np.square(window.astype(np.float64)), axis=0))
        block = AudioFrame(samples=window.astype(np.float32), timestamp=0.0,
                           frame_index=0, sample_rate=sample_rate)
        doa = estimate_doa(block, classroom.array)
        if not doa.valid or doa.angle_degrees is None or doa.confidence < min_confidence:
            result.frames_low_confidence += 1
            continue
        result.observations.append(Observation(
            lag_samples=float(doa.tdoa_samples),
            tdoa_us=float(doa.tdoa_seconds * 1e6),
            bearing_degrees=float(doa.angle_degrees),
            confidence=float(doa.confidence),
            rms_ch0=float(levels[0]),
            rms_ch1=float(levels[1]) if len(levels) > 1 else 0.0,
        ))

    if result.n == 0:
        result.note = (
            f"no usable transient - {len(onsets)} onset(s) found above a "
            f"{result.noise_floor:.4f} noise floor. Clap harder, or closer."
            if onsets else
            f"NO CLAP DETECTED above the {result.noise_floor:.4f} noise floor. "
            f"The room is louder than the claps; clap harder or nearer the array."
        )
    return result


# --- the sign check, reported on its own -------------------------------------

def coherence(result: StationResult | None, max_lag_samples: float
              ) -> tuple[bool, str]:
    """Did this station measure anything at all? See the precondition above."""
    if result is None or result.n == 0:
        return False, "no usable frames"
    _, spread = result.bearing
    if spread is not None and spread > MAX_COHERENT_SPREAD_DEGREES:
        return False, (f"spread {spread:.0f} deg over {result.n} frames "
                       f"(limit {MAX_COHERENT_SPREAD_DEGREES:.0f})")
    if result.station.role.startswith("endfire"):
        lag, _ = result.lag
        fraction = abs(lag) / max_lag_samples if max_lag_samples else 0.0
        if fraction < MIN_ENDFIRE_LAG_FRACTION:
            return False, (
                f"|lag| {abs(lag):.2f} is {100 * fraction:.0f}% of the {max_lag_samples:.2f}"
                f" physical maximum; end-fire must approach it")
    return True, "coherent"


def sign_verdict(results: dict[str, StationResult],
                 max_lag_samples: float = 6.30) -> tuple[str, str]:
    """Is the channel-0 side of the array really the +90 side?

    Judged ONLY from end-fire stations that passed the coherence precondition.
    A station that measured nothing cannot vote on the sign, however confident
    the arithmetic on its median looks.
    """
    ch0 = results.get("near_mic1")
    ch1 = results.get("near_mic2")
    ok0, why0 = coherence(ch0, max_lag_samples)
    ok1, why1 = coherence(ch1, max_lag_samples)
    got0 = ch0.median_bearing if (ch0 and ok0) else None
    got1 = ch1.median_bearing if (ch1 and ok1) else None

    if got0 is None and got1 is None:
        return NA, (
            "NO COHERENT TDOA - INCONCLUSIVE. Neither end-fire station produced "
            f"a meaningful measurement (mic 1: {why0}; mic 2: {why1}).\n"
            "  The sign convention is UNTESTED. It is NOT confirmed, and it is "
            "NOT mirrored either - there is simply no signal to judge. Do not "
            "change wiring, L/R straps or classroom.yaml on the strength of "
            "this run.\n"
            "  What would make it conclusive: an end-fire station whose "
            f"frame-to-frame spread is under {MAX_COHERENT_SPREAD_DEGREES:.0f} "
            f"deg AND whose |lag| reaches at least "
            f"{100 * MIN_ENDFIRE_LAG_FRACTION:.0f}% of the "
            f"{max_lag_samples:.2f}-sample geometric maximum. Run "
            "tools/analyse_claps.py on a raw recording to find out why not."
        )

    votes = []
    if got0 is not None:
        votes.append(("mic 1 (channel 0)", got0, +1))
    if got1 is not None:
        votes.append(("mic 2 (channel 1)", got1, -1))

    correct = [name for name, got, want in votes if math.copysign(1, got) == want]
    wrong = [(name, got) for name, got, want in votes if math.copysign(1, got) != want]

    if wrong and not correct:
        detail = ", ".join(f"{name} read {got:+.1f} deg" for name, got in wrong)
        return FAIL, (
            "MIRRORED. " + detail + ". Section 5 requires sound at mic 1 to read "
            "POSITIVE and sound at mic 2 to read NEGATIVE. Every bearing in the "
            "system is inverted: the heatmap would be confidently backwards.\n"
            "  Cause is one of: the microphones' L/R straps are swapped, the two "
            "SD lines are crossed, or microphones[0] in classroom.yaml is not the "
            "mic physically at lower x. Fix the CAUSE - do not negate the angle "
            "somewhere downstream."
        )
    if wrong:
        detail = ", ".join(f"{name} read {got:+.1f} deg" for name, got in wrong)
        return FAIL, (
            "INCONSISTENT. " + detail + ", while the other end-fire station "
            "agreed with section 5. One station disagreeing with its own mirror "
            "is not a sign error; it is a bad measurement or a dead channel. "
            "Re-run before concluding anything."
        )
    got = ", ".join(f"{name} read {g:+.1f} deg" for name, g, _ in votes)
    return PASS, (
        "CORRECT. " + got + ", matching section 5: +90 is toward channel 0. "
        "Bearings are not mirrored."
    )


def effective_spacing(
    results: dict[str, StationResult], classroom: ClassroomConfig, sample_rate: int
) -> tuple[float | None, str]:
    """Least-squares fit of lag = (d*fs/c) * sin(angle) over the angle sweep.

    End-fire stations are EXCLUDED: the estimator clamps there, so including
    them would bias the fit toward whatever the clamp allows.
    """
    xs, ys = [], []
    for result in results.values():
        if result.station.role != "angle" or result.n == 0:
            continue
        if result.station.expected_degrees is None:
            continue
        lag, _ = result.lag
        xs.append(math.sin(math.radians(result.station.expected_degrees)))
        ys.append(lag)

    if len(xs) < 2 or not any(abs(x) > 1e-6 for x in xs):
        return None, "not enough off-broadside angles to estimate spacing"

    slope = float(np.polyfit(np.asarray(xs), np.asarray(ys), 1)[0])
    spacing = abs(slope) * classroom.speed_of_sound / sample_rate
    configured = classroom.array.spacing
    delta = 100.0 * (spacing - configured) / configured
    return spacing, (
        f"fitted {spacing * 100:.1f} cm vs {configured * 100:.1f} cm configured "
        f"({delta:+.0f}%)"
    )


# --- reporting ----------------------------------------------------------------

def verdict(results: dict[str, StationResult], max_lag: float = 6.30
            ) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    state, message = sign_verdict(results, max_lag)
    checks.append(("sign convention (section 5)", state, message.split(".")[0] + "."))

    def station_check(key: str, name: str, tolerance: float) -> None:
        result = results.get(key)
        if result is None or result.n == 0:
            checks.append((name, NA, "no usable frames"))
            return
        ok, why = coherence(result, max_lag)
        if not ok:
            # Incoherent, so neither a pass nor a fail: nothing was measured.
            checks.append((name, NA, f"no coherent TDOA - {why}"))
            return
        expected = result.station.expected_degrees or 0.0
        got = result.median_bearing
        ok = abs(got - expected) <= tolerance
        checks.append((name, PASS if ok else FAIL,
                       f"median {got:+.1f} deg vs {expected:+.0f} expected "
                       f"(tolerance {tolerance:.0f})"))

    station_check("near_mic1", "end-fire toward mic 1", ENDFIRE_TOLERANCE)
    station_check("midpoint", "broadside at the midpoint", BROADSIDE_TOLERANCE)
    station_check("near_mic2", "end-fire toward mic 2", ENDFIRE_TOLERANCE)

    sweep = [r for r in results.values() if r.station.role == "angle" and r.n]
    if not sweep:
        checks.append(("angle sweep accuracy", NA, "no usable frames"))
    else:
        errors = [r.mean_abs_error for r in sweep if r.mean_abs_error is not None]
        mean_error = statistics.fmean(errors)
        checks.append((
            "angle sweep accuracy",
            PASS if mean_error <= ERROR_BUDGET_DEGREES else FAIL,
            f"mean |error| {mean_error:.1f} deg over {len(sweep)} angles "
            f"(budget {ERROR_BUDGET_DEGREES:.0f})",
        ))

    # Coherence, not frame count. 82 frames of near-random bearings is not
    # "data produced"; it is 82 measurements of the room. Judging this on
    # len(observations) is how an incoherent run scored a PASS line.
    incoherent = [r.station.label for r in results.values()
                  if not coherence(r, max_lag)[0]]
    checks.append((
        "every station measured something coherent",
        PASS if not incoherent else FAIL,
        "all stations coherent" if not incoherent
        else f"{len(incoherent)} of {len(results)} produced no coherent TDOA",
    ))
    return checks


def report(results: dict[str, StationResult], classroom: ClassroomConfig,
           sample_rate: int) -> bool:
    max_lag = max_delay_samples(classroom.array.spacing, sample_rate)
    state, message = sign_verdict(results, max_lag)

    print()
    print("=" * 78)
    print("SIGN CONVENTION CHECK  (the point of Phase 4a)")
    print("=" * 78)
    print(f"  [{state}]  {message}")
    print()

    print("--- per-station measurements ---")
    header = (f"{'station':<34}{'exp':>6}{'n':>4}{'bearing':>11}{'sd':>7}"
              f"{'lag':>8}{'tdoa us':>10}{'conf':>7}{'|err|':>8}")
    print(header)
    print("-" * len(header))
    for result in results.values():
        st = result.station
        expected = "--" if st.expected_degrees is None else f"{st.expected_degrees:+.0f}"
        if result.n == 0:
            print(f"{st.label[:33]:<34}{expected:>6}{0:>4}{'  no usable frames':>36}")
            continue
        bearing, sd = result.bearing
        lag, _ = result.lag
        tdoa, _ = result.tdoa
        conf, _ = result.confidence
        err = result.mean_abs_error
        print(f"{st.label[:33]:<34}{expected:>6}{result.n:>4}{bearing:>+11.1f}{sd:>7.1f}"
              f"{lag:>+8.2f}{tdoa:>+10.1f}{conf:>7.2f}"
              f"{('--' if err is None else f'{err:.1f}'):>8}")
    print()

    print("--- frame selection: ONSETS, not levels ---")
    for result in results.values():
        print(f"  {result.station.label[:32]:<34} captured {result.frames_captured:>4}"
              f"   floor {result.noise_floor:.4f}"
              f"   claps {result.onsets_found:>3}"
              f"   below gate {result.frames_below_gate:>4}"
              f"   low conf {result.frames_low_confidence:>3}")
    print()

    spacing, note = effective_spacing(results, classroom, sample_rate)
    print("--- effective microphone spacing ---")
    print(f"  {note}")
    if spacing is not None and abs(spacing - classroom.array.spacing) > 0.1 * classroom.array.spacing:
        print("  This differs from the configured value by more than 10%.")
        print("  Per section 10, set it EXPLICITLY in config/classroom.yaml by moving")
        print("  the microphone x positions. Never carry it as a silent constant.")
    print()

    checks = verdict(results, max_lag)
    print("--- verdict ---")
    for name, check_state, detail in checks:
        print(f"  [{check_state:<4}]  {name:<30} {detail}")
    print()

    passed = all(s == PASS for _, s, _ in checks)
    if passed:
        print("VERDICT: PASS - the physical array localizes, and the sign is right.")
        print("Phase 4b (whisper range) may proceed.")
    else:
        print("VERDICT: FAIL - do NOT proceed to Phase 4b.")
        if state == NA:
            print()
            print(message)
            print()
            print("Nothing here justifies a hardware change. Diagnose the raw")
            print("audio first:  python tools/analyse_claps.py record --port COMx")
        if state == FAIL:
            print()
            print(message)
            print()
            print("Fix the sign before measuring anything else. Every number that")
            print("follows a mirrored array is mirrored too.")
    return passed


def results_to_json(results: dict[str, StationResult]) -> str:
    return json.dumps(
        [{"station": asdict(r.station),
          "n": r.n,
          "median_bearing_degrees": r.median_bearing,
          "mean_abs_error_degrees": r.mean_abs_error,
          "frames_captured": r.frames_captured,
          "observations": [asdict(o) for o in r.observations]}
         for r in results.values()],
        indent=2,
    )


def run_stations(stations, source_factory, classroom, *, prompt=None, **kw
                 ) -> dict[str, StationResult]:
    results: dict[str, StationResult] = {}
    for index, station in enumerate(stations, start=1):
        if prompt is not None:
            prompt(f"\n[{index}/{len(stations)}] {station.label}\n  "
                   f"{station.instruction}\n  Press Enter when ready...")
        results[station.key] = measure_station(
            source_factory(station), station, classroom, **kw)
        result = results[station.key]
        bearing = result.median_bearing
        print(f"  -> {result.n} usable frames, median "
              f"{'n/a' if bearing is None else f'{bearing:+.1f} deg'}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 4a: prove the physical array localizes, sign included.")
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "esp32"])
    parser.add_argument("--port", default=None, help="COM port when --source esp32")
    parser.add_argument("--angles", type=float, nargs="*", default=None,
                        help=f"angle sweep in degrees (default {DEFAULT_ANGLES})")
    parser.add_argument("--frames", type=int, default=200,
                        help="frames captured per station")
    parser.add_argument("--rms-gate", type=float, default=DEFAULT_RMS_GATE)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--noise", type=float, default=0.005,
                        help="synthetic source noise amplitude")
    parser.add_argument("--json", default=None, help="also write raw results here")
    parser.add_argument("--no-prompt", action="store_true",
                        help="do not wait for Enter between stations")
    return parser


def main(argv: list[str] | None = None, prompt=input) -> int:
    args = build_parser().parse_args(argv)
    config = load_audio_config()
    classroom = load_classroom_config()

    if args.source == "esp32":
        source_factory = esp32_source_factory(config, args.port)
        sample_rate = config.transport.transmit_sample_rate
        print("Source: ESP32 over USB serial.")
        print("The board must be on a STREAMING build and the Serial Monitor closed.")
    else:
        source_factory = synthetic_source_factory(config, classroom, args.noise)
        sample_rate = config.transport.transmit_sample_rate
        print("Source: SYNTHETIC. This checks the measurement, NOT the microphones.")
        print("It cannot tell you anything about the physical sign convention.")

    print(f"Array: {classroom.array.num_channels} mics, "
          f"{classroom.array.spacing * 100:.1f} cm apart, at {sample_rate} Hz.")

    stations = default_stations(args.angles)
    try:
        results = run_stations(
            stations, source_factory, classroom,
            prompt=None if (args.no_prompt or args.source == "synthetic") else prompt,
            num_frames=args.frames, rms_gate=args.rms_gate,
            min_confidence=args.min_confidence,
        )
    except AudioSourceError as exc:
        # No board, no port, or the port is held by the Serial Monitor. Say so
        # plainly; a traceback here reads like a bug in the measurement.
        print(f"\nCannot reach the microphone array: {exc}", file=sys.stderr)
        print("This tool needs real hardware. Run it without --source esp32 to "
              "self-test the measurement only.", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        # This tool is interactive by design; it cannot be piped.
        print("\nInterrupted before any station completed - nothing measured.",
              file=sys.stderr)
        print("Run it from a terminal, or pass --no-prompt.", file=sys.stderr)
        return 2

    passed = report(results, classroom, sample_rate)
    if args.json:
        Path(args.json).write_text(results_to_json(results), encoding="utf-8")
        print(f"raw results written to {args.json}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
