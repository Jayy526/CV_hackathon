"""Phase 4a diagnosis: is there a real inter-channel delay in the raw audio?

The station sweep produced near-zero bearings with 37-55 deg spread and uniform
0.47-0.52 confidence everywhere. That is not a mirrored array; it is an ABSENT
measurement. Mirroring flips the sign, it does not collapse the magnitude: a
mirrored array clapped 5 cm from mic 1 reads -90 deg, not -2.3 deg.

The suspicion this tool tests is frame selection. 200 frames of 1024 samples at
16 kHz is 12.8 s per station, and a clap's direct sound is ~2 ms - about 0.06%
of the recording. Everything else is reverberation, which arrives from every
direction at once and carries no bearing. Taking a median over ~85 RMS-gated
frames therefore measures the room, not the source.

So: record raw audio once, then analyse it OFFLINE, comparing the delay at the
onset against the delay over the whole frame, with a plain cross-correlation as
an independent check that does not depend on GCC-PHAT being right.

    python tools/analyse_claps.py record --seconds 10 -o claps.wav
    python tools/analyse_claps.py analyse claps.wav --expect 6.30 --plot claps.png

The WAV is kept so this can be re-run without the hardware.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from heimdall.audio.analysis import read_wav, write_wav  # noqa: E402
from heimdall.audio.config import load_audio_config  # noqa: E402
from heimdall.audio.gcc_phat import gcc_phat, max_delay_samples  # noqa: E402
from heimdall.audio.geometry import load_classroom_config  # noqa: E402
from heimdall.audio.receiver import AudioReceiver  # noqa: E402
from heimdall.audio.sources import AudioSourceError  # noqa: E402

# A clap's direct sound is ~2 ms. 256 samples at 16 kHz is 16 ms: long enough
# for GCC-PHAT to have something to correlate, short enough that the first
# wall reflection (a 2 m path difference is 6 ms) has not yet dominated.
ONSET_WINDOW = 256
# Channel balance is judged on the DIRECT sound only. Measured over the full
# 256-sample window it is worthless: reverberation reaches both microphones at
# similar level, so a nearly dead channel still looks only a few dB down.
DIRECT_WINDOW = 64
# What the station sweep used, for the side-by-side comparison.
FRAME_WINDOW = 1024
# Envelope smoothing for onset detection, and the refractory gap between claps.
ENVELOPE_HOP = 16
MIN_CLAP_GAP_SAMPLES = 4000        # 250 ms
ONSET_FRACTION = 0.2               # attack = first crossing of 20% of the peak


@dataclass
class Clap:
    index: int
    onset: int                      # sample index of the attack
    peak_index: int
    peak_ch0: float
    peak_ch1: float
    rms_ch0: float
    rms_ch1: float
    lag_onset: float | None = None          # GCC-PHAT, 256-sample onset window
    lag_frame: float | None = None          # GCC-PHAT, 1024-sample frame
    lag_xcorr: float | None = None          # plain cross-correlation, onset window
    confidence_onset: float = 0.0
    confidence_frame: float = 0.0
    note: str = ""

    @property
    def channel_imbalance_db(self) -> float:
        lo = max(min(self.peak_ch0, self.peak_ch1), 1e-9)
        hi = max(self.peak_ch0, self.peak_ch1, 1e-9)
        return float(20.0 * np.log10(hi / lo))


def envelope(signal: np.ndarray, hop: int = ENVELOPE_HOP) -> np.ndarray:
    """Coarse |x| envelope, one value per `hop` samples."""
    usable = (len(signal) // hop) * hop
    if usable == 0:
        return np.zeros(0)
    return np.abs(signal[:usable]).reshape(-1, hop).max(axis=1)


def find_claps(
    samples: np.ndarray,
    *,
    threshold: float = 0.15,
    hop: int = ENVELOPE_HOP,
    min_gap: int = MIN_CLAP_GAP_SAMPLES,
) -> list[int]:
    """Peak sample index of each transient loud enough to be a clap.

    Detection runs on the channel sum so a clap is found even if one microphone
    is weak; per-channel health is then reported separately rather than being
    allowed to hide the event.
    """
    mono = samples.sum(axis=1)
    env = envelope(mono, hop)
    if env.size == 0:
        return []

    ceiling = float(env.max())
    if ceiling <= 0:
        return []

    peaks: list[int] = []
    for i, value in enumerate(env):
        if value < threshold * ceiling:
            continue
        centre = i * hop
        if peaks and centre - peaks[-1] < min_gap:
            # Same clap, later reflection: keep whichever is louder.
            if value > abs(mono[peaks[-1]]):
                peaks[-1] = int(np.argmax(np.abs(mono[centre:centre + hop])) + centre)
            continue
        peaks.append(int(np.argmax(np.abs(mono[centre:centre + hop])) + centre))
    return peaks


def find_onset(mono: np.ndarray, peak_index: int, lookback: int = 512) -> int:
    """Walk back from the peak to the start of the attack.

    GCC-PHAT must see the direct sound, not the decay: by the time the envelope
    peaks, the first reflections are usually already present.
    """
    begin = max(0, peak_index - lookback)
    window = np.abs(mono[begin:peak_index + 1])
    if window.size == 0:
        return peak_index
    above = np.nonzero(window >= ONSET_FRACTION * float(window.max()))[0]
    return int(begin + above[0]) if above.size else peak_index


def plain_cross_correlation_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Integer-lag cross-correlation, deliberately naive.

    An independent check on GCC-PHAT: no whitening, no coherence mask, no
    sub-sample interpolation. If the two disagree wildly on the same window,
    the problem is in the estimator; if they agree, it is in the audio.

    Sign matches gcc_phat: positive means `a` arrived LATER than `b`.
    """
    a = np.asarray(a, dtype=np.float64) - float(np.mean(a))
    b = np.asarray(b, dtype=np.float64) - float(np.mean(b))
    if not np.any(a) or not np.any(b):
        return 0.0
    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        shifted = np.roll(b, lag)
        score = float(np.dot(a, shifted))
        if score > best_score:
            best_score, best_lag = score, lag
    return float(best_lag)


def analyse_clap(
    samples: np.ndarray, sample_rate: int, peak_index: int, index: int, max_lag: float
) -> Clap:
    mono = samples.sum(axis=1)
    onset = find_onset(mono, peak_index)

    def window(begin: int, length: int) -> np.ndarray:
        stop = min(len(samples), begin + length)
        return samples[begin:stop]

    onset_block = window(onset, ONSET_WINDOW)
    direct = window(onset, DIRECT_WINDOW)
    # The frame window is centred the way the station sweep's frames fell:
    # the clap somewhere inside a 64 ms block, mostly reverberation.
    frame_block = window(max(0, onset - FRAME_WINDOW // 4), FRAME_WINDOW)

    clap = Clap(
        index=index,
        onset=onset,
        peak_index=peak_index,
        peak_ch0=float(np.max(np.abs(direct[:, 0]))) if direct.size else 0.0,
        peak_ch1=float(np.max(np.abs(direct[:, 1]))) if direct.size else 0.0,
        rms_ch0=float(np.sqrt(np.mean(direct[:, 0] ** 2))) if direct.size else 0.0,
        rms_ch1=float(np.sqrt(np.mean(direct[:, 1] ** 2))) if direct.size else 0.0,
    )

    if onset_block.shape[0] < 32:
        clap.note = "onset too close to the end of the recording"
        return clap

    # gcc_phat(signal, reference): positive when `signal` arrived LATER.
    # Channel 0 first, so a source near mic 1 gives a NEGATIVE lag.
    # max_tau, not the whole correlation range: an unbounded search happily
    # returns physically impossible delays.
    max_tau = max_lag / sample_rate
    result = gcc_phat(onset_block[:, 0], onset_block[:, 1],
                      sample_rate=sample_rate, max_tau=max_tau)
    if result.valid:
        clap.lag_onset = float(result.delay_samples)
    clap.confidence_onset = float(result.confidence)

    if frame_block.shape[0] >= 64:
        framed = gcc_phat(frame_block[:, 0], frame_block[:, 1],
                          sample_rate=sample_rate, max_tau=max_tau)
        if framed.valid:
            clap.lag_frame = float(framed.delay_samples)
        clap.confidence_frame = float(framed.confidence)

    clap.lag_xcorr = plain_cross_correlation_lag(
        onset_block[:, 0], onset_block[:, 1], int(np.ceil(max_lag)))
    return clap


def ascii_waveform(samples: np.ndarray, begin: int, length: int = 48,
                   width: int = 58) -> list[str]:
    """Both channels at sample resolution, so a delay is visible by eye."""
    stop = min(len(samples), begin + length)
    block = samples[begin:stop]
    if block.size == 0:
        return ["  (no samples)"]
    scale = max(float(np.max(np.abs(block))), 1e-9)
    lines = []
    for channel in range(block.shape[1]):
        lines.append(f"    ch{channel}:")
        for offset in range(block.shape[0]):
            value = block[offset, channel] / scale
            column = int((value + 1.0) / 2.0 * (width - 1))
            row = [" "] * width
            row[(width - 1) // 2] = "|"
            row[column] = "#"
            lines.append(f"      {begin + offset:>7} {''.join(row)}")
    return lines


def plot_claps(samples, sample_rate, claps, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(len(claps), 4)
    if count == 0:
        return None
    figure, axes = plt.subplots(count, 1, figsize=(11, 2.6 * count), squeeze=False)
    for row, clap in enumerate(claps[:count]):
        axis = axes[row][0]
        begin = max(0, clap.onset - 32)
        stop = min(len(samples), clap.onset + ONSET_WINDOW)
        t = np.arange(begin, stop)
        axis.plot(t, samples[begin:stop, 0], label="ch0 (mic 1)", linewidth=0.9)
        axis.plot(t, samples[begin:stop, 1], label="ch1 (mic 2)", linewidth=0.9)
        axis.axvline(clap.onset, color="k", linestyle=":", linewidth=0.8)
        lag = "n/a" if clap.lag_onset is None else f"{clap.lag_onset:+.2f}"
        axis.set_title(f"clap {clap.index}: onset lag {lag} samples", fontsize=9)
        axis.legend(fontsize=7)
        axis.set_xlabel("sample")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def channel_levels(samples: np.ndarray) -> list[tuple[float, float]]:
    """Whole-recording (rms, peak) per channel. A dead microphone shows here
    unmistakably, whatever the claps did."""
    return [(float(np.sqrt(np.mean(samples[:, c] ** 2))),
             float(np.max(np.abs(samples[:, c]))))
            for c in range(samples.shape[1])]


def overall_imbalance_db(samples: np.ndarray) -> float:
    levels = [rms for rms, _ in channel_levels(samples)]
    lo, hi = max(min(levels), 1e-9), max(max(levels), 1e-9)
    return float(20.0 * np.log10(hi / lo))


def diagnose(claps: list[Clap], max_lag: float,
             overall_imbalance: float = 0.0) -> tuple[str, str]:
    """Return (case, explanation) for cases (a), (b) and (c)."""
    if not claps:
        return "none", (
            "No claps were detected at all. Either the recording is silent or "
            "the threshold is too high. Check the WAV in an audio editor before "
            "concluding anything about the array."
        )

    imbalance = float(np.median([c.channel_imbalance_db for c in claps]))
    weak = [c for c in claps if min(c.peak_ch0, c.peak_ch1) < 0.02]
    # 8 dB, not 12: two identical microphones a hand's width apart should differ
    # by ~1 dB on the direct sound. 8 dB is already a fault, not a tolerance.
    if imbalance > 8.0 or len(weak) > len(claps) / 2 or overall_imbalance > 8.0:
        return "c", (
            f"ONE CHANNEL IS MUCH WEAKER. Direct-sound peak imbalance is "
            f"{imbalance:.1f} dB per clap, {overall_imbalance:.1f} dB over the "
            f"whole recording. Both """
            f"microphones should see a clap at nearly the same level from a "
            f"metre away. THIS is when wiring is worth examining: check the SD "
            f"line, the 3V3 and GND on the quiet microphone, and that it is "
            f"firmly seated."
        )

    usable = [c for c in claps if c.lag_onset is not None]
    if not usable:
        return "b", (
            "GCC-PHAT returned nothing usable even at the onset, though both "
            "channels show the clap."
        )

    onset_lags = np.array([c.lag_onset for c in usable])
    frame_lags = np.array([c.lag_frame for c in usable if c.lag_frame is not None])
    onset_median = float(np.median(np.abs(onset_lags)))
    frame_median = float(np.median(np.abs(frame_lags))) if frame_lags.size else 0.0
    onset_spread = float(np.std(onset_lags))

    coherent = onset_median >= 0.4 * max_lag and onset_spread <= 0.25 * max_lag
    if coherent and frame_median < 0.4 * max_lag:
        return "a", (
            f"THE HARDWARE IS FINE; THE TOOL'S FRAME SELECTION IS THE BUG.\n"
            f"  At the onset the delay is {onset_median:.2f} samples "
            f"({100 * onset_median / max_lag:.0f}% of the {max_lag:.2f} maximum), "
            f"spread {onset_spread:.2f}.\n"
            f"  Over the 64 ms frame the same claps give {frame_median:.2f} "
            f"samples ({100 * frame_median / max_lag:.0f}%) - the direct sound is "
            f"swamped by reverberation.\n"
            f"  Fix: select frames on ONSET, not RMS, and correlate a short "
            f"window at the attack. Do not touch the wiring."
        )
    if coherent:
        return "a", (
            f"A coherent delay of {onset_median:.2f} samples "
            f"({100 * onset_median / max_lag:.0f}% of maximum) is present at the "
            f"onset AND survives the 64 ms frame ({frame_median:.2f}). The audio "
            f"is good; the station sweep's median-over-all-frames is what "
            f"discarded it."
        )
    return "b", (
        f"NO COHERENT DELAY, even at the onset, though both channels show the "
        f"clap at similar level.\n"
        f"  Onset lags: median |{onset_median:.2f}| samples against a physical "
        f"maximum of {max_lag:.2f}, spread {onset_spread:.2f}.\n"
        f"  This is acoustic or physical, not a code bug. Before changing any "
        f"code, check: are both microphones unobstructed, facing the same way, "
        f"firmly seated, and actually {max_lag * 343 / 16000 * 100:.1f} cm apart? "
        f"Is the clap ON the array axis rather than in front of it?"
    )


def report(claps: list[Clap], sample_rate: int, samples: np.ndarray,
           max_lag: float, waveforms: int = 2) -> str:
    print()
    print(f"--- {len(claps)} clap(s) detected in "
          f"{len(samples) / sample_rate:.1f} s at {sample_rate} Hz ---")
    print(f"physical maximum |lag| for this array: {max_lag:.2f} samples "
          f"({max_lag / sample_rate * 1e6:.0f} us)")
    print()
    print("--- per-channel level over the WHOLE recording ---")
    for channel, (level, top) in enumerate(channel_levels(samples)):
        print(f"  ch{channel}: rms {level:.4f}  peak {top:.4f}")
    print(f"  imbalance: {overall_imbalance_db(samples):.1f} dB")
    print()

    header = (f"{'clap':>5}{'onset':>9}{'pk ch0':>9}{'pk ch1':>9}{'rms ch0':>9}"
              f"{'rms ch1':>9}{'imbal':>8}{'lag onset':>11}{'lag frame':>11}"
              f"{'lag xcorr':>11}{'conf on':>9}{'conf fr':>9}")
    print(header)
    print("-" * len(header))
    for clap in claps:
        def fmt(value):
            return "  n/a" if value is None else f"{value:+.2f}"
        print(f"{clap.index:>5}{clap.onset:>9}{clap.peak_ch0:>9.3f}"
              f"{clap.peak_ch1:>9.3f}{clap.rms_ch0:>9.3f}{clap.rms_ch1:>9.3f}"
              f"{clap.channel_imbalance_db:>7.1f}dB{fmt(clap.lag_onset):>11}"
              f"{fmt(clap.lag_frame):>11}{fmt(clap.lag_xcorr):>11}"
              f"{clap.confidence_onset:>9.2f}{clap.confidence_frame:>9.2f}")
    print()

    usable = [c for c in claps if c.lag_onset is not None]
    if usable:
        onset = np.array([c.lag_onset for c in usable])
        xcorr = np.array([c.lag_xcorr for c in usable if c.lag_xcorr is not None])
        frame = np.array([c.lag_frame for c in usable if c.lag_frame is not None])
        print("--- the three lag measurements, same claps ---")
        print(f"  GCC-PHAT, 256-sample onset window : median {np.median(onset):+.2f}"
              f"  sd {np.std(onset):.2f}")
        if frame.size:
            print(f"  GCC-PHAT, 1024-sample frame       : median {np.median(frame):+.2f}"
                  f"  sd {np.std(frame):.2f}")
        if xcorr.size:
            print(f"  plain cross-correlation (onset)   : median {np.median(xcorr):+.2f}"
                  f"  sd {np.std(xcorr):.2f}")
            agreement = float(np.median(np.abs(onset[:len(xcorr)] - xcorr)))
            print(f"  GCC-PHAT vs plain xcorr disagree by median {agreement:.2f} samples")
            if agreement > 1.0:
                print("  ^ they disagree: suspect the ESTIMATOR, not the audio.")
            else:
                print("  ^ they agree: the estimator is not the problem.")
        print()

    for clap in claps[:waveforms]:
        print(f"--- clap {clap.index} at sample resolution, from the onset ---")
        for line in ascii_waveform(samples, max(0, clap.onset - 4), length=40):
            print(line)
        print()

    case, message = diagnose(claps, max_lag, overall_imbalance_db(samples))
    print("=" * 78)
    print(f"DIAGNOSIS: case ({case})" if case in "abc" else "DIAGNOSIS")
    print("=" * 78)
    print(f"  {message}")
    print()
    return case


# --- recording ---------------------------------------------------------------

def record(port: str | None, seconds: float, output: Path) -> Path:
    from heimdall.audio.sources import ESP32AudioSource

    config = load_audio_config()
    source = ESP32AudioSource(port=port, config=config)
    rate = source.sample_rate
    needed = int(seconds * rate)

    print(f"Recording {seconds:.0f} s at {rate} Hz. Clap 3-4 times, hard, "
          f"~5 cm from MIC 1 on the array axis.")
    blocks: list[np.ndarray] = []
    captured = 0
    with AudioReceiver(source) as receiver:
        while captured < needed:
            frame = receiver.read_frame(timeout=3.0)
            if frame is None:
                break
            blocks.append(frame.samples)
            captured += frame.num_samples

    if not blocks:
        raise AudioSourceError("no audio captured - the board sent nothing")
    samples = np.vstack(blocks)[:needed]
    write_wav(output, samples, rate)
    print(f"wrote {output} ({len(samples) / rate:.1f} s, "
          f"{samples.shape[1]} channels, {rate} Hz)")
    print(f"dropped packets: {source.packets_dropped}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    rec = sub.add_parser("record", help="capture raw audio from the ESP32 to a WAV")
    rec.add_argument("--port", default=None)
    rec.add_argument("--seconds", type=float, default=10.0)
    rec.add_argument("-o", "--output", default="claps.wav")

    ana = sub.add_parser("analyse", help="analyse a WAV offline, no hardware needed")
    ana.add_argument("wav")
    ana.add_argument("--threshold", type=float, default=0.15,
                     help="clap detection threshold, fraction of the loudest peak")
    ana.add_argument("--spacing", type=float, default=None,
                     help="microphone spacing in metres (default: classroom.yaml)")
    ana.add_argument("--plot", default=None, help="write a PNG of the onsets here")
    ana.add_argument("--waveforms", type=int, default=2,
                     help="how many claps to print at sample resolution")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "record":
        try:
            record(args.port, args.seconds, Path(args.output))
        except AudioSourceError as exc:
            print(f"\nCannot record: {exc}", file=sys.stderr)
            print("This mode needs the ESP32. Use 'analyse' on an existing WAV "
                  "to work without hardware.", file=sys.stderr)
            return 2
        return 0

    samples, sample_rate = read_wav(args.wav)
    if samples.shape[1] < 2:
        print(f"{args.wav} has {samples.shape[1]} channel(s); need 2.",
              file=sys.stderr)
        return 2

    spacing = args.spacing
    if spacing is None:
        spacing = load_classroom_config().array.spacing
    max_lag = max_delay_samples(spacing, sample_rate)

    peaks = find_claps(samples, threshold=args.threshold)
    claps = [analyse_clap(samples, sample_rate, peak, i + 1, max_lag)
             for i, peak in enumerate(peaks)]

    case = report(claps, sample_rate, samples, max_lag, waveforms=args.waveforms)
    if args.plot and claps:
        written = plot_claps(samples, sample_rate, claps, args.plot)
        if written:
            print(f"waveform plot written to {written}")
    return 0 if case == "a" else 1


if __name__ == "__main__":
    raise SystemExit(main())
