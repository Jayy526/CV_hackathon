"""Phase E: hardware validation and diagnostics.

Shows waveform, RMS energy over time, per-channel energy and a spectrogram, so
you can confirm by eye that both microphones are hearing the same event.

Works today against the synthetic source; point it at the ESP32 source later
and nothing below changes.

    python tools/monitor_audio.py                       # synthetic, saves a PNG
    python tools/monitor_audio.py --seconds 3 --angle 30
    python tools/monitor_audio.py --text                # no plotting at all
    python tools/monitor_audio.py --wav recordings/test.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from heimdall.audio.analysis import (  # noqa: E402
    analyse_frame,
    channel_rms,
    concatenate_frames,
    db_fs,
    spectrogram,
    write_wav,
)
from heimdall.audio.config import load_audio_config  # noqa: E402
from heimdall.audio.geometry import load_classroom_config  # noqa: E402
from heimdall.audio.receiver import AudioReceiver  # noqa: E402
from heimdall.audio.sources import SyntheticAudioSource  # noqa: E402


def build_source(args, audio_config, classroom):
    """The one place that knows which kind of hardware we are talking to."""
    if args.source == "synthetic":
        return SyntheticAudioSource(
            sample_rate=audio_config.sample_rate,
            num_channels=audio_config.num_channels,
            frame_size=audio_config.frame_size,
            angle_degrees=args.angle,
            mic_spacing_m=classroom.array.spacing,
            burst_frames=args.burst_frames,
            silence_frames=args.silence_frames,
            noise_amplitude=args.noise,
        )

    if args.source == "esp32":
        from heimdall.audio.sources import ESP32AudioSource

        return ESP32AudioSource(config=audio_config)

    raise SystemExit("unknown source %r" % args.source)


def capture(source, seconds):
    """Pull frames for `seconds` and return (frames, stats)."""
    frames = []
    stats = []
    target = int(np.ceil(seconds * source.sample_rate / source.frame_size))

    with AudioReceiver(source) as receiver:
        for _ in range(target):
            frame = receiver.read_frame(timeout=2.0)
            if frame is None:
                break
            frames.append(frame)
            stats.append(analyse_frame(frame))

    return frames, stats


def print_text_report(frames, stats, sample_rate):
    if not frames:
        print("No audio captured.")
        return

    samples = concatenate_frames(frames)
    channels = samples.shape[1]

    print("Audio device connected")
    print("Sample rate:      %d Hz" % sample_rate)
    print("Channels:         %d" % channels)
    print("Frames received:  %d" % len(frames))
    print("Samples received: %d per channel" % samples.shape[0])
    print("Duration:         %.2f s" % (samples.shape[0] / sample_rate))
    print()

    overall_rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    overall_peak = float(np.max(np.abs(samples)))
    print("RMS:  %.6f  (%.1f dBFS)" % (overall_rms, db_fs(overall_rms)))
    print("Peak: %.6f  (%.1f dBFS)" % (overall_peak, db_fs(overall_peak)))
    print()

    print("Per channel:")
    print("  %-10s %-12s %-12s %-10s" % ("CHANNEL", "RMS", "PEAK", "RMS dBFS"))
    for channel in range(channels):
        column = samples[:, channel].astype(np.float64)
        rms_value = float(np.sqrt(np.mean(column**2)))
        print(
            "  channel_%-2d %-12.6f %-12.6f %-10.1f"
            % (channel, rms_value, float(np.max(np.abs(column))), db_fs(rms_value))
        )

    if channels >= 2:
        left = samples[:, 0].astype(np.float64)
        right = samples[:, 1].astype(np.float64)
        left_rms = float(np.sqrt(np.mean(left**2)))
        right_rms = float(np.sqrt(np.mean(right**2)))
        print()
        if min(left_rms, right_rms) < 1e-9:
            print("WARNING: a channel is silent. Check wiring before trusting anything else.")
        else:
            imbalance_db = db_fs(left_rms) - db_fs(right_rms)
            print("Channel balance: %+.1f dB (channel_0 relative to channel_1)" % imbalance_db)
            if abs(imbalance_db) > 6.0:
                print(
                    "WARNING: channels differ by more than 6 dB. One microphone may be "
                    "faulty, misplaced, or much closer to the source than the other."
                )

    loudest = max(stats, key=lambda s: s.rms)
    print()
    print(
        "Loudest frame: #%d at t=%.3f s, RMS %.6f, speech-band ratio %.2f"
        % (loudest.frame_index, loudest.timestamp, loudest.rms, loudest.speech_band_ratio)
    )


def plot(frames, stats, sample_rate, output_path, show):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = concatenate_frames(frames)
    channels = samples.shape[1]
    time_axis = np.arange(samples.shape[0]) / sample_rate
    frame_times = np.array([s.timestamp for s in stats])

    figure, axes = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)

    # 1. Waveform, one trace per microphone.
    for channel in range(channels):
        axes[0].plot(
            time_axis, samples[:, channel], linewidth=0.6, alpha=0.8,
            label="MIC %d (channel_%d)" % (channel + 1, channel),
        )
    axes[0].set_title("Waveform")
    axes[0].set_ylabel("amplitude")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    # 2. Overall RMS energy per frame.
    axes[1].plot(frame_times, [s.rms for s in stats], color="tab:purple")
    axes[1].set_title("RMS energy per frame")
    axes[1].set_ylabel("RMS")
    axes[1].grid(alpha=0.3)

    # 3. Per-channel energy - the two traces should rise together for one event.
    per_channel = np.array([channel_rms(f) for f in frames])
    for channel in range(channels):
        axes[2].plot(
            frame_times, per_channel[:, channel],
            label="MIC %d" % (channel + 1),
        )
    axes[2].set_title("Per-channel energy (both traces should track the same event)")
    axes[2].set_ylabel("RMS")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    # 4. Spectrogram of channel 0.
    freqs, times, magnitude = spectrogram(samples[:, 0], sample_rate)
    decibels = 20.0 * np.log10(magnitude + 1e-10)
    axes[3].pcolormesh(times, freqs, decibels, shading="auto", cmap="magma")
    axes[3].set_title("Spectrogram (MIC 1)")
    axes[3].set_ylabel("Hz")
    axes[3].set_xlabel("time (s)")

    if show:
        plt.show()
    else:
        figure.savefig(output_path, dpi=110)
        print("\nPlot written to %s" % output_path)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "esp32"])
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--angle", type=float, default=30.0,
                        help="synthetic source bearing, in degrees")
    parser.add_argument("--noise", type=float, default=0.002,
                        help="synthetic background noise amplitude")
    parser.add_argument("--burst-frames", type=int, default=10)
    parser.add_argument("--silence-frames", type=int, default=10)
    parser.add_argument("--wav", type=Path, default=None,
                        help="also write the capture to this WAV file")
    parser.add_argument("--plot", type=Path, default=Path("recordings/monitor.png"))
    parser.add_argument("--show", action="store_true", help="open a window instead of saving")
    parser.add_argument("--text", action="store_true", help="text report only, no plot")
    args = parser.parse_args()

    audio_config = load_audio_config()
    classroom = load_classroom_config()

    try:
        source = build_source(args, audio_config, classroom)
    except NotImplementedError as exc:
        print("Cannot monitor real hardware yet:")
        print("  %s" % exc)
        return 2

    frames, stats = capture(source, args.seconds)
    print_text_report(frames, stats, source.sample_rate)

    if not frames:
        return 1

    if args.wav is not None:
        path = write_wav(args.wav, concatenate_frames(frames), source.sample_rate)
        print("\nWAV written to %s (channel_0 = MIC 1, channel_1 = MIC 2)" % path)

    if not args.text:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(frames, stats, source.sample_rate, args.plot, args.show)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
