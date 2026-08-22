"""Live per-channel meter. Which microphone is actually working?

    python tools/mic_meter.py            # live bars, Ctrl-C to stop
    python tools/mic_meter.py --seconds 20

Tap ONE microphone at a time and watch which bar moves. On a healthy pair:

  * tapping mic 1 moves ch0 and barely touches ch1
  * tapping mic 2 moves ch1 and barely touches ch0
  * with both quiet, COHERENCE is high (0.3-0.9): they hear the same room

COHERENCE IS THE ONE THAT MATTERS. Two microphones a hand's width apart hear
the same sounds, so their signals must share structure. Near 0.00 means the two
channels have NOTHING in common - which no amount of software can fix, because
there is no shared signal to find a delay in. A pair that both show healthy
levels but zero coherence is still broken.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from acoustic_array.analysis import db_fs  # noqa: E402
from acoustic_array.sources import AudioSourceError, ESP32AudioSource  # noqa: E402

BAR = 34


def bar(value: float, floor_db: float = -60.0) -> str:
    db = db_fs(max(value, 1e-9))
    filled = int(np.clip((db - floor_db) / (0.0 - floor_db), 0.0, 1.0) * BAR)
    return "#" * filled + " " * (BAR - filled)


def coherence(a: np.ndarray, b: np.ndarray, n: int = 512) -> float:
    """Mean magnitude-squared coherence. The real health check for a PAIR."""
    if len(a) < n * 2:
        return 0.0
    window = np.hanning(n)
    p00 = p11 = p01 = 0.0
    count = 0
    for i in range(0, len(a) - n, n // 2):
        x = np.fft.rfft(a[i:i + n] * window)
        y = np.fft.rfft(b[i:i + n] * window)
        p00 = p00 + np.abs(x) ** 2
        p11 = p11 + np.abs(y) ** 2
        p01 = p01 + x * np.conj(y)
        count += 1
    if count < 4:
        return 0.0
    return float(np.mean(np.abs(p01) ** 2 / (p00 * p11 + 1e-30)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Live two-channel microphone meter.")
    parser.add_argument("--port", default=None)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    try:
        source = ESP32AudioSource(port=args.port)
        source.start()
    except AudioSourceError as exc:
        print(f"\nCannot reach the array: {exc}", file=sys.stderr)
        return 2

    print(f"port {source.port_name}   {source.sample_rate} Hz   "
          f"{source.num_channels} channels")
    print()
    print("Tap MIC 1, then MIC 2. Each should move its OWN bar only.")
    print("Then stay quiet and watch coherence - it must be well above 0.05.")
    print()
    print(f"{'ch0 (mic 1)':<36} {'ch1 (mic 2)':<36} {'coh':>6}  verdict")

    recent: list[np.ndarray] = []
    began = time.monotonic()
    try:
        while time.monotonic() - began < args.seconds:
            frame = source.read_frame()
            if frame is None:
                print("stream ended", file=sys.stderr)
                break
            recent.append(frame.samples)
            recent = recent[-16:]                     # ~1 s of history
            audio = np.vstack(recent)
            r0 = float(np.sqrt(np.mean(audio[:, 0] ** 2)))
            r1 = float(np.sqrt(np.mean(audio[:, 1] ** 2)))
            coh = coherence(audio[:, 0].astype(float), audio[:, 1].astype(float))

            note = ""
            if r0 < 1e-4 and r1 < 1e-4:
                note = "BOTH SILENT"
            elif r0 < 1e-4:
                note = "CH0 SILENT - mic 1 dead"
            elif r1 < 1e-4:
                note = "CH1 SILENT - mic 2 dead"
            elif coh < 0.05:
                note = "NO SHARED SIGNAL - the pair is broken"
            elif coh < 0.20:
                note = "weak coherence"
            else:
                note = "pair looks healthy"

            print(f"[{bar(r0)}] {db_fs(r0):6.1f}  [{bar(r1)}] {db_fs(r1):6.1f} "
                  f"{coh:6.3f}  {note}", end="\r", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
