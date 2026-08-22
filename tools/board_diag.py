"""Ask the ESP32 what IT sees, before any decimation or transport.

    python tools/board_diag.py

The host-side meter (tools/mic_meter.py) sees audio after the FIR, the
decimation, the packet layer and the receiver. This sees the raw I2S levels on
the board itself, so a silent channel here is unambiguously a microphone or
wiring fault - nothing downstream can have caused it.

Works on a STREAMING build (START_IN_DIAG 0): it sends 'd' to switch the
firmware into text diagnostics, reads them, then sends 's' to resume streaming.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acoustic_array.config import load_audio_config  # noqa: E402


def read_diagnostics(port_name: str, baud: int, settle: float, seconds: float) -> str:
    import serial

    port = serial.Serial(port_name, baud, timeout=0.4)
    with port:
        # Opening asserts DTR/RTS and resets the board; wait for the CP2102 to
        # re-enumerate before it will answer anything.
        time.sleep(settle)
        port.reset_input_buffer()
        port.write(b"d")
        port.flush()
        deadline = time.monotonic() + seconds
        data = b""
        while time.monotonic() < deadline:
            data += port.read(max(1, port.in_waiting))
            time.sleep(0.1)
        port.write(b"s")            # leave it streaming again
        port.flush()

    text = data.decode("ascii", "replace")
    # Drop any binary that was still in flight when the mode switched.
    return "\n".join(
        line for line in text.splitlines()
        if line and sum(1 for c in line if 32 <= ord(c) < 127) > len(line) * 0.8
    )


def verdict(text: str) -> tuple[bool, str]:
    """Read the per-channel levels the board reported."""
    levels = [l for l in text.splitlines() if l.strip().startswith("levels")]
    if not levels:
        return False, ("The board did not report levels. Is it on a STREAMING "
                       "build (START_IN_DIAG 0) and is the Serial Monitor closed?")
    last = levels[-1]
    try:
        parts = last.split()
        ch0 = int(parts[parts.index("ch0") + 2])
        ch1 = int(parts[parts.index("ch1") + 2])
    except (ValueError, IndexError):
        return False, f"could not parse: {last}"

    if ch0 == 0 and ch1 == 0:
        return False, ("BOTH CHANNELS DEAD at the I2S input. Neither microphone "
                       "is driving the data line. Check 3V3 and GND on both, "
                       "then SD -> GPIO 33.")
    if ch0 == 0:
        return False, ("MIC 1 (ch0) IS DEAD at the I2S input - the board reads "
                       "exactly zero. Check mic 1's 3V3, GND, its SD wire to "
                       "GPIO 33, and that its L/R pin is tied to GND.")
    if ch1 == 0:
        return False, ("MIC 2 (ch1) IS DEAD at the I2S input. Check mic 2's "
                       "3V3, GND, SD to GPIO 33, and L/R tied to 3V3.")
    ratio = max(ch0, ch1) / max(min(ch0, ch1), 1)
    if ratio > 4.0:
        return False, (f"Both channels alive but badly unbalanced "
                       f"({ch0} vs {ch1}, {ratio:.1f}x). Two identical mics side "
                       f"by side should be within about 2x.")
    return True, f"Both channels alive and balanced: ch0 {ch0}, ch1 {ch1}."


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read the ESP32's own I2S diagnostics.")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--settle", type=float, default=4.0)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args(argv)

    transport = load_audio_config().transport
    port_name = args.port or transport.port
    if not port_name:
        print("No port. Set transport.port in config/audio.yaml or pass --port.",
              file=sys.stderr)
        return 2

    try:
        text = read_diagnostics(port_name, args.baud or transport.baud_rate,
                                args.settle, args.seconds)
    except Exception as exc:  # noqa: BLE001 - any open/read failure is equally fatal
        print(f"Could not talk to the board on {port_name}: {exc}", file=sys.stderr)
        print("Is the Arduino Serial Monitor open? COM ports are exclusive.",
              file=sys.stderr)
        return 2

    print(text)
    ok, message = verdict(text)
    print()
    print("=" * 70)
    print(f"  [{'PASS' if ok else 'FAIL'}]  {message}")
    print("=" * 70)
    if ok:
        print("The microphones are alive. Now check they hear the same room:")
        print("  python tools/mic_meter.py    (coherence must exceed 0.3)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
