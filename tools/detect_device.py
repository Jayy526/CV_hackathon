"""Phase A diagnostic: find the ESP32 microphone board.

Usage:
    python tools/detect_device.py

Run it once with the board unplugged and once with it plugged in; the port that
appears is the board. Bluetooth virtual COM ports are listed but never matched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heimdall.audio.config import load_audio_config  # noqa: E402
from heimdall.audio.device import list_serial_devices  # noqa: E402


def main() -> int:
    config = load_audio_config()
    devices = list_serial_devices(config)

    print("Heimdall audio - device detection")
    print(f"Configured sample rate : {config.sample_rate} Hz")
    print(f"Configured channels    : {config.num_channels}")
    print(f"Configured frame size  : {config.frame_size} samples "
          f"({config.frame_duration_s * 1000:.1f} ms, {config.bytes_per_frame} bytes)")
    print()

    if not devices:
        print("No serial ports found at all.")
        return 1

    print(f"{'PORT':<8} {'VID:PID':<10} {'ESP32?':<7} DESCRIPTION")
    print("-" * 72)
    for d in devices:
        flag = "YES" if d.is_candidate else "-"
        print(f"{d.port:<8} {d.vid_pid:<10} {flag:<7} {d.description}")

    candidates = [d for d in devices if d.is_candidate]
    print()
    if not candidates:
        print(f"ESP32 candidates: 0  (of {len(devices)} ports)")
        print("No ESP32-class board detected. Plug the board in and re-run.")
        return 2

    print(f"ESP32 candidates: {len(candidates)}")
    for d in candidates:
        print(f"  {d.port}  {d.vid_pid}  {d.match_label}")
        if d.serial_number:
            print(f"      serial number: {d.serial_number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
