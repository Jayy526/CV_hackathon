"""Serial device discovery for the ESP32 microphone board (Phase A).

Enumerates serial ports and flags the ones whose USB VID/PID match a known
ESP32-class device from config/audio.yaml. Works with no board attached: it
simply reports zero candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

from serial.tools import list_ports

from heimdall.audio.config import AudioConfig, load_audio_config


@dataclass(frozen=True)
class SerialDevice:
    port: str
    description: str
    hwid: str
    vid: int | None
    pid: int | None
    serial_number: str | None
    is_candidate: bool
    match_label: str | None

    @property
    def vid_pid(self) -> str:
        if self.vid is None or self.pid is None:
            return "-"
        return f"{self.vid:04X}:{self.pid:04X}"


def list_serial_devices(config: AudioConfig | None = None) -> list[SerialDevice]:
    """Return every serial port on the system, marking ESP32 candidates."""
    config = config or load_audio_config()
    devices: list[SerialDevice] = []

    for port in sorted(list_ports.comports(), key=lambda p: p.device):
        label = None
        for known in config.serial.known_device_ids:
            if known.matches(port.vid, port.pid):
                label = known.label
                break
        devices.append(
            SerialDevice(
                port=port.device,
                description=port.description or "",
                hwid=port.hwid or "",
                vid=port.vid,
                pid=port.pid,
                serial_number=port.serial_number,
                is_candidate=label is not None,
                match_label=label,
            )
        )

    return devices


def find_esp32_devices(config: AudioConfig | None = None) -> list[SerialDevice]:
    """Return only the ports that look like an ESP32 board."""
    return [d for d in list_serial_devices(config) if d.is_candidate]
