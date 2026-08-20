"""Configuration loading for the audio module.

Nothing in the audio pipeline should hard-code a sample rate, channel count or
device id. Everything comes from config/audio.yaml via load_audio_config().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "audio.yaml"


@dataclass(frozen=True)
class DeviceId:
    """A USB VID/PID pattern that identifies an ESP32-class board."""

    vid: str
    pid: str | None
    label: str

    def matches(self, vid: int | None, pid: int | None) -> bool:
        if vid is None:
            return False
        if int(self.vid, 16) != vid:
            return False
        if self.pid is None:
            return True
        return pid is not None and int(self.pid, 16) == pid


@dataclass(frozen=True)
class SerialConfig:
    port: str | None = None
    baudrate: int = 921600
    known_device_ids: tuple[DeviceId, ...] = ()


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 48000
    sample_width_bits: int = 32
    num_channels: int = 2
    frame_size: int = 1024
    serial: SerialConfig = field(default_factory=SerialConfig)

    @property
    def frame_duration_s(self) -> float:
        return self.frame_size / self.sample_rate

    @property
    def bytes_per_frame(self) -> int:
        return self.frame_size * self.num_channels * (self.sample_width_bits // 8)


def load_audio_config(path: str | Path | None = None) -> AudioConfig:
    """Load config/audio.yaml (or an explicit path) into an AudioConfig."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    audio = raw.get("audio", {})
    serial_raw = raw.get("serial", {})
    devices = tuple(
        DeviceId(vid=d["vid"], pid=d.get("pid"), label=d.get("label", ""))
        for d in serial_raw.get("known_device_ids", [])
    )

    return AudioConfig(
        sample_rate=int(audio.get("sample_rate", 48000)),
        sample_width_bits=int(audio.get("sample_width_bits", 32)),
        num_channels=int(audio.get("num_channels", 2)),
        frame_size=int(audio.get("frame_size", 1024)),
        serial=SerialConfig(
            port=serial_raw.get("port"),
            baudrate=int(serial_raw.get("baudrate", 921600)),
            known_device_ids=devices,
        ),
    )
