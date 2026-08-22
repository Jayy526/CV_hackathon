"""Configuration loading for the audio module.

Nothing in the audio pipeline should hard-code a sample rate, channel count or
device id. Everything comes from config/audio.yaml via load_audio_config().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# One level shallower than the old heimdall.audio location. Optional: the
# package must be usable by someone who has the hardware and none of this
# repo's configuration, so an absent file falls back to the built defaults.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "audio.yaml"

# Standard rates a CP210x bridge divides its clock down to. Anything else is
# a typo or an unreliable overclock, and a wrong baud rate corrupts every
# byte silently rather than failing loudly.
SUPPORTED_BAUD_RATES = frozenset(
    {9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600, 1000000, 2000000}
)

# Bytes of binary packet header preceding each continuous PCM payload: magic,
# protocol version, sequence number, sample count, payload length, timestamp and
# CRC. Fixed size, so the receiver can resynchronise after a corrupt stretch by
# scanning for the magic without parsing anything variable-length.
HEADER_BYTES = 16

# Ceiling on how much of the serial link the continuous stream may occupy.
# Driving a UART near capacity drops bytes silently under USB scheduling jitter,
# so headroom is a correctness requirement, not a comfort margin.
MAX_LINK_UTILISATION = 0.85


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
class WireFormat:
    """The byte-level contract between the ESP32 firmware and the receiver.

    Both sides read this from config/audio.yaml; neither may hard-code it. The
    ESP32 acquires 32-bit I2S slots and narrows to int16 before transmitting,
    so the wire width is deliberately NOT AudioConfig.sample_width_bits.
    """

    sample_format: str = "int16"
    byte_order: str = "little"
    channel_layout: str = "interleaved"
    num_channels: int = 2

    def __post_init__(self) -> None:
        if self.sample_format != "int16":
            raise ValueError(
                "only int16 is supported on the wire, got %r" % (self.sample_format,)
            )
        if self.byte_order not in ("little", "big"):
            raise ValueError("byte_order must be 'little' or 'big', got %r" % (self.byte_order,))
        if self.channel_layout != "interleaved":
            raise ValueError(
                "only interleaved channels are supported, got %r" % (self.channel_layout,)
            )
        if self.num_channels < 1:
            raise ValueError("num_channels must be >= 1, got %r" % (self.num_channels,))

    @property
    def bytes_per_sample(self) -> int:
        return 2

    @property
    def bytes_per_sample_frame(self) -> int:
        """Bytes for one time instant across all channels."""
        return self.bytes_per_sample * self.num_channels

    @property
    def numpy_dtype(self) -> str:
        """numpy dtype string, byte order included: '<i2' or '>i2'."""
        return ("<" if self.byte_order == "little" else ">") + "i2"


@dataclass(frozen=True)
class TransportConfig:
    """How audio actually reaches the laptop: USB serial, offline, no network.

    The stream is CONTINUOUS, not event-triggered. Continuous 48 kHz stereo
    int16 is 192,000 B/s and an 8N1 link carries baud/10 bytes/sec - 92,160 B/s
    at 921600 baud - so the full acquisition rate cannot cross the wire. The
    ESP32 therefore keeps acquiring at 48 kHz and decimates by
    `decimation_factor` down to `transmit_sample_rate` before sending.

    The decimation MUST be anti-aliased. Measured through this repo's own DOA
    pipeline at 0.135 m spacing: a proper FIR decimation to 16 kHz costs 0.70
    degrees of mean bearing error, while dropping every third sample costs 6.1
    degrees mean and 30 degrees worst case. Aliasing destroys the inter-channel
    phase that GCC-PHAT depends on.

    `port` is the audio data link. It is deliberately NOT the same thing as
    SerialConfig, which exists only to identify the board by USB VID/PID.
    """

    type: str = "serial"
    port: str | None = None
    baud_rate: int = 921600
    acquisition_sample_rate: int = 48000
    transmit_sample_rate: int = 16000
    samples_per_packet: int = 256
    wire_format: WireFormat = field(default_factory=WireFormat)

    def __post_init__(self) -> None:
        if self.type != "serial":
            raise ValueError(
                "only 'serial' transport is supported (Wi-Fi/TCP was dropped "
                "deliberately), got %r" % (self.type,)
            )
        if self.baud_rate not in SUPPORTED_BAUD_RATES:
            raise ValueError(
                "baud_rate %r is not a rate the CP210x bridge supports; "
                "expected one of %r" % (self.baud_rate, sorted(SUPPORTED_BAUD_RATES))
            )
        if self.acquisition_sample_rate <= 0:
            raise ValueError(
                "acquisition_sample_rate must be positive, got %r"
                % (self.acquisition_sample_rate,)
            )
        if self.transmit_sample_rate <= 0:
            raise ValueError(
                "transmit_sample_rate must be positive, got %r"
                % (self.transmit_sample_rate,)
            )
        if self.transmit_sample_rate > self.acquisition_sample_rate:
            raise ValueError(
                "transmit_sample_rate (%d) cannot exceed acquisition_sample_rate "
                "(%d): the ESP32 decimates, it does not interpolate"
                % (self.transmit_sample_rate, self.acquisition_sample_rate)
            )
        if self.acquisition_sample_rate % self.transmit_sample_rate != 0:
            raise ValueError(
                "acquisition_sample_rate (%d) must be an exact multiple of "
                "transmit_sample_rate (%d); a fractional ratio needs a resampler, "
                "not an integer decimator"
                % (self.acquisition_sample_rate, self.transmit_sample_rate)
            )
        if self.samples_per_packet < 1:
            raise ValueError(
                "samples_per_packet must be >= 1, got %r" % (self.samples_per_packet,)
            )
        if not self.sustainable():
            raise ValueError(
                "continuous stream does not fit the link: %d Hz x %d ch x %d B needs "
                "%.0f B/s on the wire, which is %.1f%% of the %d B/s a %d-baud 8N1 "
                "link carries (limit %.0f%%). Lower transmit_sample_rate or raise "
                "baud_rate; this is not clamped silently."
                % (
                    self.transmit_sample_rate,
                    self.wire_format.num_channels,
                    self.wire_format.bytes_per_sample,
                    self.wire_bytes_per_second,
                    100.0 * self.link_utilisation,
                    self.max_bytes_per_second,
                    self.baud_rate,
                    100.0 * MAX_LINK_UTILISATION,
                )
            )

    # --- decimation ---------------------------------------------------------

    @property
    def decimation_factor(self) -> int:
        """Integer 48000 / 16000 = 3. Validated exact in __post_init__."""
        return self.acquisition_sample_rate // self.transmit_sample_rate

    # --- framing ------------------------------------------------------------

    @property
    def bytes_per_packet(self) -> int:
        """Audio payload of one packet, header excluded."""
        return self.samples_per_packet * self.wire_format.bytes_per_sample_frame

    @property
    def wire_bytes_per_packet(self) -> int:
        """What one packet actually costs on the wire, header included."""
        return self.bytes_per_packet + HEADER_BYTES

    @property
    def packets_per_second(self) -> float:
        return self.transmit_sample_rate / float(self.samples_per_packet)

    @property
    def framing_overhead(self) -> float:
        """Header cost as a fraction of the wire bytes, in [0, 1)."""
        return HEADER_BYTES / float(self.wire_bytes_per_packet)

    # --- bandwidth ----------------------------------------------------------

    @property
    def max_bytes_per_second(self) -> int:
        """Payload ceiling of the link: 8N1 costs 10 bits per byte."""
        return self.baud_rate // 10

    @property
    def payload_bytes_per_second(self) -> int:
        """Audio bytes per second, framing excluded."""
        return self.transmit_sample_rate * self.wire_format.bytes_per_sample_frame

    @property
    def wire_bytes_per_second(self) -> float:
        """Audio bytes per second, framing INCLUDED. This is what must fit."""
        return self.packets_per_second * self.wire_bytes_per_packet

    @property
    def link_utilisation(self) -> float:
        """Fraction of the link the continuous stream occupies."""
        return self.wire_bytes_per_second / float(self.max_bytes_per_second)

    def sustainable(self) -> bool:
        """True when the continuous stream fits the link with headroom.

        A serial link driven near 100% has no room for USB scheduling jitter or
        a momentarily late reader, and the failure mode is silently dropped
        bytes rather than an error - so the limit sits below raw capacity.
        """
        return self.link_utilisation <= MAX_LINK_UTILISATION

    def packet_duration_seconds(self) -> float:
        """How much audio time one packet contains, at the TRANSMITTED rate."""
        return self.samples_per_packet / float(self.transmit_sample_rate)

    def continuous_bytes_per_second(self, sample_rate: int) -> int:
        """What continuous streaming at `sample_rate` would cost on the wire.

        Used to show why the full 48 kHz acquisition rate cannot be transmitted.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive, got %r" % (sample_rate,))
        return sample_rate * self.wire_format.bytes_per_sample_frame

    def supports_continuous_stream(self, sample_rate: int) -> bool:
        """True only if the link could carry unbroken audio at `sample_rate`."""
        return (
            self.continuous_bytes_per_second(sample_rate)
            <= self.max_bytes_per_second * MAX_LINK_UTILISATION
        )


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 48000
    sample_width_bits: int = 32
    num_channels: int = 2
    frame_size: int = 1024
    serial: SerialConfig = field(default_factory=SerialConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)

    @property
    def frame_duration_s(self) -> float:
        return self.frame_size / self.sample_rate

    @property
    def bytes_per_frame(self) -> int:
        return self.frame_size * self.num_channels * (self.sample_width_bits // 8)


def default_transmit_sample_rate(
    acquisition_sample_rate: int,
    wire_format: "WireFormat",
    baud_rate: int,
) -> int:
    """Highest integer-divisor rate of the acquisition rate that fits the link.

    Used ONLY when the YAML omits transmit_sample_rate. An explicit value is
    never adjusted - it is validated and rejected if it does not fit, because
    silently rewriting a rate the user asked for would mean the config file and
    the wire disagree. A default is different: nobody asked for it, so it has to
    be one that works.

    With 48 kHz stereo at 921600 baud this picks 16000 (48000 and 24000 both
    overrun the link).
    """
    capacity = (baud_rate // 10) * MAX_LINK_UTILISATION
    per_sample = wire_format.bytes_per_sample_frame
    for factor in range(1, acquisition_sample_rate + 1):
        if acquisition_sample_rate % factor:
            continue
        rate = acquisition_sample_rate // factor
        # Charge for framing at the packet size the caller will end up using.
        if rate * per_sample * (1.0 + HEADER_BYTES / 1024.0) <= capacity:
            return rate
    return 1


def load_audio_config(path: str | Path | None = None) -> AudioConfig:
    """Load config/audio.yaml (or an explicit path) into an AudioConfig."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        # No config file: the dataclass defaults already encode the section 13
        # contract (48 kHz acquired, 16 kHz transmitted, 921600 baud, 256
        # samples per packet), so this is a working configuration, not a stub.
        return AudioConfig()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    audio = raw.get("audio", {})
    serial_raw = raw.get("serial", {})
    transport_raw = raw.get("transport", {}) or {}
    wire_raw = transport_raw.get("wire_format", {}) or {}
    devices = tuple(
        DeviceId(vid=d["vid"], pid=d.get("pid"), label=d.get("label", ""))
        for d in serial_raw.get("known_device_ids", [])
    )

    wire = WireFormat(
        sample_format=str(wire_raw.get("sample_format", "int16")),
        byte_order=str(wire_raw.get("byte_order", "little")),
        channel_layout=str(wire_raw.get("channel_layout", "interleaved")),
        num_channels=int(
            wire_raw.get("num_channels", audio.get("num_channels", 2))
        ),
    )
    transport = TransportConfig(
        type=str(transport_raw.get("type", "serial")),
        port=transport_raw.get("port"),
        baud_rate=int(transport_raw.get("baud_rate", 921600)),
        # The ESP32 keeps acquiring at audio.sample_rate; only the transmitted
        # rate is negotiable, so the acquisition rate is taken from the audio
        # block rather than duplicated in the transport block.
        acquisition_sample_rate=int(audio.get("sample_rate", 48000)),
        transmit_sample_rate=int(
            transport_raw.get("transmit_sample_rate")
            or default_transmit_sample_rate(
                int(audio.get("sample_rate", 48000)),
                wire,
                int(transport_raw.get("baud_rate", 921600)),
            )
        ),
        samples_per_packet=int(transport_raw.get("samples_per_packet", 256)),
        wire_format=wire,
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
        transport=transport,
    )
