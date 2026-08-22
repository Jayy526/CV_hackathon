"""Phase A tests: configuration is loadable and sane (sample rate, channels)."""

import textwrap

import pytest

from heimdall.audio.config import (
    HEADER_BYTES,
    MAX_LINK_UTILISATION,
    SUPPORTED_BAUD_RATES,
    DeviceId,
    TransportConfig,
    WireFormat,
    load_audio_config,
)


@pytest.fixture(scope="module")
def config():
    return load_audio_config()


def test_default_sample_rate_is_48k(config):
    assert config.sample_rate == 48000


def test_sample_rate_is_a_supported_audio_rate(config):
    assert config.sample_rate in (8000, 16000, 22050, 32000, 44100, 48000, 96000)


def test_channel_count_is_two_initially(config):
    assert config.num_channels == 2


def test_sample_width_matches_inmp441_i2s_slot(config):
    # INMP441 emits 24-bit data inside a 32-bit I2S slot.
    assert config.sample_width_bits in (16, 24, 32)


def test_frame_size_derived_values(config):
    assert config.frame_size > 0
    expected_bytes = config.frame_size * config.num_channels * (config.sample_width_bits // 8)
    assert config.bytes_per_frame == expected_bytes
    assert config.frame_duration_s == pytest.approx(config.frame_size / config.sample_rate)


def test_sample_rate_is_configurable(tmp_path):
    """Changing the YAML must change the loaded rate - nothing is hard-coded."""
    path = tmp_path / "audio.yaml"
    path.write_text(
        textwrap.dedent(
            """
            audio:
              sample_rate: 16000
              num_channels: 4
              frame_size: 512
              sample_width_bits: 32
            serial:
              baudrate: 115200
            """
        ),
        encoding="utf-8",
    )
    cfg = load_audio_config(path)
    assert cfg.sample_rate == 16000
    assert cfg.num_channels == 4
    assert cfg.frame_size == 512
    assert cfg.serial.baudrate == 115200


def test_known_device_ids_are_loaded(config):
    labels = [d.label for d in config.serial.known_device_ids]
    assert any("Espressif" in label for label in labels)


def test_device_id_matching():
    espressif = DeviceId(vid="303A", pid=None, label="Espressif")
    cp210x = DeviceId(vid="10C4", pid="EA60", label="CP210x")

    assert espressif.matches(0x303A, 0x1001)  # wildcard pid
    assert not espressif.matches(0x10C4, 0x1001)
    assert cp210x.matches(0x10C4, 0xEA60)
    assert not cp210x.matches(0x10C4, 0x0001)  # wrong pid
    assert not espressif.matches(None, None)  # bluetooth port, no vid


# --- USB serial transport: the wire contract --------------------------------
#
# Offline USB only, and CONTINUOUS rather than event-triggered. Wi-Fi/TCP and
# the earlier burst design were both dropped deliberately; nothing here may
# reintroduce a network concept or an event trigger.


def test_transport_is_offline_usb_serial(config):
    t = config.transport
    assert t.type == "serial"
    assert t.baud_rate == 921600
    # The port is a real machine-specific device name (COM9 here), not null:
    # tools fall back to it so no command has to carry a hand-written port.
    # What must hold is that it is a SERIAL port, never a network address.
    assert t.port is None or isinstance(t.port, str)
    assert t.port is None or not t.port.count(".") == 3


def test_transport_config_has_no_network_concept(config):
    """Guards the Wi-Fi/TCP removal: a host field must not creep back in."""
    assert not hasattr(config.transport, "host")
    assert not hasattr(config, "network")


def test_transport_is_continuous_not_event_triggered(config):
    """Guards the burst-design removal. A continuous stream has no trigger and
    no pre-roll; if these come back, the transport has silently changed shape."""
    t = config.transport
    assert not hasattr(t, "pre_trigger_samples")
    assert not hasattr(t, "trigger_threshold")
    assert not hasattr(t, "burst_samples")
    # Packets arrive back to back, so packet rate x packet duration is exactly 1.
    assert t.packets_per_second * t.packet_duration_seconds() == pytest.approx(1.0)


def test_device_detection_config_stays_separate_from_the_data_link(config):
    """`serial:` identifies the board by USB VID/PID; `transport:` carries the
    audio. Two different jobs, deliberately not merged."""
    assert config.serial.known_device_ids
    assert not hasattr(config.transport, "known_device_ids")
    assert config.serial.baudrate == 921600  # unchanged by the rename


def test_wire_format_is_interleaved_int16_little_endian(config):
    wire = config.transport.wire_format
    assert wire.sample_format == "int16"
    assert wire.byte_order == "little"
    assert wire.channel_layout == "interleaved"
    assert wire.num_channels == config.num_channels


def test_wire_width_is_independent_of_the_i2s_slot_width(config):
    """The ESP32 acquires 32-bit slots but narrows to int16 before sending."""
    assert config.sample_width_bits == 32
    assert config.transport.wire_format.bytes_per_sample == 2


def test_wire_format_numpy_dtype_carries_byte_order():
    assert WireFormat().numpy_dtype == "<i2"
    assert WireFormat(byte_order="big").numpy_dtype == ">i2"


def test_wire_format_rejects_formats_the_receiver_cannot_decode():
    with pytest.raises(ValueError):
        WireFormat(sample_format="int32")
    with pytest.raises(ValueError):
        WireFormat(byte_order="middle")
    with pytest.raises(ValueError):
        WireFormat(channel_layout="planar")
    with pytest.raises(ValueError):
        WireFormat(num_channels=0)


# --- decimation: 48 kHz in, 16 kHz out --------------------------------------


def test_acquisition_rate_is_unchanged_by_the_transport(config):
    """The ESP32 still acquires at 48 kHz; only the transmitted rate is lowered."""
    assert config.sample_rate == 48000
    assert config.transport.acquisition_sample_rate == 48000
    assert config.transport.transmit_sample_rate == 16000


def test_decimation_factor_is_three():
    t = TransportConfig(acquisition_sample_rate=48000, transmit_sample_rate=16000)
    assert t.decimation_factor == 3


def test_decimation_factor_is_derived_not_configured():
    """Changing the rates changes the factor; it is never independently set."""
    assert TransportConfig(transmit_sample_rate=8000).decimation_factor == 6
    assert (
        TransportConfig(
            acquisition_sample_rate=16000, transmit_sample_rate=16000
        ).decimation_factor
        == 1
    )
    assert (
        TransportConfig(
            acquisition_sample_rate=44100, transmit_sample_rate=11025, baud_rate=921600
        ).decimation_factor
        == 4
    )


def test_non_integer_decimation_ratios_are_rejected():
    """A fractional ratio needs a resampler, not the integer decimator the
    firmware will implement. Fail loudly instead of rounding."""
    with pytest.raises(ValueError, match="exact multiple"):
        TransportConfig(acquisition_sample_rate=48000, transmit_sample_rate=22050)
    with pytest.raises(ValueError, match="exact multiple"):
        TransportConfig(acquisition_sample_rate=48000, transmit_sample_rate=44100)


def test_transmit_rate_cannot_exceed_acquisition_rate():
    with pytest.raises(ValueError, match="decimates, it does not interpolate"):
        TransportConfig(
            acquisition_sample_rate=16000, transmit_sample_rate=48000, baud_rate=2000000
        )


def test_rates_must_be_positive():
    with pytest.raises(ValueError):
        TransportConfig(acquisition_sample_rate=0)
    with pytest.raises(ValueError):
        TransportConfig(transmit_sample_rate=0)


# --- continuous bandwidth ---------------------------------------------------


def test_packet_size_derives_from_the_wire_format(config):
    t = config.transport
    assert t.samples_per_packet == 256
    assert t.wire_format.bytes_per_sample_frame == 2 * t.wire_format.num_channels
    assert t.bytes_per_packet == 256 * 4 == 1024
    assert t.wire_bytes_per_packet == t.bytes_per_packet + HEADER_BYTES


def test_framing_overhead_is_accounted_for_not_ignored(config):
    """The sustainability check must charge for headers, not just audio bytes."""
    t = config.transport
    assert t.payload_bytes_per_second == 64000
    assert t.wire_bytes_per_second == pytest.approx(65000.0)
    assert t.wire_bytes_per_second > t.payload_bytes_per_second
    assert t.framing_overhead == pytest.approx(HEADER_BYTES / 1040.0, rel=1e-6)
    assert t.framing_overhead < 0.02


def test_921600_baud_carries_16k_stereo_int16(config):
    """The chosen configuration, with the numbers that justify it."""
    t = config.transport
    assert t.max_bytes_per_second == 92160
    assert t.link_utilisation == pytest.approx(65000.0 / 92160.0)
    assert t.link_utilisation < MAX_LINK_UTILISATION
    assert t.sustainable()


def test_115200_baud_cannot_carry_16k_stereo_int16():
    """115200 is fine for hardware diagnostics and useless for continuous audio:
    11,520 B/s against the 64,000 B/s the stream needs."""
    with pytest.raises(ValueError, match="does not fit the link"):
        TransportConfig(baud_rate=115200, transmit_sample_rate=16000)


def test_48k_stereo_is_rejected_on_every_supported_baud_rate():
    """The measurement that ruled out transmitting the acquisition rate."""
    for baud in (115200, 230400, 460800, 921600):
        with pytest.raises(ValueError, match="does not fit the link"):
            TransportConfig(baud_rate=baud, transmit_sample_rate=48000)


def test_8k_stereo_is_the_460800_fallback():
    """Documented fallback if 921600 proves unreliable on the real board."""
    t = TransportConfig(baud_rate=460800, transmit_sample_rate=8000)
    assert t.decimation_factor == 6
    assert t.sustainable()
    assert t.link_utilisation < MAX_LINK_UTILISATION
    # ...but 16 kHz does not fit at 460800.
    with pytest.raises(ValueError, match="does not fit the link"):
        TransportConfig(baud_rate=460800, transmit_sample_rate=16000)


def test_unsustainable_config_is_rejected_never_clamped():
    """The error must name both sides of the comparison so the fix is obvious,
    and the configuration must not be quietly rewritten to something that fits."""
    with pytest.raises(ValueError) as excinfo:
        TransportConfig(baud_rate=230400, transmit_sample_rate=16000)
    message = str(excinfo.value)
    assert "230400" in message and "23040" in message
    assert "transmit_sample_rate" in message and "baud_rate" in message


def test_continuous_cost_helpers_explain_the_rejection(config):
    t = config.transport
    assert t.continuous_bytes_per_second(48000) == 192000
    assert t.continuous_bytes_per_second(16000) == 64000
    assert not t.supports_continuous_stream(48000)
    assert t.supports_continuous_stream(16000)


def test_derived_rates_reject_a_nonsense_sample_rate(config):
    with pytest.raises(ValueError):
        config.transport.continuous_bytes_per_second(0)
    with pytest.raises(ValueError):
        config.transport.continuous_bytes_per_second(-1)


def test_packet_duration_uses_the_transmitted_rate(config):
    """256 samples at 16 kHz is 16 ms of audio, not 5.3 ms at 48 kHz."""
    t = config.transport
    assert t.packet_duration_seconds() == pytest.approx(256 / 16000.0)
    assert t.packet_duration_seconds() == pytest.approx(0.016)


def test_transport_rejects_impossible_values():
    with pytest.raises(ValueError):
        TransportConfig(type="tcp")           # Wi-Fi/TCP is gone for good
    with pytest.raises(ValueError):
        TransportConfig(baud_rate=250000)     # not a CP210x rate
    with pytest.raises(ValueError):
        TransportConfig(samples_per_packet=0)


def test_supported_baud_rates_include_the_usable_cp210x_set():
    assert 921600 in SUPPORTED_BAUD_RATES
    assert 115200 in SUPPORTED_BAUD_RATES
    assert 250000 not in SUPPORTED_BAUD_RATES


# --- configurability --------------------------------------------------------


def test_transport_settings_are_configurable(tmp_path):
    """Nothing about the transport is hard-coded either."""
    path = tmp_path / "audio.yaml"
    path.write_text(
        textwrap.dedent(
            """
            audio:
              sample_rate: 48000
              num_channels: 2
            transport:
              type: serial
              port: COM9
              baud_rate: 460800
              transmit_sample_rate: 8000
              samples_per_packet: 128
              wire_format:
                sample_format: int16
                byte_order: little
                channel_layout: interleaved
                num_channels: 2
            """
        ),
        encoding="utf-8",
    )
    t = load_audio_config(path).transport
    assert t.port == "COM9"
    assert t.baud_rate == 460800
    assert t.transmit_sample_rate == 8000
    assert t.decimation_factor == 6
    assert t.samples_per_packet == 128
    assert t.bytes_per_packet == 128 * 4
    assert t.max_bytes_per_second == 46080
    assert t.sustainable()


def test_acquisition_rate_comes_from_the_audio_block(tmp_path):
    """The transport must not duplicate audio.sample_rate; it reads it."""
    path = tmp_path / "audio.yaml"
    path.write_text(
        textwrap.dedent(
            """
            audio:
              sample_rate: 32000
            transport:
              transmit_sample_rate: 16000
            """
        ),
        encoding="utf-8",
    )
    t = load_audio_config(path).transport
    assert t.acquisition_sample_rate == 32000
    assert t.decimation_factor == 2


def test_a_yaml_that_cannot_fit_the_link_fails_to_load(tmp_path):
    """A bad configuration must fail at load time, not at 3 a.m. on the wire."""
    path = tmp_path / "audio.yaml"
    path.write_text(
        textwrap.dedent(
            """
            audio:
              sample_rate: 48000
            transport:
              baud_rate: 115200
              transmit_sample_rate: 16000
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not fit the link"):
        load_audio_config(path)


def test_missing_transport_block_falls_back_to_defaults(tmp_path):
    path = tmp_path / "audio.yaml"
    path.write_text("audio:\n  sample_rate: 48000\n", encoding="utf-8")
    t = load_audio_config(path).transport
    assert t.type == "serial"
    assert t.baud_rate == 921600
    assert t.transmit_sample_rate == 16000
    assert t.decimation_factor == 3
    assert t.wire_format.sample_format == "int16"


def test_wire_channels_default_to_the_audio_channel_count(tmp_path):
    path = tmp_path / "audio.yaml"
    path.write_text(
        textwrap.dedent(
            """
            audio:
              num_channels: 4
            transport:
              type: serial
              transmit_sample_rate: 8000
            """
        ),
        encoding="utf-8",
    )
    t = load_audio_config(path).transport
    assert t.wire_format.num_channels == 4
    assert t.wire_format.bytes_per_sample_frame == 8
