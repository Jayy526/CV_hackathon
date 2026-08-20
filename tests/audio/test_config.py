"""Phase A tests: configuration is loadable and sane (sample rate, channels)."""

import textwrap

import pytest

from heimdall.audio.config import DeviceId, load_audio_config


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
