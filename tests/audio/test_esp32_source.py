"""ESP32AudioSource against a fake serial port. No hardware, no COM port.

Section 14 measured the wire as LOSS-FREE BUT OCCASIONALLY CORRUPT, so these
tests are mostly about what happens when a packet is bad: it must be dropped
whole, counted, and never spliced over.
"""

import importlib.util
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from heimdall.audio import packets as packet_mod
from heimdall.audio.config import load_audio_config
from heimdall.audio.receiver import AudioReceiver
from heimdall.audio.sources import AudioSourceError, ESP32AudioSource

PACKET = 1040


@pytest.fixture(scope="module")
def config():
    return load_audio_config()


@pytest.fixture(scope="module")
def transport(config):
    return config.transport


class PortDied(Exception):
    """Stands in for serial.SerialException."""


class FakePort:
    """Hands out `data` a chunk at a time, then optionally dies or goes quiet."""

    def __init__(self, data, chunk=PACKET, die_after=None):
        self.data = data
        self.chunk = chunk
        self.die_after = die_after
        self.pos = 0
        self.closed = False
        self.buffer_reset = False

    def _check(self):
        if self.die_after is not None and self.pos >= self.die_after:
            raise PortDied("ClearCommError failed (Access is denied.)")

    @property
    def in_waiting(self):
        self._check()
        return max(0, len(self.data) - self.pos)

    def read(self, n):
        self._check()
        n = min(n, self.chunk)
        out = self.data[self.pos:self.pos + n]
        self.pos += len(out)
        return out

    def reset_input_buffer(self):
        self.buffer_reset = True

    def close(self):
        self.closed = True


def make_source(data, *, chunk=PACKET, die_after=None, stall=0.05, **kw):
    holder = {}

    def factory(port, baud, timeout):
        holder["port"] = FakePort(data, chunk=chunk, die_after=die_after)
        holder["baud"] = baud
        return holder["port"]

    source = ESP32AudioSource(
        port="COM_FAKE", serial_factory=factory,
        settle_seconds=0.0, stall_timeout=stall, **kw,
    )
    source._holder = holder
    return source


def payload_for(ch0, ch1, samples=256):
    """One packet payload with each channel held at a constant value."""
    frame = np.empty(samples * 2, dtype="<i2")
    frame[0::2] = ch0
    frame[1::2] = ch1
    return frame.tobytes()


def build(transport, count, start=0, payload=None):
    return b"".join(
        packet_mod.build_packet(transport, start + i, payload=payload)
        for i in range(count)
    )


def drain(source, limit=100):
    frames = []
    with source:
        while len(frames) < limit:
            frame = source.read_frame()
            if frame is None:
                break
            frames.append(frame)
    return frames


# --- the format contract ----------------------------------------------------

def test_the_sample_rate_is_16000_not_48000(transport):
    source = make_source(b"")
    assert source.sample_rate == 16000
    assert source.sample_rate == transport.transmit_sample_rate
    # The ESP32 still ACQUIRES at 48 kHz; that is not what arrives here.
    assert load_audio_config().sample_rate == 48000


def test_frames_carry_the_transmitted_rate_not_the_acquisition_rate(transport):
    frames = drain(make_source(build(transport, 4)))
    assert len(frames) == 1
    assert frames[0].sample_rate == 16000


def test_frame_shape_matches_the_configured_frame_size(transport, config):
    frames = drain(make_source(build(transport, 8)))
    assert len(frames) == 2
    for frame in frames:
        assert frame.samples.shape == (config.frame_size, 2)
        assert frame.samples.dtype == np.float32


def test_four_packets_make_one_frame(transport, config):
    assert config.frame_size // transport.samples_per_packet == 4
    assert len(drain(make_source(build(transport, 3)))) == 0     # not enough
    assert len(drain(make_source(build(transport, 4)))) == 1


def test_channels_are_deinterleaved_the_right_way_round(transport):
    data = build(transport, 4, payload=payload_for(ch0=1000, ch1=-2000))
    (frame,) = drain(make_source(data))
    assert np.allclose(frame.channel(0), 1000 / 32768.0)
    assert np.allclose(frame.channel(1), -2000 / 32768.0)


def test_int16_is_scaled_into_the_float_range(transport):
    data = build(transport, 4, payload=payload_for(ch0=16384, ch1=-32768))
    (frame,) = drain(make_source(data))
    assert np.allclose(frame.channel(0), 0.5)
    assert np.allclose(frame.channel(1), -1.0)
    assert frame.samples.min() >= -1.0 and frame.samples.max() <= 1.0


def test_a_frame_size_that_is_not_whole_packets_is_refused(config):
    import dataclasses

    bad = dataclasses.replace(config, frame_size=1000)      # 1000 / 256 is not whole
    with pytest.raises(AudioSourceError) as excinfo:
        ESP32AudioSource(port="COM_FAKE", config=bad, serial_factory=lambda *a: None)
    assert "whole number of packets" in str(excinfo.value)


# --- the clean stream -------------------------------------------------------

def test_a_clean_stream_yields_every_frame_with_nothing_dropped(transport):
    source = make_source(build(transport, 40))
    frames = drain(source)

    assert len(frames) == 10
    assert source.packets_dropped == 0
    assert source.stats.frames_abandoned == 0
    assert source.stats.discontinuities == 0
    assert source.packet_counters.sequence_gaps == 0


def test_timestamps_advance_by_the_frame_duration(transport):
    frames = drain(make_source(build(transport, 12)))
    assert [round(f.timestamp, 6) for f in frames] == [0.0, 0.064, 0.128]
    assert frames[0].duration == pytest.approx(0.064)


def test_frame_indices_are_monotonic(transport):
    frames = drain(make_source(build(transport, 20)))
    assert [f.frame_index for f in frames] == list(range(5))


def test_a_stream_that_does_not_start_at_sequence_zero_is_anchored(transport):
    frames = drain(make_source(build(transport, 8, start=9000)))
    assert frames[0].timestamp == 0.0        # relative to the first packet seen
    assert frames[1].timestamp == pytest.approx(0.064)


def test_reads_that_do_not_land_on_packet_boundaries_still_work(transport):
    for chunk in (1, 7, 333, 1039, 1041, 65536):
        frames = drain(make_source(build(transport, 8), chunk=chunk))
        assert len(frames) == 2, chunk


def test_lead_in_garbage_from_the_dtr_reset_is_absorbed(transport):
    data = b"\x11\x22\x33" * 400 + build(transport, 8)
    source = make_source(data)
    frames = drain(source)

    assert len(frames) == 2
    assert source.packet_counters.lead_in_bytes == 1200
    assert source.packets_dropped == 0


# --- corruption: drop whole, count, never splice ----------------------------

def test_a_corrupt_payload_is_dropped_and_counted_not_passed_on(transport):
    data = bytearray(build(transport, 8))
    data[2 * PACKET + 16 + 5] ^= 0xFF               # inside packet 2's payload
    source = make_source(bytes(data))
    frames = drain(source)

    assert source.packets_dropped_payload_crc == 1
    assert source.packets_dropped == 1
    # Packets 0,1 were pending when 3 arrived out of order -> discarded.
    assert source.stats.frames_abandoned == 1
    assert len(frames) == 1
    assert frames[0].timestamp == pytest.approx(3 * 256 / 16000)


def test_a_corrupt_magic_is_dropped_and_counted(transport):
    data = bytearray(build(transport, 8))
    data[2 * PACKET] ^= 0x01                        # flip a bit inside A5 5A
    source = make_source(bytes(data))
    drain(source)

    counters = source.packet_counters
    assert counters.stray_bytes == PACKET           # the whole packet discarded
    assert counters.resyncs >= 1
    assert counters.header_crc_failures == 0        # magic failed before the CRC
    assert source.stats.discontinuities == 1


def test_the_discarded_total_is_the_same_however_the_reads_land(transport):
    # The resync COUNT depends on where chunk boundaries fall - a discard can be
    # split across several - but the bytes discarded always sum to one packet.
    # Same reason the byte ledger survives a false A5 5A inside a payload.
    totals = set()
    for chunk in (1, 64, 1040, 4096, 1 << 20):
        data = bytearray(build(transport, 8))
        data[2 * PACKET] ^= 0x01
        source = make_source(bytes(data), chunk=chunk)
        drain(source)
        totals.add(source.packet_counters.stray_bytes)
    assert totals == {PACKET}


def test_a_corrupt_header_crc_is_counted_separately_from_the_payload(transport):
    data = bytearray(build(transport, 8))
    data[2 * PACKET + 4] ^= 0xFF                    # packet 2's sequence field
    source = make_source(bytes(data))
    drain(source)

    assert source.packets_dropped_header_crc == 1
    assert source.packets_dropped_payload_crc == 0
    assert source.packets_dropped == 1


def test_header_and_payload_failures_are_never_conflated(transport):
    data = bytearray(build(transport, 20))
    data[2 * PACKET + 4] ^= 0xFF                    # header
    data[9 * PACKET + 16] ^= 0xFF                   # payload
    source = make_source(bytes(data))
    drain(source)

    assert source.packets_dropped_header_crc == 1
    assert source.packets_dropped_payload_crc == 1
    d = source.diagnostics()
    assert d["packets_dropped_header_crc"] == 1
    assert d["packets_dropped_payload_crc"] == 1
    assert d["packets_dropped_total"] == 2


# --- THE gap policy: discard the partial frame, never splice ----------------

def test_a_mid_frame_gap_discards_the_partial_frame(transport):
    # Packets 0,1,2 arrive; 3 never does; 4..7 follow. A splice would emit a
    # frame of 0,1,2,4 - 16 ms of missing time hidden inside one frame.
    data = build(transport, 3) + build(transport, 4, start=4)
    source = make_source(data)
    frames = drain(source)

    assert source.stats.frames_abandoned == 1
    assert source.stats.discontinuities == 1
    assert len(frames) == 1
    assert frames[0].timestamp == pytest.approx(4 * 256 / 16000)


def test_the_discarded_audio_is_not_present_in_any_emitted_frame(transport):
    # Mark the pre-gap packets distinctly; none of it may survive into a frame.
    before = b"".join(packet_mod.build_packet(transport, i,
                                              payload=payload_for(9999, 9999))
                      for i in range(3))
    after = b"".join(packet_mod.build_packet(transport, i,
                                             payload=payload_for(100, 200))
                     for i in range(4, 8))
    (frame,) = drain(make_source(before + after))

    assert np.allclose(frame.channel(0), 100 / 32768.0)
    assert not np.any(np.isclose(frame.samples, 9999 / 32768.0))


def test_a_gap_is_never_zero_filled(transport):
    data = build(transport, 3, payload=payload_for(500, 500)) + \
           build(transport, 4, start=4, payload=payload_for(500, 500))
    (frame,) = drain(make_source(data))
    # A zero-filled gap would put a step to silence inside the frame: a
    # broadband transient correlated across both channels at zero lag.
    assert np.all(frame.samples != 0.0)


def test_a_gap_between_frames_costs_nothing(transport):
    # Gap lands exactly on a frame boundary: nothing is part-assembled.
    data = build(transport, 4) + build(transport, 4, start=5)
    source = make_source(data)
    frames = drain(source)

    assert len(frames) == 2
    assert source.stats.frames_abandoned == 0
    assert source.stats.discontinuities == 1
    assert frames[1].timestamp == pytest.approx(5 * 256 / 16000)


def test_the_timestamp_jump_makes_the_lost_time_visible(transport):
    data = build(transport, 4) + build(transport, 4, start=8)
    frames = drain(make_source(data))
    # 4 packets skipped = 64 ms, so the second frame starts 128 ms in, not 64.
    assert frames[1].timestamp - frames[0].timestamp == pytest.approx(0.128)


# --- the port misbehaving ---------------------------------------------------

def test_a_dropout_mid_stream_raises_rather_than_truncating_silently(transport):
    source = make_source(build(transport, 40), die_after=10 * PACKET)
    source.start()
    with pytest.raises(AudioSourceError) as excinfo:
        for _ in range(20):
            source.read_frame()
    assert "serial read failed" in str(excinfo.value)
    source.stop()


def test_a_quiet_port_reports_end_of_stream_rather_than_hanging(transport):
    source = make_source(build(transport, 4))
    with source:
        assert source.read_frame() is not None
        began = time.monotonic()
        assert source.read_frame() is None
        assert time.monotonic() - began < 2.0


def test_start_without_a_port_says_how_to_find_one(config):
    import dataclasses

    no_port = dataclasses.replace(
        config, transport=dataclasses.replace(config.transport, port=None))
    source = ESP32AudioSource(config=no_port, serial_factory=lambda *a: None)
    with pytest.raises(AudioSourceError) as excinfo:
        source.start()
    assert "detect_device.py" in str(excinfo.value)


def test_a_port_that_will_not_open_is_reported_clearly():
    def factory(*args):
        raise OSError("Access is denied.")

    source = ESP32AudioSource(port="COM_FAKE", serial_factory=factory)
    with pytest.raises(AudioSourceError) as excinfo:
        source.start()
    assert "Serial Monitor" in str(excinfo.value)


def test_read_frame_before_start_is_refused(transport):
    with pytest.raises(AudioSourceError):
        make_source(build(transport, 4)).read_frame()


def test_stop_closes_the_port(transport):
    source = make_source(build(transport, 4))
    source.start()
    source.stop()
    assert source._holder["port"].closed is True
    assert source.is_running is False


def test_the_port_is_opened_at_the_configured_baud(transport):
    source = make_source(build(transport, 4))
    source.start()
    assert source._holder["baud"] == transport.baud_rate == 921600
    source.stop()


def test_the_input_buffer_is_flushed_after_the_reset_settles(transport):
    source = make_source(build(transport, 4))
    source.start()
    assert source._holder["port"].buffer_reset is True
    source.stop()


def test_settle_defaults_to_the_same_four_seconds_as_the_tool():
    # The CP2102 must finish re-enumerating after the DTR reset on open.
    assert ESP32AudioSource(port="COM_FAKE").settle_seconds == 4.0


# --- backpressure stays the receiver's job ----------------------------------

def test_backpressure_drops_oldest_via_the_existing_receiver(transport):
    source = make_source(build(transport, 120), stall=0.05)
    receiver = AudioReceiver(source, queue_size=1)
    receiver.start()

    deadline = time.monotonic() + 10.0
    while receiver.stats.frames_received < 30 and time.monotonic() < deadline:
        time.sleep(0.01)
    receiver.stop()

    assert receiver.stats.frames_received == 30
    # Nothing consumed while it ran, and the queue holds one: the rest dropped.
    assert receiver.stats.frames_dropped >= 28


def test_the_source_does_not_reimplement_queueing(transport):
    source = make_source(b"")
    assert not hasattr(source, "_queue")
    assert not isinstance(getattr(source, "_thread", None), threading.Thread)


# --- one decoder, not two ---------------------------------------------------

def test_the_source_uses_the_same_decoder_the_hardware_runs_proved():
    tools = Path(__file__).resolve().parents[2] / "tools"
    spec = importlib.util.spec_from_file_location(
        "tool_vss", tools / "verify_serial_stream.py")
    tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)

    # If these ever became separate implementations, three 300 s hardware runs
    # would stop being evidence about what the receiver actually does.
    assert tool.StreamVerifier is packet_mod.PacketDecoder
    assert tool.crc16_ccitt_false is packet_mod.crc16_ccitt_false
    assert tool.build_packet is packet_mod.build_packet
