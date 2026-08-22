"""tools/verify_serial_stream.py, driven headlessly with no hardware attached.

Every test here builds a byte stream in memory and feeds it to the verifier, so
the framing logic is exercised without a COM port, a board, or pyserial.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load_tool(name):
    spec = importlib.util.spec_from_file_location("tool_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verify():
    return load_tool("verify_serial_stream")


@pytest.fixture(scope="module")
def transport():
    from heimdall.audio.config import load_audio_config

    return load_audio_config().transport


def stream(verify, transport, count, start=0, **kw):
    return b"".join(
        verify.build_packet(transport, start + i, **kw) for i in range(count)
    )


def run(verify, transport, data, chunk=4096, elapsed=None):
    v = verify.StreamVerifier(transport)
    for i in range(0, len(data), chunk):
        v.feed(data[i:i + chunk])
    if elapsed is None:
        # Wall time a healthy link would have taken to deliver these bytes.
        elapsed = v.stats.total_bytes / transport.wire_bytes_per_second
    v.stats.elapsed = elapsed
    return v.stats


def states(verify, stats, transport):
    """Criterion name -> PASS / FAIL / N/A."""
    return {name: state for name, state, _ in verify.verdict(stats, transport)}


def all_pass(verify, stats, transport):
    # N/A is deliberately not a pass: `all(...)` over the raw strings would be
    # true for every state, which is exactly the bug this API change fixes.
    return all(state == "PASS" for state in states(verify, stats, transport).values())


# --- the CRC itself --------------------------------------------------------

def test_crc16_matches_the_ccitt_false_check_vector(verify):
    # The one published value that distinguishes CCITT-FALSE from the half-dozen
    # other things people call "CRC-16 CCITT".
    assert verify.crc16_ccitt_false(b"123456789") == 0x29B1


def test_crc16_of_empty_input_is_the_init_value(verify):
    assert verify.crc16_ccitt_false(b"") == 0xFFFF


def test_crc16_detects_a_single_flipped_bit(verify):
    assert verify.crc16_ccitt_false(b"\x00" * 64) != verify.crc16_ccitt_false(
        b"\x00" * 63 + b"\x01"
    )


# --- the packet layout, decoded independently of the builder ---------------

def test_built_packet_matches_the_section_13_layout(verify, transport):
    packet = verify.build_packet(transport, sequence=7, flags=0x03)

    assert len(packet) == 1040
    assert packet[0] == 0xA5 and packet[1] == 0x5A
    assert packet[2] == 1                                   # protocol version
    assert packet[3] == 0x03                                # flags
    assert int.from_bytes(packet[4:8], "little") == 7       # sequence
    assert int.from_bytes(packet[8:10], "little") == 256    # samples per channel
    assert int.from_bytes(packet[10:12], "little") == 1024  # payload length
    assert int.from_bytes(packet[12:14], "little") == verify.crc16_ccitt_false(packet[:12])
    assert int.from_bytes(packet[14:16], "little") == verify.crc16_ccitt_false(packet[16:])


def test_packet_size_comes_from_config_not_a_literal(verify, transport):
    assert len(verify.build_packet(transport, 0)) == transport.wire_bytes_per_packet


# --- the clean case --------------------------------------------------------

def test_a_clean_stream_passes_every_check(verify, transport):
    stats = run(verify, transport, stream(verify, transport, 200))

    assert stats.packets == 200
    assert stats.sequence_gaps == 0
    assert stats.missing_packets == 0
    assert stats.header_crc_failures == 0
    assert stats.payload_crc_failures == 0
    assert stats.stray_bytes == 0
    assert stats.resyncs == 0
    assert all_pass(verify, stats, transport)


@pytest.mark.parametrize("chunk", [1, 7, 1039, 1040, 1041, 4096])
def test_framing_is_independent_of_how_reads_land(verify, transport, chunk):
    # A serial read never respects packet boundaries; the odd sizes are the point.
    stats = run(verify, transport, stream(verify, transport, 20), chunk=chunk)
    assert stats.packets == 20
    assert stats.stray_bytes == 0


def test_payload_bytes_survive_intact(verify, transport):
    payload = bytes(range(256)) * 4
    stats = run(verify, transport, verify.build_packet(transport, 0, payload=payload))
    assert stats.packets == 1
    assert stats.payload_crc_failures == 0


# --- the DTR/RTS reset gotcha ---------------------------------------------

def test_lead_in_garbage_is_expected_and_not_charged_as_drift(verify, transport):
    # Opening the port resets the ESP32, so the first bytes are mid-packet.
    # Counting these as a framing error would fail every run on a healthy link.
    stats = run(verify, transport, b"\x11\x22\x33" * 400 + stream(verify, transport, 50))

    assert stats.packets == 50
    assert stats.lead_in_bytes == 1200
    assert stats.stray_bytes == 0
    assert stats.resyncs == 0
    assert all_pass(verify, stats, transport)


def test_a_stream_that_starts_mid_packet_still_locks_on(verify, transport):
    data = stream(verify, transport, 10)
    stats = run(verify, transport, data[500:])          # sliced into packet 0
    assert stats.packets == 9
    assert stats.stray_bytes == 0


# --- corruption ------------------------------------------------------------

def test_a_corrupt_payload_byte_is_caught_by_the_payload_crc(verify, transport):
    data = bytearray(stream(verify, transport, 5))
    data[1040 + 20] ^= 0xFF                              # inside packet 1's payload
    stats = run(verify, transport, bytes(data))

    assert stats.packets == 5                            # framing still intact
    assert stats.payload_crc_failures == 1
    assert stats.header_crc_failures == 0
    checks = states(verify, stats, transport)
    assert checks["payload CRC"] == "FAIL"
    assert checks["packet boundaries held"] == "PASS"


def test_a_corrupt_header_is_caught_and_charged_as_lost_framing(verify, transport):
    data = bytearray(stream(verify, transport, 5))
    data[1040 + 4] ^= 0xFF                               # packet 1's sequence field
    stats = run(verify, transport, bytes(data))

    assert stats.header_crc_failures == 1
    # The whole packet is unusable, so its bytes are discarded to the next magic.
    assert stats.stray_bytes == 1040
    assert stats.packets == 4
    assert not all_pass(verify, stats, transport)


def test_inserted_bytes_are_reported_as_boundary_drift(verify, transport):
    data = (verify.build_packet(transport, 0) + b"\x00" * 5
            + verify.build_packet(transport, 1)
            + verify.build_packet(transport, 2))
    stats = run(verify, transport, data)

    assert stats.packets == 3
    assert stats.resyncs == 1
    assert stats.stray_bytes == 5
    checks = states(verify, stats, transport)
    assert checks["packet boundaries held"] == "FAIL"


def test_pure_garbage_never_reports_a_packet(verify, transport):
    stats = run(verify, transport, bytes(50_000), elapsed=1.0)
    assert stats.packets == 0
    assert not all_pass(verify, stats, transport)


# --- sequence numbers ------------------------------------------------------

def test_a_dropped_packet_shows_up_as_a_sequence_gap(verify, transport):
    data = (verify.build_packet(transport, 0)
            + verify.build_packet(transport, 1)
            + verify.build_packet(transport, 3))       # 2 never arrived
    stats = run(verify, transport, data)

    assert stats.sequence_gaps == 1
    assert stats.missing_packets == 1


def test_a_long_burst_of_loss_is_counted_in_full(verify, transport):
    data = verify.build_packet(transport, 100) + verify.build_packet(transport, 400)
    stats = run(verify, transport, data)
    assert stats.missing_packets == 299


def test_the_uint32_sequence_wrap_is_not_a_gap(verify, transport):
    data = (verify.build_packet(transport, 0xFFFFFFFE)
            + verify.build_packet(transport, 0xFFFFFFFF)
            + verify.build_packet(transport, 0))
    stats = run(verify, transport, data)
    assert stats.sequence_gaps == 0


# --- the header flag bits --------------------------------------------------

def test_ring_overrun_flag_is_surfaced(verify, transport):
    stats = run(verify, transport, stream(verify, transport, 4, flags=0x01))
    assert stats.flag_overrun_packets == 4
    assert stats.flag_i2s_fail_packets == 0
    checks = states(verify, stats, transport)
    assert checks["no ring overruns (flag bit0)"] == "FAIL"


def test_i2s_failure_flag_is_surfaced(verify, transport):
    stats = run(verify, transport, stream(verify, transport, 4, flags=0x02))
    assert stats.flag_i2s_fail_packets == 4
    checks = states(verify, stats, transport)
    assert checks["no I2S failures (flag bit1)"] == "FAIL"


def test_both_flags_at_once(verify, transport):
    stats = run(verify, transport, stream(verify, transport, 3, flags=0x03))
    assert stats.flag_overrun_packets == 3
    assert stats.flag_i2s_fail_packets == 3


def test_an_unexpected_protocol_version_is_rejected_not_ignored(verify, transport):
    stats = run(verify, transport, stream(verify, transport, 2, version=9))
    assert stats.bad_version == 2
    checks = states(verify, stats, transport)
    assert checks["protocol version"] == "FAIL"


# --- throughput ------------------------------------------------------------

def test_throughput_passes_at_the_contracted_rate(verify, transport):
    assert transport.wire_bytes_per_second == 65000
    stats = run(verify, transport, stream(verify, transport, 625), elapsed=10.0)
    assert abs(stats.bytes_per_second - 65000) < 1.0
    checks = states(verify, stats, transport)
    assert checks["throughput"] == "PASS"


def test_a_slow_link_fails_throughput_even_with_perfect_framing(verify, transport):
    # Every packet valid, no gaps - but only half the audio arrived in real time.
    stats = run(verify, transport, stream(verify, transport, 625), elapsed=20.0)
    checks = states(verify, stats, transport)
    assert checks["throughput"] == "FAIL"
    assert checks["sequence continuity"] == "PASS"


def test_zero_elapsed_does_not_divide_by_zero(verify, transport):
    stats = run(verify, transport, stream(verify, transport, 1), elapsed=0.0)
    assert stats.bytes_per_second == 0.0
    assert stats.packets_per_second == 0.0
    verify.verdict(stats, transport)


# --- the printed verdict ---------------------------------------------------

def test_report_returns_true_and_says_pass_on_a_clean_stream(verify, transport, capsys):
    stats = run(verify, transport, stream(verify, transport, 625), elapsed=10.0)
    assert verify.report(stats, transport) is True
    assert "VERDICT: PASS" in capsys.readouterr().out


def test_report_returns_false_and_names_the_fallback_on_overruns(
    verify, transport, capsys
):
    stats = run(verify, transport, stream(verify, transport, 625, flags=0x01),
                elapsed=10.0)
    assert verify.report(stats, transport) is False
    out = capsys.readouterr().out
    assert "VERDICT: FAIL" in out
    assert "460800" in out                    # the section 11 fallback, named


def test_report_lists_every_criterion_individually(verify, transport, capsys):
    stats = run(verify, transport, stream(verify, transport, 10))
    verify.report(stats, transport)
    out = capsys.readouterr().out
    for name, _, _ in verify.verdict(stats, transport):
        assert name in out


# --- the tool refuses to guess --------------------------------------------

def test_the_port_falls_back_to_the_configured_one(verify):
    # --port used to be required, which is why a hand-written placeholder kept
    # leaking into commands. It now defaults to transport.port.
    assert verify.build_parser().parse_args([]).port is None


def test_no_port_anywhere_is_refused_with_instructions(verify, capsys, monkeypatch):
    import dataclasses

    real = verify.load_audio_config

    def unconfigured(*args, **kwargs):
        config = real(*args, **kwargs)
        return dataclasses.replace(
            config, transport=dataclasses.replace(config.transport, port=None))

    monkeypatch.setattr(verify, "load_audio_config", unconfigured)
    assert verify.main([]) == 2
    assert "detect_device.py" in capsys.readouterr().err


def test_a_port_that_cannot_be_opened_exits_nonzero_without_traceback(verify, capsys):
    pytest.importorskip("serial")
    assert verify.main(["--port", "COM_NOPE", "--duration", "0.1"]) == 2
    assert "Could not open" in capsys.readouterr().err


# --- a run that received nothing must not look green -----------------------
#
# Every criterion counts faults, so with no packets there is nothing to count.
# Reporting those as PASS made a totally failed run read as mostly passing.

PACKET_ONLY_CRITERIA = [
    "sequence continuity",
    "header CRC",
    "payload CRC",
    "protocol version",
    "packet boundaries held",
    "no ring overruns (flag bit0)",
    "no I2S failures (flag bit1)",
]


def test_a_zero_packet_run_produces_no_pass_lines(verify, transport, capsys):
    stats = run(verify, transport, bytes(100_000), elapsed=10.0)
    assert stats.packets == 0

    # The one criterion that can pass with nothing received is the only one
    # not judged from packets: whether the port itself stayed open. It earns
    # that PASS - it separates "nothing arrived" from "the port died".
    passing = {n for n, state in states(verify, stats, transport).items()
               if state == "PASS"}
    assert passing == {"port stayed open for the whole run"}

    assert verify.report(stats, transport) is False
    assert capsys.readouterr().out.count("[PASS") == 1


def test_zero_packets_makes_packet_only_criteria_not_applicable(verify, transport):
    stats = run(verify, transport, bytes(100_000), elapsed=10.0)
    checks = states(verify, stats, transport)
    for name in PACKET_ONLY_CRITERIA:
        assert checks[name] == "N/A", name
    # These two are judgeable with no packets at all, and both must fail.
    assert checks["received a stream at all"] == "FAIL"
    assert checks["throughput"] == "FAIL"


def test_not_applicable_is_never_counted_as_a_pass(verify, transport):
    stats = run(verify, transport, bytes(100_000), elapsed=10.0)
    assert not all_pass(verify, stats, transport)


def test_the_report_explains_what_na_means(verify, transport, capsys):
    stats = run(verify, transport, bytes(100_000), elapsed=10.0)
    verify.report(stats, transport)
    assert "Not a pass" in capsys.readouterr().out


def test_a_healthy_run_has_no_na_lines_at_all(verify, transport):
    stats = run(verify, transport, stream(verify, transport, 625), elapsed=10.0)
    assert "N/A" not in states(verify, stats, transport).values()


# --- zero packets must not be blamed on the link ---------------------------

def test_zero_packets_does_not_recommend_the_baud_fallback(verify, transport, capsys):
    # Following that advice from here means a DECIMATION 6 / FIR redesign to fix
    # what is usually a wrong #define or the wrong COM port.
    stats = run(verify, transport, bytes(100_000), elapsed=10.0)
    verify.report(stats, transport)
    out = capsys.readouterr().out

    assert "460800" not in out
    assert "NOT evidence against 921600" in out
    assert "START_IN_DIAG" in out
    assert "--port" in out


def test_a_gapped_stream_does_recommend_the_fallback(verify, transport, capsys):
    data = b"".join(verify.build_packet(transport, i) for i in range(0, 1000, 2))
    stats = run(verify, transport, data, elapsed=8.0)
    assert stats.sequence_gaps > 0
    verify.report(stats, transport)
    out = capsys.readouterr().out

    assert "460800" in out
    assert "START_IN_DIAG" not in out


# --- intact bytes, mismatched contract -------------------------------------

def mismatched(samples, payload_len):
    """A firmware build whose #defines disagree with config/audio.yaml."""
    return SimpleNamespace(samples_per_packet=samples, bytes_per_packet=payload_len)


def test_a_contract_mismatch_is_not_charged_as_header_corruption(verify, transport):
    wrong = mismatched(128, 512)
    data = b"".join(verify.build_packet(wrong, i) for i in range(20))
    stats = run(verify, transport, data)

    assert stats.contract_mismatches > 0
    assert stats.header_crc_failures == 0      # the bytes were never corrupt
    assert stats.packets == 0


def test_a_contract_mismatch_is_counted_before_lock(verify, transport):
    # It is counted precisely because the stream never locks: otherwise the one
    # fault that explains the silence would leave no trace at all.
    data = b"".join(verify.build_packet(mismatched(128, 512), i) for i in range(20))
    stats = run(verify, transport, data)
    assert stats.packets == 0
    assert stats.contract_mismatches >= 15


def test_the_mismatched_sizes_are_recorded_and_named(verify, transport, capsys):
    data = b"".join(verify.build_packet(mismatched(128, 512), i) for i in range(20))
    stats = run(verify, transport, data, elapsed=10.0)
    assert (128, 512) in stats.contract_seen

    checks = states(verify, stats, transport)
    assert checks["header contract matches config"] == "FAIL"
    verify.report(stats, transport)
    out = capsys.readouterr().out
    assert "128 samples / 512 B" in out
    assert "config says 256 / 1024 B" in out
    assert "THIS ONE" in out
    assert "460800" not in out


def test_the_contract_check_is_na_when_nothing_arrived_at_all(verify, transport):
    # No mismatch seen and no packets: unproven, not proven good.
    stats = run(verify, transport, bytes(100_000), elapsed=10.0)
    assert states(verify, stats, transport)["header contract matches config"] == "N/A"


def test_the_contract_check_passes_on_a_clean_stream(verify, transport):
    stats = run(verify, transport, stream(verify, transport, 100))
    assert stats.contract_mismatches == 0
    assert states(verify, stats, transport)["header contract matches config"] == "PASS"


# --- payload CRC failures are reported, not merely counted -----------------
#
# A hardware run produced exactly one payload CRC failure in 19.5 MB with zero
# gaps and zero framing drift. Counting alone cannot tell a single flipped bit
# on the wire from a sender emitting garbage, so each failure is logged.

def corrupt(verify, transport, count, bad_indices, byte_offset=20):
    data = bytearray(stream(verify, transport, count))
    for i in bad_indices:
        data[i * 1040 + 16 + byte_offset] ^= 0xFF
    return bytes(data)


def test_a_payload_failure_records_sequence_flags_and_both_crcs(verify, transport):
    data = bytearray(stream(verify, transport, 5, flags=0x02))
    data[2 * 1040 + 16] ^= 0xFF
    stats = run(verify, transport, bytes(data))

    assert stats.payload_crc_failures == 1
    (failure,) = stats.payload_failures
    assert failure.sequence == 2
    assert failure.flags == 0x02
    assert failure.computed != failure.expected
    assert failure.expected == verify.crc16_ccitt_false(bytes(1024))


def test_a_payload_failure_hexdumps_the_first_and_last_32_bytes(verify, transport):
    payload = bytes(range(256)) * 4
    packet = bytearray(verify.build_packet(transport, 0, payload=payload))
    packet[16] ^= 0xFF
    stats = run(verify, transport, bytes(packet) + verify.build_packet(transport, 1))

    (failure,) = stats.payload_failures
    assert len(failure.head) == 32
    assert len(failure.tail) == 32
    assert failure.head[0] == payload[0] ^ 0xFF        # the flipped byte itself
    assert failure.tail == payload[-32:]


def test_isolated_failures_are_counted_as_separate_bursts(verify, transport):
    stats = run(verify, transport, corrupt(verify, transport, 20, [2, 7, 15]))
    assert stats.payload_crc_failures == 3
    assert stats.payload_crc_bursts == 3
    assert stats.longest_payload_burst == 1
    assert not any(f.consecutive for f in stats.payload_failures)


def test_consecutive_failures_are_one_burst(verify, transport):
    stats = run(verify, transport, corrupt(verify, transport, 20, [5, 6, 7]))
    assert stats.payload_crc_failures == 3
    assert stats.payload_crc_bursts == 1
    assert stats.longest_payload_burst == 3
    assert [f.consecutive for f in stats.payload_failures] == [False, True, True]


def test_bursts_and_isolated_failures_are_told_apart_in_one_run(verify, transport):
    stats = run(verify, transport, corrupt(verify, transport, 30, [2, 10, 11, 12, 25]))
    assert stats.payload_crc_failures == 5
    assert stats.payload_crc_bursts == 3
    assert stats.longest_payload_burst == 3


def test_logged_failures_are_bounded_but_the_count_is_not(verify, transport):
    stats = run(verify, transport, corrupt(verify, transport, 60, range(50)))
    assert stats.payload_crc_failures == 50          # complete
    assert len(stats.payload_failures) == verify.MAX_LOGGED_FAILURES
    assert stats.payload_failures_unlogged == 50 - verify.MAX_LOGGED_FAILURES


def test_a_clean_run_logs_nothing_and_prints_no_failure_section(
    verify, transport, capsys
):
    stats = run(verify, transport, stream(verify, transport, 100))
    assert stats.payload_failures == []
    assert stats.payload_crc_bursts == 0
    verify.report(stats, transport)
    assert "payload CRC failures:" not in capsys.readouterr().out


def test_the_report_prints_the_failure_detail(verify, transport, capsys):
    stats = run(verify, transport, corrupt(verify, transport, 10, [3]))
    verify.report(stats, transport)
    out = capsys.readouterr().out

    assert "payload CRC failures: 1 in 1 burst(s)" in out
    assert "seq 3" in out
    assert "Every failure was isolated" in out
    assert "first 32 B:" in out and "last  32 B:" in out


def test_the_report_says_when_logging_was_capped(verify, transport, capsys):
    stats = run(verify, transport, corrupt(verify, transport, 60, range(50)))
    verify.report(stats, transport)
    out = capsys.readouterr().out
    assert "and 30 more, not logged" in out
    assert "The count above is complete." in out


# --- the port vanishing mid-run --------------------------------------------

class FakePortDied(Exception):
    """Stands in for serial.SerialException."""


class FakeSerial:
    """A port that yields `data` and then optionally dies."""

    def __init__(self, data, die_after=None, chunk=1040):
        self._data = data
        self._chunk = chunk
        self._pos = 0
        self._die_after = die_after
        self.closed = False

    @property
    def in_waiting(self):
        self._check()
        return max(0, len(self._data) - self._pos)

    def read(self, n):
        self._check()
        # A real read returns what the driver happens to have, never the whole
        # run at once; one packet per call keeps the dropout point exact.
        n = min(n, self._chunk)
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def _check(self):
        if self._die_after is not None and self._pos >= self._die_after:
            raise FakePortDied("ClearCommError failed (Access is denied.)")

    def reset_input_buffer(self):
        pass

    def close(self):
        self.closed = True


def install_fake_serial(monkeypatch, data, die_after=None):
    import types

    holder = {}

    def factory(port, baud, timeout=None):
        holder["port"] = FakeSerial(data, die_after)
        return holder["port"]

    fake = types.SimpleNamespace(Serial=factory, SerialException=FakePortDied)
    monkeypatch.setitem(sys.modules, "serial", fake)
    return holder


def test_a_port_that_dies_mid_run_still_reports_what_it_measured(
    verify, transport, monkeypatch, capsys
):
    data = stream(verify, transport, 100)
    install_fake_serial(monkeypatch, data, die_after=50 * 1040)

    assert verify.main(["--port", "COM_FAKE", "--duration", "60", "--settle", "0"]) == 1
    out, err = capsys.readouterr()

    assert "the port disappeared mid-run" in err
    assert "the port disappeared mid-run" in out       # named in the criteria
    assert "50 packets" in out                          # the measurement survived
    assert "Traceback" not in out and "Traceback" not in err


def test_the_dropout_criterion_passes_when_nothing_goes_wrong(verify, transport):
    stats = run(verify, transport, stream(verify, transport, 10))
    assert states(verify, stats, transport)["port stayed open for the whole run"] == "PASS"


def test_a_dropout_does_not_get_blamed_on_the_baud_rate(
    verify, transport, monkeypatch, capsys
):
    install_fake_serial(monkeypatch, stream(verify, transport, 100),
                        die_after=50 * 1040)
    verify.main(["--port", "COM_FAKE", "--duration", "60", "--settle", "0"])
    assert "460800" not in capsys.readouterr().out


def test_the_port_is_closed_even_when_it_died(verify, transport, monkeypatch):
    holder = install_fake_serial(monkeypatch, stream(verify, transport, 100),
                                 die_after=50 * 1040)
    verify.main(["--port", "COM_FAKE", "--duration", "60", "--settle", "0"])
    assert holder["port"].closed is True


# --- the settle default -----------------------------------------------------

def test_settle_defaults_to_four_seconds(verify):
    # 1.5 s was not enough for the CP2102 to re-enumerate after the DTR reset;
    # the first hardware attempt died with "Access is denied".
    assert verify.build_parser().parse_args(["--port", "X"]).settle == 4.0


def test_settle_is_still_overridable(verify):
    assert verify.build_parser().parse_args(["--port", "X", "--settle", "0"]).settle == 0.0


# --- byte conservation: corruption is not loss ------------------------------
#
# Three 300 s hardware runs reconciled to the byte. Run 2 showed a sequence gap
# and 1040 stray bytes, and the tool told the user to drop to 460800 - but the
# byte total was identical to the clean run's. Nothing was lost; a flipped bit
# in the magic made the resync discard a packet that had fully arrived.

def test_a_clean_stream_conserves_every_byte(verify, transport):
    stats = run(verify, transport, stream(verify, transport, 500))
    assert verify.sender_packets(stats) == 500
    assert verify.bytes_lost(stats, transport) == 0


def test_a_corrupted_magic_looks_like_a_gap_but_loses_no_bytes(verify, transport):
    # Exactly the run 2 event: flip one bit inside the A5 5A of packet 7.
    data = bytearray(stream(verify, transport, 20))
    data[7 * 1040] ^= 0x01
    stats = run(verify, transport, bytes(data))

    assert stats.packets == 19
    assert stats.sequence_gaps == 1                 # looks like a dropped packet
    assert stats.stray_bytes == 1040                # but the bytes are all here
    assert stats.header_crc_failures == 0           # magic failed before the CRC
    assert verify.bytes_lost(stats, transport) == 0
    assert states(verify, stats, transport)["no bytes lost"] == "PASS"


def test_a_genuinely_missing_packet_is_counted_as_lost_bytes(verify, transport):
    data = b"".join(verify.build_packet(transport, i) for i in [0, 1, 3, 4])
    stats = run(verify, transport, data)

    assert stats.sequence_gaps == 1
    assert stats.stray_bytes == 0
    assert verify.bytes_lost(stats, transport) == 1040
    assert states(verify, stats, transport)["no bytes lost"] == "FAIL"


def test_the_two_are_told_apart_by_the_ledger_not_the_gap(verify, transport):
    # Both runs have exactly one sequence gap. Only one lost bytes.
    corrupted = bytearray(stream(verify, transport, 20))
    corrupted[7 * 1040] ^= 0x01
    dropped = b"".join(verify.build_packet(transport, i)
                       for i in list(range(7)) + list(range(8, 20)))

    a = run(verify, transport, bytes(corrupted))
    b = run(verify, transport, dropped)
    assert a.sequence_gaps == b.sequence_gaps == 1
    assert verify.bytes_lost(a, transport) == 0
    assert verify.bytes_lost(b, transport) == 1040


def test_a_trailing_partial_packet_is_not_charged_as_loss(verify, transport):
    stats = run(verify, transport, stream(verify, transport, 10)[:-500])
    assert stats.residual_bytes == 540
    assert verify.bytes_lost(stats, transport) == 0


def test_sender_packets_spans_the_sequence_numbers_not_the_received_count(
    verify, transport
):
    data = b"".join(verify.build_packet(transport, i) for i in [10, 11, 15])
    stats = run(verify, transport, data)
    assert stats.packets == 3
    assert verify.sender_packets(stats) == 6        # 10..15 inclusive


def test_sender_packets_handles_the_uint32_wrap(verify, transport):
    data = (verify.build_packet(transport, 0xFFFFFFFE)
            + verify.build_packet(transport, 0xFFFFFFFF)
            + verify.build_packet(transport, 0))
    stats = run(verify, transport, data)
    assert verify.sender_packets(stats) == 3
    assert verify.bytes_lost(stats, transport) == 0


def test_no_packets_means_no_conservation_claim(verify, transport):
    stats = run(verify, transport, bytes(50_000), elapsed=1.0)
    assert verify.sender_packets(stats) == 0
    assert verify.bytes_lost(stats, transport) == 0
    assert states(verify, stats, transport)["no bytes lost"] == "N/A"


def test_the_byte_ledger_is_printed_and_balances(verify, transport, capsys):
    stats = run(verify, transport, b"\x00" * 16 + stream(verify, transport, 100))
    verify.report(stats, transport)
    out = capsys.readouterr().out

    assert "--- byte ledger ---" in out
    assert "BYTES LOST" in out
    accounted = (stats.packets * 1040 + stats.stray_bytes
                 + stats.lead_in_bytes + stats.residual_bytes)
    assert accounted == stats.total_bytes


def test_no_ledger_is_printed_when_nothing_locked(verify, transport, capsys):
    stats = run(verify, transport, bytes(50_000), elapsed=1.0)
    verify.report(stats, transport)
    assert "byte ledger" not in capsys.readouterr().out


# --- the three diagnoses ----------------------------------------------------

def test_a_sequence_gap_alone_never_recommends_dropping_the_baud(
    verify, transport, capsys
):
    # THE run 2 REGRESSION. This exact input printed the 460800 advice.
    data = bytearray(stream(verify, transport, 18_000))
    data[7 * 1040] ^= 0x01
    stats = run(verify, transport, bytes(data))
    assert stats.sequence_gaps == 1

    assert verify.report(stats, transport) is False
    out = capsys.readouterr().out
    assert "460800" not in out
    assert "CORRUPTION, NOT LOSS" in out
    assert "NOT evidence against 921600" in out
    assert "The link is keeping up." in out


def test_a_payload_crc_failure_alone_is_diagnosed_as_corruption(
    verify, transport, capsys
):
    # The run 1 event.
    stats = run(verify, transport, corrupt(verify, transport, 5000, [9]))
    verify.report(stats, transport)
    out = capsys.readouterr().out
    assert "CORRUPTION, NOT LOSS" in out
    assert "460800" not in out


def test_real_byte_loss_does_recommend_the_fallback(verify, transport, capsys):
    data = b"".join(verify.build_packet(transport, i) for i in range(0, 400, 2))
    stats = run(verify, transport, data, elapsed=3.2)
    assert verify.bytes_lost(stats, transport) > 0

    verify.report(stats, transport)
    out = capsys.readouterr().out
    assert "genuine LOSS" in out
    assert "460800" in out
    assert "CORRUPTION, NOT LOSS" not in out


def test_ring_overrun_flags_still_recommend_the_fallback(verify, transport, capsys):
    stats = run(verify, transport, stream(verify, transport, 625, flags=0x01),
                elapsed=10.0)
    assert verify.bytes_lost(stats, transport) == 0     # nothing lost on the wire
    verify.report(stats, transport)
    out = capsys.readouterr().out
    assert "460800" in out                               # the firmware dropped frames
    assert "CORRUPTION, NOT LOSS" not in out


def test_nothing_locked_still_gets_the_four_causes(verify, transport, capsys):
    stats = run(verify, transport, bytes(100_000), elapsed=10.0)
    verify.report(stats, transport)
    out = capsys.readouterr().out
    assert "Nothing ever locked on" in out
    assert "START_IN_DIAG" in out
    assert "460800" not in out
    assert "CORRUPTION, NOT LOSS" not in out


def test_the_three_diagnoses_are_mutually_exclusive(verify, transport, capsys):
    corrupted = bytearray(stream(verify, transport, 100))
    corrupted[7 * 1040] ^= 0x01
    cases = [
        bytes(corrupted),
        b"".join(verify.build_packet(transport, i) for i in range(0, 100, 2)),
        bytes(50_000),
    ]
    for data in cases:
        stats = run(verify, transport, data, elapsed=1.0)
        verify.report(stats, transport)
        out = capsys.readouterr().out
        headline = sum(marker in out for marker in
                       ("CORRUPTION, NOT LOSS", "genuine LOSS", "Nothing ever locked on"))
        assert headline == 1, out
