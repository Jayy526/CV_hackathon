"""Phase 1 / CONTEXT.md section 14, Tests 2 and 3: prove the wire, nothing else.

This is a THROWAWAY DIAGNOSTIC, not the receiver. It deliberately imports no
AudioSource, no AudioFrame and no part of the pipeline. It only reads bytes off
a COM port and checks them against the section 13 packet contract.

It does import load_audio_config, because section 13 says the Python side is the
authority on every number in the contract and section 12 says to reuse rather
than duplicate. Re-typing 1040 here would let the tool and the firmware disagree
without anyone noticing, which is the exact class of fault it exists to catch.

Usage:
    python tools/verify_serial_stream.py --duration 300

The board must be running a STREAMING build (START_IN_DIAG 0) and the Arduino
Serial Monitor must be CLOSED - Windows COM ports are exclusive.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heimdall.audio.config import HEADER_BYTES, load_audio_config  # noqa: E402

from heimdall.audio.packets import (  # noqa: E402
    HEXDUMP_BYTES,
    MAGIC,
    MAX_LOGGED_FAILURES,
    PROTOCOL_VERSION,
    PacketCounters,
    PacketDecoder,
    PayloadFailure,
    build_packet,
    bytes_lost,
    crc16_ccitt_false,
    hexdump,
    sender_packets,
)

# The framing lives in heimdall.audio.packets so this diagnostic and the real
# ESP32AudioSource cannot drift apart. These aliases keep the tool's own
# vocabulary; they are the same objects.
Stats = PacketCounters
StreamVerifier = PacketDecoder


# Verdict states. N/A exists because most criteria count faults AMONG RECEIVED
# PACKETS: with no packets there is nothing to count, and calling that PASS
# makes a run that received nothing look mostly green.
PASS, FAIL, NA = "PASS", "FAIL", "N/A"


# The throughput window is wall clock, so it inherits USB scheduling jitter.
# 2% is loose enough not to fail a healthy link, and tight enough that losing
# even one packet per second (1.6%) shows up.
RATE_TOLERANCE = 0.02


def verdict(stats: Stats, transport) -> list[tuple[str, str, str]]:
    """The section 14 pass criteria, each judged separately and tri-state.

    Returns (name, PASS|FAIL|N/A, detail). Most criteria count faults among
    RECEIVED packets, so with zero packets they are N/A, never PASS: a run that
    received nothing must not read as eight lines of green.
    """
    expected_rate = transport.wire_bytes_per_second
    lost = bytes_lost(stats, transport)
    rate_ok = (
        stats.elapsed > 0
        and abs(stats.bytes_per_second - expected_rate) <= expected_rate * RATE_TOLERANCE
    )

    def among_packets(ok: bool) -> str:
        """Judgeable only from packets that actually arrived."""
        if stats.packets == 0:
            return NA
        return PASS if ok else FAIL

    if stats.contract_mismatches:
        contract_state = FAIL
    elif stats.packets == 0:
        # No mismatch seen, but nothing locked either: unproven, not proven good.
        contract_state = NA
    else:
        contract_state = PASS
    contract_detail = f"{stats.contract_mismatches} headers with a valid CRC but "\
                      f"the wrong sample count or payload length"
    if stats.contract_seen:
        saw = ", ".join(f"{c} samples / {n} B" for c, n in sorted(stats.contract_seen))
        contract_detail += (f" (saw {saw}; config says {transport.samples_per_packet}"
                            f" / {transport.bytes_per_packet} B)")

    return [
        ("received a stream at all", PASS if stats.packets else FAIL,
         f"{stats.packets} packets in {stats.elapsed:.1f} s"),
        ("port stayed open for the whole run",
         FAIL if stats.port_error else PASS,
         stats.port_error or "no dropout"),
        ("throughput", PASS if rate_ok else FAIL,
         f"{stats.bytes_per_second:,.0f} B/s vs {expected_rate:,.0f} expected "
         f"(+/-{RATE_TOLERANCE * 100:.0f}%)"),
        ("header contract matches config", contract_state, contract_detail),
        ("no bytes lost", among_packets(lost == 0),
         f"{lost} bytes lost over {sender_packets(stats)} packets the sender "
         f"emitted" + ("  <- corruption, not loss" if stats.sequence_gaps and not lost
                       else "")),
        ("sequence continuity", among_packets(stats.sequence_gaps == 0),
         f"{stats.sequence_gaps} gaps, {stats.missing_packets} packets missing"),
        ("header CRC", among_packets(stats.header_crc_failures == 0),
         f"{stats.header_crc_failures} failures"),
        ("payload CRC", among_packets(stats.payload_crc_failures == 0),
         f"{stats.payload_crc_failures} failures"
         + (f" in {stats.payload_crc_bursts} burst(s), longest "
            f"{stats.longest_payload_burst} consecutive"
            if stats.payload_crc_failures else "")),
        ("protocol version", among_packets(stats.bad_version == 0),
         f"{stats.bad_version} packets not version {PROTOCOL_VERSION}"),
        ("packet boundaries held",
         among_packets(stats.stray_bytes == 0 and stats.resyncs == 0),
         f"{stats.resyncs} resyncs, {stats.stray_bytes} stray bytes after lock"),
        ("no ring overruns (flag bit0)",
         among_packets(stats.flag_overrun_packets == 0),
         f"{stats.flag_overrun_packets} packets flagged"),
        ("no I2S failures (flag bit1)",
         among_packets(stats.flag_i2s_fail_packets == 0),
         f"{stats.flag_i2s_fail_packets} packets flagged"),
    ]


def _print_byte_ledger(stats: Stats, transport) -> None:
    """The arithmetic that separates corruption from loss, shown in full.

    A sequence gap is not loss. Printing the ledger every run means the question
    is settled by the numbers on screen rather than reconstructed afterwards.
    """
    if stats.packets == 0:
        return
    packet = transport.wire_bytes_per_packet
    span = sender_packets(stats)
    lost = bytes_lost(stats, transport)
    print("--- byte ledger ---")
    print(f"  received in whole packets : {stats.packets:>12,} x {packet} = "
          f"{stats.packets * packet:,}")
    print(f"  discarded after lock      : {stats.stray_bytes:>12,}  (resyncs: "
          f"{stats.resyncs})")
    print(f"  discarded before lock     : {stats.lead_in_bytes:>12,}  "
          f"(the DTR reset)")
    print(f"  trailing partial packet   : {stats.residual_bytes:>12,}")
    print(f"  ---------------------------------------------")
    print(f"  accounted                 : "
          f"{stats.packets * packet + stats.stray_bytes + stats.lead_in_bytes + stats.residual_bytes:>12,}"
          f"   of {stats.total_bytes:,} read")
    print(f"  sender emitted            : {span:>12,} packets = {span * packet:,} B")
    print(f"  BYTES LOST                : {lost:>12,}")
    print()


def _print_payload_failures(stats: Stats) -> None:
    """Dump each logged payload CRC failure. Silent when there are none."""
    if not stats.payload_failures:
        return
    print(f"--- payload CRC failures: {stats.payload_crc_failures} in "
          f"{stats.payload_crc_bursts} burst(s), longest run "
          f"{stats.longest_payload_burst} ---")
    if stats.payload_crc_bursts == stats.payload_crc_failures:
        print("Every failure was isolated - no two consecutive sequence numbers.")
    for f in stats.payload_failures:
        print(f"  seq {f.sequence}  flags 0x{f.flags:02x}  "
              f"crc computed 0x{f.computed:04x} vs header 0x{f.expected:04x}"
              f"{'  (consecutive with the previous failure)' if f.consecutive else ''}")
        print(f"    first {len(f.head)} B: {hexdump(f.head)}")
        print(f"    last  {len(f.tail)} B: {hexdump(f.tail)}")
    if stats.payload_failures_unlogged:
        print(f"  ... and {stats.payload_failures_unlogged} more, not logged "
              f"(cap is {MAX_LOGGED_FAILURES}). The count above is complete.")
    print()


def report(stats: Stats, transport) -> bool:
    checks = verdict(stats, transport)
    # N/A is not a pass. A criterion nothing could be judged against has not
    # been satisfied; it has merely not been tested.
    passed = all(state == PASS for _, state, _ in checks)
    utilisation = 100.0 * stats.bytes_per_second / transport.max_bytes_per_second
    print()
    print("--- Tests 2 and 3: sustained throughput and framing integrity ---")
    print(f"lead-in discarded : {stats.lead_in_bytes} bytes "
          f"(expected - opening the port resets the board)")
    print(f"packets           : {stats.packets}  ({stats.packets_per_second:.2f}/s, "
          f"expect {transport.packets_per_second:.2f}/s)")
    print(f"bytes             : {stats.total_bytes:,}  "
          f"({stats.bytes_per_second:,.0f} B/s)")
    print(f"link utilisation  : {utilisation:.1f}% of "
          f"{transport.max_bytes_per_second:,} B/s")
    print()
    for name, state, detail in checks:
        print(f"  [{state:<4}]  {name:<32} {detail}")
    if any(state == NA for _, state, _ in checks):
        print()
        print("  N/A = nothing arrived to judge this against. Not a pass.")
    print()
    _print_byte_ledger(stats, transport)
    _print_payload_failures(stats)

    if passed:
        print("VERDICT: PASS - the wire is clean. Tests 2 and 3 satisfied.")
        print("Now reopen the Serial Monitor and press 'd': short writes must also")
        print("read 0. That counter lives on the ESP32 and is invisible from here.")
        return True

    print("VERDICT: FAIL - do not build the receiver on this.")
    print()
    if stats.packets == 0:
        # Zero packets is NOT link saturation. A saturated link still delivers
        # packets, gapped and overrun-flagged. Nothing locked on at all means
        # the two ends never agreed on what to send or where to send it, and
        # sending someone to the 460800 fallback from here would cost a FIR
        # redesign to fix a wrong #define.
        print("Nothing ever locked on. This is NOT evidence against 921600: a")
        print("saturated link still delivers packets, just gapped and flagged.")
        print("Do not drop the baud on the strength of this run. Check instead:")
        print("  1. Is the board on a STREAMING build? START_IN_DIAG 1 refuses")
        print("     to stream at all and runs the port at 115200.")
        print("  2. Right --port, and the Arduino Serial Monitor closed?")
        print(f"  3. Baud agrees? config/audio.yaml says {transport.baud_rate}.")
        print("  4. Do the firmware #defines still match config/audio.yaml?")
        if stats.contract_mismatches:
            print("     ^^ THIS ONE. Intact headers arrived carrying a sample")
            print("        count or payload length the config disagrees with.")
            print("        See 'header contract matches config' above.")
    elif bytes_lost(stats, transport) > 0 or stats.flag_overrun_packets:
        # LOSS. Bytes the sender emitted never arrived, or the firmware itself
        # reported dropping frames. This is the only shape that indicts the baud.
        print("Bytes the sender emitted did not arrive, or the firmware flagged")
        print("ring overruns. This is genuine LOSS: 921600 does not hold on this")
        print("board. Per section 11 the answer is 460800 + 8 kHz stereo. We do")
        print("not ship something lossy.")
    elif stats.sequence_gaps or stats.payload_crc_failures or stats.header_crc_failures:
        # CORRUPTION with every byte present. A flipped bit in the magic makes
        # the resync discard a whole packet, which looks exactly like a dropped
        # packet in the sequence numbers - but the byte ledger balances.
        print("CORRUPTION, NOT LOSS. Every byte the sender emitted arrived; the")
        print("byte ledger above balances. The link is keeping up.")
        print("This is NOT evidence against 921600 and does not justify dropping")
        print("the baud. A single flipped bit on the unprotected ESP32 -> CP2102")
        print("UART hop corrupts a byte without losing one; if it lands in the")
        print("magic, the whole packet is discarded and shows up as a gap.")
        print("The fix is for the receiver to DROP corrupt packets and COUNT")
        print("them - never to slow the link down or repair the payload.")
    elif stats.contract_mismatches:
        print("Some headers carried a sample count or payload length the config")
        print("disagrees with. Reconcile the firmware #defines with")
        print("config/audio.yaml before reading anything else into this run.")
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the ESP32 packet stream.")
    parser.add_argument("--port", default=None,
                        help="COM port of the ESP32; defaults to transport.port "
                             "in config/audio.yaml (tools/detect_device.py finds it)")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="measurement window in seconds (default 300 = 5 min)")
    # 4.0, not 1.5. Opening the port asserts DTR/RTS and resets the ESP32, which
    # makes the CP2102 re-enumerate on USB. At 1.5 s the handle was still stale
    # and the first read died with "ClearCommError failed (Access is denied)".
    # The wait is for USB re-enumeration, not for the firmware to boot.
    parser.add_argument("--settle", type=float, default=4.0,
                        help="seconds to wait after opening before counting "
                             "(default 4; the CP2102 must finish re-enumerating)")
    parser.add_argument("--baud", type=int, default=None,
                        help="override the configured baud rate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    transport = load_audio_config().transport
    baud = args.baud or transport.baud_rate
    port_name = args.port or transport.port
    if not port_name:
        print("No serial port. Set transport.port in config/audio.yaml or pass "
              "--port; tools/detect_device.py finds it.", file=sys.stderr)
        return 2

    try:
        import serial
    except ImportError:
        print("pyserial is not installed.", file=sys.stderr)
        return 2

    print(f"Opening {port_name} at {baud} baud; expecting "
          f"{transport.wire_bytes_per_packet}-byte packets at "
          f"{transport.packets_per_second:.1f}/s.")
    try:
        port = serial.Serial(port_name, baud, timeout=0.2)
    except Exception as exc:  # noqa: BLE001 - any open failure is equally fatal here
        print(f"Could not open {port_name}: {exc}", file=sys.stderr)
        print("Is the Serial Monitor still open? Windows COM ports are exclusive.",
              file=sys.stderr)
        return 2

    verifier = StreamVerifier(transport)
    start = time.monotonic()
    try:
        # Opening asserted DTR/RTS, which reset the ESP32. Everything readable
        # during the settle window is boot noise and mid-packet garbage.
        time.sleep(args.settle)
        port.reset_input_buffer()
        print(f"Counting for {args.duration:.0f} s. "
              f"Ctrl-C stops early and still reports.")
        start = time.monotonic()
        while time.monotonic() - start < args.duration:
            chunk = port.read(max(1, port.in_waiting))
            if chunk:
                verifier.feed(chunk)
    except KeyboardInterrupt:
        print("\ninterrupted - reporting what was measured")
    except serial.SerialException as exc:
        # A dropout at minute 4 of 5 used to raise and lose the whole run.
        # Record it as a named failure and report the four minutes we have.
        verifier.stats.port_error = f"the port disappeared mid-run: {exc}"
        print(f"\n{verifier.stats.port_error}", file=sys.stderr)
    finally:
        verifier.stats.elapsed = time.monotonic() - start
        try:
            port.close()
        except Exception:  # noqa: BLE001 - a vanished port cannot be closed cleanly
            pass

    return 0 if report(verifier.stats, transport) else 1


if __name__ == "__main__":
    raise SystemExit(main())
