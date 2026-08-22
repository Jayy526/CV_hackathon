"""The section 13 packet contract: framing, CRC, and the byte-level decoder.

ONE implementation, shared by two callers that must never disagree:

  * tools/verify_serial_stream.py, which proved the wire over three 300 s
    hardware runs (section 14);
  * acoustic_array.sources.ESP32AudioSource, which feeds the real pipeline.

If those two framed bytes differently, what those hardware runs proved would
stop being evidence about what the receiver actually does. So the state
machine, the CRC and the resync accounting live here and nowhere else.

Section 14 measured the wire as LOSS-FREE BUT OCCASIONALLY CORRUPT
(BER ~4e-9). The decoder therefore emits only packets passing BOTH CRCs, and
counts every one it drops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from acoustic_array.config import HEADER_BYTES


@dataclass(frozen=True)
class DecodedPacket:
    """A packet that passed the header CRC, the contract check and the payload
    CRC. Nothing failing any of those is ever handed out."""

    sequence: int
    flags: int
    payload: bytes


MAGIC = b"\xa5\x5a"
PROTOCOL_VERSION = 1

# Why a header was rejected. Intact bytes carrying the wrong contract is not
# corruption, and telling them apart is most of the point of this tool.
_BAD_CRC = "header-crc"
_BAD_CONTRACT = "contract"

# A payload CRC failure is rare and interesting, so each one is logged in full.
# Bounded because a genuinely broken link would otherwise emit a gigabyte of
# hexdumps; the count keeps rising after logging stops.
MAX_LOGGED_FAILURES = 20
HEXDUMP_BYTES = 32


@dataclass
class PayloadFailure:
    """One packet whose payload CRC did not match its header.

    `consecutive` distinguishes a burst from a lone flip: a burst points at the
    sender or the buffering, a single isolated bit at the wire.
    """

    sequence: int
    flags: int
    computed: int
    expected: int
    head: bytes
    tail: bytes
    consecutive: bool


def hexdump(data: bytes) -> str:
    return " ".join(f"{b:02x}" for b in data)

def crc16_ccitt_false(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no xorout.

    Pinned by test against the standard check vector: crc(b"123456789") == 0x29B1.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass
class PacketCounters:
    """Everything the verdict is computed from. Counted, never estimated."""

    packets: int = 0
    total_bytes: int = 0
    # Bytes thrown away BEFORE the first valid header. Expected, not a fault:
    # opening the port asserts DTR/RTS and resets the ESP32 (the section 14
    # gotcha), so the first bytes read are always mid-packet garbage.
    lead_in_bytes: int = 0
    # Bytes thrown away AFTER lock. This IS the "1040-byte boundaries drift"
    # measurement: a clean link never discards a byte once synchronised.
    stray_bytes: int = 0
    resyncs: int = 0
    header_crc_failures: int = 0
    # Header CRC VALID, but sample count or payload length disagrees with
    # config/audio.yaml. The bytes are intact; the two ends disagree about what
    # a packet is. Counted whether or not the stream ever locked, because with
    # a contract mismatch it never will - and this is then the only evidence of
    # why. A valid 12-byte CRC behind the magic does not happen by chance.
    contract_mismatches: int = 0
    payload_crc_failures: int = 0
    bad_version: int = 0
    sequence_gaps: int = 0
    missing_packets: int = 0
    flag_overrun_packets: int = 0
    flag_i2s_fail_packets: int = 0
    # Set when the port vanished mid-run (USB dropout, board reset, unplug).
    # Recorded rather than raised, so a failure at minute 4 of 5 still reports
    # the four minutes that were measured.
    port_error: str | None = None
    # Sequence span actually emitted by the sender, and the bytes still sitting
    # in the reassembly buffer when the run ended. Both feed byte conservation.
    first_seq: int | None = None
    last_seq: int | None = None
    residual_bytes: int = 0
    # Detail on the first MAX_LOGGED_FAILURES payload CRC failures.
    payload_failures: list[PayloadFailure] = field(default_factory=list)
    payload_failures_unlogged: int = 0
    # Maximal runs of consecutively-numbered failing packets. bursts == failures
    # means every failure was isolated.
    payload_crc_bursts: int = 0
    longest_payload_burst: int = 0
    elapsed: float = 0.0
    flags_seen: set[int] = field(default_factory=set)
    # (samples_per_packet, payload_length) pairs seen on mismatched headers, so
    # the report can name what the firmware actually claimed.
    contract_seen: set[tuple[int, int]] = field(default_factory=set)

    @property
    def bytes_per_second(self) -> float:
        return self.total_bytes / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def packets_per_second(self) -> float:
        return self.packets / self.elapsed if self.elapsed > 0 else 0.0


class PacketDecoder:
    """Byte-stream state machine: resynchronise on magic, validate, tally.

    Fed arbitrary chunks. It never assumes a read lands on a packet boundary,
    because it never does.
    """

    def __init__(self, transport) -> None:
        self.packet_bytes = transport.wire_bytes_per_packet
        self.payload_bytes = transport.bytes_per_packet
        self.samples_per_packet = transport.samples_per_packet
        self.stats = PacketCounters()
        self._buf = bytearray()
        self._locked = False
        self._last_seq: int | None = None
        self._last_bad_seq: int | None = None
        self._burst_len = 0
        self._decoded: list[DecodedPacket] = []

    def feed(self, data: bytes) -> list["DecodedPacket"]:
        """Absorb bytes; return the packets that passed EVERY check.

        A packet failing the header CRC, the contract check or the payload CRC
        is counted and discarded whole. It is never repaired, never partially
        used, and never returned here - section 14 makes that a requirement.
        """
        self.stats.total_bytes += len(data)
        self._buf += data
        self._decoded: list[DecodedPacket] = []
        self._drain()
        self.stats.residual_bytes = len(self._buf)
        return self._decoded

    def _drain(self) -> None:
        while len(self._buf) >= self.packet_bytes:
            if self._buf[:2] == MAGIC:
                header = self._parse_header()
                if isinstance(header, dict):
                    self._consume(header)
                    continue
                if header == _BAD_CONTRACT:
                    # Counted even before lock: a contract mismatch is why the
                    # stream never locks, so charging it only after lock would
                    # hide the one fault that explains the silence.
                    self.stats.contract_mismatches += 1
                elif self._locked:
                    # Magic at a real boundary but the CRC disagrees: that is
                    # corruption. Before lock it is just garbage, not evidence.
                    self.stats.header_crc_failures += 1
            self._resync()

    def _parse_header(self) -> dict | str:
        head = bytes(self._buf[:HEADER_BYTES])
        if crc16_ccitt_false(head[:12]) != int.from_bytes(head[12:14], "little"):
            return _BAD_CRC
        count = int.from_bytes(head[8:10], "little")
        length = int.from_bytes(head[10:12], "little")
        if count != self.samples_per_packet or length != self.payload_bytes:
            self.stats.contract_seen.add((count, length))
            return _BAD_CONTRACT
        return {
            "version": head[2],
            "flags": head[3],
            "sequence": int.from_bytes(head[4:8], "little"),
            "payload_crc": int.from_bytes(head[14:16], "little"),
        }

    def _consume(self, header: dict) -> None:
        s = self.stats
        payload = bytes(self._buf[HEADER_BYTES:self.packet_bytes])
        del self._buf[:self.packet_bytes]
        self._locked = True
        s.packets += 1

        if header["version"] != PROTOCOL_VERSION:
            s.bad_version += 1

        computed = crc16_ccitt_false(payload)
        if computed != header["payload_crc"]:
            self._log_payload_failure(header, payload, computed)
        elif header["version"] == PROTOCOL_VERSION:
            # Both CRCs good and the protocol understood: the only path on
            # which a payload is handed to a caller.
            self._decoded.append(DecodedPacket(
                sequence=header["sequence"], flags=header["flags"], payload=payload))

        flags = header["flags"]
        s.flags_seen.add(flags)
        if flags & 0x01:
            s.flag_overrun_packets += 1
        if flags & 0x02:
            s.flag_i2s_fail_packets += 1

        seq = header["sequence"]
        if s.first_seq is None:
            s.first_seq = seq
        s.last_seq = seq
        if self._last_seq is not None:
            expected = (self._last_seq + 1) & 0xFFFFFFFF
            if seq != expected:
                s.sequence_gaps += 1
                s.missing_packets += (seq - expected) & 0xFFFFFFFF
        self._last_seq = seq

    def _log_payload_failure(self, header: dict, payload: bytes, computed: int) -> None:
        """Record a payload CRC failure in enough detail to reason about it.

        Counting alone cannot distinguish a single flipped bit on the wire from
        a sender that is emitting garbage, and those need different fixes.
        """
        s = self.stats
        s.payload_crc_failures += 1
        seq = header["sequence"]

        consecutive = (
            self._last_bad_seq is not None
            and seq == (self._last_bad_seq + 1) & 0xFFFFFFFF
        )
        if consecutive:
            self._burst_len += 1
        else:
            s.payload_crc_bursts += 1
            self._burst_len = 1
        s.longest_payload_burst = max(s.longest_payload_burst, self._burst_len)
        self._last_bad_seq = seq

        if len(s.payload_failures) < MAX_LOGGED_FAILURES:
            s.payload_failures.append(PayloadFailure(
                sequence=seq,
                flags=header["flags"],
                computed=computed,
                expected=header["payload_crc"],
                head=payload[:HEXDUMP_BYTES],
                tail=payload[-HEXDUMP_BYTES:],
                consecutive=consecutive,
            ))
        else:
            s.payload_failures_unlogged += 1

    def _resync(self) -> None:
        """Drop to the next plausible boundary, charging the bytes honestly."""
        at = self._buf.find(MAGIC, 1)
        if at < 0:
            # No magic anywhere in the buffer. Keep the last byte in case a
            # 0xA5 straddles the chunk boundary; the rest is genuinely garbage.
            at = len(self._buf) - 1
        del self._buf[:at]
        if self._locked:
            self.stats.stray_bytes += at
            self.stats.resyncs += 1
        else:
            self.stats.lead_in_bytes += at


def sender_packets(stats: PacketCounters) -> int:
    """How many packets the SENDER emitted over the observed sequence span.

    Taken from the sequence numbers, which is what makes conservation a real
    measurement rather than a restatement of the tool's own bookkeeping.
    """
    if stats.packets == 0 or stats.first_seq is None or stats.last_seq is None:
        return 0
    return ((stats.last_seq - stats.first_seq) & 0xFFFFFFFF) + 1


def bytes_lost(stats: PacketCounters, transport) -> int:
    """Bytes the sender emitted that never reached us, over the locked span.

    A sequence gap alone does NOT mean loss. A packet whose magic was corrupted
    is discarded whole by the resync, which shows up as a gap even though every
    one of its bytes arrived - that is the difference between corruption and a
    link that cannot keep up, and only this check can tell them apart.

    Trailing partial bytes are excluded on both sides: the span ends at the last
    COMPLETE packet, so a run that stops mid-packet is not charged for it.

    Robust to a false magic: if A5 5A happens to occur inside a payload after a
    magic corruption, the intermediate resyncs each discard a partial span, but
    those spans sum to the same 1040 stray bytes. The ledger still balances - it
    simply reports more resyncs and a header CRC failure. Verified, not assumed.
    """
    span = sender_packets(stats)
    if span == 0:
        return 0
    packet = transport.wire_bytes_per_packet
    return span * packet - (stats.packets * packet + stats.stray_bytes)


def build_packet(
    transport,
    sequence: int,
    payload: bytes | None = None,
    flags: int = 0,
    version: int = PROTOCOL_VERSION,
) -> bytes:
    """Construct one contract-conformant packet.

    Used by the tests, and by anyone who wants to check this tool against a
    stream it did not produce.
    """
    if payload is None:
        payload = bytes(transport.bytes_per_packet)
    header = bytearray(HEADER_BYTES)
    header[0:2] = MAGIC
    header[2] = version
    header[3] = flags
    header[4:8] = (sequence & 0xFFFFFFFF).to_bytes(4, "little")
    header[8:10] = transport.samples_per_packet.to_bytes(2, "little")
    header[10:12] = len(payload).to_bytes(2, "little")
    header[12:14] = crc16_ccitt_false(bytes(header[:12])).to_bytes(2, "little")
    header[14:16] = crc16_ccitt_false(payload).to_bytes(2, "little")
    return bytes(header) + payload
