"""The hardware boundary.

    AudioSource                (abstract)
    |-- SyntheticAudioSource   (deterministic, no hardware)
    |-- ESP32AudioSource       (real: USB serial, 16 kHz stereo int16)

Everything downstream of this file depends on AudioSource only, so swapping
synthetic audio for the ESP32 changes nothing else in the pipeline.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from acoustic_array import synthetic
from acoustic_array.config import AudioConfig, load_audio_config
from acoustic_array.frame import AudioFrame
from acoustic_array.packets import PacketCounters, PacketDecoder


class AudioSourceError(RuntimeError):
    """Raised when a source cannot be opened or has failed irrecoverably."""


class AudioSource(ABC):
    """A producer of fixed-size multi-channel AudioFrames."""

    def __init__(self, sample_rate: int, num_channels: int, frame_size: int) -> None:
        self._sample_rate = int(sample_rate)
        self._num_channels = int(num_channels)
        self._frame_size = int(frame_size)
        self._running = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def frame_size(self) -> int:
        return self._frame_size

    @property
    def is_running(self) -> bool:
        return self._running

    @abstractmethod
    def start(self) -> None:
        """Open the underlying device/generator."""

    @abstractmethod
    def read_frame(self) -> AudioFrame | None:
        """Return the next frame, or None if the source is exhausted."""

    @abstractmethod
    def stop(self) -> None:
        """Release the underlying device."""

    def __enter__(self) -> "AudioSource":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


class SyntheticAudioSource(AudioSource):
    """Deterministic mock source: no ESP32, no COM port, no sound card.

    By default it produces a repeating pattern of silence and speech-like bursts
    arriving from `angle_degrees`, so the full pipeline (GCC-PHAT -> DOA -> seat
    mapping -> events) can be exercised end to end in tests.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        num_channels: int = 2,
        frame_size: int = 1024,
        *,
        angle_degrees: float = 0.0,
        mic_spacing_m: float = 0.3,
        noise_amplitude: float = 0.002,
        burst_frames: int = 8,
        silence_frames: int = 8,
        max_frames: int | None = None,
        seed: int = 0,
        buffer: np.ndarray | None = None,
    ) -> None:
        super().__init__(sample_rate, num_channels, frame_size)
        self.angle_degrees = angle_degrees
        self.mic_spacing_m = mic_spacing_m
        self.noise_amplitude = noise_amplitude
        self.burst_frames = max(int(burst_frames), 0)
        self.silence_frames = max(int(silence_frames), 0)
        self.max_frames = max_frames
        self.seed = seed
        self._buffer = buffer
        self._frame_index = 0

    @classmethod
    def from_buffer(
        cls,
        buffer: np.ndarray,
        sample_rate: int,
        frame_size: int = 1024,
    ) -> "SyntheticAudioSource":
        """Replay a fixed (num_samples, num_channels) array frame by frame."""
        buffer = np.atleast_2d(np.asarray(buffer, dtype=np.float32))
        if buffer.shape[0] < buffer.shape[1]:
            raise ValueError(
                "buffer must be shaped (num_samples, num_channels); "
                f"got {buffer.shape} which looks transposed"
            )
        return cls(
            sample_rate=sample_rate,
            num_channels=buffer.shape[1],
            frame_size=frame_size,
            buffer=buffer,
        )

    @classmethod
    def from_config(cls, config: AudioConfig | None = None, **kwargs: object) -> "SyntheticAudioSource":
        config = config or load_audio_config()
        return cls(
            sample_rate=config.sample_rate,
            num_channels=config.num_channels,
            frame_size=config.frame_size,
            **kwargs,  # type: ignore[arg-type]
        )

    def start(self) -> None:
        self._running = True
        self._frame_index = 0

    def stop(self) -> None:
        self._running = False

    def _delays(self) -> np.ndarray:
        """Per-channel delays in samples for a uniform linear array."""
        tdoa = synthetic.tdoa_for_angle(
            self.angle_degrees, self.mic_spacing_m, self.sample_rate
        )
        # tdoa_for_angle gives the delay of channel 0 relative to channel 1, so
        # channel c is delayed by -c * tdoa for a uniform linear array.
        return -np.arange(self.num_channels, dtype=np.float64) * tdoa

    def _generate_frame(self, index: int) -> np.ndarray:
        period = self.burst_frames + self.silence_frames
        # silence_frames=0 means a continuous sound; burst_frames=0 means silence.
        in_burst = period > 0 and (index % period) < self.burst_frames

        if in_burst:
            source = synthetic.speech_like(
                self.frame_size, self.sample_rate, seed=self.seed + index, amplitude=0.3
            )
        else:
            source = np.zeros(self.frame_size)

        return synthetic.simulate_array_signals(
            source,
            self._delays(),
            noise_amplitude=self.noise_amplitude,
            seed=self.seed + index,
        )

    def read_frame(self) -> AudioFrame | None:
        if not self._running:
            raise AudioSourceError("read_frame() called before start()")

        index = self._frame_index
        if self.max_frames is not None and index >= self.max_frames:
            return None

        if self._buffer is not None:
            begin = index * self.frame_size
            if begin >= self._buffer.shape[0]:
                return None
            chunk = self._buffer[begin : begin + self.frame_size]
            if chunk.shape[0] < self.frame_size:
                pad = np.zeros(
                    (self.frame_size - chunk.shape[0], chunk.shape[1]), dtype=np.float32
                )
                chunk = np.vstack([chunk, pad])
            samples = chunk.astype(np.float32)
        else:
            samples = self._generate_frame(index)

        self._frame_index += 1
        return AudioFrame(
            samples=samples,
            timestamp=index * self.frame_size / self.sample_rate,
            frame_index=index,
            sample_rate=self.sample_rate,
        )


@dataclass
class ESP32SourceStats:
    """What the link is doing, exposed so a degrading wire is never silent.

    Section 14 measured the wire as loss-free but occasionally corrupt, so a
    non-zero drop count is expected, not alarming. A RISING one is the signal.
    """

    frames_emitted: int = 0
    # Partial frames thrown away because a packet went missing part-way through
    # assembling them. See the gap policy in ESP32AudioSource.
    frames_abandoned: int = 0
    packets_used: int = 0
    discontinuities: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "frames_emitted": self.frames_emitted,
            "frames_abandoned": self.frames_abandoned,
            "packets_used": self.packets_used,
            "discontinuities": self.discontinuities,
        }


class ESP32AudioSource(AudioSource):
    """Continuous 16 kHz stereo int16 off the ESP32, over USB serial.

    Framing is NOT implemented here. It comes from acoustic_array.packets, the
    same decoder tools/verify_serial_stream.py used to prove the wire over
    three 300 s hardware runs, so the two cannot drift apart.

    SAMPLE RATE IS 16000, NOT 48000. The ESP32 acquires at 48 kHz and decimates
    3:1 behind an anti-aliasing FIR before transmitting, because the link
    physically cannot carry 48 kHz stereo (section 13). `sample_rate` here is
    what arrives, which is what every downstream stage must use.

    WHAT HAPPENS ACROSS A DROPPED PACKET
    ------------------------------------
    Any packet failing either CRC is dropped whole by the decoder, so it simply
    never appears - which shows up as a jump in the sequence numbers. When that
    happens part-way through assembling a frame, THE PARTIAL FRAME IS DISCARDED
    and assembly restarts at the next good packet.

    It is not zero-filled and the two sides are not spliced together. A 16 ms
    splice is a phase discontinuity, and inter-channel phase is the entire
    signal GCC-PHAT measures; zero-fill is worse still, because a step to
    silence and back is a broadband transient correlated across both channels
    at zero lag, which would drag the bearing toward 0 degrees. Both would be
    inventing audio that never existed.

    The cost is bounded and small: at the measured BER, roughly one dropped
    packet every 7.5 minutes discards at most one 1024-sample frame, or about
    64 ms - well under 0.02% of the stream. Losing that is cheap. A fabricated
    bearing is not.
    """

    def __init__(
        self,
        port: str | None = None,
        config: AudioConfig | None = None,
        *,
        serial_factory: object | None = None,
        settle_seconds: float = 4.0,
        read_timeout: float = 0.2,
        stall_timeout: float = 2.0,
    ) -> None:
        config = config or load_audio_config()
        transport = config.transport
        wire = transport.wire_format

        super().__init__(
            sample_rate=transport.transmit_sample_rate,
            num_channels=wire.num_channels,
            frame_size=config.frame_size,
        )
        if config.frame_size % transport.samples_per_packet:
            raise AudioSourceError(
                "frame_size (%d) must be a whole number of packets of %d samples; "
                "a partial packet per frame would put a gap in the middle of one"
                % (config.frame_size, transport.samples_per_packet)
            )

        self.config = config
        self.transport = transport
        self.port_name = port or transport.port
        # 4.0 s, matching tools/verify_serial_stream.py: opening the port
        # asserts DTR/RTS, which resets the ESP32 and makes the CP2102
        # re-enumerate on USB. At 1.5 s the handle was still stale and the
        # first read failed with "ClearCommError failed (Access is denied)".
        self.settle_seconds = settle_seconds
        self.read_timeout = read_timeout
        # No bytes at all for this long means the board stopped talking.
        self.stall_timeout = stall_timeout

        self._serial_factory = serial_factory or _open_serial_port
        self._port: object | None = None
        self._decoder = PacketDecoder(transport)
        self._dtype = wire.numpy_dtype
        self._packets_per_frame = config.frame_size // transport.samples_per_packet
        self._pending: list[tuple[int, bytes]] = []
        self._last_seq: int | None = None
        self._anchor_seq: int | None = None
        self._frame_index = 0
        self.stats = ESP32SourceStats()

    # --- link health ---------------------------------------------------------

    @property
    def packet_counters(self) -> PacketCounters:
        """The decoder's own tally: CRC failures, resyncs, sequence gaps."""
        return self._decoder.stats

    @property
    def packets_dropped_header_crc(self) -> int:
        return self._decoder.stats.header_crc_failures

    @property
    def packets_dropped_payload_crc(self) -> int:
        return self._decoder.stats.payload_crc_failures

    @property
    def packets_dropped(self) -> int:
        return self.packets_dropped_header_crc + self.packets_dropped_payload_crc

    def diagnostics(self) -> dict[str, int]:
        """Everything a caller needs to see the link degrading."""
        counters = self._decoder.stats
        return {
            **self.stats.as_dict(),
            "packets_decoded": counters.packets,
            "packets_dropped_header_crc": counters.header_crc_failures,
            "packets_dropped_payload_crc": counters.payload_crc_failures,
            "packets_dropped_total": self.packets_dropped,
            "contract_mismatches": counters.contract_mismatches,
            "sequence_gaps": counters.sequence_gaps,
            "missing_packets": counters.missing_packets,
            "resyncs": counters.resyncs,
            "stray_bytes": counters.stray_bytes,
            "flag_overrun_packets": counters.flag_overrun_packets,
            "flag_i2s_fail_packets": counters.flag_i2s_fail_packets,
            "total_bytes": counters.total_bytes,
        }

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        if not self.port_name:
            raise AudioSourceError(
                "no serial port configured. Set transport.port in config/audio.yaml "
                "or pass port=..., and run tools/detect_device.py to find it."
            )
        try:
            self._port = self._serial_factory(
                self.port_name, self.transport.baud_rate, self.read_timeout
            )
        except AudioSourceError:
            raise
        except Exception as exc:  # noqa: BLE001 - every open failure is equally fatal
            raise AudioSourceError(
                f"could not open {self.port_name}: {exc}. Is the Arduino Serial "
                "Monitor still open? Windows COM ports are exclusive."
            ) from exc

        # The open reset the board; wait for the bridge to come back, then throw
        # away the boot noise. The decoder would charge it as lead-in anyway,
        # but starting clean keeps the counters meaningful.
        if self.settle_seconds > 0:
            time.sleep(self.settle_seconds)
        try:
            self._port.reset_input_buffer()
        except Exception:  # noqa: BLE001 - not fatal; the decoder resyncs regardless
            pass

        self._pending.clear()
        self._last_seq = None
        self._anchor_seq = None
        self._frame_index = 0
        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._port is not None:
            try:
                self._port.close()
            except Exception:  # noqa: BLE001 - a vanished port cannot close cleanly
                pass
            self._port = None

    # --- reading -------------------------------------------------------------

    def read_frame(self) -> AudioFrame | None:
        if not self._running:
            raise AudioSourceError("read_frame() called before start()")

        deadline = time.monotonic() + self.stall_timeout
        while True:
            frame = self._emit_if_ready()
            if frame is not None:
                return frame
            chunk = self._read_bytes()
            if chunk:
                deadline = time.monotonic() + self.stall_timeout
                self._absorb(chunk)
            elif time.monotonic() >= deadline:
                # The board stopped talking. Exhausted, not an error: the
                # receiver treats None as end of stream.
                return None

    def _read_bytes(self) -> bytes:
        try:
            waiting = getattr(self._port, "in_waiting", 0)
            return self._port.read(max(1, waiting))
        except Exception as exc:  # noqa: BLE001 - wrapped so callers need no pyserial
            raise AudioSourceError(
                f"serial read failed on {self.port_name}: {exc}"
            ) from exc

    def _absorb(self, chunk: bytes) -> None:
        for packet in self._decoder.feed(chunk):
            if self._anchor_seq is None:
                self._anchor_seq = packet.sequence

            expected = (
                None if self._last_seq is None
                else (self._last_seq + 1) & 0xFFFFFFFF
            )
            if expected is not None and packet.sequence != expected:
                self.stats.discontinuities += 1
                if self._pending:
                    # Mid-frame gap: throw the partial frame away rather than
                    # splice across it. See the class docstring.
                    self._pending.clear()
                    self.stats.frames_abandoned += 1

            self._pending.append((packet.sequence, packet.payload))
            self._last_seq = packet.sequence
            self.stats.packets_used += 1

    def _emit_if_ready(self) -> AudioFrame | None:
        if len(self._pending) < self._packets_per_frame:
            return None

        block = self._pending[: self._packets_per_frame]
        del self._pending[: self._packets_per_frame]

        first_seq = block[0][0]
        raw = b"".join(payload for _, payload in block)
        interleaved = np.frombuffer(raw, dtype=self._dtype)
        samples = (
            interleaved.reshape(-1, self.num_channels).astype(np.float32) / 32768.0
        )

        # Timestamps come from the SENDER's sequence numbers, relative to the
        # first packet of the session, so a gap shows up as a jump in time
        # rather than being quietly closed up.
        offset = (first_seq - (self._anchor_seq or 0)) & 0xFFFFFFFF
        timestamp = offset * self.transport.samples_per_packet / self.sample_rate

        frame = AudioFrame(
            samples=samples,
            timestamp=timestamp,
            frame_index=self._frame_index,
            sample_rate=self.sample_rate,
        )
        self._frame_index += 1
        self.stats.frames_emitted += 1
        return frame


def _open_serial_port(port: str, baud: int, timeout: float) -> object:
    """Default factory. Imported lazily so tests never need pyserial."""
    try:
        import serial
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise AudioSourceError(
            "pyserial is not installed, so the ESP32 cannot be opened."
        ) from exc
    return serial.Serial(port, baud, timeout=timeout)
