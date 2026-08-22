"""The public surface of the acoustic direction sensor.

Everything a consumer needs is here:

    from acoustic_array import AcousticArray, AcousticEvent

    with AcousticArray.synthetic(angle_degrees=-20.0) as array:
        for event in array.stream():
            print(event.to_dict())

WHAT THIS SENSOR IS
-------------------
Two microphones in a line, measuring the BEARING of the loudest sound. That is
all. It knows microphones, its own geometry, angles and confidence. It does not
know about rooms, seats, people, cameras or what the sound means.

WHAT IT DELIBERATELY DOES NOT REPORT
------------------------------------
There is no `position`, no `seat_id` and no distance field, and that is not an
omission to be filled in later - a linear array cannot measure them:

  * NO RANGE. Two microphones in a line give one number, the time difference
    between them. A source 1 m away and a source 5 m away on the same bearing
    produce identical measurements.
  * NO ELEVATION. Same reason. The measurement collapses everything onto one
    angle about the array axis.
  * FRONT/BACK IS AMBIGUOUS. A sound 30 degrees in front and the same sound 30
    degrees behind are physically indistinguishable to this array. Nothing here
    resolves that; a caller with a second sensor must.

A bearing is not a location. Anything that turns one into a location is adding
an assumption, and that assumption belongs to the caller, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from acoustic_array.config import AudioConfig, load_audio_config
from acoustic_array.doa import DoaResult, angular_resolution_degrees, estimate_doa
from acoustic_array.events import AudioEventDetector, DetectedEvent, EventType
from acoustic_array.frame import AudioFrame
from acoustic_array.geometry import MicrophoneArray, default_array
from acoustic_array.receiver import AudioReceiver
from acoustic_array.sources import AudioSource, SyntheticAudioSource

# What produced the audio. The consumer is expected to DISPLAY this: a demo
# that looks identical whether or not microphones are attached is a trap.
SOURCE_SYNTHETIC = "synthetic"
SOURCE_HARDWARE = "hardware"


@dataclass(frozen=True)
class AcousticEvent:
    """One direction measurement, with an explicit account of its own limits.

    `direction_degrees` follows the section 5 convention: 0 is broadside,
    +90 is along the array axis toward channel 0, -90 toward the last channel.

    When the sensor declines to answer, `direction_degrees` is None and
    `reason` says why. It never guesses.
    """

    timestamp: float
    event_type: str
    direction_degrees: float | None
    confidence: float
    localization_confidence: float | None
    angular_resolution_degrees: float | None
    duration: float
    channel_rms: tuple[float, ...]
    source_kind: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "direction_degrees": self.direction_degrees,
            "confidence": self.confidence,
            "localization_confidence": self.localization_confidence,
            "angular_resolution_degrees": self.angular_resolution_degrees,
            "duration": self.duration,
            "channel_rms": list(self.channel_rms),
            "source_kind": self.source_kind,
            "reason": self.reason,
        }

    @property
    def has_direction(self) -> bool:
        return self.direction_degrees is not None

    @property
    def is_live(self) -> bool:
        """True only when real microphones produced this. Display it."""
        return self.source_kind == SOURCE_HARDWARE


class AcousticArray:
    """The sensor: frames in, direction events out.

    Construct with `synthetic()` for development or `hardware()` for the real
    array. Nothing downstream should need to know which it got, except to say
    so on screen - hence `source_kind` on every event.
    """

    def __init__(
        self,
        source: AudioSource,
        *,
        array: MicrophoneArray | None = None,
        config: AudioConfig | None = None,
        source_kind: str = SOURCE_SYNTHETIC,
        detector: AudioEventDetector | None = None,
        min_confidence: float = 0.30,
        # 32 frames is 2.0 s of audio at 64 ms/frame. For a live overlay that
        # is 2 s of lag the moment the consumer falls behind even briefly.
        # 4 frames caps it at 256 ms; the receiver drops oldest, which for a
        # display is exactly right - stale bearings are worthless.
        queue_size: int = 4,
    ) -> None:
        self.config = config or load_audio_config()
        self.array = array or default_array()
        self.source_kind = source_kind
        self.min_confidence = min_confidence
        self.detector = detector or AudioEventDetector()
        self.receiver = AudioReceiver(source, queue_size=queue_size)
        self._pending: list[DoaResult] = []
        self._channel_rms: list[tuple[float, ...]] = []

    # --- construction --------------------------------------------------------

    @classmethod
    def synthetic(
        cls,
        angle_degrees: float = 0.0,
        *,
        config: AudioConfig | None = None,
        array: MicrophoneArray | None = None,
        **kwargs: object,
    ) -> "AcousticArray":
        """Hardware-free. Every event it emits is labelled SOURCE_SYNTHETIC."""
        config = config or load_audio_config()
        array = array or default_array()
        source = SyntheticAudioSource(
            sample_rate=config.transport.transmit_sample_rate,
            num_channels=array.num_channels,
            frame_size=config.frame_size,
            angle_degrees=angle_degrees,
            mic_spacing_m=array.spacing,
            **kwargs,  # type: ignore[arg-type]
        )
        return cls(source, array=array, config=config,
                   source_kind=SOURCE_SYNTHETIC)

    @classmethod
    def hardware(
        cls,
        port: str | None = None,
        *,
        config: AudioConfig | None = None,
        array: MicrophoneArray | None = None,
        **kwargs: object,
    ) -> "AcousticArray":
        """The real ESP32 array over USB serial.

        Corrupt packets are dropped whole and counted by the source; see
        `link_diagnostics()`. A dropped packet discards the partial frame
        rather than splicing across the gap.
        """
        from acoustic_array.sources import ESP32AudioSource

        config = config or load_audio_config()
        source = ESP32AudioSource(port=port, config=config, **kwargs)  # type: ignore[arg-type]
        return cls(source, array=array or default_array(), config=config,
                   source_kind=SOURCE_HARDWARE)

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self.receiver.start()

    def stop(self) -> None:
        self.receiver.stop()

    def __enter__(self) -> "AcousticArray":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def sample_rate(self) -> int:
        return self.receiver.sample_rate

    @property
    def is_live(self) -> bool:
        return self.source_kind == SOURCE_HARDWARE

    def link_diagnostics(self) -> dict[str, int]:
        """Dropped-packet counts, when running on hardware. Empty otherwise."""
        source = self.receiver.source
        getter = getattr(source, "diagnostics", None)
        return dict(getter()) if callable(getter) else {}

    # --- the stream ----------------------------------------------------------

    def process_frame(self, frame: AudioFrame) -> list[AcousticEvent]:
        """Feed one frame; return any events that just completed."""
        from acoustic_array.analysis import channel_rms

        doa = estimate_doa(frame, self.array, min_confidence=self.min_confidence)
        self._pending.append(doa)
        self._channel_rms.append(tuple(float(v) for v in channel_rms(frame)))

        completed = self.detector.process(frame)
        events = [self._build(detected) for detected in completed]
        if completed:
            # Cleared on every run boundary, not only on emission: frames from a
            # discarded run would otherwise leak into the next event and drag
            # its bearing off.
            self._pending.clear()
            self._channel_rms.clear()
        return events

    def flush(self) -> list[AcousticEvent]:
        events = [self._build(d) for d in self.detector.flush()]
        self._pending.clear()
        self._channel_rms.clear()
        return events

    def stream_live(self, timeout: float = 1.0) -> Iterator[AcousticEvent]:
        """One event per FRAME, emitted as soon as the frame arrives.

        USE THIS FOR A LIVE DISPLAY. `stream()` emits on run COMPLETION: a
        continuous sound produces nothing at all until it stops, and a merged
        run reports the average of its whole duration. That is right for
        logging discrete events and badly wrong for a real-time overlay, where
        it reads as seconds of lag.

        The cost is honest: each event covers one 64 ms frame, so it is noisier
        than a merged run. The display's own decay window does the smoothing.
        """
        from acoustic_array.analysis import channel_rms

        while True:
            frame = self.receiver.read_frame(timeout=timeout)
            if frame is None:
                if self.receiver.is_exhausted:
                    return
                continue

            doa = estimate_doa(frame, self.array, min_confidence=self.min_confidence)
            stats = self.detector.classify_frame(frame)
            rms = tuple(float(v) for v in channel_rms(frame))
            usable = (doa.valid and doa.angle_degrees is not None
                      and doa.confidence >= self.min_confidence)
            yield AcousticEvent(
                timestamp=frame.timestamp,
                event_type=stats[0].value,
                direction_degrees=float(doa.angle_degrees) if usable else None,
                confidence=float(stats[1]),
                localization_confidence=float(doa.confidence) if usable else None,
                angular_resolution_degrees=(
                    doa.angular_resolution_degrees if usable else None),
                duration=frame.duration,
                channel_rms=rms,
                source_kind=self.source_kind,
                reason="" if usable else (doa.reason or
                    f"localization confidence {doa.confidence:.2f} below "
                    f"{self.min_confidence:.2f}"),
            )

    def stream(self, timeout: float = 1.0) -> Iterator[AcousticEvent]:
        """Yield events until the source is exhausted."""
        while True:
            frame = self.receiver.read_frame(timeout=timeout)
            if frame is None:
                if self.receiver.is_exhausted:
                    break
                continue
            for event in self.process_frame(frame):
                yield event
        for event in self.flush():
            yield event

    # --- assembling one event -------------------------------------------------

    def _build(self, detected: DetectedEvent) -> AcousticEvent:
        usable = [d for d in self._pending
                  if d.valid and d.angle_degrees is not None
                  and d.confidence >= self.min_confidence]
        rms = self._mean_channel_rms()

        if not usable:
            return AcousticEvent(
                timestamp=detected.timestamp,
                event_type=detected.event_type.value,
                direction_degrees=None,
                confidence=detected.confidence,
                localization_confidence=None,
                angular_resolution_degrees=None,
                duration=detected.duration,
                channel_rms=rms,
                source_kind=self.source_kind,
                reason=self._decline_reason(),
            )

        weights = [d.confidence for d in usable]
        total = sum(weights) or 1.0
        bearing = sum(d.angle_degrees * w for d, w in zip(usable, weights)) / total
        localization = total / len(usable)
        resolution = next(
            (d.angular_resolution_degrees for d in usable
             if d.angular_resolution_degrees is not None),
            angular_resolution_degrees(bearing, self.array.spacing, self.sample_rate),
        )
        return AcousticEvent(
            timestamp=detected.timestamp,
            event_type=detected.event_type.value,
            direction_degrees=float(bearing),
            confidence=detected.confidence,
            localization_confidence=float(localization),
            angular_resolution_degrees=resolution,
            duration=detected.duration,
            channel_rms=rms,
            source_kind=self.source_kind,
        )

    def _mean_channel_rms(self) -> tuple[float, ...]:
        if not self._channel_rms:
            return tuple(0.0 for _ in range(self.array.num_channels))
        columns = zip(*self._channel_rms)
        return tuple(sum(values) / len(self._channel_rms) for values in columns)

    def _decline_reason(self) -> str:
        """Why no bearing. Always a sentence, never an empty string."""
        if not self._pending:
            return "no frames were analysed for direction"
        invalid = [d for d in self._pending if not d.valid]
        if len(invalid) == len(self._pending):
            reasons = {d.reason for d in invalid if d.reason}
            return ("no valid correlation: "
                    + ("; ".join(sorted(reasons)) if reasons else "silent or degenerate input"))
        best = max((d.confidence for d in self._pending), default=0.0)
        return (f"localization confidence {best:.2f} is below the "
                f"{self.min_confidence:.2f} threshold; the correlation peak is "
                f"not distinguishable from a sidelobe")
