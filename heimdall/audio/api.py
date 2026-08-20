"""Phase K: the public interface to the audio module.

This is the ONLY module the rest of Heimdall should import:

    from heimdall.audio.api import AudioEvent, AudioModule

Vision and fusion do not need to know that GCC-PHAT exists, what a TDOA is, or
how many microphones are attached. They get AudioEvents.

An AudioEvent is EVIDENCE, not a judgement. `event_type` says what the sound
resembled; `seat_id` says which seat best matches the measured direction, which
with a two-microphone array is a bearing match and not a position fix. Both can
be wrong. Fusion weighs them against vision; the audio module never concludes
that anyone cheated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from heimdall.audio.config import AudioConfig, load_audio_config
from heimdall.audio.doa import DoaResult, estimate_doa
from heimdall.audio.events import AudioEventDetector, DetectedEvent, EventType
from heimdall.audio.frame import AudioFrame
from heimdall.audio.geometry import ClassroomConfig, load_classroom_config
from heimdall.audio.receiver import AudioReceiver
from heimdall.audio.seat_mapper import DEFAULT_MIN_CONFIDENCE, SeatMatch, map_audio_to_seat
from heimdall.audio.sources import AudioSource, SyntheticAudioSource

SOURCE_NAME = "microphone_array"


@dataclass(frozen=True)
class AudioEvent:
    """One piece of audio evidence, ready for the fusion engine."""

    timestamp: float
    event_type: str
    confidence: float
    duration: float
    seat_id: str | None = None
    direction_degrees: float | None = None
    position: dict[str, float] | None = None
    source: str = SOURCE_NAME

    # Honesty fields. Fusion may ignore them, but it must be able to see them.
    localization_confidence: float | None = None
    seat_ambiguous: bool = False
    candidate_seats: tuple[str, ...] = ()
    angular_resolution_degrees: float | None = None
    rms: float = 0.0
    peak: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "seat_id": self.seat_id,
            "direction_degrees": self.direction_degrees,
            "position": self.position,
            "confidence": self.confidence,
            "duration": self.duration,
            "source": self.source,
            "localization_confidence": self.localization_confidence,
            "seat_ambiguous": self.seat_ambiguous,
            "candidate_seats": list(self.candidate_seats),
            "angular_resolution_degrees": self.angular_resolution_degrees,
            "notes": self.notes,
        }


@dataclass
class StageTimings:
    """Per-stage latency in milliseconds, for the performance report."""

    samples: dict[str, list[float]] = field(default_factory=dict)

    def record(self, stage: str, seconds: float) -> None:
        self.samples.setdefault(stage, []).append(seconds * 1000.0)

    def report(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for stage, values in self.samples.items():
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            out[stage] = {
                "mean_ms": float(np.mean(array)),
                "p95_ms": float(np.percentile(array, 95)),
                "max_ms": float(np.max(array)),
                "count": int(array.size),
            }
        return out


class AudioModule:
    """The whole audio pipeline behind one object.

        module = AudioModule.synthetic()          # no hardware needed
        with module:
            for event in module.stream(seconds=2.0):
                print(event.to_dict())

    Swapping SyntheticAudioSource for a future ESP32AudioSource is the only
    change needed to run on real microphones.
    """

    def __init__(
        self,
        source: AudioSource,
        classroom: ClassroomConfig | None = None,
        *,
        detector: AudioEventDetector | None = None,
        min_localization_confidence: float = DEFAULT_MIN_CONFIDENCE,
        queue_size: int = 32,
    ) -> None:
        self.classroom = classroom or load_classroom_config()
        self.receiver = AudioReceiver(source, queue_size=queue_size)
        self.detector = detector or AudioEventDetector()
        self.min_localization_confidence = min_localization_confidence
        self.timings = StageTimings()
        self._pending: list[DoaResult] = []

    @classmethod
    def synthetic(
        cls,
        classroom: ClassroomConfig | None = None,
        audio_config: AudioConfig | None = None,
        *,
        detector: AudioEventDetector | None = None,
        min_localization_confidence: float = DEFAULT_MIN_CONFIDENCE,
        queue_size: int = 32,
        **source_kwargs: object,
    ) -> "AudioModule":
        """Build a fully working pipeline with no hardware attached.

        Module-level options are named explicitly; everything else is forwarded
        to SyntheticAudioSource, so a typo in a source argument still fails
        loudly instead of being silently accepted here.
        """
        classroom = classroom or load_classroom_config()
        audio_config = audio_config or load_audio_config()
        source_kwargs.setdefault("mic_spacing_m", classroom.array.spacing)
        source = SyntheticAudioSource(
            sample_rate=audio_config.sample_rate,
            num_channels=max(audio_config.num_channels, classroom.array.num_channels),
            frame_size=audio_config.frame_size,
            **source_kwargs,  # type: ignore[arg-type]
        )
        return cls(
            source,
            classroom,
            detector=detector,
            min_localization_confidence=min_localization_confidence,
            queue_size=queue_size,
        )

    @property
    def sample_rate(self) -> int:
        return self.receiver.sample_rate

    @property
    def num_channels(self) -> int:
        return self.receiver.num_channels

    def start(self) -> None:
        self.receiver.start()

    def stop(self) -> None:
        self.receiver.stop()

    def __enter__(self) -> "AudioModule":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def process_frame(self, frame: AudioFrame) -> list[AudioEvent]:
        """Run one frame through localization and detection.

        Returns the events that completed on this frame, which is usually none.
        """
        started = time.perf_counter()
        localization = estimate_doa(frame, self.classroom.array)
        self.timings.record("doa", time.perf_counter() - started)

        previous_state = self.detector.current_state

        detect_started = time.perf_counter()
        completed = self.detector.process(frame)
        self.timings.record("detect", time.perf_counter() - detect_started)

        events: list[AudioEvent] = []

        # `_pending` holds the localizations of the run that just ended. It must
        # be cleared on every run boundary, not only when an event is emitted:
        # the detector drops runs shorter than min_duration, and those dropped
        # frames would otherwise leak into the next event and drag its bearing
        # away from the true one.
        if completed or self.detector.current_state is not previous_state:
            for detected in completed:
                events.append(self._build_event(detected, self._pending))
            self._pending = []

        self._pending.append(localization)
        return events

    def flush(self) -> list[AudioEvent]:
        """Emit any event still in progress. Call at end of stream."""
        events = [self._build_event(d, self._pending) for d in self.detector.flush()]
        self._pending = []
        return events

    def poll(self, timeout: float | None = 1.0) -> list[AudioEvent]:
        """Read one frame if available and return any completed events."""
        frame = self.receiver.read_frame(timeout=timeout)
        if frame is None:
            return []
        return self.process_frame(frame)

    def stream(self, seconds: float | None = None, max_frames: int | None = None):
        """Yield AudioEvents until the source is exhausted or the time is up."""
        deadline = None if seconds is None else time.monotonic() + seconds
        frames_read = 0

        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if max_frames is not None and frames_read >= max_frames:
                break

            frame = self.receiver.read_frame(timeout=0.5)
            if frame is None:
                if self.receiver.is_exhausted:
                    break
                continue

            frames_read += 1
            for event in self.process_frame(frame):
                yield event

        for event in self.flush():
            yield event

    def performance_report(self) -> dict[str, dict[str, float]]:
        """Mean/p95/max latency per pipeline stage, in milliseconds."""
        return self.timings.report()

    def _aggregate(self, localizations: list[DoaResult]) -> DoaResult | None:
        """Combine per-frame bearings into one estimate for the whole event.

        Only frames whose correlation was trustworthy contribute; if none were,
        the event is reported with no direction rather than a made-up one.
        """
        usable = [
            loc
            for loc in localizations
            if loc.valid
            and loc.angle_degrees is not None
            and loc.confidence >= self.min_localization_confidence
        ]
        if not usable:
            return None

        weights = np.array([loc.confidence for loc in usable], dtype=np.float64)
        angles = np.array([loc.angle_degrees for loc in usable], dtype=np.float64)
        total = float(np.sum(weights))
        if total <= 0:
            return None

        mean_angle = float(np.sum(angles * weights) / total)
        mean_confidence = float(np.mean(weights))
        resolutions = [
            loc.angular_resolution_degrees
            for loc in usable
            if loc.angular_resolution_degrees is not None
        ]

        best = max(usable, key=lambda loc: loc.confidence)
        return DoaResult(
            angle_degrees=mean_angle,
            confidence=mean_confidence,
            tdoa_seconds=float(np.mean([loc.tdoa_seconds for loc in usable])),
            tdoa_samples=float(np.mean([loc.tdoa_samples for loc in usable])),
            valid=True,
            ambiguous=best.ambiguous,
            alternative_angle_degrees=best.alternative_angle_degrees,
            position=best.position,
            angular_resolution_degrees=float(np.mean(resolutions)) if resolutions else None,
            num_channels=best.num_channels,
        )

    def _build_event(self, detected: DetectedEvent, localizations: list[DoaResult]) -> AudioEvent:
        started = time.perf_counter()
        localization = self._aggregate(localizations)

        seat_match: SeatMatch | None = None
        if localization is not None:
            seat_match = map_audio_to_seat(
                localization,
                self.classroom,
                min_confidence=self.min_localization_confidence,
            )
        self.timings.record("seat_mapping", time.perf_counter() - started)

        position = None
        if localization is not None and localization.position is not None:
            position = {"x": float(localization.position[0]), "y": float(localization.position[1])}

        notes = ""
        if localization is None:
            notes = "no trustworthy direction estimate for this event"
        elif seat_match is not None and not seat_match.matched:
            notes = seat_match.reason
        elif seat_match is not None and seat_match.ambiguous:
            notes = seat_match.reason

        return AudioEvent(
            timestamp=detected.timestamp,
            event_type=detected.event_type.value,
            confidence=detected.confidence,
            duration=detected.duration,
            seat_id=seat_match.seat_id if seat_match else None,
            direction_degrees=localization.angle_degrees if localization else None,
            position=position,
            localization_confidence=localization.confidence if localization else None,
            seat_ambiguous=bool(seat_match.ambiguous) if seat_match else False,
            candidate_seats=tuple(c.seat_id for c in seat_match.candidates) if seat_match else (),
            angular_resolution_degrees=(
                localization.angular_resolution_degrees if localization else None
            ),
            rms=detected.rms,
            peak=detected.peak,
            notes=notes,
        )


__all__ = ["AudioEvent", "AudioModule", "EventType", "SOURCE_NAME"]
