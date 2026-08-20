"""Phase J: audio event detection.

WHAT THIS DOES AND DOES NOT MEAN
--------------------------------
This module reports that a sound occurred, roughly how loud it was, and whether
it looked speech-like. That is all. Speech in an exam room is evidence, not a
verdict: a cough, a question to the invigilator, a chair scraping and a
whispered answer all pass through here the same way. Nothing in this file
decides that anyone cheated, and no caller should treat POSSIBLE_SPEECH as if
it had. The fusion engine combines this with vision and makes that call.

The classifier is deliberately swappable. `FrameClassifier` is the seam: the
current one is energy and spectrum heuristics with no model and no extra
dependencies. A trained VAD or speech model drops in behind the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from heimdall.audio.analysis import FrameStats, analyse_frame
from heimdall.audio.frame import AudioFrame


class EventType(str, Enum):
    SILENCE = "SILENCE"
    SOUND_DETECTED = "SOUND_DETECTED"
    POSSIBLE_SPEECH = "POSSIBLE_SPEECH"
    POSSIBLE_WHISPER = "POSSIBLE_WHISPER"

    @property
    def is_active(self) -> bool:
        """True for anything other than silence."""
        return self is not EventType.SILENCE


@dataclass(frozen=True)
class DetectedEvent:
    """One contiguous run of frames sharing an event type.

    `localization` and `seat_id` are filled in by whoever knows about geometry -
    this module never computes them.
    """

    event_type: EventType
    timestamp: float
    duration: float
    confidence: float
    rms: float
    peak: float
    num_frames: int
    localization: object | None = None
    seat_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "confidence": self.confidence,
            "rms": self.rms,
            "peak": self.peak,
            "seat_id": self.seat_id,
        }


class FrameClassifier(ABC):
    """The swap point for a smarter model. One frame in, one label out."""

    @abstractmethod
    def classify(self, stats: FrameStats, noise_floor: float) -> tuple[EventType, float]:
        """Return (event type, confidence in [0, 1]) for a single frame."""


@dataclass
class EnergyClassifier(FrameClassifier):
    """Energy plus coarse spectral shape. No model, no training data.

    A frame counts as sound when it rises `activation_db` above the tracked
    noise floor. It is then called speech-like when most of its energy sits in
    the speech band and the spectrum is harmonic rather than noise-like.
    Whisper detection is the same test at a much lower level, and is the least
    reliable thing here - treat it as a hint, never as a finding.
    """

    activation_db: float = 8.0
    whisper_ceiling_db: float = 18.0
    speech_band_ratio_min: float = 0.35
    spectral_flatness_max: float = 0.55
    zero_crossing_rate_max: float = 0.35
    absolute_floor: float = 1e-5

    def classify(self, stats: FrameStats, noise_floor: float) -> tuple[EventType, float]:
        floor = max(noise_floor, self.absolute_floor)
        excess_db = 20.0 * np.log10(max(stats.rms, 1e-12) / floor)

        if excess_db < self.activation_db:
            return EventType.SILENCE, float(np.clip(1.0 - excess_db / self.activation_db, 0.0, 1.0))

        # How far above the activation threshold, saturating at +20 dB.
        level_confidence = float(np.clip((excess_db - self.activation_db) / 20.0, 0.0, 1.0))

        speech_like = (
            stats.speech_band_ratio >= self.speech_band_ratio_min
            and stats.spectral_flatness <= self.spectral_flatness_max
            and stats.zero_crossing_rate <= self.zero_crossing_rate_max
        )

        if not speech_like:
            return EventType.SOUND_DETECTED, float(np.clip(0.4 + 0.6 * level_confidence, 0.0, 1.0))

        # Speech-like but barely above the floor: possibly a whisper. This is a
        # weak heuristic, so its confidence is capped well below normal speech.
        if excess_db < self.whisper_ceiling_db:
            return EventType.POSSIBLE_WHISPER, float(np.clip(0.25 + 0.25 * level_confidence, 0.0, 0.6))

        shape_confidence = float(
            np.clip(stats.speech_band_ratio, 0.0, 1.0)
            * np.clip(1.0 - stats.spectral_flatness, 0.0, 1.0)
        )
        return EventType.POSSIBLE_SPEECH, float(
            np.clip(0.5 * shape_confidence + 0.5 * level_confidence, 0.0, 1.0)
        )


@dataclass
class NoiseFloorTracker:
    """Tracks the background level. Falls quickly, rises slowly, so a long
    burst of speech cannot drag the floor up and hide itself.

    The first `warmup_frames` frames are a calibration period during which the
    floor follows the input quickly in both directions. Without it the floor
    starts at an arbitrary guess, and a room whose ambient level sits above that
    guess reports its own background hiss as a sound event until the slow
    release finally catches up.

    `maximum` is what stops calibration from backfiring. A stream that opens
    with someone already talking would otherwise calibrate the floor to speech
    level and then hear nothing for the rest of the session. A real classroom
    background is never that loud, so if the tracker believes it is, the tracker
    is wrong and gets clamped. The default of 0.01 is about -40 dBFS, well below
    conversational speech and well above a quiet room.
    """

    value: float = 1e-4
    attack: float = 0.05   # weight when the level drops (adapt fast)
    release: float = 0.001  # weight when the level rises (adapt slowly)
    minimum: float = 1e-6
    maximum: float = 0.01
    warmup_frames: int = 8
    _seen: int = field(default=0, init=False, repr=False)

    @property
    def is_warming_up(self) -> bool:
        return self._seen < self.warmup_frames

    def update(self, rms: float) -> float:
        rms = max(rms, 0.0)
        if self.is_warming_up:
            # Converge on the real ambient level instead of the constructor
            # default, fast enough to be settled within a fraction of a second.
            weight = 1.0 if self._seen == 0 else self.attack * 4.0
        else:
            weight = self.attack if rms < self.value else self.release

        self._seen += 1
        updated = (1.0 - weight) * self.value + weight * rms
        self.value = min(max(updated, self.minimum), max(self.maximum, self.minimum))
        return self.value


@dataclass
class AudioEventDetector:
    """Turns a stream of frames into a stream of events.

    Frames are classified individually, then runs of the same label are merged.
    A run must last at least `min_duration` to be emitted, which suppresses
    single-frame blips. Call `flush()` at the end of a stream to emit whatever
    was still in progress.
    """

    classifier: FrameClassifier = field(default_factory=EnergyClassifier)
    noise_floor: NoiseFloorTracker = field(default_factory=NoiseFloorTracker)
    min_duration: float = 0.10
    emit_silence: bool = False

    _current_type: EventType | None = field(default=None, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    _end_time: float = field(default=0.0, init=False, repr=False)
    _confidences: list[float] = field(default_factory=list, init=False, repr=False)
    _rms: list[float] = field(default_factory=list, init=False, repr=False)
    _peaks: list[float] = field(default_factory=list, init=False, repr=False)
    _frames: int = field(default=0, init=False, repr=False)

    @property
    def current_state(self) -> EventType:
        return self._current_type or EventType.SILENCE

    def reset(self) -> None:
        self._current_type = None
        self._confidences.clear()
        self._rms.clear()
        self._peaks.clear()
        self._frames = 0

    def process(self, frame: AudioFrame) -> list[DetectedEvent]:
        """Feed one frame. Returns any events that just completed."""
        stats = analyse_frame(frame)
        floor = self.noise_floor.update(stats.rms)
        event_type, confidence = self.classifier.classify(stats, floor)

        completed: list[DetectedEvent] = []
        if self._current_type is not None and event_type is not self._current_type:
            finished = self._finish()
            if finished is not None:
                completed.append(finished)

        if self._current_type is None:
            self._current_type = event_type
            self._start_time = frame.timestamp

        self._end_time = frame.end_timestamp
        self._confidences.append(confidence)
        self._rms.append(stats.rms)
        self._peaks.append(stats.peak)
        self._frames += 1
        return completed

    def process_stats(self, stats: FrameStats, duration: float) -> list[DetectedEvent]:
        """Same as process(), for callers that already computed FrameStats."""
        floor = self.noise_floor.update(stats.rms)
        event_type, confidence = self.classifier.classify(stats, floor)

        completed: list[DetectedEvent] = []
        if self._current_type is not None and event_type is not self._current_type:
            finished = self._finish()
            if finished is not None:
                completed.append(finished)

        if self._current_type is None:
            self._current_type = event_type
            self._start_time = stats.timestamp

        self._end_time = stats.timestamp + duration
        self._confidences.append(confidence)
        self._rms.append(stats.rms)
        self._peaks.append(stats.peak)
        self._frames += 1
        return completed

    def flush(self) -> list[DetectedEvent]:
        """Emit the event still in progress, if any."""
        finished = self._finish()
        return [finished] if finished is not None else []

    def _finish(self) -> DetectedEvent | None:
        event_type = self._current_type
        frames = self._frames
        duration = self._end_time - self._start_time

        confidence = float(np.mean(self._confidences)) if self._confidences else 0.0
        rms = float(np.mean(self._rms)) if self._rms else 0.0
        peak = float(np.max(self._peaks)) if self._peaks else 0.0
        start = self._start_time

        self.reset()

        if event_type is None or frames == 0:
            return None
        if duration + 1e-9 < self.min_duration:
            return None
        if event_type is EventType.SILENCE and not self.emit_silence:
            return None

        return DetectedEvent(
            event_type=event_type,
            timestamp=start,
            duration=duration,
            confidence=confidence,
            rms=rms,
            peak=peak,
            num_frames=frames,
        )
