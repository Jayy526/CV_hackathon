"""Audio event detection on synthetic frames.

Covers the no-audio condition, run merging, the swappable classifier, and the
boundary the module must not cross (speech is evidence, not a verdict).
"""

import numpy as np
import pytest

from heimdall.audio import synthetic
from heimdall.audio.analysis import FrameStats, analyse_frame
from heimdall.audio.events import (
    AudioEventDetector,
    EnergyClassifier,
    EventType,
    FrameClassifier,
    NoiseFloorTracker,
)
from heimdall.audio.frame import AudioFrame
from heimdall.audio.sources import SyntheticAudioSource

SAMPLE_RATE = 48000
FRAME_SIZE = 1024
FRAME_DURATION = FRAME_SIZE / SAMPLE_RATE


def frame_from_mono(mono, index, num_channels=2):
    samples = np.stack([mono] * num_channels, axis=1).astype(np.float32)
    return AudioFrame(samples, index * FRAME_DURATION, index, SAMPLE_RATE)


def silence_frame(index, level=0.0):
    mono = np.full(FRAME_SIZE, level, dtype=np.float64)
    return frame_from_mono(mono, index)


def speech_frame(index, amplitude=0.3):
    mono = synthetic.speech_like(FRAME_SIZE, SAMPLE_RATE, seed=index, amplitude=amplitude)
    return frame_from_mono(mono, index)


def noise_frame(index, amplitude=0.3):
    mono = synthetic.white_noise(FRAME_SIZE, seed=index, amplitude=amplitude)
    return frame_from_mono(mono, index)


def run(detector, frames):
    events = []
    for frame in frames:
        events += detector.process(frame)
    events += detector.flush()
    return events


# --- the no-audio condition --------------------------------------------------

def test_pure_silence_produces_no_events():
    detector = AudioEventDetector()
    events = run(detector, [silence_frame(i) for i in range(20)])
    assert events == []
    assert detector.current_state is EventType.SILENCE


def test_silence_can_be_emitted_when_asked():
    detector = AudioEventDetector(emit_silence=True)
    events = run(detector, [silence_frame(i) for i in range(20)])
    assert len(events) == 1
    assert events[0].event_type is EventType.SILENCE


def test_a_quiet_room_floor_is_still_silence():
    """Low-level background hiss must not be reported as an event."""
    detector = AudioEventDetector()
    frames = [frame_from_mono(synthetic.white_noise(FRAME_SIZE, seed=i, amplitude=2e-4), i)
              for i in range(25)]
    events = run(detector, frames)
    assert all(e.event_type is not EventType.POSSIBLE_SPEECH for e in events)


# --- detecting sound ---------------------------------------------------------

def test_speech_like_audio_is_classified_as_possible_speech():
    detector = AudioEventDetector()
    frames = [silence_frame(i, 1e-5) for i in range(5)]
    frames += [speech_frame(i) for i in range(5, 15)]
    frames += [silence_frame(i, 1e-5) for i in range(15, 20)]

    events = run(detector, frames)
    assert any(e.event_type is EventType.POSSIBLE_SPEECH for e in events)


def test_broadband_noise_is_sound_but_not_speech():
    """A chair scrape or a bang must not be reported as speech."""
    detector = AudioEventDetector()
    frames = [silence_frame(i, 1e-5) for i in range(4)]
    frames += [noise_frame(i) for i in range(4, 14)]

    events = run(detector, frames)
    types = {e.event_type for e in events}
    assert EventType.SOUND_DETECTED in types
    assert EventType.POSSIBLE_SPEECH not in types


def whisper_frames(lead_in=12):
    """A whisper is defined RELATIVE to the room, not in absolute terms: 8-18 dB
    above the noise floor. So the room needs a realistic floor to whisper over,
    plus enough of it for the tracker's calibration window to settle."""
    frames = [
        frame_from_mono(synthetic.white_noise(FRAME_SIZE, seed=i, amplitude=1e-4), i)
        for i in range(lead_in)
    ]
    frames += [
        speech_frame(i, amplitude=0.0015) for i in range(lead_in, lead_in + 12)
    ]
    return frames


def test_quiet_speech_is_flagged_as_a_possible_whisper():
    events = run(AudioEventDetector(), whisper_frames())
    assert any(e.event_type is EventType.POSSIBLE_WHISPER for e in events)


def test_whisper_confidence_is_capped_below_certainty():
    """The whisper heuristic is weak and must never look confident."""
    for event in run(AudioEventDetector(), whisper_frames()):
        if event.event_type is EventType.POSSIBLE_WHISPER:
            assert event.confidence <= 0.6


def test_audio_during_the_calibration_window_is_absorbed():
    """A documented cost of adaptive calibration: a whisper that starts before
    the floor has settled is taken for background and missed."""
    events = run(AudioEventDetector(), whisper_frames(lead_in=2))
    assert not any(e.event_type is EventType.POSSIBLE_WHISPER for e in events)


# --- event timing and merging ------------------------------------------------

def test_a_run_of_frames_becomes_one_event():
    detector = AudioEventDetector()
    events = run(detector, [speech_frame(i) for i in range(12)])
    speech = [e for e in events if e.event_type is EventType.POSSIBLE_SPEECH]
    assert len(speech) == 1
    assert speech[0].num_frames == 12


def test_event_duration_matches_the_frames_it_covers():
    detector = AudioEventDetector()
    events = run(detector, [speech_frame(i) for i in range(10)])
    assert events[0].duration == pytest.approx(10 * FRAME_DURATION, rel=1e-6)


def test_event_timestamp_is_the_start_of_the_run():
    detector = AudioEventDetector()
    frames = [silence_frame(i, 1e-5) for i in range(6)]
    frames += [speech_frame(i) for i in range(6, 18)]

    events = run(detector, frames)
    active = [e for e in events if e.event_type.is_active][0]
    assert active.timestamp == pytest.approx(6 * FRAME_DURATION, rel=1e-6)


def test_two_separated_bursts_produce_two_events():
    detector = AudioEventDetector()
    frames = []
    frames += [speech_frame(i) for i in range(10)]
    frames += [silence_frame(i, 1e-6) for i in range(10, 20)]
    frames += [speech_frame(i) for i in range(20, 30)]

    events = run(detector, frames)
    speech = [e for e in events if e.event_type is EventType.POSSIBLE_SPEECH]
    assert len(speech) == 2
    assert speech[0].timestamp < speech[1].timestamp


def test_a_single_frame_blip_is_suppressed_by_min_duration():
    detector = AudioEventDetector(min_duration=0.5)
    frames = [silence_frame(i, 1e-5) for i in range(4)]
    frames += [speech_frame(4)]
    frames += [silence_frame(i, 1e-5) for i in range(5, 12)]

    events = run(detector, frames)
    assert not [e for e in events if e.event_type.is_active]


def test_min_duration_is_configurable():
    frames = [silence_frame(i, 1e-5) for i in range(3)]
    frames += [speech_frame(i) for i in range(3, 9)]
    frames += [silence_frame(i, 1e-5) for i in range(9, 14)]

    permissive = run(AudioEventDetector(min_duration=0.05), list(frames))
    strict = run(AudioEventDetector(min_duration=5.0), list(frames))

    assert [e for e in permissive if e.event_type.is_active]
    assert not [e for e in strict if e.event_type.is_active]


def test_flush_emits_an_event_still_in_progress():
    detector = AudioEventDetector()
    for i in range(10):
        detector.process(speech_frame(i))
    assert detector.flush()


def test_flush_is_idempotent():
    detector = AudioEventDetector()
    for i in range(10):
        detector.process(speech_frame(i))
    detector.flush()
    assert detector.flush() == []


def test_event_reports_level_statistics():
    detector = AudioEventDetector()
    events = run(detector, [speech_frame(i) for i in range(10)])
    assert events[0].rms > 0
    assert events[0].peak >= events[0].rms


def test_event_is_json_friendly():
    detector = AudioEventDetector()
    payload = run(detector, [speech_frame(i) for i in range(10)])[0].as_dict()
    assert set(payload) >= {"event_type", "timestamp", "duration", "confidence"}
    assert isinstance(payload["event_type"], str)


def test_localization_and_seat_start_empty():
    """This module never computes geometry - the API layer fills these in."""
    detector = AudioEventDetector()
    event = run(detector, [speech_frame(i) for i in range(10)])[0]
    assert event.localization is None
    assert event.seat_id is None


# --- the noise floor tracker -------------------------------------------------

def test_noise_floor_adapts_downward_quickly():
    tracker = NoiseFloorTracker(value=0.1)
    for _ in range(200):
        tracker.update(0.001)
    assert tracker.value < 0.01


def test_noise_floor_rises_slowly_so_speech_cannot_hide_itself():
    tracker = NoiseFloorTracker(value=1e-4)
    for _ in range(50):
        tracker.update(0.5)
    assert tracker.value < 0.05


def test_noise_floor_never_reaches_zero():
    tracker = NoiseFloorTracker(value=1e-3)
    for _ in range(5000):
        tracker.update(0.0)
    assert tracker.value >= tracker.minimum


# --- the classifier is swappable ---------------------------------------------

def test_classifier_can_be_replaced():
    class AlwaysSpeech(FrameClassifier):
        def classify(self, stats: FrameStats, noise_floor: float):
            return EventType.POSSIBLE_SPEECH, 0.99

    detector = AudioEventDetector(classifier=AlwaysSpeech())
    events = run(detector, [silence_frame(i) for i in range(10)])
    assert len(events) == 1
    assert events[0].event_type is EventType.POSSIBLE_SPEECH
    assert events[0].confidence == pytest.approx(0.99)


def test_energy_classifier_thresholds_are_configurable():
    # A 0.3-amplitude burst sits about 60 dB above a 1e-4 floor, so the strict
    # threshold has to clear that to prove the setting is honoured.
    loose = EnergyClassifier(activation_db=1.0)
    strict = EnergyClassifier(activation_db=100.0)
    stats = analyse_frame(speech_frame(0))

    assert loose.classify(stats, 1e-4)[0].is_active
    assert strict.classify(stats, 1e-4)[0] is EventType.SILENCE


def test_classifier_confidence_is_always_in_range():
    classifier = EnergyClassifier()
    for frame in [silence_frame(0), speech_frame(1), noise_frame(2), speech_frame(3, 0.001)]:
        _, confidence = classifier.classify(analyse_frame(frame), 1e-4)
        assert 0.0 <= confidence <= 1.0


def test_process_stats_matches_process():
    frames = [speech_frame(i) for i in range(8)]

    by_frame = run(AudioEventDetector(), frames)

    detector = AudioEventDetector()
    events = []
    for frame in frames:
        events += detector.process_stats(analyse_frame(frame), FRAME_DURATION)
    events += detector.flush()

    assert [e.event_type for e in events] == [e.event_type for e in by_frame]
    assert events[0].num_frames == by_frame[0].num_frames


# --- integration with the mock source ----------------------------------------

def test_detector_runs_over_a_synthetic_source():
    source = SyntheticAudioSource(
        SAMPLE_RATE, 2, FRAME_SIZE, burst_frames=10, silence_frames=10, max_frames=60
    )
    detector = AudioEventDetector()
    source.start()
    events = []
    while True:
        frame = source.read_frame()
        if frame is None:
            break
        events += detector.process(frame)
    events += detector.flush()
    source.stop()

    speech = [e for e in events if e.event_type is EventType.POSSIBLE_SPEECH]
    assert len(speech) == 3  # three bursts in sixty frames


def test_event_type_is_a_plain_string_enum():
    """Fusion serialises these, so they must survive a JSON round trip."""
    assert EventType.POSSIBLE_SPEECH == "POSSIBLE_SPEECH"
    assert EventType.SILENCE.value == "SILENCE"
    assert not EventType.SILENCE.is_active
    assert EventType.SOUND_DETECTED.is_active
