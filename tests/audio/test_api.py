"""The public interface, exercised the way the fusion engine will use it.

If these tests pass, vision/fusion can consume audio events without knowing
that GCC-PHAT, TDOAs or microphones exist.
"""

import json

import numpy as np
import pytest

from heimdall.audio.api import AudioEvent, AudioModule, EventType
from heimdall.audio.events import AudioEventDetector, FrameClassifier
from heimdall.audio.frame import AudioFrame
from heimdall.audio.geometry import load_classroom_config
from heimdall.audio.sources import SyntheticAudioSource

SAMPLE_RATE = 48000
FRAME_SIZE = 1024


@pytest.fixture
def room():
    return load_classroom_config()


def module_aimed_at(room, seat_id, **kwargs):
    """A pipeline fed by a synthetic source placed on a real seat's bearing."""
    bearing = room.array.bearing_to(room.seat(seat_id).position)
    kwargs.setdefault("burst_frames", 12)
    kwargs.setdefault("silence_frames", 8)
    kwargs.setdefault("max_frames", 40)
    return AudioModule.synthetic(classroom=room, angle_degrees=bearing, **kwargs)


def collect(module):
    with module:
        return list(module.stream())


# --- the public surface ------------------------------------------------------

def test_public_imports_are_available():
    from heimdall.audio.api import AudioEvent as Event
    from heimdall.audio.api import AudioModule as Module

    assert Event is AudioEvent and Module is AudioModule


def test_module_builds_without_any_hardware():
    module = AudioModule.synthetic(max_frames=2)
    assert module.sample_rate > 0
    assert module.num_channels >= 2


def test_module_reports_the_configured_format(room):
    module = AudioModule.synthetic(classroom=room, max_frames=2)
    assert module.sample_rate == 48000
    assert module.num_channels == 2


# --- end to end --------------------------------------------------------------

def test_a_speech_burst_becomes_an_audio_event(room):
    events = collect(module_aimed_at(room, "B4"))
    assert events
    assert all(isinstance(e, AudioEvent) for e in events)
    assert any(e.event_type == EventType.POSSIBLE_SPEECH.value for e in events)


def test_event_carries_the_fields_fusion_expects(room):
    event = collect(module_aimed_at(room, "B4"))[0]
    payload = event.to_dict()

    for key in (
        "timestamp",
        "event_type",
        "seat_id",
        "direction_degrees",
        "confidence",
        "duration",
        "source",
    ):
        assert key in payload


def test_event_is_json_serialisable(room):
    event = collect(module_aimed_at(room, "B4"))[0]
    restored = json.loads(json.dumps(event.to_dict()))
    assert restored["event_type"] == event.event_type


def test_source_is_labelled_as_the_microphone_array(room):
    event = collect(module_aimed_at(room, "B4"))[0]
    assert event.source == "microphone_array"


@pytest.mark.parametrize("seat_id", ["A1", "A6", "C3", "E6"])
def test_the_targeted_seat_is_reported(room, seat_id):
    events = collect(module_aimed_at(room, seat_id))
    assert events
    assert events[0].seat_id == seat_id


def test_direction_matches_the_seat_bearing(room):
    expected = room.array.bearing_to(room.seat("A1").position)
    event = collect(module_aimed_at(room, "A1"))[0]
    assert event.direction_degrees == pytest.approx(expected, abs=1.5)


def test_event_duration_and_timestamp_are_sane(room):
    events = collect(module_aimed_at(room, "B4"))
    for event in events:
        assert event.duration > 0
        assert event.timestamp >= 0
    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps)


def test_confidences_are_in_range(room):
    for event in collect(module_aimed_at(room, "B4")):
        assert 0.0 <= event.confidence <= 1.0
        assert event.localization_confidence is None or 0.0 <= event.localization_confidence <= 1.0


def test_repeated_bursts_produce_repeated_events(room):
    events = collect(module_aimed_at(room, "C3", burst_frames=8, silence_frames=8, max_frames=64))
    speech = [e for e in events if e.event_type == EventType.POSSIBLE_SPEECH.value]
    assert len(speech) >= 3

    # The winning seat may flip between seats that share the bearing - that is
    # the linear-array ambiguity, not instability. What must hold every time is
    # that the true seat is offered and the bearing stays put.
    for event in speech:
        assert "C3" in event.candidate_seats
        assert event.direction_degrees == pytest.approx(
            room.array.bearing_to(room.seat("C3").position), abs=1.5
        )


# --- what the API must not claim ---------------------------------------------

def test_two_microphones_never_report_a_position(room):
    for event in collect(module_aimed_at(room, "B4")):
        assert event.position is None


def test_bearing_ambiguity_is_exposed_to_fusion(room):
    """Fusion must be able to see that several seats share the bearing."""
    event = collect(module_aimed_at(room, "B4"))[0]
    assert event.seat_ambiguous
    assert len(event.candidate_seats) > 1
    assert event.seat_id in event.candidate_seats
    assert "range is unknown" in event.notes


def test_angular_resolution_is_reported(room):
    event = collect(module_aimed_at(room, "B4"))[0]
    assert event.angular_resolution_degrees is not None
    assert event.angular_resolution_degrees > 0


def test_event_types_are_evidence_not_verdicts(room):
    """Nothing in the vocabulary asserts cheating."""
    for event in collect(module_aimed_at(room, "B4")):
        assert event.event_type in {t.value for t in EventType}
        assert "CHEAT" not in event.event_type.upper()


# --- silence and degenerate input --------------------------------------------

def test_a_silent_room_produces_no_events(room):
    module = AudioModule.synthetic(classroom=room, burst_frames=0, max_frames=30)
    assert collect(module) == []


def test_uncorrelated_channels_yield_no_direction(room):
    """Two mics hearing unrelated noise must not be turned into a seat."""
    left = np.random.default_rng(1).standard_normal(FRAME_SIZE * 20) * 0.3
    right = np.random.default_rng(2).standard_normal(FRAME_SIZE * 20) * 0.3
    buffer = np.stack([left, right], axis=1).astype(np.float32)

    source = SyntheticAudioSource.from_buffer(buffer, SAMPLE_RATE, frame_size=FRAME_SIZE)
    module = AudioModule(source, room)
    events = collect(module)

    for event in events:
        assert event.seat_id is None
        assert event.direction_degrees is None
        assert "no trustworthy direction" in event.notes


def test_events_without_direction_are_still_reported(room):
    """Losing the direction must not lose the fact that a sound happened."""
    left = np.random.default_rng(3).standard_normal(FRAME_SIZE * 20) * 0.3
    right = np.random.default_rng(4).standard_normal(FRAME_SIZE * 20) * 0.3
    buffer = np.stack([left, right], axis=1).astype(np.float32)

    source = SyntheticAudioSource.from_buffer(buffer, SAMPLE_RATE, frame_size=FRAME_SIZE)
    events = collect(AudioModule(source, room))
    assert events
    assert all(e.event_type != EventType.SILENCE.value for e in events)


def test_high_confidence_threshold_suppresses_seat_assignment(room):
    bearing = room.array.bearing_to(room.seat("B4").position)
    module = AudioModule.synthetic(
        classroom=room, angle_degrees=bearing, max_frames=30, min_localization_confidence=0.999
    )
    for event in collect(module):
        assert event.seat_id is None


# --- frame-level API ---------------------------------------------------------

def test_process_frame_can_be_driven_manually(room):
    module = AudioModule.synthetic(classroom=room, max_frames=1)
    source = SyntheticAudioSource(
        SAMPLE_RATE, 2, FRAME_SIZE, angle_degrees=0.0, mic_spacing_m=room.array.spacing
    )
    source.start()
    events = []
    for _ in range(20):
        events += module.process_frame(source.read_frame())
    events += module.flush()
    source.stop()

    assert events
    assert isinstance(events[0], AudioEvent)


def test_poll_returns_a_list_even_with_no_data(room):
    module = AudioModule.synthetic(classroom=room, max_frames=0)
    module.start()
    assert module.poll(timeout=0.1) == []
    module.stop()


def test_stream_stops_when_the_source_is_exhausted(room):
    module = module_aimed_at(room, "B4", max_frames=12)
    with module:
        events = list(module.stream())
    assert isinstance(events, list)


def test_stream_respects_max_frames(room):
    module = AudioModule.synthetic(classroom=room, max_frames=1000)
    with module:
        list(module.stream(max_frames=5))
    assert module.receiver.stats.frames_received >= 5


def test_context_manager_stops_the_receiver(room):
    module = AudioModule.synthetic(classroom=room, max_frames=4)
    with module:
        assert module.receiver.is_running
    assert not module.receiver.is_running


# --- performance -------------------------------------------------------------

def test_performance_report_has_mean_and_p95(room):
    module = module_aimed_at(room, "B4")
    collect(module)
    report = module.performance_report()

    assert {"doa", "detect", "seat_mapping"} <= set(report)
    for stage in report.values():
        assert stage["mean_ms"] >= 0
        assert stage["p95_ms"] >= stage["mean_ms"] * 0.5
        assert stage["count"] > 0


def test_processing_keeps_up_with_real_time(room):
    """Per-frame work must cost less than the frame's own duration."""
    module = module_aimed_at(room, "B4", max_frames=60)
    collect(module)
    report = module.performance_report()

    frame_duration_ms = 1000.0 * FRAME_SIZE / SAMPLE_RATE
    per_frame_ms = report["doa"]["mean_ms"] + report["detect"]["mean_ms"]
    assert per_frame_ms < frame_duration_ms


# --- customisation -----------------------------------------------------------

def test_detector_can_be_replaced(room):
    from heimdall.audio.events import EventType as InternalEventType

    class AlwaysSpeech(FrameClassifier):
        def classify(self, stats, noise_floor):
            return InternalEventType.POSSIBLE_SPEECH, 0.9

    module = AudioModule.synthetic(
        classroom=room,
        burst_frames=0,
        max_frames=20,
        detector=AudioEventDetector(classifier=AlwaysSpeech()),
    )
    events = collect(module)
    assert events
    assert events[0].event_type == "POSSIBLE_SPEECH"


def test_a_different_classroom_changes_the_seat_ids(tmp_path):
    path = tmp_path / "classroom.yaml"
    path.write_text(
        "classroom: {width: 6.0, height: 8.0}\n"
        "microphones:\n"
        "  - {id: m1, x: 2.7, y: 0.0}\n"
        "  - {id: m2, x: 3.3, y: 0.0}\n"
        "array: {orientation_degrees: 90.0}\n"
        "seats:\n"
        "  grid: {rows: 2, columns: 2, row_spacing: 1.0, column_spacing: 1.0,\n"
        "         origin_x: 2.0, origin_y: 3.0}\n",
        encoding="utf-8",
    )

    from heimdall.audio.geometry import load_classroom_config as load

    room = load(path)
    events = collect(module_aimed_at(room, "B2"))
    assert events
    assert events[0].seat_id in {"A1", "A2", "B1", "B2"}


def test_four_channel_frames_are_rejected_not_silently_mishandled(room):
    """Until a 4-mic geometry is configured, a 4-channel frame has no direction."""
    module = AudioModule.synthetic(classroom=room, max_frames=1)
    samples = np.random.default_rng(0).standard_normal((FRAME_SIZE, 4)).astype(np.float32) * 0.2
    frame = AudioFrame(samples, 0.0, 0, SAMPLE_RATE)

    module.process_frame(frame)
    events = module.flush()
    for event in events:
        assert event.direction_degrees is None
        assert event.seat_id is None
