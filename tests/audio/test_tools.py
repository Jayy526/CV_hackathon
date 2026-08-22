"""The diagnostic and calibration tools, driven headlessly.

These run the same code paths a human would, but never open a window, never
touch a COM port and never need a sound card.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load_tool(name):
    """Import a tools/*.py script as a module."""
    spec = importlib.util.spec_from_file_location("tool_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def monitor():
    return load_tool("monitor_audio")


@pytest.fixture(scope="module")
def calibrate():
    return load_tool("calibrate_audio")


@pytest.fixture(scope="module")
def detect():
    return load_tool("detect_device")


@pytest.fixture(scope="module")
def benchmark():
    return load_tool("benchmark_audio")


@pytest.fixture(scope="module")
def room():
    from heimdall.audio.geometry import load_classroom_config

    return load_classroom_config()


@pytest.fixture(scope="module")
def audio_config():
    from heimdall.audio.config import load_audio_config

    return load_audio_config()


# --- detect_device -----------------------------------------------------------

def test_device_detection_runs_without_hardware(detect, capsys):
    exit_code = detect.main()
    output = capsys.readouterr().out
    # 0 = board found, 2 = no board, 1 = no serial ports at all.
    assert exit_code in (0, 1, 2)
    assert "device detection" in output.lower()


# --- monitor_audio -----------------------------------------------------------

def monitor_args(**overrides):
    args = dict(
        source="synthetic",
        seconds=0.3,
        angle=30.0,
        noise=0.002,
        burst_frames=10,
        silence_frames=10,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_monitor_captures_frames(monitor, audio_config, room):
    source = monitor.build_source(monitor_args(), audio_config, room)
    frames, stats = monitor.capture(source, 0.3)

    assert frames
    assert len(frames) == len(stats)
    assert frames[0].num_channels == 2
    assert frames[0].sample_rate == audio_config.sample_rate


def test_monitor_text_report_mentions_both_channels(monitor, audio_config, room, capsys):
    source = monitor.build_source(monitor_args(), audio_config, room)
    frames, stats = monitor.capture(source, 0.3)
    monitor.print_text_report(frames, stats, source.sample_rate)

    output = capsys.readouterr().out
    assert "channel_0" in output and "channel_1" in output
    assert "Sample rate:      %d Hz" % audio_config.sample_rate in output
    assert "Channel balance" in output


def test_monitor_reports_a_silent_channel_as_a_warning(monitor, capsys):
    """The single most useful wiring diagnostic: one mic dead."""
    from heimdall.audio.analysis import analyse_frame
    from heimdall.audio.frame import AudioFrame
    from heimdall.audio import synthetic

    samples = np.zeros((1024, 2), dtype=np.float32)
    samples[:, 0] = synthetic.speech_like(1024, 48000, seed=1)
    frame = AudioFrame(samples, 0.0, 0, 48000)

    monitor.print_text_report([frame], [analyse_frame(frame)], 48000)
    assert "a channel is silent" in capsys.readouterr().out


def test_monitor_warns_on_a_large_channel_imbalance(monitor, capsys):
    from heimdall.audio.analysis import analyse_frame
    from heimdall.audio.frame import AudioFrame
    from heimdall.audio import synthetic

    mono = synthetic.speech_like(1024, 48000, seed=1)
    samples = np.stack([mono, mono * 0.05], axis=1).astype(np.float32)
    frame = AudioFrame(samples, 0.0, 0, 48000)

    monitor.print_text_report([frame], [analyse_frame(frame)], 48000)
    assert "differ by more than 6 dB" in capsys.readouterr().out


def test_monitor_handles_no_audio(monitor, capsys):
    monitor.print_text_report([], [], 48000)
    assert "No audio captured" in capsys.readouterr().out


def test_monitor_writes_a_two_channel_wav(monitor, audio_config, room, tmp_path):
    from heimdall.audio.analysis import concatenate_frames, read_wav, write_wav

    source = monitor.build_source(monitor_args(), audio_config, room)
    frames, _ = monitor.capture(source, 0.3)

    path = write_wav(tmp_path / "capture.wav", concatenate_frames(frames), source.sample_rate)
    loaded, sample_rate = read_wav(path)

    assert sample_rate == source.sample_rate
    assert loaded.shape[1] == 2


def test_monitor_plot_is_written_headlessly(monitor, audio_config, room, tmp_path):
    source = monitor.build_source(monitor_args(), audio_config, room)
    frames, stats = monitor.capture(source, 0.3)

    output = tmp_path / "monitor.png"
    monitor.plot(frames, stats, source.sample_rate, output, show=False)

    assert output.exists() and output.stat().st_size > 0


def test_monitor_builds_a_real_esp32_source_and_refuses_to_guess_a_port(
    monitor, audio_config, room
):
    # Was "refuses: not implemented". It is implemented now, so the honest
    # failure moved from construction to start(), where the port is missing.
    from heimdall.audio.sources import AudioSourceError, ESP32AudioSource

    source = monitor.build_source(monitor_args(source="esp32"), audio_config, room)
    assert isinstance(source, ESP32AudioSource)
    assert source.sample_rate == 16000

    # Clear the configured port rather than assuming it is unset: the shipped
    # config names a real device, and a test must not open the board.
    source.port_name = None
    with pytest.raises(AudioSourceError) as excinfo:
        source.start()
    assert "detect_device.py" in str(excinfo.value)


# --- calibrate_audio ---------------------------------------------------------

def test_calibration_recovers_known_angles(calibrate, audio_config, room):
    factory = calibrate.synthetic_source_factory(audio_config, room, noise=0.002)
    points = calibrate.run_calibration(
        [0.0, 30.0, 60.0], factory, room, num_frames=5,
        min_confidence=0.3, sample_rate=audio_config.sample_rate,
    )

    assert len(points) == 3
    for point in points:
        assert point.estimated_angle_degrees is not None
        assert abs(point.error_degrees) < 2.0
        assert point.frames_used > 0


def test_calibration_records_the_known_angle_and_error(calibrate, audio_config, room):
    factory = calibrate.synthetic_source_factory(audio_config, room, noise=0.002)
    point = calibrate.run_calibration(
        [45.0], factory, room, num_frames=5,
        min_confidence=0.3, sample_rate=audio_config.sample_rate,
    )[0]

    assert point.known_angle_degrees == 45.0
    assert point.error_degrees == pytest.approx(
        point.estimated_angle_degrees - 45.0, abs=1e-9
    )
    assert point.expected_resolution_degrees > 0


def test_calibration_reports_endfire_as_least_reliable(calibrate, audio_config, room):
    """90 degrees is along the array axis, where resolution collapses."""
    factory = calibrate.synthetic_source_factory(audio_config, room, noise=0.002)
    points = calibrate.run_calibration(
        [0.0, 90.0], factory, room, num_frames=5,
        min_confidence=0.3, sample_rate=audio_config.sample_rate,
    )

    broadside, endfire = points
    assert endfire.expected_resolution_degrees > broadside.expected_resolution_degrees
    assert "least reliable" in endfire.note


def test_calibration_does_not_hide_bad_results(calibrate, audio_config, room):
    """A hopeless SNR must produce visible error, not a quietly fudged number."""
    factory = calibrate.synthetic_source_factory(audio_config, room, noise=3.0)
    points = calibrate.run_calibration(
        [30.0], factory, room, num_frames=5,
        min_confidence=0.5, sample_rate=audio_config.sample_rate,
    )
    point = points[0]

    failed = point.estimated_angle_degrees is None
    inaccurate = point.error_degrees is not None and abs(point.error_degrees) > 2.0
    assert failed or inaccurate
    assert point.frames_rejected > 0 or inaccurate


def test_calibration_counts_rejected_frames(calibrate, audio_config, room):
    factory = calibrate.synthetic_source_factory(audio_config, room, noise=0.002)
    point = calibrate.run_calibration(
        [0.0], factory, room, num_frames=5,
        min_confidence=0.999, sample_rate=audio_config.sample_rate,
    )[0]

    assert point.estimated_angle_degrees is None
    assert point.frames_rejected == 5
    assert "could not resolve" in point.note


def test_calibration_report_prints_a_verdict(calibrate, audio_config, room, capsys):
    factory = calibrate.synthetic_source_factory(audio_config, room, noise=0.002)
    points = calibrate.run_calibration(
        [0.0, 30.0], factory, room, num_frames=5,
        min_confidence=0.3, sample_rate=audio_config.sample_rate,
    )
    calibrate.print_report(points, room, audio_config.sample_rate, synthetic=True)

    output = capsys.readouterr().out
    assert "VERDICT" in output
    assert "SYNTHETIC" in output
    assert "mean" in output and "p95" in output


def test_calibration_report_says_synthetic_proves_nothing_about_hardware(
    calibrate, audio_config, room, capsys
):
    factory = calibrate.synthetic_source_factory(audio_config, room, noise=0.002)
    points = calibrate.run_calibration(
        [0.0], factory, room, num_frames=3,
        min_confidence=0.3, sample_rate=audio_config.sample_rate,
    )
    calibrate.print_report(points, room, audio_config.sample_rate, synthetic=True)
    assert "nothing about whether your microphones work" in capsys.readouterr().out


def test_calibration_measure_returns_no_angle_for_silence(calibrate, room):
    from heimdall.audio.sources import SyntheticAudioSource

    source = SyntheticAudioSource(48000, 2, 1024, burst_frames=0, noise_amplitude=0.0)
    angle, confidence, used, rejected = calibrate.measure_angle(source, room, 5, 0.3)

    assert angle is None
    assert used == 0
    assert rejected == 5


def test_calibration_esp32_factory_builds_a_real_source(calibrate, audio_config, room, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    from heimdall.audio.sources import AudioSourceError, ESP32AudioSource

    factory = calibrate.esp32_source_factory(audio_config, room, noise=0.0)
    source = factory(0.0)
    assert isinstance(source, ESP32AudioSource)
    with pytest.raises(AudioSourceError):
        source.start()          # no port configured; it will not invent one


def test_calibration_point_is_json_friendly(calibrate, audio_config, room):
    import json
    from dataclasses import asdict

    factory = calibrate.synthetic_source_factory(audio_config, room, noise=0.002)
    point = calibrate.run_calibration(
        [15.0], factory, room, num_frames=3,
        min_confidence=0.3, sample_rate=audio_config.sample_rate,
    )[0]

    payload = json.loads(json.dumps(asdict(point)))
    assert payload["known_angle_degrees"] == 15.0
    assert "error_degrees" in payload


# --- benchmark_audio ---------------------------------------------------------

def benchmark_args(**overrides):
    args = dict(
        source="synthetic",
        angle=25.0,
        noise=0.002,
        burst_frames=6,
        silence_frames=4,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def run_benchmark(benchmark, audio_config, room, frames=12, warmup=2, **source_overrides):
    source = benchmark.build_source(benchmark_args(**source_overrides), audio_config, room)
    return benchmark.benchmark(source, room, frames, warmup_frames=warmup)


def test_benchmark_times_every_pipeline_stage(benchmark, audio_config, room):
    result = run_benchmark(benchmark, audio_config, room)

    for stage in ("capture", "gcc_phat", "doa", "detect", "frame_total"):
        assert stage in result.stages, stage
        entry = result.stages[stage]
        assert entry["count"] == result.frames_measured
        assert entry["mean_ms"] >= 0.0
        assert entry["max_ms"] >= entry["p95_ms"] >= 0.0


def test_benchmark_breaks_out_capture_and_gcc_phat_separately(benchmark, audio_config, room):
    """The gap performance_report() left: these two were never reported alone."""
    result = run_benchmark(benchmark, audio_config, room)

    assert "capture" in result.stages
    assert "gcc_phat" in result.stages
    # GCC-PHAT is the bulk of DOA; the geometry on top of it is arithmetic.
    assert result.stages["gcc_phat"]["mean_ms"] <= result.stages["doa"]["mean_ms"] * 2.0


def test_benchmark_discards_warmup_frames(benchmark, audio_config, room):
    result = run_benchmark(benchmark, audio_config, room, frames=10, warmup=4)

    assert result.warmup_frames == 4
    assert result.frames_measured == 10
    assert result.stages["doa"]["count"] == 10


def test_benchmark_frame_total_excludes_the_extra_gcc_phat_pass(benchmark, audio_config, room):
    """frame_total must be what api.process_frame costs, not what this tool costs."""
    result = run_benchmark(benchmark, audio_config, room, frames=15)

    stages = result.stages
    expected = (
        stages["capture"]["mean_ms"] + stages["doa"]["mean_ms"] + stages["detect"]["mean_ms"]
    )
    assert result.per_frame_cost_ms == pytest.approx(expected, rel=1e-9)


def test_benchmark_compares_against_the_real_time_budget(benchmark, audio_config, room):
    result = run_benchmark(benchmark, audio_config, room)

    expected_ms = 1000.0 * audio_config.frame_size / audio_config.sample_rate
    assert result.frame_duration_ms == pytest.approx(expected_ms)
    assert result.realtime_factor == pytest.approx(
        result.frame_duration_ms / result.per_frame_cost_ms
    )
    assert result.realtime_factor > 1.0, "pipeline is slower than real time"


def test_benchmark_times_seat_mapping_only_when_an_event_completes(benchmark, audio_config, room):
    """Seat mapping is per event. Silence must not fabricate samples for it."""
    silent = run_benchmark(
        benchmark, audio_config, room, frames=12, burst_frames=0, silence_frames=8, noise=0.0
    )

    assert silent.events_emitted == 0
    assert "seat_mapping" not in silent.stages
    assert silent.stages["doa"]["count"] == 12


def test_benchmark_report_labels_synthetic_capture_as_not_acquisition(
    benchmark, audio_config, room, capsys
):
    result = run_benchmark(benchmark, audio_config, room)
    benchmark.print_report(result)

    output = capsys.readouterr().out
    assert "SYNTHETIC" in output
    assert "Acquisition latency needs hardware" in output
    assert "VERDICT" in output
    assert "MEAN ms" in output and "P95 ms" in output


def test_benchmark_report_calls_synthetic_frame_drops_meaningless(
    benchmark, audio_config, room, capsys
):
    """An unpaced source always overruns the queue; that is not a real drop."""
    result = run_benchmark(benchmark, audio_config, room)
    result.frames_dropped = 7
    benchmark.print_report(result)

    assert "means nothing here" in capsys.readouterr().out


def test_benchmark_report_handles_no_frames(benchmark, capsys):
    empty = benchmark.BenchmarkResult(
        source="synthetic", sample_rate=48000, frame_size=1024, num_channels=2,
        frames_measured=0, frames_dropped=0, events_emitted=0, warmup_frames=5,
        frame_duration_ms=21.33,
    )
    benchmark.print_report(empty)

    assert "Nothing to measure" in capsys.readouterr().out
    assert empty.per_frame_cost_ms == 0.0


def test_benchmark_result_is_json_friendly(benchmark, audio_config, room):
    import json
    from dataclasses import asdict

    result = run_benchmark(benchmark, audio_config, room)
    payload = json.loads(json.dumps(asdict(result)))

    assert payload["sample_rate"] == audio_config.sample_rate
    assert payload["stages"]["gcc_phat"]["p95_ms"] >= 0.0


def test_benchmark_builds_a_real_esp32_source(benchmark, audio_config, room):
    from heimdall.audio.sources import AudioSourceError, ESP32AudioSource

    source = benchmark.build_source(benchmark_args(source="esp32"), audio_config, room)
    assert isinstance(source, ESP32AudioSource)
    with pytest.raises(AudioSourceError):
        source.start()
