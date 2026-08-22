"""tools/verify_localization.py, driven headlessly with no hardware attached.

The test that matters most here is the MIRRORED one: if the tool cannot detect
a flipped sign convention on synthetic audio, it cannot be trusted to detect it
on hardware, and section 5's "verified by inspection" would stay unverified.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load_tool(name):
    spec = importlib.util.spec_from_file_location("tool_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def loc():
    return load_tool("verify_localization")


@pytest.fixture(scope="module")
def config():
    from heimdall.audio.config import load_audio_config

    return load_audio_config()


@pytest.fixture(scope="module")
def room():
    from heimdall.audio.geometry import load_classroom_config

    return load_classroom_config()


def source_factory(loc, config, room, *, mirror=False, spacing=None, noise=0.005,
                   silent=False):
    """Synthetic stand-in for the array, optionally wired backwards."""
    from heimdall.audio.sources import SyntheticAudioSource

    def make(station):
        angle = station.expected_degrees or 0.0
        return SyntheticAudioSource(
            sample_rate=config.transport.transmit_sample_rate,
            num_channels=2,
            frame_size=config.frame_size,
            angle_degrees=-angle if mirror else angle,
            mic_spacing_m=spacing or room.array.spacing,
            noise_amplitude=noise,
            # Clap-LIKE: a burst with quiet between, so the onset detector has
            # an attack to find. A continuous burst has no transient at all and
            # would skip the selection path the tool now depends on.
            burst_frames=0 if silent else 1,
            silence_frames=1 if silent else 6,
        )
    return make


def run(loc, config, room, *, angles=None, frames=70, **kw):
    stations = loc.default_stations(angles if angles is not None else [-30.0, 0.0, 30.0])
    return loc.run_stations(
        stations, source_factory(loc, config, room, **kw), room,
        prompt=None, num_frames=frames,
    )


# --- the stations themselves -------------------------------------------------

def test_the_three_geometric_stations_come_first(loc):
    stations = loc.default_stations([0.0])
    assert [s.key for s in stations[:3]] == ["near_mic1", "midpoint", "near_mic2"]


def test_the_geometric_stations_encode_the_section_5_convention(loc):
    by_key = {s.key: s for s in loc.default_stations([])}
    # +90 toward channel 0, -90 toward the last channel, 0 at broadside.
    assert by_key["near_mic1"].expected_degrees == 90.0
    assert by_key["midpoint"].expected_degrees == 0.0
    assert by_key["near_mic2"].expected_degrees == -90.0


def test_the_angle_sweep_is_configurable(loc):
    stations = loc.default_stations([-45.0, 45.0])
    sweep = [s for s in stations if s.role == "angle"]
    assert [s.expected_degrees for s in sweep] == [-45.0, 45.0]


def test_every_station_tells_the_operator_what_to_do(loc):
    for station in loc.default_stations():
        assert station.instruction.strip()
        assert "clap" in station.instruction.lower()


# --- THE sign check ----------------------------------------------------------

def test_a_correctly_wired_array_passes_the_sign_check(loc, config, room):
    results = run(loc, config, room)
    state, message = loc.sign_verdict(results)
    assert state == "PASS"
    assert "CORRECT" in message
    assert "not mirrored" in message.lower()


def test_a_mirrored_array_is_caught(loc, config, room):
    # Both microphones swapped: exactly what crossed SD lines or a flipped L/R
    # strap would produce. Every bearing in the system would be inverted.
    results = run(loc, config, room, mirror=True)
    state, message = loc.sign_verdict(results)

    assert state == "FAIL"
    assert "MIRRORED" in message
    assert "backwards" in message


def test_the_mirrored_message_names_the_physical_causes_not_a_workaround(
    loc, config, room
):
    _, message = loc.sign_verdict(run(loc, config, room, mirror=True))
    assert "L/R" in message
    assert "classroom.yaml" in message
    # It must not suggest negating the angle downstream to paper over wiring.
    assert "Fix the CAUSE" in message


def test_a_mirrored_array_fails_the_whole_run(loc, config, room, capsys):
    results = run(loc, config, room, mirror=True)
    assert loc.report(results, room, 16000) is False
    out = capsys.readouterr().out
    assert "do NOT proceed to Phase 4b" in out
    assert "MIRRORED" in out


def test_one_station_disagreeing_with_its_mirror_is_called_inconsistent(loc, config, room):
    results = run(loc, config, room)
    # Corrupt just one end-fire station's observations to the wrong sign.
    bad = results["near_mic2"]
    bad.observations = [
        type(o)(o.lag_samples, o.tdoa_us, abs(o.bearing_degrees), o.confidence,
                o.rms_ch0, o.rms_ch1)
        for o in bad.observations
    ]
    state, message = loc.sign_verdict(results)
    assert state == "FAIL"
    assert "INCONSISTENT" in message
    assert "Re-run" in message


def test_no_data_means_the_sign_is_untested_not_confirmed(loc, config, room):
    results = run(loc, config, room, silent=True)
    state, message = loc.sign_verdict(results)
    assert state == "N/A"
    assert "UNTESTED" in message
    assert "NOT confirmed" in message
    assert "INCONCLUSIVE" in message
    assert "MIRRORED" not in message


def test_the_sign_result_is_printed_first_and_on_its_own(loc, config, room, capsys):
    loc.report(run(loc, config, room), room, 16000)
    out = capsys.readouterr().out
    assert out.index("SIGN CONVENTION CHECK") < out.index("per-station measurements")


# --- measurement -------------------------------------------------------------

def test_a_correctly_wired_array_recovers_its_known_angles(loc, config, room):
    results = run(loc, config, room, angles=[-60.0, -30.0, 0.0, 30.0, 60.0], frames=70)
    for result in results.values():
        if result.station.role != "angle":
            continue
        assert result.n > 0, result.station.label
        assert result.mean_abs_error < 5.0, result.station.label


def test_lag_and_tdoa_are_recorded_alongside_the_bearing(loc, config, room):
    results = run(loc, config, room, angles=[60.0])
    result = results["angle_p60"]
    observation = result.observations[0]
    assert observation.lag_samples != 0.0
    assert observation.tdoa_us != 0.0
    # +60 deg is toward channel 0, so channel 0 leads: a negative gcc lag.
    assert observation.lag_samples < 0
    assert observation.rms_ch0 > 0 and observation.rms_ch1 > 0


def test_silent_frames_are_gated_out_and_counted(loc, config, room):
    results = run(loc, config, room, silent=True, frames=70)
    result = results["midpoint"]
    assert result.n == 0
    assert result.frames_below_gate > 0
    # The message now names the actual cause and what to do about it.
    assert "NO CLAP DETECTED" in result.note
    assert "noise floor" in result.note


def test_rejected_frames_are_reported_not_hidden(loc, config, room, capsys):
    loc.report(run(loc, config, room, silent=True, frames=70), room, 16000)
    out = capsys.readouterr().out
    assert "ONSETS, not levels" in out
    assert "below gate" in out
    assert "floor" in out


def test_the_confidence_gate_matches_the_pipeline_default(loc):
    from heimdall.audio.seat_mapper import DEFAULT_MIN_CONFIDENCE

    # If this tool were more permissive than the pipeline it would report
    # bearings the pipeline itself would refuse to act on. The tool keeps its
    # own copy (seat_mapper is parked), so pin them together here.
    assert loc.DEFAULT_MIN_CONFIDENCE == DEFAULT_MIN_CONFIDENCE == 0.30


def test_spread_is_reported_not_just_the_best_trial(loc, config, room, capsys):
    results = run(loc, config, room, noise=0.05, frames=70)
    loc.report(results, room, 16000)
    out = capsys.readouterr().out
    assert "sd" in out
    result = results["midpoint"]
    mean, sd = result.bearing
    assert sd is not None and sd >= 0.0


# --- effective spacing --------------------------------------------------------

def test_the_configured_spacing_is_recovered_when_it_is_right(loc, config, room):
    results = run(loc, config, room, angles=[-45.0, -30.0, 0.0, 30.0, 45.0], frames=70)
    spacing, note = loc.effective_spacing(results, room, 16000)
    assert spacing == pytest.approx(room.array.spacing, abs=0.005)
    assert "13.5 cm configured" in note


def test_a_different_effective_spacing_is_surfaced_not_absorbed(loc, config, room, capsys):
    results = run(loc, config, room, angles=[-45.0, -30.0, 30.0, 45.0],
                  spacing=0.16, frames=70)
    spacing, _ = loc.effective_spacing(results, room, 16000)
    assert spacing == pytest.approx(0.16, abs=0.01)

    loc.report(results, room, 16000)
    out = capsys.readouterr().out
    assert "differs from the configured value" in out
    assert "classroom.yaml" in out
    assert "silent constant" in out


def test_endfire_stations_are_excluded_from_the_spacing_fit(loc, config, room):
    # They clamp at +/-90, so including them would bias the slope.
    results = run(loc, config, room, angles=[-45.0, 45.0], frames=70)
    for key in ("near_mic1", "near_mic2"):
        assert results[key].station.role != "angle"


def test_spacing_needs_off_broadside_angles(loc, config, room):
    results = run(loc, config, room, angles=[0.0], frames=70)
    spacing, note = loc.effective_spacing(results, room, 16000)
    assert spacing is None
    assert "not enough" in note


# --- verdict and reporting ----------------------------------------------------

def test_a_good_run_passes_every_criterion(loc, config, room):
    results = run(loc, config, room, angles=[-60.0, -30.0, 0.0, 30.0, 60.0], frames=70)
    assert all(state == "PASS" for _, state, _ in loc.verdict(results))
    assert loc.report(results, room, 16000) is True


def test_a_run_with_no_data_is_never_a_pass(loc, config, room):
    results = run(loc, config, room, silent=True, frames=70)
    states = {state for _, state, _ in loc.verdict(results)}
    assert "PASS" not in states
    assert loc.report(results, room, 16000) is False


def test_the_verdict_separates_the_sign_from_the_error_average(loc, config, room):
    names = [name for name, _, _ in loc.verdict(run(loc, config, room))]
    assert names[0] == "sign convention (section 5)"
    assert "angle sweep accuracy" in names


def test_results_serialise_to_json(loc, config, room):
    import json

    payload = json.loads(loc.results_to_json(run(loc, config, room, frames=70)))
    assert payload[0]["station"]["key"] == "near_mic1"
    assert payload[0]["n"] > 0
    assert "observations" in payload[0]


# --- headless and hardware-free ----------------------------------------------

def test_the_synthetic_self_test_runs_end_to_end(loc):
    assert loc.main(["--no-prompt", "--frames", "6", "--angles", "-30", "0", "30"]) == 0


def test_the_synthetic_run_says_it_proves_nothing_about_the_microphones(loc, capsys):
    loc.main(["--no-prompt", "--frames", "4", "--angles", "0"])
    out = capsys.readouterr().out
    assert "NOT the microphones" in out
    assert "cannot tell you anything about the physical sign convention" in out


def test_esp32_without_hardware_refuses_clearly_and_without_a_traceback(
    loc, capsys, monkeypatch
):
    # Force "no port configured" so the test never touches the real board,
    # whatever config/audio.yaml happens to say on this machine.
    import dataclasses

    real = loc.load_audio_config

    def unconfigured(*args, **kwargs):
        config = real(*args, **kwargs)
        return dataclasses.replace(
            config, transport=dataclasses.replace(config.transport, port=None))

    monkeypatch.setattr(loc, "load_audio_config", unconfigured)
    assert loc.main(["--source", "esp32", "--no-prompt", "--frames", "4"]) == 2
    err = capsys.readouterr().err
    assert "Cannot reach the microphone array" in err
    assert "detect_device.py" in err
    assert "Traceback" not in err


def test_json_output_is_written_when_asked(loc, tmp_path):
    out = tmp_path / "results.json"
    loc.main(["--no-prompt", "--frames", "4", "--angles", "0", "--json", str(out)])
    assert out.exists() and out.read_text(encoding="utf-8").startswith("[")


def test_the_tool_never_opens_a_port_in_synthetic_mode(loc, monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", None)
    assert loc.main(["--no-prompt", "--frames", "4", "--angles", "0"]) == 0


# --- the coherence precondition ---------------------------------------------
#
# THE REGRESSION. The tool reported MIRRORED from a station whose median bearing
# was -2.3 deg with 52 deg of spread over 82 frames: 0.4 standard errors from
# zero, and a lag of 0.16 samples against a physical maximum of 6.30. Mirroring
# flips the SIGN; it does not collapse the MAGNITUDE. A mirrored array clapped
# 5 cm from mic 1 reads -90 deg. Acting on that verdict meant rewiring a
# correctly-wired array.

MAX_LAG = 6.30


def synthetic_station(loc, key, bearings, lag, *, role=None, expected=None):
    """A StationResult with observations chosen to order."""
    stations = {s.key: s for s in loc.default_stations([-30.0, 0.0, 30.0])}
    station = stations[key]
    if role is not None or expected is not None:
        import dataclasses

        station = dataclasses.replace(
            station,
            role=role or station.role,
            expected_degrees=station.expected_degrees if expected is None else expected,
        )
    result = loc.StationResult(station=station)
    result.frames_captured = len(bearings)
    result.observations = [
        loc.Observation(lag_samples=lag, tdoa_us=lag / 16000 * 1e6,
                        bearing_degrees=b, confidence=0.50,
                        rms_ch0=0.05, rms_ch1=0.05)
        for b in bearings
    ]
    return result


def noisy_results(loc, seed=0):
    """The real failed run: near-zero medians, huge spread, negligible lag."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return {
        "near_mic1": synthetic_station(
            loc, "near_mic1", list(rng.normal(-2.3, 52.0, 82)), 0.16),
        "near_mic2": synthetic_station(
            loc, "near_mic2", list(rng.normal(1.1, 48.0, 82)), 2.08),
    }


def coherent_results(loc, sign=+1):
    """Both end-fire stations tight and at nearly the full geometric lag."""
    import numpy as np

    rng = np.random.default_rng(1)
    return {
        "near_mic1": synthetic_station(
            loc, "near_mic1", list(rng.normal(sign * 88.0, 3.0, 40)), -6.1 * sign),
        "near_mic2": synthetic_station(
            loc, "near_mic2", list(rng.normal(-sign * 87.0, 3.0, 40)), 6.1 * sign),
    }


def test_wide_spread_near_zero_is_inconclusive_not_mirrored(loc):
    state, message = loc.sign_verdict(noisy_results(loc), MAX_LAG)

    assert state == "N/A"
    assert "NO COHERENT TDOA - INCONCLUSIVE" in message
    assert "MIRRORED" not in message


def test_the_inconclusive_message_forbids_touching_the_hardware(loc):
    _, message = loc.sign_verdict(noisy_results(loc), MAX_LAG)
    assert "NOT mirrored either" in message
    assert "Do not change wiring" in message
    assert "classroom.yaml" in message


def test_the_inconclusive_message_says_what_would_make_it_conclusive(loc):
    _, message = loc.sign_verdict(noisy_results(loc), MAX_LAG)
    assert "What would make it conclusive" in message
    assert "spread" in message
    assert "geometric maximum" in message
    assert "analyse_claps.py" in message


def test_a_genuinely_mirrored_array_is_still_caught(loc):
    # Tight spread, full-magnitude lag, wrong sign: this IS mirroring, and the
    # precondition must not suppress it.
    state, message = loc.sign_verdict(coherent_results(loc, sign=-1), MAX_LAG)
    assert state == "FAIL"
    assert "MIRRORED" in message


def test_a_correctly_wired_coherent_array_still_passes(loc):
    state, _ = loc.sign_verdict(coherent_results(loc, sign=+1), MAX_LAG)
    assert state == "PASS"


def test_an_endfire_lag_far_below_the_geometric_maximum_is_incoherent(loc):
    # Tight spread but the lag is only 2.5% of maximum: the source cannot have
    # been on the array axis, so this station cannot vote on the sign.
    results = {
        "near_mic1": synthetic_station(loc, "near_mic1", [-2.0] * 40, 0.16),
        "near_mic2": synthetic_station(loc, "near_mic2", [1.0] * 40, 0.20),
    }
    state, message = loc.sign_verdict(results, MAX_LAG)
    assert state == "N/A"
    assert "physical maximum" in message


def test_coherence_names_which_precondition_failed(loc):
    wide = synthetic_station(loc, "near_mic1", [-2.0, 60.0, -70.0, 5.0] * 10, -6.1)
    ok, why = loc.coherence(wide, MAX_LAG)
    assert ok is False and "spread" in why

    short = synthetic_station(loc, "near_mic1", [-2.0] * 20, 0.16)
    ok, why = loc.coherence(short, MAX_LAG)
    assert ok is False and "physical maximum" in why

    good = synthetic_station(loc, "near_mic1", [88.0, 89.0, 90.0] * 8, -6.1)
    ok, why = loc.coherence(good, MAX_LAG)
    assert ok is True and why == "coherent"


def test_an_incoherent_station_is_na_in_the_verdict_not_a_fail(loc):
    checks = dict((name, state) for name, state, _ in
                  loc.verdict(noisy_results(loc), MAX_LAG))
    assert checks["end-fire toward mic 1"] == "N/A"
    assert checks["end-fire toward mic 2"] == "N/A"
    assert checks["sign convention (section 5)"] == "N/A"


def test_an_incoherent_run_never_passes(loc):
    states = {s for _, s, _ in loc.verdict(noisy_results(loc), MAX_LAG)}
    assert "PASS" not in states


def test_the_incoherent_report_points_at_the_raw_audio_tool(loc, room, capsys):
    assert loc.report(noisy_results(loc), room, 16000) is False
    out = capsys.readouterr().out
    assert "INCONCLUSIVE" in out
    assert "Nothing here justifies a hardware change" in out
    assert "analyse_claps.py record" in out
    assert "MIRRORED" not in out


def test_the_spread_limit_is_stated_not_hidden(loc):
    assert loc.MAX_COHERENT_SPREAD_DEGREES == 25.0
    assert loc.MIN_ENDFIRE_LAG_FRACTION == 0.40
    _, message = loc.sign_verdict(noisy_results(loc), MAX_LAG)
    assert "25" in message and "40%" in message


# --- THE HARDWARE REGRESSION: a steady room source must not win -------------
#
# On hardware, every station returned ~-22 deg with `below gate 0` out of 200
# frames, whatever the operator did. Reproduced here: a steady source at a
# fixed bearing plus a few claps elsewhere. Level gating admits the whole
# recording, so the steady source - which never moves - decides every estimate.
# Onset selection measures the claps instead.

FS = 16000
MAXLAG = 0.135 * FS / 343.0


def _delayed(signal, lag):
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal))
    return np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * lag), len(signal))


def room_with_claps(clap_bearing, steady_bearing=-25.0, seconds=8.0,
                    n_claps=4, steady_level=0.030, clap_level=0.60, seed=0):
    """A running fan at one bearing, and claps from another."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)

    steady = rng.normal(0, 1, n)
    steady = steady / np.std(steady) * steady_level
    ch0 = steady.copy()
    ch1 = _delayed(steady, MAXLAG * np.sin(np.radians(steady_bearing)))

    clap_lag = MAXLAG * np.sin(np.radians(clap_bearing))
    for k in range(n_claps):
        at = int((k + 1) * n / (n_claps + 1))
        burst = rng.normal(0, 1, 4000) * np.exp(-np.arange(4000) / 30.0)
        burst = burst / (np.max(np.abs(burst)) + 1e-9) * clap_level
        ch0[at:at + 4000] += burst
        ch1[at:at + 4000] += _delayed(burst, clap_lag)
    return np.stack([ch0, ch1], axis=1).astype(np.float32)


def buffered_source(buffer):
    from acoustic_array.sources import SyntheticAudioSource

    return SyntheticAudioSource.from_buffer(buffer, FS, frame_size=1024)


def test_a_steady_room_source_does_not_hijack_every_station(loc, room):
    # THE regression. Claps at +40, fan at -25. The answer must be the claps.
    stations = {s.key: s for s in loc.default_stations([40.0])}
    buffer = room_with_claps(clap_bearing=40.0, steady_bearing=-25.0, seed=3)

    result = loc.measure_station(buffered_source(buffer), stations["angle_p40"],
                                 room, num_frames=200)

    assert result.n > 0, result.note
    assert result.median_bearing == pytest.approx(40.0, abs=12.0), result.median_bearing
    # And emphatically NOT the steady source's bearing.
    assert abs(result.median_bearing - (-25.0)) > 25.0


@pytest.mark.parametrize("clap", [-50.0, -20.0, 0.0, 20.0, 50.0])
def test_the_measured_bearing_follows_the_clap_not_the_fan(loc, room, clap):
    stations = {s.key: s for s in loc.default_stations([clap])}
    key = f"angle_{clap:+.0f}".replace("+", "p").replace("-", "m")
    buffer = room_with_claps(clap_bearing=clap, steady_bearing=-25.0, seed=7)

    result = loc.measure_station(buffered_source(buffer), stations[key], room,
                                 num_frames=200)
    assert result.n > 0, result.note
    assert result.median_bearing == pytest.approx(clap, abs=15.0)


def test_level_gating_would_have_returned_the_fan_instead(loc, room):
    """Documents WHY the fix exists, by measuring the old behaviour directly."""
    from acoustic_array.gcc_phat import gcc_phat

    buffer = room_with_claps(clap_bearing=40.0, steady_bearing=-25.0, seed=3)
    lags = []
    for i in range(0, len(buffer) - 1024, 1024):
        block = buffer[i:i + 1024]
        if max(float(np.sqrt(np.mean(block[:, c] ** 2))) for c in (0, 1)) < 0.01:
            continue                                    # the old absolute gate
        r = gcc_phat(block[:, 0], block[:, 1], FS, max_tau=MAXLAG / FS)
        if r.valid and r.confidence >= 0.30:
            lags.append(r.delay_samples)

    assert len(lags) > 50, "the old gate admitted almost everything - that is the bug"
    old_bearing = math.degrees(math.asin(
        float(np.clip(-np.median(lags) / MAXLAG, -1, 1))))
    # The old path lands on the fan, not the claps at +40.
    assert abs(old_bearing - (-25.0)) < 12.0
    assert abs(old_bearing - 40.0) > 40.0


def test_a_room_louder_than_the_claps_refuses_rather_than_guessing(loc, room):
    stations = {s.key: s for s in loc.default_stations([30.0])}
    # Fan far louder than the claps: nothing clears the onset threshold.
    buffer = room_with_claps(clap_bearing=30.0, steady_level=0.30,
                             clap_level=0.20, seed=5)
    result = loc.measure_station(buffered_source(buffer), stations["angle_p30"],
                                 room, num_frames=200)

    assert result.n == 0
    assert "NO CLAP DETECTED" in result.note or "no usable transient" in result.note
    assert "clap harder" in result.note.lower()


def test_the_noise_floor_is_measured_not_assumed(loc, room):
    stations = {s.key: s for s in loc.default_stations([0.0])}
    quiet = room_with_claps(0.0, steady_level=0.004, seed=1)
    loud = room_with_claps(0.0, steady_level=0.040, seed=1)

    a = loc.measure_station(buffered_source(quiet), stations["angle_p0"], room,
                            num_frames=200)
    b = loc.measure_station(buffered_source(loud), stations["angle_p0"], room,
                            num_frames=200)
    assert b.noise_floor > a.noise_floor * 3
    # Both still find the claps, because the threshold moves with the room.
    assert a.n > 0 and b.n > 0


def test_onsets_are_counted_and_reported(loc, room):
    stations = {s.key: s for s in loc.default_stations([0.0])}
    buffer = room_with_claps(0.0, n_claps=4, seed=2)
    result = loc.measure_station(buffered_source(buffer), stations["angle_p0"],
                                 room, num_frames=200)
    assert result.onsets_found == 4
    assert result.frames_captured > result.onsets_found


def test_the_report_names_onsets_rather_than_levels(loc, room, capsys):
    stations = {s.key: s for s in loc.default_stations([0.0])}
    results = {"angle_p0": loc.measure_station(
        buffered_source(room_with_claps(0.0, seed=2)), stations["angle_p0"],
        room, num_frames=200)}
    loc.report(results, room, FS)
    out = capsys.readouterr().out
    assert "ONSETS, not levels" in out
    assert "floor" in out and "claps" in out
