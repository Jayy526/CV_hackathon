"""The heatmap overlay, headless: no webcam, no board, no OpenCV.

These tests exist to pin the honesty rules, not the aesthetics. A blob instead
of a band, a band pinned to the frame edge for an off-frame sound, or a missing
synthetic banner would each turn the demo into a lie, and each has a test here.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from acoustic_camera import CameraConfig
from acoustic_camera.overlay import (
    OFF_FRAME_COLOUR,
    SPEECH_COLOUR,
    SYNTHETIC_BANNER,
    WHISPER_COLOUR,
    BearingRing,
    Echo,
    OverlayConfig,
    banner_text,
    build_hud,
    colour_for,
    composite,
    draw_banner,
    echo_from_event,
    energy_from_rms,
)
from acoustic_camera.projection import project_band

TOOLS = Path(__file__).resolve().parents[2] / "tools"


@pytest.fixture
def camera():
    return CameraConfig(width=640, height=360, horizontal_fov_degrees=70.0)


@pytest.fixture
def overlay():
    return OverlayConfig()


def blank(camera, value=0):
    return np.full((camera.height, camera.width, 3), value, dtype=np.uint8)


class FakeEvent:
    """Stands in for an AcousticEvent. Duck-typed, as the overlay expects."""

    def __init__(self, direction=-10.0, event_type="POSSIBLE_WHISPER",
                 resolution=4.55, localization_confidence=0.8, rms=0.05,
                 reason="", timestamp=0.0):
        self.timestamp = timestamp
        self.event_type = event_type
        self.direction_degrees = direction
        self.angular_resolution_degrees = resolution
        self.localization_confidence = localization_confidence
        self.confidence = 0.7
        self.channel_rms = (rms, rms)
        self.reason = reason


def echo_for(event, camera, overlay, at=0.0):
    return echo_from_event(event, camera, overlay, display_time=at)


def painted_columns(before, after):
    """Columns whose pixels changed, ignoring row position."""
    diff = np.any(before != after, axis=(0, 2))
    return np.nonzero(diff)[0]


# --- a located sound is a FULL-HEIGHT BAND, never a blob ---------------------

def test_a_located_sound_paints_every_row_of_its_columns(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=0.0), camera, overlay))
    before = blank(camera)
    after = composite(before, ring, 0.0, overlay)

    columns = painted_columns(before, after)
    assert columns.size > 0
    # Full height: every painted column differs in EVERY row. A blob would
    # leave the top and bottom of the frame untouched.
    for column in columns:
        assert np.all(np.any(before[:, column] != after[:, column], axis=1)), column


def test_the_overlay_never_paints_only_part_of_a_column(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=15.0), camera, overlay))
    after = composite(blank(camera), ring, 0.0, overlay)

    top_row = np.any(after[0] != 0, axis=1)
    bottom_row = np.any(after[-1] != 0, axis=1)
    middle_row = np.any(after[camera.height // 2] != 0, axis=1)
    assert np.array_equal(top_row, middle_row)
    assert np.array_equal(bottom_row, middle_row)


def test_the_band_sits_where_the_projection_says(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=-20.0), camera, overlay))
    after = composite(blank(camera), ring, 0.0, overlay)

    band = project_band(-20.0, 4.55, 0.8, camera)
    columns = painted_columns(blank(camera), after)
    assert columns.min() == pytest.approx(int(band.left_column), abs=1)
    assert columns.max() == pytest.approx(int(band.right_column), abs=1)


# --- width is the uncertainty, and it widens ---------------------------------

def test_the_band_widens_visibly_as_confidence_drops(camera, overlay):
    widths = []
    for confidence in (0.95, 0.7, 0.45, 0.2):
        ring = BearingRing(overlay)
        ring.add(echo_for(FakeEvent(direction=0.0, localization_confidence=confidence),
                          camera, overlay))
        after = composite(blank(camera), ring, 0.0, overlay)
        widths.append(painted_columns(blank(camera), after).size)

    assert widths == sorted(widths)
    assert widths[-1] > widths[0] * 2


def test_the_same_uncertainty_is_wider_at_the_frame_edge(camera, overlay):
    def width_at(bearing):
        ring = BearingRing(overlay)
        ring.add(echo_for(FakeEvent(direction=bearing), camera, overlay))
        return painted_columns(blank(camera),
                               composite(blank(camera), ring, 0.0, overlay)).size

    # Not a width computed once at the centre and reused: the tangent map
    # stretches the same angle into more pixels away from the axis.
    assert width_at(28.0) > width_at(0.0) * 1.2


def test_the_band_is_never_a_fixed_width_bar(camera, overlay):
    def width_at(bearing):
        ring = BearingRing(overlay)
        ring.add(echo_for(FakeEvent(direction=bearing), camera, overlay))
        return painted_columns(blank(camera),
                               composite(blank(camera), ring, 0.0, overlay)).size

    # Width grows away from the optical axis. It is SYMMETRIC about it, so
    # +28 and -28 rightly match - that is the projection being correct, not a
    # fixed-width bar.
    assert width_at(28.0) > width_at(14.0) > width_at(0.0)
    assert width_at(-28.0) == width_at(28.0)
    assert width_at(-14.0) == width_at(14.0)


# --- intensity decays over the window ----------------------------------------

def test_intensity_fades_with_age(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=0.0), camera, overlay, at=0.0))

    levels = [float(composite(blank(camera), ring, t, overlay).mean())
              for t in (0.0, 0.4, 0.8, 1.2)]
    assert levels == sorted(levels, reverse=True)
    assert levels[0] > levels[-1]


def test_an_echo_older_than_the_window_is_gone_entirely(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=0.0), camera, overlay, at=0.0))

    assert ring.active(1.49)
    assert ring.active(1.51) == []
    after = composite(blank(camera), ring, 1.6, overlay)
    assert np.array_equal(after, blank(camera))


def test_the_decay_window_is_configurable(camera):
    short = OverlayConfig(decay_seconds=0.4)
    ring = BearingRing(short)
    ring.add(echo_for(FakeEvent(direction=0.0), camera, short, at=0.0))
    assert ring.active(0.3)
    assert ring.active(0.5) == []


def test_concurrent_directions_accumulate(camera, overlay):
    both = BearingRing(overlay)
    both.add(echo_for(FakeEvent(direction=-20.0), camera, overlay))
    both.add(echo_for(FakeEvent(direction=20.0), camera, overlay))
    painted_both = painted_columns(blank(camera),
                                   composite(blank(camera), both, 0.0, overlay))

    one = BearingRing(overlay)
    one.add(echo_for(FakeEvent(direction=-20.0), camera, overlay))
    painted_one = painted_columns(blank(camera),
                                  composite(blank(camera), one, 0.0, overlay))
    assert painted_both.size > painted_one.size * 1.8


def test_a_louder_sound_draws_more_strongly(camera, overlay):
    def level(rms):
        ring = BearingRing(overlay)
        ring.add(echo_for(FakeEvent(direction=0.0, rms=rms), camera, overlay))
        return float(composite(blank(camera), ring, 0.0, overlay).mean())

    assert level(0.20) > level(0.02)


def test_energy_maps_through_dbfs_and_is_bounded():
    config = OverlayConfig()
    assert energy_from_rms(0.0, config) == 0.0
    assert energy_from_rms(1.0, config) == 1.0
    assert 0.0 < energy_from_rms(0.01, config) < 1.0


# --- colour distinguishes whisper ---------------------------------------------

def test_whisper_speech_and_other_are_three_different_colours():
    assert colour_for("POSSIBLE_WHISPER") == WHISPER_COLOUR
    assert colour_for("POSSIBLE_SPEECH") == SPEECH_COLOUR
    assert len({colour_for("POSSIBLE_WHISPER"), colour_for("POSSIBLE_SPEECH"),
                colour_for("SOUND_DETECTED")}) == 3


def test_a_whisper_paints_a_different_colour_than_speech(camera, overlay):
    def painted(event_type):
        ring = BearingRing(overlay)
        ring.add(echo_for(FakeEvent(direction=0.0, event_type=event_type),
                          camera, overlay))
        after = composite(blank(camera), ring, 0.0, overlay)
        return after[camera.height // 2, camera.width // 2].tolist()

    assert painted("POSSIBLE_WHISPER") != painted("POSSIBLE_SPEECH")


# --- OFF-FRAME is a wedge, never a band at the edge --------------------------

def test_an_off_frame_sound_produces_no_band(camera, overlay):
    echo = echo_for(FakeEvent(direction=60.0), camera, overlay)
    assert echo.is_off_frame is True
    assert echo.is_located is False
    assert echo.band is None
    assert echo.off_frame_side == "right"


def test_the_off_frame_marker_is_not_a_full_height_band(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=60.0), camera, overlay))
    before = blank(camera)
    after = composite(before, ring, 0.0, overlay)

    # A wedge: painted at the vertical centre, absent at the very top and
    # bottom. A band pinned to the edge would paint every row identically and
    # read as "the sound is over there", which is not what off-frame means.
    changed = before != after
    assert np.any(changed[camera.height // 2])
    assert not np.any(changed[0])
    assert not np.any(changed[-1])


def test_the_off_frame_marker_appears_on_the_correct_side(camera, overlay):
    for bearing, side in ((60.0, "right"), (-60.0, "left")):
        ring = BearingRing(overlay)
        ring.add(echo_for(FakeEvent(direction=bearing), camera, overlay))
        after = composite(blank(camera), ring, 0.0, overlay)
        row = after[camera.height // 2]
        if side == "right":
            assert np.any(row[-8:] != 0) and not np.any(row[:8] != 0)
        else:
            assert np.any(row[:8] != 0) and not np.any(row[-8:] != 0)


def test_the_off_frame_marker_uses_its_own_colour(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=60.0), camera, overlay))
    after = composite(blank(camera), ring, 0.0, overlay)
    pixel = after[camera.height // 2, camera.width - 2]
    assert pixel.tolist() != [0, 0, 0]
    # Tinted toward the off-frame colour, not the whisper colour.
    assert pixel[2] > pixel[1]


def test_off_frame_is_never_drawn_as_a_band_at_the_edge_column(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=75.0), camera, overlay))
    after = composite(blank(camera), ring, 0.0, overlay)
    # The extreme edge column must NOT be uniformly painted top to bottom.
    edge = after[:, camera.width - 1]
    assert not np.all(np.any(edge != 0, axis=1))


# --- a CLIPPED band is a different thing, and legitimate ---------------------

def test_a_clipped_band_is_still_drawn_as_a_band(camera, overlay):
    echo = echo_for(FakeEvent(direction=31.0, localization_confidence=0.25),
                    camera, overlay)
    assert echo.is_located is True
    assert echo.is_off_frame is False
    assert echo.band.clipped_right is True


def test_a_clipped_band_is_marked_but_not_confused_with_off_frame(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=31.0, localization_confidence=0.25),
                      camera, overlay))
    before = blank(camera)
    after = composite(before, ring, 0.0, overlay)

    changed = before != after
    # Unlike the off-frame wedge, the top row IS painted: the source is in shot.
    assert np.any(changed[0])
    assert np.any(changed[camera.height // 2])


def test_clipping_and_off_frame_never_both_apply(camera, overlay):
    for bearing in (-70.0, -31.0, 0.0, 31.0, 70.0):
        echo = echo_for(FakeEvent(direction=bearing, localization_confidence=0.25),
                        camera, overlay)
        clipped = echo.is_located and (echo.band.clipped_left or echo.band.clipped_right)
        assert not (clipped and echo.is_off_frame)


# --- declining to answer draws NOTHING ----------------------------------------

def test_a_declined_event_draws_nothing_at_all(camera, overlay):
    event = FakeEvent(direction=None, reason="localization confidence 0.11 is below 0.30")
    ring = BearingRing(overlay)
    ring.add(echo_for(event, camera, overlay))

    before = blank(camera)
    after = composite(before, ring, 0.0, overlay)
    assert np.array_equal(before, after)


def test_a_declined_event_keeps_its_reason_for_display(camera, overlay):
    event = FakeEvent(direction=None, reason="uncorrelated channels")
    ring = BearingRing(overlay)
    ring.add(echo_for(event, camera, overlay))

    assert ring.latest_decline(0.0) == "uncorrelated channels"
    hud = build_hud(source_kind="synthetic", ring=ring, at_time=0.0, camera=camera)
    assert "uncorrelated channels" in hud.all_text()


def test_a_declined_event_never_becomes_a_guess(camera, overlay):
    echo = echo_for(FakeEvent(direction=None, reason="silent"), camera, overlay)
    assert echo.declined is True
    assert echo.band is None
    assert echo.off_frame_side == ""
    assert echo.bearing_degrees is None


def test_a_missing_resolution_also_declines_rather_than_assuming_one(camera, overlay):
    echo = echo_for(FakeEvent(direction=10.0, resolution=None), camera, overlay)
    assert echo.band is None
    assert "uncertainty is unknown" in echo.reason


# --- the synthetic banner ------------------------------------------------------

def test_the_banner_is_present_whenever_the_source_is_synthetic():
    assert banner_text("synthetic") == SYNTHETIC_BANNER
    assert "NOT A LIVE MEASUREMENT" in banner_text("synthetic")


def test_there_is_no_banner_on_genuinely_live_audio():
    assert banner_text("hardware") is None


def test_the_hud_carries_the_banner_for_synthetic(camera, overlay):
    ring = BearingRing(overlay)
    hud = build_hud(source_kind="synthetic", ring=ring, at_time=0.0, camera=camera)
    assert hud.banner == SYNTHETIC_BANNER
    assert SYNTHETIC_BANNER in hud.all_text()


def test_the_banner_paints_an_unmissable_bar(camera):
    frame = blank(camera)
    painted = draw_banner(frame.copy(), present=True)
    assert not np.array_equal(painted[0:40], frame[0:40])
    # Big: tens of rows across the full width, not a small grey caption.
    assert np.all(np.any(painted[10] != frame[10], axis=1))


def test_no_bar_is_painted_for_hardware(camera):
    frame = blank(camera)
    assert np.array_equal(draw_banner(frame.copy(), present=False), frame)


@pytest.mark.parametrize("kind", ["synthetic", "", "unknown", "mock"])
def test_anything_that_is_not_hardware_gets_the_banner(kind):
    assert banner_text(kind) == SYNTHETIC_BANNER


# --- sync: the newest bearing is not the newest frame ------------------------

def test_rendering_selects_by_timestamp_not_by_recency(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=-25.0), camera, overlay, at=0.0))
    ring.add(echo_for(FakeEvent(direction=25.0), camera, overlay, at=10.0))

    # The +25 echo is the most recently ADDED, but it belongs 10 s later.
    active = ring.active(0.1)
    assert len(active) == 1
    assert active[0][0].bearing_degrees == -25.0


def test_an_echo_from_the_future_is_not_drawn_yet(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=0.0), camera, overlay, at=5.0))
    assert ring.active(0.0) == []


def test_a_small_lead_is_tolerated_rather_than_hidden(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=0.0), camera, overlay, at=0.1))
    assert ring.active(0.0)          # within sync_tolerance_seconds


def test_the_ring_prunes_what_it_can_no_longer_show(camera, overlay):
    ring = BearingRing(overlay)
    for i in range(5):
        ring.add(echo_for(FakeEvent(direction=0.0), camera, overlay, at=float(i)))
    ring.prune(10.0)
    assert len(ring) == 0


# --- the HUD says what it must -------------------------------------------------

def test_the_hud_reports_bearing_type_and_confidence(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=-12.0, event_type="POSSIBLE_WHISPER"),
                      camera, overlay))
    text = build_hud(source_kind="synthetic", ring=ring, at_time=0.0,
                     camera=camera).all_text()
    assert "POSSIBLE_WHISPER" in text
    assert "-12.0 deg" in text
    assert "confidence" in text


def test_the_hud_reports_link_diagnostics(camera, overlay):
    hud = build_hud(
        source_kind="hardware", ring=BearingRing(overlay), at_time=0.0, camera=camera,
        link_diagnostics={"packets_dropped_total": 7,
                          "packets_dropped_header_crc": 2,
                          "packets_dropped_payload_crc": 5,
                          "frames_abandoned": 3})
    text = hud.all_text()
    assert "7 packets dropped" in text
    assert "header CRC 2" in text and "payload CRC 5" in text
    assert "LINK DEGRADING" in hud.warnings[0]


def test_a_healthy_link_raises_no_warning(camera, overlay):
    hud = build_hud(source_kind="hardware", ring=BearingRing(overlay), at_time=0.0,
                    camera=camera,
                    link_diagnostics={"packets_dropped_total": 0,
                                      "frames_abandoned": 0})
    assert hud.warnings == []


def test_the_hud_shows_latency_fps_and_overlay_cost(camera, overlay):
    text = build_hud(source_kind="synthetic", ring=BearingRing(overlay), at_time=0.0,
                     camera=camera, video_fps=29.7, overlay_ms=0.42,
                     av_latency_ms=83.0, av_offset_ms=-20.0).all_text()
    assert "29.7 fps" in text
    assert "0.42 ms/frame" in text
    assert "83 ms" in text
    assert "-20 ms" in text


def test_the_hud_states_the_front_back_ambiguity(camera, overlay):
    text = build_hud(source_kind="synthetic", ring=BearingRing(overlay), at_time=0.0,
                     camera=camera).all_text()
    assert "BEHIND" in text
    assert "cannot be shown" in text


def test_the_hud_says_how_much_of_the_world_is_out_of_shot(camera, overlay):
    text = build_hud(source_kind="synthetic", ring=BearingRing(overlay), at_time=0.0,
                     camera=camera).all_text()
    assert "-35 to +35 deg" in text


def test_the_hud_surfaces_the_parallax_warning(camera, overlay):
    hud = build_hud(source_kind="synthetic", ring=BearingRing(overlay), at_time=0.0,
                    camera=camera, parallax_note="camera is 20 cm off the array centre")
    assert "PARALLAX" in hud.warnings[0]


def test_the_hud_names_off_frame_explicitly(camera, overlay):
    ring = BearingRing(overlay)
    ring.add(echo_for(FakeEvent(direction=60.0), camera, overlay))
    text = build_hud(source_kind="synthetic", ring=ring, at_time=0.0,
                     camera=camera).all_text()
    assert "OFF-FRAME to the right" in text
    assert "not located in shot" in text


# --- the app shell, without a camera or a board -------------------------------

def load_tool():
    spec = importlib.util.spec_from_file_location(
        "tool_heatmap", TOOLS / "whisper_heatmap_webcam.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return load_tool()


def test_the_app_imports_without_opencv(app):
    assert callable(app.main)
    assert callable(app.benchmark)


def test_no_camera_exits_nonzero_without_a_traceback(app, capsys):
    assert app.main([]) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert err.strip()


def test_the_typical_overlay_is_well_under_the_frame_budget(app):
    # Measured ~3.1 ms at 720p with one band: 9% of the 33.3 ms budget.
    per_frame = app.benchmark(width=1280, height=720, bands=1, iterations=60)
    assert per_frame < (1000.0 / 30.0) / 4, f"{per_frame:.2f} ms/frame is too slow"


def test_even_a_busy_frame_fits_inside_the_budget(app):
    # Four overlapping wide bands is the worst realistic case: ~10.5 ms at 720p,
    # 31% of budget. It fits, but it is NOT "well under" - which is exactly why
    # the HUD announces it rather than letting video quietly fall behind.
    per_frame = app.benchmark(width=1280, height=720, bands=4, iterations=60)
    assert per_frame < 1000.0 / 30.0, f"{per_frame:.2f} ms/frame does not fit"


def test_the_hud_announces_an_overlay_that_is_eating_the_budget(camera, overlay):
    hud = build_hud(source_kind="synthetic", ring=BearingRing(overlay), at_time=0.0,
                    camera=camera, overlay_ms=25.0)
    assert any("OVERLAY COST" in w for w in hud.warnings)
    assert "video may not keep up" in hud.all_text()


def test_a_cheap_overlay_raises_no_budget_warning(camera, overlay):
    hud = build_hud(source_kind="synthetic", ring=BearingRing(overlay), at_time=0.0,
                    camera=camera, overlay_ms=3.0)
    assert not any("OVERLAY COST" in w for w in hud.warnings)


def test_the_benchmark_mode_reports_and_succeeds(app, capsys):
    assert app.main(["--benchmark"]) == 0
    out = capsys.readouterr().out
    assert "frame budget at 30 fps" in out
    # Typical AND worst case, both stated: a single averaged figure would hide
    # that four overlapping bands cost three times what one does.
    assert "typical (1 band)" in out
    assert "busy (4 overlapping bands)" in out
    assert "% of budget" in out


def test_a_real_acoustic_event_flows_all_the_way_to_pixels(camera, overlay):
    from acoustic_array import AcousticArray

    with AcousticArray.synthetic(angle_degrees=-15.0, max_frames=24) as array:
        events = [e for e in array.stream() if e.has_direction]
    assert events

    ring = BearingRing(overlay)
    ring.add(echo_from_event(events[0], camera, overlay, display_time=0.0))
    after = composite(blank(camera), ring, 0.0, overlay)
    assert not np.array_equal(after, blank(camera))

    hud = build_hud(source_kind="synthetic", ring=ring, at_time=0.0, camera=camera)
    assert hud.banner == SYNTHETIC_BANNER


def test_the_synthetic_source_is_paced_to_real_time(app, camera, overlay):
    # The synthetic source is UNPACED by construction (CONTEXT section 8) and
    # outruns real time by orders of magnitude. Unpaced, half a second would
    # deliver hundreds of events, fill the ring instantly and make the 1.5 s
    # decay window meaningless.
    import time

    from acoustic_array import AcousticArray

    ring = BearingRing(overlay)
    array = AcousticArray.synthetic(angle_degrees=-18.0)
    array.start()
    pump = app.AudioPump(array, ring, camera, overlay, 0.0, pace=True)
    pump.start()
    time.sleep(0.6)
    pump.stop()
    array.stop()

    # ~1-2 events per second of simulated audio, not hundreds.
    assert pump.events_seen < 20, pump.events_seen


def test_hardware_is_not_paced_because_the_wire_already_paces_it(app, camera, overlay):
    pump = app.AudioPump(object(), BearingRing(overlay), camera, overlay, 0.0)
    assert pump.pace is False


def test_a_synthetic_fallback_is_announced_not_silent(app):
    array, note = app.build_array(port="COM_NOPE", prefer_hardware=True)
    try:
        assert array.source_kind == "synthetic"
        assert note is not None and "SYNTHETIC" in note
    finally:
        array.stop()
