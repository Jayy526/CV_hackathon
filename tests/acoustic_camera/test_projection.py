"""Bearing -> pixel projection, against a synthetic camera. No webcam, no board.

Expected pixel columns here are computed independently from the trigonometry,
not read back out of the module under test.
"""

import math

import pytest

from acoustic_camera import (
    CameraConfig,
    CameraConfigError,
    load_camera_config,
    parallax_dominant_within_m,
    parallax_error_degrees,
    parallax_warning,
    project_band,
    project_bearing,
    project_event,
    uncertainty_degrees,
    visible_bearing_range,
)


def cam(**kw):
    base = dict(width=1280, height=720, horizontal_fov_degrees=70.0,
                azimuth_offset_degrees=0.0)
    base.update(kw)
    return CameraConfig(**base)


def expected_column(bearing, width=1280, hfov=70.0, offset=0.0):
    """The projection, worked out here from scratch."""
    f_px = (width / 2.0) / math.tan(math.radians(hfov / 2.0))
    return width / 2.0 + f_px * math.tan(math.radians(bearing - offset))


# --- the focal length --------------------------------------------------------

def test_focal_length_matches_the_formula():
    config = cam()
    assert config.focal_length_px == pytest.approx(640.0 / math.tan(math.radians(35.0)))
    assert config.focal_length_px == pytest.approx(914.01, abs=0.01)


def test_a_wider_lens_has_a_shorter_focal_length():
    assert cam(horizontal_fov_degrees=90.0).focal_length_px < \
           cam(horizontal_fov_degrees=55.0).focal_length_px


def test_the_optical_axis_is_the_frame_centre():
    assert cam().centre_x == 640.0
    assert project_bearing(0.0, cam()).column == pytest.approx(640.0)


# --- the projection is a tangent map, not a linear one ------------------------

@pytest.mark.parametrize("bearing", [-30.0, -20.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0, 30.0])
def test_projection_matches_independently_computed_values(bearing):
    result = project_bearing(bearing, cam())
    assert result.on_screen
    assert result.column == pytest.approx(expected_column(bearing))


def test_the_map_is_tangent_not_linear():
    config = cam()
    # A linear map would put 30 deg at 30/35 of the way to the edge.
    linear = config.centre_x + (30.0 / 35.0) * config.centre_x
    actual = project_bearing(30.0, config).column
    assert actual == pytest.approx(expected_column(30.0))
    # The difference is large enough to be visible: tens of pixels.
    assert abs(actual - linear) > 20.0


def test_the_linear_error_is_worst_in_the_mid_field_not_at_the_edge():
    # Both maps send 0 to the centre and +/-HFOV/2 to the frame boundary, so
    # they AGREE at both ends. Checking only the centre and the edge would
    # therefore prove nothing; the error lives in between.
    config = cam()

    def linear(bearing):
        return config.centre_x + (bearing / config.half_fov_degrees) * config.centre_x

    near = abs(project_bearing(2.0, config).column - linear(2.0))
    mid = abs(project_bearing(20.0, config).column - linear(20.0))
    far = abs(project_bearing(33.0, config).column - linear(33.0))

    assert near < 5.0                    # agree near the axis
    assert mid > 30.0                    # ~33 px out: 2.6% of frame width
    assert mid > near * 5
    assert mid > far                     # converging again toward the edge


def test_positive_bearings_go_right_and_negative_left():
    config = cam()
    assert project_bearing(20.0, config).column > config.centre_x
    assert project_bearing(-20.0, config).column < config.centre_x


def test_the_projection_is_symmetric_about_the_axis():
    config = cam()
    left = project_bearing(-25.0, config).column
    right = project_bearing(25.0, config).column
    assert (config.centre_x - left) == pytest.approx(right - config.centre_x)


# --- the tangent singularity, at the exact boundary --------------------------

@pytest.mark.parametrize("bearing", [90.0, -90.0])
def test_exactly_ninety_degrees_is_off_frame_not_infinite(bearing):
    result = project_bearing(bearing, cam(horizontal_fov_degrees=179.0))
    assert result.on_screen is False
    assert result.column is None
    assert result.side == ("right" if bearing > 0 else "left")
    assert "never meets the image plane" in result.reason


@pytest.mark.parametrize("bearing", [90.0, -90.0, 120.0, -120.0, 180.0, -180.0])
def test_at_or_beyond_ninety_never_produces_inf_or_nan(bearing):
    result = project_bearing(bearing, cam(horizontal_fov_degrees=179.0))
    assert result.column is None
    assert not math.isinf(result.relative_degrees)
    assert not math.isnan(result.relative_degrees)


def test_just_inside_ninety_is_finite():
    # 179 deg FOV so the FOV test does not mask the singularity test.
    result = project_bearing(89.9, cam(horizontal_fov_degrees=179.0))
    assert result.column is None or math.isfinite(result.column)


def test_beyond_ninety_is_not_wrapped_back_into_frame():
    # tan(100 deg) is negative, so an unguarded formula would place a source
    # behind the camera on the LEFT of the picture. It must be off-frame right.
    result = project_bearing(100.0, cam(horizontal_fov_degrees=179.0))
    assert result.on_screen is False
    assert result.side == "right"


def test_the_azimuth_offset_moves_the_singularity_with_it():
    config = cam(azimuth_offset_degrees=20.0, horizontal_fov_degrees=179.0)
    assert project_bearing(110.0, config).on_screen is False   # relative +90
    assert project_bearing(-70.0, config).on_screen is False   # relative -90


# --- off-frame is reported, never clamped ------------------------------------

def test_a_bearing_outside_the_fov_is_off_frame_with_a_side():
    config = cam()
    right = project_bearing(50.0, config)
    left = project_bearing(-50.0, config)

    assert right.on_screen is False and right.side == "right"
    assert left.on_screen is False and left.side == "left"
    assert "OFF-FRAME" in right.reason and "OFF-FRAME" in left.reason


def test_an_off_frame_bearing_is_never_clamped_to_the_edge():
    config = cam()
    for bearing in (40.0, 60.0, 89.0, -40.0, -60.0, -89.0):
        result = project_bearing(bearing, config)
        assert result.column is None, bearing
        # Nothing that could be mistaken for a drawable column.
        assert result.column not in (0.0, float(config.width - 1))


def test_the_fov_edge_itself_is_the_boundary():
    config = cam()
    just_inside = project_bearing(34.9, config)
    just_outside = project_bearing(35.1, config)
    assert just_inside.on_screen is True
    assert just_outside.on_screen is False


def test_the_last_valid_column_is_width_minus_one():
    config = cam()
    inside = project_bearing(34.9, config)
    assert inside.on_screen
    assert 0.0 <= inside.column <= config.width - 1


def test_a_narrow_lens_makes_more_of_the_world_off_frame():
    assert project_bearing(30.0, cam(horizontal_fov_degrees=90.0)).on_screen is True
    assert project_bearing(30.0, cam(horizontal_fov_degrees=55.0)).on_screen is False


def test_the_visible_bearing_range_is_reported():
    assert visible_bearing_range(cam()) == pytest.approx((-35.0, 35.0))
    assert visible_bearing_range(cam(azimuth_offset_degrees=10.0)) == \
        pytest.approx((-25.0, 45.0))


# --- a non-zero azimuth offset -----------------------------------------------

def test_the_azimuth_offset_shifts_the_whole_mapping():
    config = cam(azimuth_offset_degrees=15.0)
    # A source at +15 deg in array coordinates is now dead centre.
    assert project_bearing(15.0, config).column == pytest.approx(config.centre_x)
    assert project_bearing(0.0, config).column == pytest.approx(
        expected_column(0.0, offset=15.0))


def test_a_negative_azimuth_offset_shifts_the_other_way():
    config = cam(azimuth_offset_degrees=-12.0)
    assert project_bearing(-12.0, config).column == pytest.approx(config.centre_x)
    assert project_bearing(0.0, config).column > config.centre_x


def test_the_offset_changes_which_bearings_are_visible():
    config = cam(azimuth_offset_degrees=20.0)
    assert project_bearing(50.0, config).on_screen is True     # relative +30
    assert project_bearing(-20.0, config).on_screen is False   # relative -40
    assert project_bearing(-20.0, config).side == "left"


# --- band width: the two edges are projected separately ----------------------

def test_the_band_width_is_the_difference_of_two_projections():
    config = cam()
    band = project_band(0.0, angular_resolution_degrees=4.55, confidence=1.0,
                        config=config)
    sigma = 4.55
    assert band.left_column == pytest.approx(expected_column(-sigma))
    assert band.right_column == pytest.approx(expected_column(+sigma))
    assert band.width_px == pytest.approx(
        expected_column(sigma) - expected_column(-sigma))


def test_the_same_uncertainty_is_wider_in_pixels_at_the_edge():
    # THE POINT of projecting both edges. A width computed once at the centre
    # and reused would understate the doubt exactly where it is largest.
    config = cam()
    centre = project_band(0.0, 4.55, 1.0, config)
    edge = project_band(30.0, 4.55, 1.0, config)
    assert edge.width_px > centre.width_px * 1.3


def test_a_centre_computed_width_would_be_wrong_at_the_edge():
    config = cam()
    centre = project_band(0.0, 4.55, 1.0, config)
    edge = project_band(30.0, 4.55, 1.0, config)
    # Compare an UNCLIPPED band, or the clip would flatter the centre estimate.
    assert edge.clipped_left is False and edge.clipped_right is False
    # ~49 px of understatement: visible, not academic.
    assert edge.width_px - centre.width_px > 40.0


def test_the_band_genuinely_widens_as_confidence_drops():
    config = cam()
    widths = [project_band(0.0, 4.55, c, config).width_px
              for c in (1.0, 0.8, 0.6, 0.4, 0.2)]
    assert widths == sorted(widths)
    assert widths[-1] > widths[0] * 2


def test_the_band_never_narrows_below_the_arrays_own_resolution():
    config = cam()
    floor = project_band(0.0, 4.55, 1.0, config).width_px
    assert project_band(0.0, 4.55, 2.0, config).width_px == pytest.approx(floor)


def test_uncertainty_is_floored_so_zero_confidence_is_not_infinite():
    assert uncertainty_degrees(4.55, 0.0) == pytest.approx(4.55 / 0.05)
    assert math.isfinite(uncertainty_degrees(4.55, 0.0))


def test_uncertainty_rejects_a_nonsense_resolution():
    with pytest.raises(ValueError):
        uncertainty_degrees(0.0, 0.5)


def test_a_band_whose_centre_is_visible_may_be_clipped_at_the_edge():
    config = cam()
    band = project_band(33.0, 4.55, 0.3, config)     # wide band near the edge
    assert band.on_screen is True
    assert band.clipped_right is True
    assert band.right_column == pytest.approx(config.width - 1)
    assert band.left_column is not None


def test_an_off_frame_band_has_no_columns_at_all():
    band = project_band(60.0, 4.55, 0.9, cam())
    assert band.on_screen is False
    assert band.left_column is None and band.right_column is None
    assert band.width_px is None
    assert band.centre.side == "right"


def test_clipping_a_band_edge_is_flagged_and_not_silent():
    band = project_band(-33.0, 4.55, 0.3, cam())
    assert band.clipped_left is True
    assert band.left_column == 0.0


# --- projecting an event ------------------------------------------------------

class FakeEvent:
    def __init__(self, direction, resolution=4.55, localization_confidence=0.8):
        self.direction_degrees = direction
        self.angular_resolution_degrees = resolution
        self.localization_confidence = localization_confidence
        self.confidence = 0.5
        self.reason = ""


def test_an_event_with_a_direction_projects():
    band = project_event(FakeEvent(-15.0), cam())
    assert band is not None and band.on_screen
    assert band.centre.column == pytest.approx(expected_column(-15.0))


def test_an_event_with_no_direction_projects_to_nothing():
    # Not off-screen. Nothing. The renderer must draw nothing and show reason.
    assert project_event(FakeEvent(None), cam()) is None


def test_a_real_acoustic_event_projects():
    from acoustic_array import AcousticArray

    with AcousticArray.synthetic(angle_degrees=-20.0, max_frames=24) as array:
        events = [e for e in array.stream() if e.has_direction]
    assert events
    band = project_event(events[0], cam())
    assert band is not None and band.on_screen
    assert band.width_px > 0


# --- parallax -----------------------------------------------------------------

@pytest.mark.parametrize("offset,range_m,expected", [
    (0.02, 1.0, 1.15), (0.02, 4.0, 0.29),
    (0.05, 1.0, 2.86), (0.05, 3.0, 0.95),
    (0.10, 1.0, 5.71), (0.10, 2.0, 2.86),
    (0.20, 1.0, 11.31), (0.20, 4.0, 2.86),
])
def test_the_parallax_table_is_reproduced(offset, range_m, expected):
    assert parallax_error_degrees(offset, range_m) == pytest.approx(expected, abs=0.01)


def test_parallax_falls_below_the_array_resolution_only_beyond_a_known_range():
    assert parallax_dominant_within_m(0.02) == pytest.approx(0.25, abs=0.01)
    assert parallax_dominant_within_m(0.05) == pytest.approx(0.63, abs=0.01)
    assert parallax_dominant_within_m(0.10) == pytest.approx(1.26, abs=0.01)
    assert parallax_dominant_within_m(0.20) == pytest.approx(2.51, abs=0.01)


def test_a_camera_at_the_array_centre_needs_no_warning():
    assert parallax_warning(cam(lateral_offset_m=0.0)) is None
    assert parallax_dominant_within_m(0.0) == 0.0


def test_a_large_offset_warns_and_says_it_cannot_be_calibrated_out():
    message = parallax_warning(cam(lateral_offset_m=0.20))
    assert message is not None
    assert "CANNOT be calibrated out" in message
    assert "2.51 m" in message
    assert "azimuth_offset_degrees will not absorb it" in message


def test_a_zero_range_is_rejected_rather_than_dividing_by_zero():
    with pytest.raises(CameraConfigError):
        parallax_error_degrees(0.05, 0.0)


# --- configuration ------------------------------------------------------------

def test_the_shipped_camera_yaml_loads():
    config = load_camera_config()
    assert 0.0 < config.horizontal_fov_degrees < 180.0
    assert config.width > 0 and config.height > 0


def test_a_missing_camera_file_falls_back_to_documented_defaults(tmp_path):
    config = load_camera_config(tmp_path / "nope.yaml")
    assert config.horizontal_fov_degrees == 70.0
    assert config.index == 0


def test_the_camera_yaml_documents_the_parallax_budget():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "config" / "camera.yaml").read_text(
        encoding="utf-8")
    assert "UNCORRECTABLE IN PRINCIPLE" in text
    assert "lateral_offset_m" in text
    for figure in ("1.15", "2.86", "5.71", "11.31"):
        assert figure in text


@pytest.mark.parametrize("fov", [0.0, -10.0, 180.0, 200.0])
def test_an_impossible_field_of_view_is_rejected(fov):
    with pytest.raises(CameraConfigError):
        cam(horizontal_fov_degrees=fov)


def test_a_negative_resolution_is_rejected():
    with pytest.raises(CameraConfigError):
        cam(width=0)


def test_an_azimuth_offset_beyond_ninety_is_rejected():
    with pytest.raises(CameraConfigError):
        cam(azimuth_offset_degrees=95.0)


def test_a_negative_lateral_offset_is_rejected():
    with pytest.raises(CameraConfigError):
        cam(lateral_offset_m=-0.05)


def test_the_projection_never_imports_the_array_package():
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[2] / "acoustic_camera"
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = getattr(node, "module", None)
            if isinstance(node, ast.ImportFrom) and module:
                assert not module.startswith("acoustic_array"), path.name


# --- tools/calibrate_camera_audio.py, headless -------------------------------

import importlib.util
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load_tool(name):
    spec = importlib.util.spec_from_file_location("tool_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def calib():
    return load_tool("calibrate_camera_audio")


def point(calib, known, measured, confidence=0.8):
    return calib.CalibrationPoint(
        known_angle_degrees=known,
        measured_bearing_degrees=measured,
        projected_column=None,
        observed_column=None,
        localization_confidence=confidence,
        frames_used=20,
        frames_rejected=2,
    )


def test_a_consistent_offset_is_estimated(calib):
    # Every point implies the same +6 deg mounting angle.
    points = [point(calib, a, a + 6.0) for a in (-30, -15, 0, 15, 30)]
    offset, verdict, explanation = calib.estimate_offset(points)

    assert verdict == "OK"
    assert offset == pytest.approx(6.0, abs=0.01)
    assert "camera.yaml" in explanation


def test_a_zero_offset_is_estimated_as_zero(calib):
    points = [point(calib, a, a) for a in (-30, -15, 0, 15, 30)]
    offset, verdict, _ = calib.estimate_offset(points)
    assert verdict == "OK" and offset == pytest.approx(0.0, abs=0.01)


def test_disagreeing_points_are_unusable_not_averaged(calib):
    # A mounting angle is one fixed number. These imply -20, 0, +25.
    points = [point(calib, -30, -50), point(calib, 0, 0), point(calib, 30, 55)]
    offset, verdict, explanation = calib.estimate_offset(points)

    assert verdict == "UNUSABLE"
    assert offset is None
    assert "disagree" in explanation
    assert "not measuring it" in explanation


def test_too_few_points_is_unusable(calib):
    points = [point(calib, -30, -28), point(calib, 0, 2), point(calib, 30, None)]
    offset, verdict, explanation = calib.estimate_offset(points)
    assert verdict == "UNUSABLE" and offset is None
    assert "verify_localization.py" in explanation


def test_a_mirrored_array_is_named_not_absorbed_into_an_offset(calib):
    # Measured bearing moves opposite to the known angle. No offset fixes that.
    points = [point(calib, a, -a) for a in (-30, -15, 0, 15, 30)]
    offset, verdict, explanation = calib.estimate_offset(points)

    assert verdict == "MIRRORED"
    assert offset is None
    assert "cannot fix it" in explanation
    assert "channel 0" in explanation


def test_the_implied_offset_is_measured_minus_known(calib):
    assert point(calib, 20.0, 26.5).implied_offset_degrees == pytest.approx(6.5)
    assert point(calib, 20.0, None).implied_offset_degrees is None


def test_the_column_error_needs_both_columns(calib):
    p = point(calib, 0.0, 0.0)
    assert p.column_error_px is None
    p.projected_column, p.observed_column = 640.0, 610.0
    assert p.column_error_px == pytest.approx(30.0)


def test_an_unusable_verdict_writes_nothing(calib, capsys):
    from acoustic_camera import load_camera_config

    points = [point(calib, -30, -50), point(calib, 0, 0), point(calib, 30, 55)]
    offset, verdict, explanation = calib.estimate_offset(points)
    assert calib.report(points, offset, verdict, explanation,
                        load_camera_config()) is False
    out = capsys.readouterr().out
    assert "camera.yaml is unchanged" in out


def test_the_report_shows_every_point_including_failed_ones(calib, capsys):
    from acoustic_camera import load_camera_config

    points = [point(calib, -30, -24), point(calib, 0, 6), point(calib, 30, None)]
    offset, verdict, explanation = calib.estimate_offset(points)
    calib.report(points, offset, verdict, explanation, load_camera_config())
    out = capsys.readouterr().out
    assert "known" in out and "implied off" in out
    assert out.count("--") >= 1          # the failed point is shown, not dropped


def test_the_report_warns_about_a_large_mounting_offset(calib, capsys):
    from acoustic_camera import CameraConfig

    points = [point(calib, a, a + 3.0) for a in (-30, 0, 30)]
    offset, verdict, explanation = calib.estimate_offset(points)
    calib.report(points, offset, verdict, explanation,
                 CameraConfig(lateral_offset_m=0.20))
    out = capsys.readouterr().out
    assert "CANNOT be calibrated out" in out


# --- it refuses clearly with no hardware -------------------------------------

def test_no_camera_refuses_without_a_traceback(calib, capsys, monkeypatch):
    from acoustic_camera import CameraConfig

    def broken(_camera):
        raise RuntimeError("camera index 0 did not open.")

    monkeypatch.setattr(calib, "open_camera", broken)
    assert calib.main(["--frames", "2"], prompt=lambda *a: "") == 2
    assert "Cannot open the camera" in capsys.readouterr().err


def test_no_esp32_refuses_and_says_there_is_no_synthetic_fallback(
    calib, capsys, monkeypatch
):
    assert calib.main(["--no-camera", "--frames", "2", "--angles", "0"],
                      prompt=lambda *a: "") == 2
    err = capsys.readouterr().err
    assert "Cannot reach the microphone array" in err
    assert "no synthetic fallback" in err
    assert "Traceback" not in err


def test_an_interrupted_run_reports_nothing_measured(calib, capsys):
    def refuse(*_args):
        raise KeyboardInterrupt

    assert calib.main(["--no-camera", "--frames", "2"], prompt=refuse) == 2
    assert "nothing measured" in capsys.readouterr().err.lower()


def test_the_tool_loads_without_opencv_installed(calib, monkeypatch):
    # Importing the module must not require cv2; only open_camera does.
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert callable(calib.estimate_offset)
