"""The acoustic_array package boundary: it must stand alone.

The point of the split is that this package is a standalone acoustic direction
sensor. These tests pin the boundary itself - what it may import, what its
events may carry, and that it runs with no room configuration in existence.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "acoustic_array"
ROOT = Path(__file__).resolve().parents[2]


def module_files():
    return sorted(p for p in PACKAGE.glob("*.py"))


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


# --- the boundary ------------------------------------------------------------

def test_the_package_never_imports_heimdall():
    offenders = {p.name: sorted(n for n in imported_modules(p) if n.startswith("heimdall"))
                 for p in module_files()}
    offenders = {k: v for k, v in offenders.items() if v}
    assert offenders == {}, f"acoustic_array must stand alone: {offenders}"


def test_the_package_never_imports_seats_or_classroom_geometry():
    banned = ("seat_mapper", "classroom")
    for path in module_files():
        for name in imported_modules(path):
            assert not any(b in name for b in banned), f"{path.name} imports {name}"


def test_the_direction_path_has_no_seat_identifiers():
    # Prose explaining what this is NOT is fine; identifiers are not. events.py
    # is excluded deliberately: DetectedEvent carries a vestigial seat_id that
    # the PARKED layer fills in and an existing test pins. This package never
    # sets it - test_this_package_never_populates_the_vestigial_seat_field.
    for name in ("api", "geometry", "doa", "gcc_phat", "sources", "__init__"):
        path = PACKAGE / (name + ".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert "seat" not in node.id.lower(), f"{path.name}: {node.id}"
            if isinstance(node, ast.Attribute):
                assert "seat" not in node.attr.lower(), f"{path.name}: {node.attr}"


def test_this_package_never_populates_the_vestigial_seat_field():
    from acoustic_array import AcousticArray

    with AcousticArray.synthetic(angle_degrees=15.0, max_frames=24) as array:
        list(array.stream())
    # Nothing in the sensor path assigns it, on any code path.
    for path in module_files():
        body = path.read_text(encoding="utf-8")
        assert "seat_id=" not in body, path.name


def test_every_expected_module_moved_across():
    expected = {"config", "device", "frame", "sources", "receiver", "packets",
                "analysis", "gcc_phat", "doa", "geometry", "events", "api",
                "synthetic", "__init__"}
    assert {p.stem for p in module_files()} == expected


def test_the_readme_exists_and_states_the_limits():
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    for phrase in ("No range", "No elevation", "Front and back are ambiguous",
                   "A bearing is not a location"):
        assert phrase in readme
    # Wiring, per section 10.
    for phrase in ("GPIO 26", "GPIO 25", "GPIO 33", "0.135 m", "START_IN_DIAG"):
        assert phrase in readme


# --- it runs with no classroom.yaml ------------------------------------------

def test_the_package_runs_with_classroom_yaml_absent(tmp_path):
    """The decisive test: import and run with no room configuration at all.

    Run in a subprocess with the config directory hidden, because a module-level
    read of classroom.yaml anywhere in the import graph would only show up on a
    fresh interpreter.
    """
    script = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import acoustic_array.config as c, acoustic_array.geometry as g\n"
        "from pathlib import Path\n"
        # Point every config default at a directory that does not exist.
        f"missing = Path({str(tmp_path / 'nothing')!r})\n"
        "c.DEFAULT_CONFIG_PATH = missing / 'audio.yaml'\n"
        "g.DEFAULT_ARRAY_PATH = missing / 'array.yaml'\n"
        "assert not (missing / 'audio.yaml').exists()\n"
        "from acoustic_array import AcousticArray\n"
        "with AcousticArray.synthetic(angle_degrees=25.0, max_frames=12) as a:\n"
        "    events = [e.to_dict() for e in a.stream()]\n"
        "print(json.dumps({'n': len(events), 'kind': events[0]['source_kind'],\n"
        "                  'dir': events[0]['direction_degrees']}))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["n"] > 0
    assert payload["kind"] == "synthetic"
    assert payload["dir"] == pytest.approx(25.0, abs=5.0)


def non_docstring_strings(path):
    """Every string literal that is not a docstring. Prose may say 'classroom';
    a path the code actually opens may not."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if not body or not isinstance(body[0], ast.Expr):
                continue
            first = body[0].value
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                docstrings.add(id(first))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings]


def test_the_package_never_opens_classroom_yaml():
    for path in module_files():
        for literal in non_docstring_strings(path):
            assert "classroom" not in literal.lower(), f"{path.name}: {literal!r}"


def test_the_array_geometry_needs_no_config_file():
    from acoustic_array.geometry import default_array, linear_array

    array = default_array()
    assert array.num_channels == 2
    assert array.spacing == pytest.approx(0.135)
    assert linear_array(4, 0.05).num_channels == 4


def test_a_missing_array_file_falls_back_rather_than_raising(tmp_path):
    from acoustic_array.geometry import load_array_config

    assert load_array_config(tmp_path / "nope.yaml").spacing == pytest.approx(0.135)


# --- the event contract -------------------------------------------------------

def test_the_event_carries_exactly_the_agreed_fields():
    from acoustic_array import AcousticArray

    with AcousticArray.synthetic(angle_degrees=-20.0, max_frames=24) as array:
        events = list(array.stream())
    assert events

    assert set(events[0].to_dict()) == {
        "timestamp", "event_type", "direction_degrees", "confidence",
        "localization_confidence", "angular_resolution_degrees", "duration",
        "channel_rms", "source_kind", "reason",
    }


def test_the_event_carries_no_seat_or_position_fields():
    from acoustic_array import AcousticEvent

    fields = set(AcousticEvent.__dataclass_fields__)
    for banned in ("seat_id", "candidate_seats", "seat_ambiguous", "position"):
        assert banned not in fields


def test_the_event_is_json_serialisable():
    from acoustic_array import AcousticArray

    with AcousticArray.synthetic(angle_degrees=0.0, max_frames=16) as array:
        for event in array.stream():
            json.loads(json.dumps(event.to_dict()))


def test_per_channel_rms_is_reported():
    from acoustic_array import AcousticArray

    with AcousticArray.synthetic(angle_degrees=10.0, max_frames=24) as array:
        events = [e for e in array.stream() if e.has_direction]
    assert events
    assert len(events[0].channel_rms) == 2
    assert all(v > 0 for v in events[0].channel_rms)


def test_source_kind_says_synthetic_and_is_not_live():
    from acoustic_array import SOURCE_SYNTHETIC, AcousticArray

    with AcousticArray.synthetic(max_frames=16) as array:
        assert array.is_live is False
        for event in array.stream():
            assert event.source_kind == SOURCE_SYNTHETIC
            assert event.is_live is False


def test_hardware_is_labelled_live_even_before_it_opens():
    from acoustic_array import SOURCE_HARDWARE, AcousticArray

    array = AcousticArray.hardware(port="COM_FAKE")
    assert array.source_kind == SOURCE_HARDWARE
    assert array.is_live is True


def test_a_declined_answer_has_no_direction_and_a_reason():
    from acoustic_array import AcousticArray

    # Silence: nothing to correlate, so the sensor must refuse rather than guess.
    with AcousticArray.synthetic(burst_frames=0, silence_frames=1,
                                 noise_amplitude=0.0, max_frames=20) as array:
        array.detector.emit_silence = True
        events = list(array.stream())

    declined = [e for e in events if not e.has_direction]
    assert declined
    for event in declined:
        assert event.direction_degrees is None
        assert event.reason.strip()
        assert event.localization_confidence is None


def test_the_reason_is_empty_only_when_a_direction_was_produced():
    from acoustic_array import AcousticArray

    with AcousticArray.synthetic(angle_degrees=-30.0, max_frames=24) as array:
        for event in array.stream():
            assert bool(event.reason) != event.has_direction


# --- the angle convention survived the move -----------------------------------

@pytest.mark.parametrize("angle", [-60.0, -30.0, 0.0, 30.0, 60.0])
def test_the_section_5_angle_convention_is_unchanged(angle):
    from acoustic_array import AcousticArray

    with AcousticArray.synthetic(angle_degrees=angle, max_frames=24) as array:
        bearings = [e.direction_degrees for e in array.stream() if e.has_direction]
    assert bearings
    assert bearings[0] == pytest.approx(angle, abs=6.0)


def test_microphone_zero_is_channel_zero():
    from acoustic_array.geometry import default_array

    array = default_array()
    # Channel 0 sits at lower x, so +90 points toward it. Swapping the list
    # order mirrors every bearing in the system.
    assert array.microphones[0].x < array.microphones[1].x


def test_angular_resolution_is_reported_and_honest():
    from acoustic_array import AcousticArray

    with AcousticArray.synthetic(angle_degrees=0.0, max_frames=24) as array:
        events = [e for e in array.stream() if e.has_direction]
    assert events
    # 0.135 m at 16 kHz is ~4.6 deg at broadside. Anything much smaller would
    # be claiming precision the geometry does not have.
    assert 3.0 < events[0].angular_resolution_degrees < 7.0


# --- the section 14 obligations came across intact ----------------------------

def test_the_hardware_source_still_validates_both_crcs_and_counts_drops():
    from acoustic_array.sources import ESP32AudioSource

    source = ESP32AudioSource(port="COM_FAKE")
    diagnostics = source.diagnostics()
    for key in ("packets_dropped_header_crc", "packets_dropped_payload_crc",
                "packets_dropped_total", "frames_abandoned"):
        assert key in diagnostics


def test_link_diagnostics_are_exposed_through_the_public_api():
    from acoustic_array import AcousticArray

    array = AcousticArray.hardware(port="COM_FAKE")
    assert "packets_dropped_total" in array.link_diagnostics()


def test_the_synthetic_api_reports_no_link_diagnostics():
    from acoustic_array import AcousticArray

    assert AcousticArray.synthetic().link_diagnostics() == {}


def test_the_packet_decoder_is_the_one_the_hardware_runs_proved():
    import importlib.util

    import acoustic_array.packets as packets

    spec = importlib.util.spec_from_file_location(
        "tool_vss2", ROOT / "tools" / "verify_serial_stream.py")
    tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    assert tool.StreamVerifier is packets.PacketDecoder


# --- the parked layer is untouched ---------------------------------------------

def test_the_parked_classroom_layer_still_works():
    from heimdall.audio.api import AudioEvent, AudioModule  # noqa: F401
    from heimdall.audio.geometry import ClassroomConfig, load_classroom_config
    from heimdall.audio.seat_mapper import map_audio_to_seat  # noqa: F401

    assert isinstance(load_classroom_config(), ClassroomConfig)


def test_the_legacy_import_paths_still_resolve_to_the_moved_code():
    import acoustic_array.gcc_phat as new
    import heimdall.audio.gcc_phat as old

    assert old.gcc_phat is new.gcc_phat


def test_heimdall_geometry_still_exposes_the_array_types():
    from heimdall.audio.geometry import Microphone, MicrophoneArray, Seat  # noqa: F401
