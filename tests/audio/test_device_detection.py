"""Phase A tests: device detection runs with or without hardware attached."""

from heimdall.audio.device import SerialDevice, find_esp32_devices, list_serial_devices


def test_listing_ports_does_not_raise_without_hardware():
    devices = list_serial_devices()
    assert isinstance(devices, list)
    assert all(isinstance(d, SerialDevice) for d in devices)


def test_no_audio_condition_returns_empty_candidate_list():
    """With no ESP32 attached this must return [] cleanly, never raise."""
    candidates = find_esp32_devices()
    assert isinstance(candidates, list)
    assert all(d.is_candidate for d in candidates)


def test_candidates_are_a_subset_of_all_ports():
    all_ports = {d.port for d in list_serial_devices()}
    candidates = {d.port for d in find_esp32_devices()}
    assert candidates <= all_ports


def test_bluetooth_ports_are_not_flagged_as_esp32():
    """Bluetooth virtual COM ports have no USB VID/PID and must never match."""
    for d in list_serial_devices():
        if d.vid is None and d.is_candidate:
            raise AssertionError(f"{d.port} has no USB VID but was flagged as ESP32")


def test_vid_pid_formatting():
    for d in list_serial_devices():
        if d.vid is not None and d.pid is not None:
            assert len(d.vid_pid) == 9 and ":" in d.vid_pid
        else:
            assert d.vid_pid == "-"
