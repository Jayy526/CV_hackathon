"""Phase 6: live webcam video with the acoustic direction drawn over it.

    python tools/whisper_heatmap_webcam.py              # the REAL array
    python tools/whisper_heatmap_webcam.py --synthetic  # simulated, banner shown

Keys:  q quit    h toggle the overlay    r start/stop recording to mp4

WHAT YOU ARE LOOKING AT
-----------------------
A full-height vertical band at the bearing of the loudest sound. Full height
because two microphones in a line measure AZIMUTH ONLY - there is no range and
no elevation, so there is no y position to draw. Band width is the uncertainty;
it widens as confidence falls. Magenta is POSSIBLE_WHISPER (the thing this
demo exists to show), amber POSSIBLE_SPEECH, blue anything else.

An edge WEDGE means the sound is outside the camera's view - not located in
shot. A band that reaches the frame edge with a hatched stripe is different:
the sound IS in shot, its uncertainty simply extends past the view.

WHAT IT DOES NOT SHOW
---------------------
Sound from BEHIND the array. A linear array cannot tell front from back, and
this app resolves that by only ever showing the camera's side. Anything behind
is silently not drawn - which the on-screen notice says out loud.

The real array is the DEFAULT. There is no silent fallback to simulated audio:
if the board cannot be reached the app says so and exits, because a demo that
looks identical with and without microphones is worthless. `--synthetic` is
available deliberately, and paints a red banner on every frame when used.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from acoustic_camera import load_camera_config, parallax_warning  # noqa: E402
from acoustic_camera.overlay import (  # noqa: E402
    BearingRing,
    OverlayConfig,
    build_hud,
    composite,
    draw_banner,
    echo_from_event,
)

HUD_COLOUR = (235, 235, 235)
WARNING_COLOUR = (60, 60, 255)


class HeatmapError(RuntimeError):
    """A failure the operator can act on. Reported as a sentence, not a trace."""


class LatencyTracker:
    """Spread of audio arrival about the anchored timeline, in ms.

    NOT absolute acoustic-to-display latency: the epoch is anchored on the
    FIRST event, so any constant delay is absorbed into the anchor and cannot
    be seen from inside. The true figure needs an external reference - a clap
    visible in frame, timed against when its band appears. What this does show
    is DRIFT: a rising value means audio is falling behind video.
    """

    def __init__(self, capacity: int = 64) -> None:
        self._samples: deque[float] = deque(maxlen=capacity)

    def record(self, seconds: float) -> None:
        self._samples.append(seconds)

    @property
    def median_ms(self) -> float | None:
        if not self._samples:
            return None
        return statistics.median(self._samples) * 1000.0


def open_camera(camera):
    try:
        import cv2
    except ImportError as exc:
        raise HeatmapError(
            "opencv-python is not installed. Run `uv sync` (it is in "
            "pyproject.toml), then try again."
        ) from exc

    capture = cv2.VideoCapture(camera.index)
    if not capture.isOpened():
        capture.release()
        raise HeatmapError(
            f"no camera at index {camera.index}. Is another application using "
            f"it, or is camera.index wrong in config/camera.yaml?")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
    ok, _ = capture.read()
    if not ok:
        capture.release()
        raise HeatmapError(
            f"camera {camera.index} opened but returned no frames. It may be "
            f"in use, or blocked by a privacy setting.")
    return capture


def build_array(port: str | None, use_synthetic: bool):
    """The REAL array by default. Synthetic only when explicitly asked for.

    There is no silent fallback. This used to drop to simulated audio whenever
    the board was not reachable, which meant a demo could look perfect while
    measuring nothing - the exact trap the banner exists to prevent. Now the
    hardware path either works or says why it did not.
    """
    from acoustic_array import AcousticArray
    from acoustic_array.sources import AudioSourceError

    if use_synthetic:
        return AcousticArray.synthetic(angle_degrees=-18.0)

    try:
        array = AcousticArray.hardware(port=port)
        array.start()
        return array
    except AudioSourceError as exc:
        raise HeatmapError(
            f"cannot reach the microphone array: {exc}\n"
            f"  Check the board with:  python tools/board_diag.py\n"
            f"  To demo without hardware, pass --synthetic (a red banner will "
            f"say so on every frame)."
        ) from exc


class AudioPump:
    """Reads events off the array on a background thread and fills the ring."""

    def __init__(self, array, ring: BearingRing, camera, overlay: OverlayConfig,
                 av_offset_seconds: float, pace: bool = False) -> None:
        self.array = array
        self.ring = ring
        self.camera = camera
        self.overlay = overlay
        self.av_offset_seconds = av_offset_seconds
        # The synthetic source is UNPACED: it generates frames as fast as the
        # CPU allows and outruns real time by orders of magnitude. Left alone it
        # would fill the ring instantly, make the 1.5 s decay meaningless and
        # spin a core. Hardware paces itself on the wire, so this is only for
        # the simulation.
        self.pace = pace
        self.latency = LatencyTracker()
        self.events_seen = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._epoch: float | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="heatmap-audio",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        # stream_live, NOT stream: stream() emits only when a sound ENDS,
        # so a continuous noise draws nothing until it stops. That was most
        # of the perceived lag.
        for event in self.array.stream_live(timeout=0.5):
            if self._stop.is_set():
                return
            arrival = time.monotonic()
            if self._epoch is None:
                # Anchor the audio timeline to the wall clock on the first event.
                self._epoch = arrival - float(event.timestamp)
            audio_wall = self._epoch + float(event.timestamp)
            if self.pace:
                # Hold the simulation to real time so the decay window and the
                # a/v sync mean what they say.
                behind = audio_wall - time.monotonic()
                if behind > 0.0:
                    time.sleep(min(behind, 1.0))
            self.latency.record(max(time.monotonic() - audio_wall, 0.0))
            self.events_seen += 1
            self.ring.add(echo_from_event(
                event, self.camera, self.overlay,
                display_time=audio_wall + self.av_offset_seconds))


def draw_hud(frame, hud, cv2) -> None:
    """Blit the HUD text. The only part of the overlay that needs OpenCV."""
    height = frame.shape[0]
    y = 74 if hud.banner else 30
    if hud.banner:
        cv2.putText(frame, hud.banner, (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (255, 255, 255), 2, cv2.LINE_AA)
    for line in hud.status:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    HUD_COLOUR, 1, cv2.LINE_AA)
        y += 22
    for line in hud.warnings:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    WARNING_COLOUR, 2, cv2.LINE_AA)
        y += 22
    if hud.decline_reason:
        cv2.putText(frame, f"no direction: {hud.decline_reason}", (16, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live acoustic direction heatmap.")
    parser.add_argument("--port", default=None, help="ESP32 COM port; omit for synthetic")
    parser.add_argument("--synthetic", action="store_true",
                        help="simulated audio instead of the board (red banner on "
                             "every frame). Without this, the real array is used "
                             "and a failure is reported rather than hidden.")
    parser.add_argument("--decay", type=float, default=1.5,
                        help="seconds a bearing stays visible (default 1.5)")
    parser.add_argument("--av-offset-ms", type=float, default=0.0,
                        help="shift audio relative to video, in milliseconds")
    parser.add_argument("--record", default=None, help="write mp4 here from the start")
    parser.add_argument("--seconds", type=float, default=None,
                        help="stop automatically after this long (for demos/tests)")
    parser.add_argument("--benchmark", action="store_true",
                        help="measure the per-frame overlay cost and exit")
    return parser


def benchmark(width=1280, height=720, bands=4, iterations=200) -> float:
    """Mean milliseconds to composite one frame. No camera, no audio."""
    from acoustic_camera import CameraConfig
    from acoustic_camera.overlay import Echo
    from acoustic_camera.projection import project_band

    camera = CameraConfig(width=width, height=height)
    config = OverlayConfig()
    ring = BearingRing(config)
    for i in range(bands):
        bearing = -24.0 + i * 16.0
        ring.add(Echo(display_time=0.0, event_type="POSSIBLE_WHISPER",
                      energy=0.8, bearing_degrees=bearing, confidence=0.7,
                      band=project_band(bearing, 4.55, 0.7, camera)))
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    composite(frame, ring, 0.1, config)          # warm up
    began = time.perf_counter()
    for _ in range(iterations):
        composite(frame, ring, 0.1, config)
    return (time.perf_counter() - began) / iterations * 1000.0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.benchmark:
        budget = 1000.0 / 30.0
        print(f"frame budget at 30 fps: {budget:.1f} ms")
        worst = 0.0
        for label, bands, size in (("typical (1 band)", 1, (1280, 720)),
                                   ("busy (4 overlapping bands)", 4, (1280, 720)),
                                   ("typical at 960x540", 1, (960, 540)),
                                   ("busy at 960x540", 4, (960, 540))):
            per_frame = benchmark(width=size[0], height=size[1], bands=bands)
            worst = max(worst, per_frame)
            print(f"  {label:<30} {per_frame:6.2f} ms  "
                  f"({100 * per_frame / budget:4.1f}% of budget)")
        return 0 if worst < budget else 1

    camera = load_camera_config()
    overlay = OverlayConfig(decay_seconds=args.decay)
    ring = BearingRing(overlay)

    try:
        import cv2
    except ImportError:
        print("opencv-python is not installed. Run `uv sync`, then try again.",
              file=sys.stderr)
        return 2

    try:
        capture = open_camera(camera)
    except HeatmapError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    try:
        array = build_array(args.port, args.synthetic)
    except HeatmapError as exc:
        capture.release()
        print(f"\n{exc}", file=sys.stderr)
        return 2
    if not array.receiver.is_running:
        array.start()

    pump = AudioPump(array, ring, camera, overlay, args.av_offset_ms / 1000.0,
                     pace=array.source_kind != "hardware")
    pump.start()

    print(f"source: {array.source_kind.upper()}"
          f"{'  (SIMULATED - banner shown)' if array.source_kind != 'hardware' else ''}")
    parallax_note = parallax_warning(camera)
    frames_with_a_band = 0
    frames_rendered = 0
    writer = None
    show_overlay = True
    dropped = 0
    frame_times: deque[float] = deque(maxlen=60)
    overlay_times: deque[float] = deque(maxlen=60)

    if args.record:
        writer = _open_writer(cv2, args.record, camera)

    deadline = None if args.seconds is None else time.monotonic() + args.seconds
    try:
        while deadline is None or time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok:
                dropped += 1
                if dropped > 30:
                    print("camera stopped delivering frames", file=sys.stderr)
                    return 2
                continue

            now = time.monotonic()
            frames_rendered += 1
            frame_times.append(now)
            ring.prune(now - overlay.decay_seconds)

            if show_overlay:
                if any(e.is_located for e, _ in ring.active(now)):
                    frames_with_a_band += 1
                began = time.perf_counter()
                frame = composite(frame, ring, now, overlay)
                overlay_times.append((time.perf_counter() - began) * 1000.0)
                frame = draw_banner(frame, array.source_kind != "hardware")
                hud = build_hud(
                    source_kind=array.source_kind, ring=ring, at_time=now,
                    camera=camera, link_diagnostics=array.link_diagnostics(),
                    parallax_note=parallax_note,
                    video_fps=_fps(frame_times),
                    overlay_ms=statistics.fmean(overlay_times) if overlay_times else 0.0,
                    av_latency_ms=pump.latency.median_ms,
                    av_offset_ms=args.av_offset_ms,
                    video_frames_dropped=dropped,
                )
                draw_hud(frame, hud, cv2)

            cv2.imshow("Heimdall - acoustic direction", frame)
            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                show_overlay = not show_overlay
            if key == ord("r"):
                if writer is None:
                    path = args.record or time.strftime("heatmap-%Y%m%d-%H%M%S.mp4")
                    writer = _open_writer(cv2, path, camera)
                    print(f"recording to {path}")
                else:
                    writer.release()
                    writer = None
                    print("recording stopped")
    except KeyboardInterrupt:
        pass
    finally:
        # Order matters: stop the producer first, then the source it reads, or
        # the pump thread can still be inside stream_live() when it is torn down.
        pump.stop()
        try:
            array.stop()
        except Exception:  # noqa: BLE001 - never let cleanup hide the real exit
            pass
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        # destroyAllWindows only QUEUES the destroy; without a few waitKey
        # cycles the HighGUI window stays on screen after the process returns.
        # This is why the webcam window would not close.
        for _ in range(5):
            cv2.waitKey(1)
        print(f"source {array.source_kind}   video frames {frames_rendered}   "
              f"audio events {pump.events_seen}   "
              f"frames showing a band {frames_with_a_band}")
        if array.source_kind == "hardware":
            drops = array.link_diagnostics().get("packets_dropped_total", 0)
            print(f"link: {drops} packets dropped")
    return 0


def _open_writer(cv2, path, camera):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, 30.0, (camera.width, camera.height))


def _fps(times: deque) -> float:
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
