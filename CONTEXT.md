# HEIMDALL — Audio Module: Session Context

Handoff notes for resuming work. Written 2026-08-19, updated 2026-08-20.

**My scope:** microphones → ESP32 → laptop → VAD → GCC-PHAT → TDOA → DOA → seat
mapping → audio event API → fusion engine. I do not touch vision, pose,
object detection, fusion internals, or the final cheating decision.

**Status: every hardware-independent phase is complete and tested. 300 tests
passing. The only remaining work needs physical hardware that has not arrived.**

---

## 0. Total status

| Phase | What it is | Where | Status |
|---|---|---|---|
| A | Config + ESP32 device detection | `config.py`, `device.py`, `tools/detect_device.py` | **Done** — 13 tests |
| B | One microphone, real audio | firmware | **Blocked** — no hardware |
| C | Two microphones, real audio | firmware | **Blocked** — no hardware |
| D | Laptop-side receiver, threaded + bounded queue | `receiver.py` | **Done** — 10 tests |
| E | Frame analysis, WAV I/O, diagnostics | `analysis.py`, `tools/monitor_audio.py` | **Done** — 24 tests |
| F | GCC-PHAT time-delay estimation | `gcc_phat.py` | **Done** — 34 tests |
| G | TDOA → direction of arrival | `doa.py` | **Done** — 34 tests |
| H | Classroom + array geometry | `geometry.py` | **Done** — 28 tests |
| I | Bearing → seat | `seat_mapper.py` | **Done** — 27 tests |
| J | Event detection / VAD | `events.py` | **Done** — 28 tests |
| K | Public API for fusion | `api.py` | **Done** — 33 tests |
| L | Four microphones | firmware + calibration | **Blocked** — no hardware; software already N-channel ready |
| — | Tooling: sources, synthetic signals, frames, benchmarks | `sources.py`, `synthetic.py`, `frame.py`, `tools/` | **Done** — 69 tests |

Totals: 300 tests, ~21 s, all green. 3,545 lines of module and tool code,
2,842 lines of tests. Zero hardware required to run any of it.

**Shipped and verified**

- End-to-end pipeline: synthetic audio → GCC-PHAT → TDOA → bearing → seat →
  `AudioEvent` JSON, with the honesty fields fusion needs (§4, §6).
- Accuracy on synthetic signals: delays to ±0.25 samples, angles to <0.2° at
  broadside, 0.13° max error over a full 0°–90° sweep.
- Speed: 1.67 ms per 21.33 ms frame, 12.8× real time, measured per stage (§8).
- Four diagnostic tools, all headless and all covered by tests.
- Every "what two microphones may not claim" limit enforced in code, not prose.

**Not done, and why**

Firmware (Phases B, C, L) only. No ESP32 line has been written, deliberately:
the board's exact GPIO map is unknown, and a generic ESP32-S3 pinout would be a
guess. `ESP32AudioSource` refuses to construct and a test pins that refusal, so
nothing can silently depend on hardware that does not exist.

**Single next action:** supply the four items in §10 (board photos both sides,
INMP441 photo, USB port count, and the VID/PID that `tools/detect_device.py`
prints with the board plugged in). Everything downstream of that is already
written and waiting.

---

## 1. How to run things

```bash
cd "C:/Users/Jayyraj Mehta/OneDrive/Desktop/CV_hackathon"

.venv/Scripts/python.exe -m pytest -q              # 300 passed, ~21s
.venv/Scripts/python.exe tools/detect_device.py    # find the ESP32 on a COM port
.venv/Scripts/python.exe tools/monitor_audio.py    # waveform/RMS/spectrogram -> PNG
.venv/Scripts/python.exe tools/calibrate_audio.py  # known-vs-estimated angle table
.venv/Scripts/python.exe tools/benchmark_audio.py  # per-stage latency vs frame budget
```

Python 3.12 venv managed by `uv`. Deps: numpy, scipy, matplotlib, pyyaml,
pyserial, soundfile, sounddevice, pytest. Everything runs with **no ESP32, no
microphone, no COM port, no sound card**.

**Nothing is committed yet.** `git init` was run inside `CV_hackathon/` only;
branch `master`, zero commits, all files untracked.

## 2. Hard constraint: the home-directory git repo

`C:\Users\Jayyraj Mehta\.git` exists and is an accidental repo covering the
entire home folder (AppData, .ssh, NTUSER.DAT, …). **Do not touch, modify, or
delete it.** The project has its own repo now; that is sufficient. Deleting the
stray one is the user's call, not ours.

## 3. Layout

```
config/audio.yaml          sample rate, channels, frame size, USB VID/PIDs
config/classroom.yaml      room, mic positions, array orientation, seat grid
heimdall/audio/
  config.py                loads audio.yaml
  device.py                serial port enumeration + ESP32 identification
  frame.py                 AudioFrame: (num_samples, num_channels) float32
  synthetic.py             deterministic signal generation + delays
  sources.py               AudioSource / SyntheticAudioSource / ESP32AudioSource
  receiver.py              threaded, non-blocking, drops oldest on backpressure
  analysis.py              RMS, per-channel energy, spectral features, WAV I/O
  gcc_phat.py              TDOA estimation
  doa.py                   TDOA -> bearing
  geometry.py              MicrophoneArray, ClassroomConfig, angle convention
  seat_mapper.py           bearing/position -> seat
  events.py                SILENCE / SOUND_DETECTED / POSSIBLE_SPEECH / POSSIBLE_WHISPER
  api.py                   AudioEvent, AudioModule  <-- the only public surface
heimdall/fusion/           empty; someone else's
firmware/esp32_mic/        empty; awaiting hardware
tools/                     detect_device, monitor_audio, calibrate_audio,
                           benchmark_audio
tests/audio/               300 tests
```

## 4. The public contract

The rest of Heimdall imports exactly one thing:

```python
from heimdall.audio.api import AudioEvent, AudioModule

module = AudioModule.synthetic(angle_degrees=-8.0)   # swap source for hardware
with module:
    for event in module.stream():
        print(event.to_dict())
```

Real output, verified:

```json
{"timestamp": 0.0, "event_type": "POSSIBLE_SPEECH", "seat_id": "B4",
 "direction_degrees": -8.02, "position": null, "confidence": 0.94,
 "duration": 0.256, "source": "microphone_array",
 "localization_confidence": 0.80, "seat_ambiguous": true,
 "candidate_seats": ["B4","C4","E5","D4","E4","A4"],
 "angular_resolution_degrees": 0.69,
 "notes": "6 seats share this bearing; range is unknown with a linear array"}
```

Fusion never needs to know GCC-PHAT exists.

## 5. Angle convention (defined in geometry.py, used everywhere)

```
  0 deg  = broadside, straight out in front of the array
+90 deg  = along the array axis, toward channel 0
-90 deg  = along the array axis, toward the last channel
```

`microphones[0]` in `classroom.yaml` **is** channel 0. The list order defines
the sign of every TDOA in the system — do not shuffle it.

`gcc_phat(signal, reference)` returns a **positive** delay when `signal`
arrived **later** than `reference`.

## 6. What two microphones may not claim

These are enforced in code and pinned by tests. Do not "improve" them away.

- `position` is **always `None`** with a linear array. No range information
  exists. Only a non-collinear array earns a position.
- Front/back is physically ambiguous. Resolved by asserting, via
  `orientation_degrees`, that students are all on one side. Seats behind the
  array are dropped, never matched.
- Many seats share a bearing. Every match reports `seat_ambiguous` and the full
  `candidate_seats` list. The top candidate is a bearing match, not a location.
- `angular_resolution_degrees` degrades honestly: 0.68° at broadside, 7.6° at
  85°, 90° (meaningless) at exact end-fire.
- Silence, uncorrelated channels, and low confidence return **no seat** plus a
  reason string. Never a guess.
- `POSSIBLE_SPEECH` is evidence, not a verdict. Whisper confidence is capped at
  0.6 because the heuristic is weak. The audio module never decides cheating.

## 7. Non-obvious engineering decisions

Things a future session would otherwise get wrong or undo.

**48 kHz, not 16 kHz.** At 16 kHz one sample is 2.1 cm of sound travel; with
10 cm spacing the entire physical delay range is ±4.7 samples. 48 kHz triples
the resolution. Configurable in `config/audio.yaml`.

**Textbook PHAT is broken for our signals — do not "simplify" it back.** Pure
PHAT divides every frequency bin by its own magnitude. Bins with no real energy
contain numerical noise, and dividing amplifies it to full amplitude. A known
−7-sample delay came back as 0. Fixed with a **coherent-energy mask** requiring
energy in *both* channels before a bin contributes (`regularization=0.01`).

**Confidence is peak-to-sidelobe, not peak/RMS.** The original peak/RMS metric
ranked noisy-but-correct estimates *below* uncorrelated garbage. Current
behaviour: clean 0.83 → moderate noise 0.53 → heavy noise 0.29 →
uncorrelated 0.12. `DEFAULT_MIN_CONFIDENCE = 0.30` sits above the garbage floor.

**Noise floor has both a warm-up and a ceiling.** Warm-up (8 frames) stops a
room whose ambient level exceeds the default guess from reporting its own hiss
forever. The ceiling (0.01, ≈ −40 dBFS) stops the opposite failure: a stream
that opens with someone already talking would otherwise calibrate the floor to
speech level and hear nothing for the rest of the session. Documented cost:
audio during the calibration window is absorbed, pinned by
`test_audio_during_the_calibration_window_is_absorbed`.

**End-fire is clamped, not rejected.** A source at exactly ±90° produces the
maximum possible TDOA, so sub-sample error pushes `|sin|` just past 1.
`angle_from_tdoa` accepts overshoot within one sample of timing error and
clamps to ±90°. Before this, every 90° measurement was discarded.

**`_pending` clears on every run boundary in `api.process_frame`.** The detector
drops runs shorter than `min_duration` and emits nothing; if `_pending` only
cleared on emission, those frames leaked into the next event and dragged its
bearing off (observed: 8.49° instead of 5.81°). Fixed and verified stable.

**Synthetic signals are broadband on purpose.** `click()` is a Gaussian-enveloped
noise burst, and `speech_like()` includes 12% aspiration noise. A narrowband
"click" has an oscillating autocorrelation and a genuinely ambiguous delay —
that was masking real algorithm behaviour, not testing it.

**WAV I/O uses the stdlib `wave` module**, not soundfile/sounddevice, so tests
never need PortAudio or an audio device.

## 8. Test suite (300)

| File | N | Covers |
|---|---|---|
| test_gcc_phat.py | 34 | known ±/fractional delays, noise, graceful failure |
| test_doa.py | 34 | known angles, resolution limits, what 2 mics may not claim |
| test_api.py | 33 | end-to-end, JSON shape, performance, silence |
| test_geometry.py | 28 | configurable rooms, angle convention, 4-mic readiness |
| test_events.py | 28 | run merging, whisper cap, swappable classifier |
| test_seat_mapper.py | 27 | clear/between/outside/low-confidence/invalid |
| test_analysis.py | 24 | RMS, per-channel, spectrogram, WAV round trip |
| test_tools.py | 30 | all three tools headless, ESP32 refusal |
| test_sources.py | 16 | mock determinism, ESP32AudioSource refuses |
| test_synthetic.py | 14 | delays verified by plain cross-correlation |
| test_receiver.py | 10 | framing, drops, no-audio, device failure |
| test_frame.py | 9 | shape, channels, 4-channel readiness |
| test_config.py | 8 | sample rate configurable, VID/PID matching |
| test_device_detection.py | 5 | runs with no hardware attached |

Measured on synthetic signals: GCC-PHAT recovers delays to ±0.25 samples; DOA
recovers angles to <0.2° at broadside; full sweep 0°–90° gives max error 0.13°.

Latency per 21.3 ms frame: DOA 0.44 ms mean / 0.62 ms p95, detect 0.41 ms,
seat mapping 1.12 ms per event. ~25× real-time headroom.

`tools/benchmark_audio.py` breaks the same frame down further, capture and
GCC-PHAT included: capture 0.01 ms, GCC-PHAT 0.75 ms mean / 1.14 ms p95, DOA
0.83 ms, detect 0.84 ms, seat mapping 0.014 ms per event, 1.67 ms total against
a 21.33 ms budget — 12.8× real time. GCC-PHAT is ~90% of DOA, so it is the only
stage worth optimising if four microphones ever make the budget tight. Absolute
numbers move with machine load; the ratios do not.

## 9. Hardware boundary

```
AudioSource (abstract)
├── SyntheticAudioSource   implemented, deterministic, used by all 300 tests
└── ESP32AudioSource       raises NotImplementedError, on purpose
```

`ESP32AudioSource.__init__` refuses to construct and a test asserts it does.
Nothing downstream imports it. When hardware arrives, implement
`start()/read_frame()/stop()` there and **change nothing else**.

## 10. What is NOT done, and why

**Phase B (one mic), Phase C (two mics), Phase L (four mics) — blocked on
hardware.** No firmware exists. No wiring instructions have been given, by the
user's explicit instruction.

Still needed before any wire is connected:

1. Photos of both sides of the ESP32-S3 board, silkscreen readable. "ESP32-S3"
   covers many boards with different GPIO numbers. Do not use a generic pinout.
2. A photo of an INMP441 module showing its pin labels.
3. Whether the board has one USB-C port or two (native USB vs UART bridge).
4. Plug the board in and run `tools/detect_device.py` for the real VID/PID.

**Everything hardware-independent is now done.** `tools/benchmark_audio.py`
was the last gap and is complete: capture, GCC-PHAT, DOA, detection, seat
mapping and a per-frame total, each with mean/p95/max, compared against the
21.33 ms frame budget, with `--json` output and 11 tests.

Two decisions inside it that must not be "tidied up":

- `frame_total` is `capture + doa + detect`, summed, **not** wall time around
  the loop body. The tool calls GCC-PHAT once on its own to time it and again
  inside `estimate_doa`; charging both to the frame would overstate the cost of
  the real pipeline by roughly 45%. `api.process_frame` never runs that extra
  pass.
- Frames dropped under the synthetic source are reported but explicitly called
  meaningless. The synthetic source is unpaced and outruns the consumer by
  construction, so the queue overflows no matter how fast the pipeline is. On
  hardware the same number becomes the most important line in the report.

`capture` on synthetic audio measures queue handover, not acquisition. The
report says so instead of printing a number that looks like microphone-to-laptop
latency. That figure requires hardware.

## 11. The four-microphone plan (Phase L) — verified constraint

**Four INMP441s cannot share one I²S port.** Standard I²S has exactly two slots
and the `L/R` pin is a one-bit selector, so a third microphone has nowhere to
go. The ESP32-S3 supports TDM, but **the INMP441 is not a TDM device** and
cannot be assigned to slot 2 or 3. "ESP32-S3 supports TDM" does not rescue this.

Two mics is clean: one I²S peripheral, shared SCK/WS, both SD lines on one pin,
one mic's `L/R` → GND and the other's → VDD. One clock, hardware-aligned.

The viable path to four: both I²S controllers (I2S0 + I2S1), two mics each,
I2S0 as clock master with its BCLK/WS routed to I2S1 as slave, so all four share
one clock domain. **Sample-start alignment between two DMA engines is not
guaranteed by hardware.** It is probably a fixed offset that can be measured
with an impulse and calibrated out, but that must be *verified empirically*, not
assumed. Design already allows a per-channel calibration offset. If the offset
proves unstable, fall back to a dedicated 4-channel TDM ADC.

Software is already 4-channel ready: `AudioFrame`, `SyntheticAudioSource`,
`channel_rms`, and `classroom.yaml` all handle N channels, and
`test_four_microphones_load_without_code_changes` proves a square array loads
and is correctly identified as non-linear. `seat_mapper` already dispatches to
position mode the moment a localizer produces a real 2-D fix — the public API
does not change.

## 12. Working agreement with the user

- Keep reporting minimal: DONE / files changed / tests / next step. Long reports
  only when explicitly asked, for architecture decisions, or on failure.
- No unsolicited markdown docs, summaries, or design documents. (This file was
  explicitly requested.)
- Build one layer at a time. Test each layer. Do not dump 500 lines at once.
- Never proceed past a hardware-dependent step without asking for the hardware
  information first.
- Do not hide poor results. The calibration tool prints an honest verdict,
  including "unusable".
- The user is not an electronics expert: for hardware, explain what, why, where
  each wire goes, what to upload, what to expect, and what to do if it fails.
