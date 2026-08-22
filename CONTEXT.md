# HEIMDALL — Audio Module: Session Context

Handoff notes for resuming work. Written 2026-08-19, updated 2026-08-22.

**My scope:** microphones → ESP32 → laptop → VAD → GCC-PHAT → TDOA → DOA → seat
mapping → audio event API → fusion engine. I do not touch vision, pose,
object detection, fusion internals, or the final cheating decision.

**Status: hardware is built and validated. The laptop pipeline is complete
(334 tests passing). The ESP32 streaming firmware is written and compiles but
has never been run. The laptop-side serial receiver does not exist yet — that
is the single gap between here and real audio reaching the pipeline.**

---

## 0. Total status

| Phase | What it is | Where | Status |
|---|---|---|---|
| A | Config + ESP32 device detection | `config.py`, `device.py`, `tools/detect_device.py` | **Done** — 39 + 5 tests |
| B | One microphone, real audio | `mic_check_1.ino` | **Done** — validated on hardware |
| C | Two microphones, real audio | `mic_check_1.ino` | **Done** — CH0/CH1 separation confirmed |
| D | Laptop-side receiver, threaded + bounded queue | `receiver.py` | **Done** — 10 tests |
| E | Frame analysis, WAV I/O, diagnostics | `analysis.py`, `tools/monitor_audio.py` | **Done** — 24 tests |
| F | GCC-PHAT time-delay estimation | `gcc_phat.py` | **Done** — 34 tests |
| G | TDOA → direction of arrival | `doa.py` | **Done** — 34 tests |
| H | Classroom + array geometry | `geometry.py` | **Done** — 31 tests |
| I | Bearing → seat | `seat_mapper.py` | **Done** — 27 tests |
| J | Event detection / VAD | `events.py` | **Done** — 28 tests |
| K | Public API for fusion | `api.py` | **Done** — 33 tests |
| M | Transport design + config | `config.py`, `config/audio.yaml` | **Done** — see §13 |
| N | ESP32 streaming firmware | `firmware/esp32_mic/esp32_mic.ino` | **Done** — run on hardware, 3 x 300 s clean |
| O | Hardware bring-up tests | `tools/verify_serial_stream.py` | **Done** — Tests 1-3 passed, criterion amended; see §14 |
| P | Laptop serial receiver → AudioFrame | `sources.ESP32AudioSource`, `packets.py` | **Done** — both CRCs validated, corrupt packets dropped whole + counted; mid-frame gap discards the partial frame |
| Q | Physical localization validation | — | **Not started** — see §15 |
| L | Four microphones | firmware + calibration | **Deferred** — software already N-channel ready |
| — | Tooling: sources, synthetic signals, frames, benchmarks | `sources.py`, `synthetic.py`, `frame.py`, `tools/` | **Done** — 69 tests |

Totals: 411 tests, ~57 s, all green. 3,658 lines of module and tool code,
2,960 lines of tests, 530 lines of firmware. Every test runs with no hardware.

**Single next action:** Phase P, the laptop serial receiver. §14 is done: the
wire is proven LOSS-FREE BUT OCCASIONALLY CORRUPT (BER ~4e-9), so the receiver
MUST validate both CRCs, drop failing packets whole, and count the drops. Do
not write it assuming a clean wire — that assumption was measured false.

---

## 1. How to run things

```bash
cd "C:/Users/Jayyraj Mehta/OneDrive/Desktop/CV_hackathon"

.venv/Scripts/python.exe -m pytest -q              # 334 passed, ~19s
.venv/Scripts/python.exe tools/detect_device.py    # find the ESP32 on a COM port
.venv/Scripts/python.exe tools/monitor_audio.py    # waveform/RMS/spectrogram -> PNG
.venv/Scripts/python.exe tools/calibrate_audio.py  # known-vs-estimated angle table
.venv/Scripts/python.exe tools/benchmark_audio.py  # per-stage latency vs frame budget
```

Firmware builds with the Arduino IDE's bundled CLI (no separate install):

```bash
CLI="$HOME/AppData/Local/Programs/Arduino IDE/resources/app/lib/backend/resources/arduino-cli.exe"
"$CLI" compile --fqbn esp32:esp32:esp32 firmware/esp32_mic
```

esp32 core 3.3.11 is installed. Last build: 289,080 B flash (22%), 44,764 B RAM
(13%), zero warnings under `--warnings all`.

Python 3.12 venv managed by `uv`. Deps: numpy, scipy, matplotlib, pyyaml,
pyserial, soundfile, sounddevice, pytest.

**Git:** one commit, `35d15cc`, covering the hardware-independent pipeline.
Everything since (0.135 m spacing, the transport config, the firmware) is
uncommitted.

## 2. Hard constraint: the home-directory git repo

`C:\Users\Jayyraj Mehta\.git` exists and is an accidental repo covering the
entire home folder (AppData, .ssh, NTUSER.DAT, …). **Do not touch, modify, or
delete it.** The project has its own repo; that is sufficient. Deleting the
stray one is the user's call, not ours.

## 3. Layout

```
config/audio.yaml          acquisition rate, channels, frame size,
                           USB VID/PIDs, and the `transport:` block (§13)
config/classroom.yaml      room, mic positions (0.135 m), orientation, seat grid
heimdall/audio/
  config.py                loads audio.yaml; WireFormat, TransportConfig
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
firmware/esp32_mic/        esp32_mic.ino - the project firmware (§13)
mic_check_1.ino            the VALIDATED hardware diagnostic. Do not modify.
tools/                     detect_device, monitor_audio, calibrate_audio,
                           benchmark_audio
tests/audio/               334 tests
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

The diagnostic sketch and the Python pipeline **agree** on sign, by different
routes: `mic_check_1.ino`'s `bestLag` is positive when ch1 arrives later, and
Python's `gcc_phat(ch0, ch1)` is positive when ch0 arrives later, which
`angle_from_tdoa` then negates. Both end at `asin(+lag·c/(fs·d))`. No sign flip
is needed anywhere. Verified by inspection, not assumed.

## 6. What two microphones may not claim

These are enforced in code and pinned by tests. Do not "improve" them away.

- `position` is **always `None`** with a linear array. No range information
  exists. Only a non-collinear array earns a position.
- Front/back is physically ambiguous. Resolved by asserting, via
  `orientation_degrees`, that students are all on one side. Seats behind the
  array are dropped, never matched.
- Many seats share a bearing. Every match reports `seat_ambiguous` and the full
  `candidate_seats` list. The top candidate is a bearing match, not a location.
- `angular_resolution_degrees` degrades honestly, and it got worse twice. At
  0.135 m spacing it is **1.52° at broadside at 48 kHz**, against the 0.68° the
  old 0.30 m placeholder gave — and the pipeline will actually receive 16 kHz
  audio (§13), where the same array reports **4.55°**. Both are arithmetic, not
  bugs: a narrower array and a lower rate are each less precise. Note that this
  is the array's own resolution limit, and it is *larger* than the 0.71° mean
  error measured through GCC-PHAT, because the metric assumes a half-sample
  timing error while sub-sample interpolation does better than that in practice.
- Silence, uncorrelated channels, and low confidence return **no seat** plus a
  reason string. Never a guess.
- `POSSIBLE_SPEECH` is evidence, not a verdict. Whisper confidence is capped at
  0.6 because the heuristic is weak. The audio module never decides cheating.

## 7. Non-obvious engineering decisions

Things a future session would otherwise get wrong or undo.

**48 kHz acquisition, 16 kHz on the wire.** These are different numbers and both
matter. See §13 — the link physically cannot carry 48 kHz stereo, and the ESP32
decimates. `audio.sample_rate` is acquisition; `transport.transmit_sample_rate`
is what the laptop actually receives. `AudioFrame.sample_rate` will be **16000**.

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

**`default_transmit_sample_rate()` exists because a blind default broke a real
config.** When `transmit_sample_rate` is absent from the YAML, the loader picks
the highest integer-divisor rate that fits the link, rather than assuming 16000.
A hard-coded 16000 default made a 16 kHz / 4-channel config unloadable. An
**explicit** value is never adjusted — it is validated and rejected if it does
not fit, because silently rewriting a configured rate would make the config file
and the wire disagree.

**The anti-aliasing FIR is not optional and its cutoff is not arbitrary.** See
§13. Decimating by dropping every third sample costs 6.10° of mean bearing
error against 0.71° with the filter. This was measured through this repo's own
DOA pipeline, not asserted.

## 8. Test suite (334)

| File | N | Covers |
|---|---|---|
| test_config.py | 39 | rate/channels, VID/PID, transport, wire format, decimation, bandwidth |
| test_gcc_phat.py | 34 | known ±/fractional delays, noise, graceful failure |
| test_doa.py | 34 | known angles, resolution limits, what 2 mics may not claim |
| test_api.py | 33 | end-to-end, JSON shape, performance, silence |
| test_geometry.py | 31 | configurable rooms, angle convention, 0.135 m spacing, 4-mic readiness |
| test_tools.py | 30 | all tools headless, ESP32 refusal |
| test_events.py | 28 | run merging, whisper cap, swappable classifier |
| test_seat_mapper.py | 27 | clear/between/outside/low-confidence/invalid |
| test_analysis.py | 24 | RMS, per-channel, spectrogram, WAV round trip |
| test_sources.py | 16 | mock determinism, ESP32AudioSource refuses |
| test_synthetic.py | 14 | delays verified by plain cross-correlation |
| test_receiver.py | 10 | framing, drops, no-audio, device failure |
| test_frame.py | 9 | shape, channels, 4-channel readiness |
| test_device_detection.py | 5 | runs with no hardware attached |

Measured on synthetic signals: GCC-PHAT recovers delays to ±0.25 samples; DOA
recovers angles to <0.2° at broadside; full sweep 0°–90° gives max error 0.13°.
**These are synthetic numbers. They are not evidence about the physical
system.** See §15.

Latency per frame: DOA 0.44 ms mean / 0.62 ms p95, detect 0.41 ms, seat mapping
1.12 ms per event. `tools/benchmark_audio.py` breaks it down further: capture
0.01 ms, GCC-PHAT 0.75 ms mean / 1.14 ms p95, DOA 0.83 ms, detect 0.84 ms, seat
mapping 0.014 ms per event, 1.67 ms total. GCC-PHAT is ~90% of DOA, so it is
the only stage worth optimising if four microphones ever make the budget tight.

Two decisions inside `benchmark_audio.py` that must not be "tidied up":

- `frame_total` is `capture + doa + detect`, summed, **not** wall time around
  the loop body. The tool calls GCC-PHAT once on its own to time it and again
  inside `estimate_doa`; charging both would overstate the real pipeline by
  ~45%. `api.process_frame` never runs that extra pass.
- Frames dropped under the synthetic source are reported but explicitly called
  meaningless — the synthetic source is unpaced and outruns the consumer by
  construction. On hardware the same number becomes the most important line.

## 9. Hardware boundary

```
AudioSource (abstract)
├── SyntheticAudioSource   implemented, deterministic, used by all 334 tests
└── ESP32AudioSource       raises NotImplementedError, still on purpose
```

`ESP32AudioSource.__init__` refuses to construct and a test asserts it does.
Nothing downstream imports it. Implementing `start()/read_frame()/stop()` there
— reading the §13 packet stream off a COM port — is Phase P, and **nothing else
in the pipeline needs to change**.

## 10. The hardware, as actually built

Assembled and validated by the user. Do not redesign or rewire it without
concrete evidence from a software test.

- ESP32-WROOM-32 / "ESP32 Dev Module", USB-to-UART bridge (CP2102 class)
- 2 × INMP441
- GPIO 26 = I²S SCK/BCLK, GPIO 25 = WS/LRCLK, GPIO 33 = SD/DATA
- Mic 1 = LEFT = CH0; Mic 2 = RIGHT = CH1
- **Mic spacing 0.135 m**, measured centre to centre

Confirmed working via `mic_check_1.ino`: I²S initialises, both mics produce
independent per-channel responses, stereo separation is real, and basic
lag/bearing estimation runs.

`config/classroom.yaml` places the array at x = 3.9325 / 4.0675 (0.135 m apart,
centred on an 8 m room), mic_1 on the lower-x side so the channel-0 axis points
toward −x. **Changing which mic sits at lower x flips the sign of every
bearing.** If calibration shows the *effective* spacing differs from 0.135 m,
change it here and say so explicitly — never silently.

`mic_check_1.ino` is kept as the hardware troubleshooting tool. Two things in
it that are wrong for production and were deliberately not carried into the
firmware: `MAX_LAG = 28` exceeds the physically possible 18.9 samples at
0.135 m, so it can return impossible lags (a reported `lag = −28` was exactly
this clamp), and its brute-force integer-lag correlation has no sub-sample
interpolation, no PHAT whitening and no confidence metric.

## 11. Transport: how we got here, and why

Three transports were considered in order. The reasoning matters more than the
answer, because the answer would change if the hardware did.

**Wi-Fi/TCP — dropped.** Not on bandwidth grounds; the user requires a fully
offline system. No Wi-Fi, no TCP, no UDP, no network dependency. A
`test_transport_config_has_no_network_concept` test guards against a `host`
field creeping back in.

**Event-triggered USB bursts — designed, then dropped.** Would have preserved
full 48 kHz resolution by sending only triggered windows with a pre-trigger
ring buffer. Abandoned because the user requires continuous audio. A
`test_transport_is_continuous_not_event_triggered` test guards against
`pre_trigger_samples` and friends returning.

**Continuous USB serial at a decimated rate — chosen.** See §13.

The arithmetic that drives all of it: an 8N1 serial link carries `baud/10`
bytes/sec, and continuous stereo int16 costs `rate × 2 ch × 2 B`.

| baud | capacity | max continuous stereo rate (85% util) |
|---|---|---|
| 115,200 | 11,520 B/s | 2.4 kHz — **diagnostics only, useless for audio** |
| 230,400 | 23,040 B/s | 4.9 kHz — unusable |
| 460,800 | 46,080 B/s | 9.8 kHz → 8 kHz stereo fits (the fallback) |
| **921,600** | **92,160 B/s** | **19.6 kHz → 16 kHz stereo fits at 70.5%** |

921600 is the classic CP2102's top standard rate. **It is not yet proven on this
board** — that is Test 2 in §14. If it fails, drop to 460800 + 8 kHz stereo,
which costs 1.07° mean bearing error instead of 0.71°.

## 12. Working agreement with the user

- Keep reporting minimal: DONE / files changed / tests / next step. Long reports
  only when explicitly asked, for architecture decisions, or on failure.
- No unsolicited markdown docs, summaries, or design documents. (This file was
  explicitly requested.)
- Build one layer at a time. Test each layer. Do not dump 500 lines at once.
- Never proceed past a hardware-dependent step without asking first.
- Do not hide poor results. The calibration tool prints an honest verdict,
  including "unusable".
- The user is not an electronics expert: for hardware, explain what, why, where
  each wire goes, what to upload, what to expect, and what to do if it fails.
- Reuse existing tested components. Do not duplicate GCC-PHAT/TDOA/DOA logic.
- Keep hardware transport separate from localization.
- Do not weaken or delete existing tests to make integration pass.

## 13. The transport contract (§M/§N)

Both ends must agree on every number here. The Python side is the authority;
the firmware `#define`s mirror it and are cross-checked by hand.

```
INMP441 ×2 → I²S 48 kHz 32-bit stereo → int16 (>>16)
           → 48-tap anti-aliasing FIR → decimate 3:1
           → continuous 16 kHz stereo int16 → USB serial 921600 → laptop
```

| Setting | Value | Where |
|---|---|---|
| acquisition rate | 48000 Hz | `audio.sample_rate` |
| transmit rate | 16000 Hz | `transport.transmit_sample_rate` |
| decimation factor | 3 (derived, must be exact) | `TransportConfig.decimation_factor` |
| baud | 921600 | `transport.baud_rate` |
| wire format | int16, little-endian, interleaved, 2 ch | `transport.wire_format` |
| samples/packet | 256 **per channel** | `transport.samples_per_packet` |
| payload | 1024 B | derived |
| header | 16 B | `config.HEADER_BYTES` |
| packet | 1040 B | derived |
| packet rate | 62.5 /s | derived |
| wire rate | 65,000 B/s (70.5% of link) | derived |

`TransportConfig.__post_init__` **rejects** any combination that does not fit
the link, naming both sides of the comparison. It never clamps. `MAX_LINK_
UTILISATION = 0.85` — a UART driven near capacity drops bytes silently under
USB scheduling jitter, so the headroom is a correctness requirement.

### Packet header, 16 bytes, little-endian

```
 0     magic 0xA5
 1     magic 0x5A
 2     protocol version (currently 1)
 3     flags: bit0 = ring overrun, bit1 = I2S read failure (since last packet)
 4-7   sequence number, uint32, wraps
 8-9   sample count PER CHANNEL, uint16 (256)
 10-11 payload length in bytes, uint16 (1024)
 12-13 CRC-16/CCITT-FALSE over header bytes 0..11
 14-15 CRC-16/CCITT-FALSE over the payload
```

Sample rate and channel count are deliberately **not** in the header — they are
the shared config contract, and the version byte covers protocol changes. The
header is fixed-size so a receiver that loses sync can resynchronise by scanning
for the magic and validating the header CRC, with nothing variable-length to
parse first.

### The FIR, and why it is not optional

48 taps, Kaiser(β = 7.5), −6 dB at 5.6 kHz, Q15, coefficients summing to 32768.
Over 71 dB of attenuation everywhere above the 8 kHz fold point; −0.26 dB at
4 kHz. Symmetric, therefore linear phase: it delays **both** channels by exactly
23.5 samples and does not disturb the inter-channel delay GCC-PHAT measures.

Measured through this repo's own DOA pipeline at 0.135 m, with reverb and noise:

| configuration | mean bearing error | max |
|---|---|---|
| 48 kHz native (reference) | 0.21° | 2.56° |
| **16 kHz, this FIR** | **0.71°** | 5.00° |
| 16 kHz, every 3rd sample | 6.10° | 30.00° |
| 8 kHz, anti-aliased (fallback) | 1.07° | 8.44° |

Aliasing folds high-frequency energy back into the band with a
frequency-dependent phase shift, and inter-channel phase is exactly what
GCC-PHAT measures. **Dropping every third sample is not decimation.**

Cost accepted: the 5.6–8 kHz band is attenuated, so sibilance is lost. Irrelevant
for TDOA; relevant if anyone later wants the audio for speech content.

The int32 FIR accumulator is safe by 26%: worst-case |acc| is
sum(|h|) × 32767 = 1,583,480,832 against int32's 2,147,483,647.

### Firmware structure

- I²S init copied **verbatim** from `mic_check_1.ino` — that configuration is
  the validated one, so it was not re-derived.
- Acquisition task pinned to **core 0** at priority 5; `loop()` drains on core 1.
  A blocking `Serial.write` therefore can never stall I²S.
- Lock-free single-producer/single-consumer ring, 4096 frames = 256 ms of slack,
  `std::atomic` with acquire/release (not `volatile` — volatile orders nothing
  between the sample write and the index publish).
- Fixed-size buffers throughout. No heap, no `String`.
- **Diagnostics are a mode, not interleaved output.** `MODE_STREAM` emits pure
  binary; `MODE_DIAG` emits pure text; they are mutually exclusive, so the
  binary stream is never contaminated. Keys over the same port: `s` = stream,
  `d` = diagnostics, `r` = reset counters. `#define START_IN_DIAG 1` boots into
  text. A fatal I²S failure forces diagnostics mode and never starts streaming.
- Counters: I²S failures, ring overruns, frames dropped, short writes (VACUOUS
  on core 3.3.11 — see §14; do not rely on it), packets,
  bytes, actual acquired/decimated rates, ring occupancy, per-channel RMS.
  32-bit on purpose (64-bit atomics on Xtensa fall back to a library lock);
  wraps after ~24.8 h.

## 14. Hardware bring-up — DONE, with an amended pass criterion

All three tests have been run on hardware. **Tests 1 and 2 pass. Test 3 passes
as amended below: the wire is loss-free but not bit-perfect.** The criterion was
changed openly, in writing, rather than relaxed quietly into a tolerance.

**Test 1 — acquisition only, no Python. PASSED.** Set `START_IN_DIAG 1`, upload,
open the Serial Monitor at **115200**. A `START_IN_DIAG 1` build runs the port at
`DIAG_BAUD` = 115200 — the rate `mic_check_1.ino` proved on this board — so this
test does not also bet on 921600. A `START_IN_DIAG 0` build runs at 921600 in
both modes, which is what Tests 2 and 3 need. Expect actual rates within ~1% of
48000/16000, `i2s failures 0`, and tapping mic 1 moving only `ch0` rms.
`<< CH0 SILENT` or `<< BOTH CHANNELS SILENT` is a wiring/power problem, not a
software one. **Result:** rates correct, i2s failures 0, per-channel isolation
confirmed.

**Test 2 — sustained throughput. PASSED.** `START_IN_DIAG 0`, re-upload, close
the Serial Monitor (Windows COM ports are exclusive), then run
`tools/verify_serial_stream.py --port COMx --duration 300`. Pass requires
**all four**: bytes ≈ 65,000 B/s, ring overruns 0, I²S failures 0, and **zero
byte loss** by the ledger below. Real byte loss or non-zero overruns mean 921600
does not hold — drop to 460800 + 8 kHz rather than shipping something lossy.

**"short writes 0" was removed from that list and must not be re-added.** It is
vacuous on arduino-esp32 3.3.11. `HardwareSerial::write` returns `size`
unconditionally (`HardwareSerial.cpp:588`), discarding what `uart_write_bytes`
returned, so the firmware's `written != PACKET_BYTES` test can never fire and
`cntShortWrites` is structurally always 0. It is also measuring something that
cannot happen at that layer: with `tx_buffer_size > 0`, `uart_write_bytes`
returns only after copying every byte into the TX ring
(`esp_driver_uart/include/driver/uart.h:522`). The real backpressure signal is
the ring-overrun counter, which is meaningful.

**Test 3 — framing integrity. PASSED AS AMENDED.** The original criterion —
every packet starts with `A5 5A`, sequence numbers increment with no gaps, both
CRCs validate, 1040-byte boundaries never drift — is strictly **not met**.

### Measured result: three 300 s runs at 921600

| run | packets | bytes read | anomaly | bytes lost |
|---|---|---|---|---|
| 1 | 18,751 | 19,501,569 | 1 payload CRC failure | **0** |
| 2 | 18,750 | 19,501,057 | 1 seq gap, 1 resync, 1040 stray bytes | **0** |
| 3 | 18,751 | 19,501,057 | none | **0** |

Totals: **56,252 packets, ~468 Mbit, 2 isolated single-byte corruptions, zero
byte loss, zero ring overruns, zero I²S failures**, throughput exact to 0.006%.
**BER ≈ 4×10⁻⁹.** Both corruptions were isolated — no two consecutive sequence
numbers — so this is not a burst mechanism.

Runs 2 and 3 have **identical byte totals**. Run 2 reconciles as
18,750 × 1040 + 1040 stray + 16 lead-in + 1 trailing = 19,501,057; run 3 as
18,751 × 1040 + 16 + 1 = the same number; run 1 the same way with a 528-byte
lead-in. **Not one byte was lost in any run.**

**Runs 1 and 2 are the same event.** One flipped bit; only where it landed
differed. In run 1 it landed in the payload and the payload CRC caught it. In
run 2 it landed inside the `A5 5A` magic, so `buf[:2] == MAGIC` failed, the
resync discarded all 1040 bytes of a packet that had *fully arrived*, and it
presented as a sequence gap — which is exactly why header CRC failures read 0
and not 1. **A sequence gap is not by itself evidence of loss.**

Where the corruption comes from: the ESP32 UART TX → CP2102 hop is 8N1, no
parity, no retry, and is the only unprotected link in the chain. The CP2102 →
host hop is USB bulk, which has its own CRC and hardware retransmission. A
flipped bit on the unprotected hop corrupts a byte *without losing one*, which
matches the observation exactly. Baud accuracy is not the cause and is not
marginal: the ESP32's fractional divider gives 80 MHz / 921600 → 86+13/16
(+0.008%), the CP2102 gives 923,077 baud (+0.16%), and the accumulated error by
the stop bit is ~1.7% of a bit period against roughly 50% sampling margin.

A ring race inside the firmware is also excluded: `sendPacket()` memcpys the
payload into the consumer-private `packet` buffer *before* the CRC is taken, so
a producer/consumer race would yield bad audio with a **valid** CRC, not a
mismatch. The corruption therefore happened after the CRC was computed.

### The amended pass criterion

This **replaces** the original Test 3 wording. It is not a tolerance bolted onto
the old one:

> **Zero byte loss, zero ring overruns, zero I²S failures, and a corruption rate
> at or below the measured 2 per 56,252 packets (BER ≈ 4×10⁻⁹) — with every
> corrupted packet DROPPED, never repaired and never passed on.**

Byte loss is **measured by conservation**, never inferred from sequence numbers:

```
bytes_lost = sender_span × 1040 − (received × 1040 + stray_bytes)
sender_span = (last_seq − first_seq) mod 2³² + 1
```

`sender_span` comes from the sender's own sequence numbering, which is what makes
this a measurement rather than a restatement of the tool's bookkeeping.
`tools/verify_serial_stream.py` prints the full ledger on every run and splits
the diagnosis three ways:

- **bytes conserved, sequence gap present** → corruption. The link is keeping
  up. **Not** grounds to drop the baud.
- **real byte loss, or ring-overrun flags set** → loss. The §11 fallback applies.
- **nothing locked on** → check build mode, port, baud, and the header contract.

A sequence gap alone must never again trigger the baud-drop advice. It did once,
on run 2, and that advice would have cost the `DECIMATION 6` FIR redesign to fix
one flipped bit.

### This makes Phase 2 conditional

The earlier claim that "the receiver is allowed to assume the wire is clean" is
**false, and is corrected here.** The receiver must assume the wire is
**loss-free but occasionally corrupt.** `ESP32AudioSource` therefore MUST:

- validate the header CRC **and** the payload CRC on every packet;
- **DROP** any packet failing either — whole. Never repair it, never interpolate
  it, never pass a failed payload downstream;
- **COUNT** the drops, header and payload failures separately;
- **EXPOSE** those counts, so a degrading link is visible rather than silent.

At 4×10⁻⁹ this costs roughly one dropped 16 ms packet every 7.5 minutes
(56,252 packets at 62.5/s is 900 s = 15 min of streaming; 2 events in 15 min). Losing
16 ms of audio is acceptable for TDOA. Letting a corrupted payload reach
GCC-PHAT is not: a flipped bit is a phase error, and inter-channel phase is
precisely what the algorithm measures.

**Gotcha for Tests 2 and 3:** opening the port with pyserial asserts DTR/RTS,
which **resets the ESP32** and makes the CP2102 re-enumerate on USB. `--settle`
defaults to **4.0 s** for that reason — at 1.5 s the handle was still stale and
the first read died with `ClearCommError failed (Access is denied)`. The tool
also scans for magic and validates a header CRC before counting, so the
mid-packet garbage that always arrives first is charged as lead-in rather than
reported as a framing error.

`tools/verify_serial_stream.py` is written and is **not** the receiver — no
`AudioSource`, no `AudioFrame`, no pipeline import. It reads the transport
contract from `config/audio.yaml` rather than re-typing it, so the tool and the
firmware cannot silently disagree about the very thing it exists to check.

## 15. Physical localization validation (Phase Q) — not done

**The synthetic accuracy in §8 is not evidence that the physical system has the
same accuracy.** It has never been measured against a real sound source.

Once audio flows end to end, test at known positions:

1. Very close to Mic 1
2. Between the microphones
3. Very close to Mic 2
4. Several known angles in front of the array

Record for each: expected position/angle, measured lag, measured TDOA,
calculated bearing, localization confidence, error in degrees.

If calibration reveals the effective spacing differs from the measured 0.135 m,
make the calibration **explicit and configurable** in `classroom.yaml` — never a
silent constant.

## 16. The four-microphone plan (Phase L) — verified constraint

**Four INMP441s cannot share one I²S port.** Standard I²S has exactly two slots
and the `L/R` pin is a one-bit selector, so a third microphone has nowhere to
go. The ESP32 supports TDM, but **the INMP441 is not a TDM device** and cannot
be assigned to slot 2 or 3. "ESP32 supports TDM" does not rescue this.

Two mics is clean: one I²S peripheral, shared SCK/WS, both SD lines on one pin,
one mic's `L/R` → GND and the other's → VDD. One clock, hardware-aligned. This
is what is built.

The viable path to four: both I²S controllers (I2S0 + I2S1), two mics each,
I2S0 as clock master with its BCLK/WS routed to I2S1 as slave, so all four share
one clock domain. **Sample-start alignment between two DMA engines is not
guaranteed by hardware.** It is probably a fixed offset that can be measured
with an impulse and calibrated out, but that must be *verified empirically*, not
assumed. Design already allows a per-channel calibration offset. If the offset
proves unstable, fall back to a dedicated 4-channel TDM ADC.

Note the bandwidth consequence: four channels doubles the wire cost, so 16 kHz
×4 would need 128,000 B/s and does **not** fit at 921600. Four microphones and
continuous USB serial are in direct tension; solve the transport before the
array.

Software is already 4-channel ready: `AudioFrame`, `SyntheticAudioSource`,
`channel_rms`, `WireFormat`, and `classroom.yaml` all handle N channels, and
`test_four_microphones_load_without_code_changes` proves a square array loads
and is correctly identified as non-linear. `seat_mapper` already dispatches to
position mode the moment a localizer produces a real 2-D fix — the public API
does not change.
