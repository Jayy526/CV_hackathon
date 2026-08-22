# acoustic_array

A two-microphone acoustic direction sensor. Two INMP441 microphones on an ESP32
stream audio to a laptop over USB; this package turns that into a **bearing** —
the angle a sound came from — with an honest confidence attached.

It knows microphones, its own geometry, angles and confidence. It knows nothing
about rooms, seats, people or cameras.

```python
from acoustic_array import AcousticArray

with AcousticArray.hardware() as array:
    for event in array.stream():
        print(event.to_dict())
```

The port comes from `transport.port` in `config/audio.yaml` (COM9 on the
machine this was built on) - pass `port=` only to override it.

Swap `hardware()` for `synthetic(angle_degrees=-20.0)` and everything
works with no hardware at all — useful for development, and **every event says
which one produced it** in `source_kind`.

---

## What it CANNOT tell you

Read this before building anything on top. These are not gaps to be filled in
later; they are what two microphones in a line physically cannot measure.

**No range.** Two microphones give exactly one number: the time difference
between them. A source 1 m away and a source 5 m away on the same bearing
produce identical measurements. There is no distance field and there will not
be one.

**No elevation.** Same reason. Everything collapses onto a single angle about
the array axis. A sound above you and a sound level with you, on the same
bearing, are the same measurement.

**Front and back are ambiguous.** A sound 30° in front of the array and the
same sound 30° behind it are physically indistinguishable. This package does
not resolve that and does not pretend to. If you need it resolved, you need a
second sensor — a camera, say — or you must physically guarantee that sources
can only be on one side.

**A bearing is not a location.** Turning an angle into a position requires an
assumption the sensor cannot make for you. Whatever you assume, it is yours,
and it should be visible in your code rather than buried in here.

**Whisper confidence is capped.** The classifier that labels something
`POSSIBLE_WHISPER` is a weak heuristic, and its confidence is deliberately
limited. `POSSIBLE_SPEECH` is evidence, not a verdict. This sensor never
decides what a sound *means*.

When it cannot answer — silence, uncorrelated channels, confidence below
threshold — `direction_degrees` is `None` and `reason` says why. It never
guesses. Check `has_direction` before using the angle.

---

## The angle convention

```
        0°  broadside, straight out in front of the array
      +90°  along the array axis, toward CHANNEL 0  (mic 1)
      -90°  along the array axis, toward the last channel (mic 2)
```

Angles are always in [-90, +90].

**`microphones[0]` IS channel 0.** The order of the microphone list defines the
sign of every bearing in the system. Swapping the two entries mirrors every
angle the sensor will ever report. Do not reorder it, and if you rebuild the
array physically, check the sign again by measurement rather than by reading
the code.

---

## The hardware

An ESP32-WROOM-32 dev board ("ESP32 Dev Module") with a CP2102-class USB-UART
bridge, and two INMP441 I²S microphone breakouts.

### Wiring

Both microphones share all three I²S lines. This is normal: standard I²S
carries two channels on one data line, and the `L/R` pin selects which slot
each microphone speaks in.

| Signal | ESP32 pin | Goes to |
|---|---|---|
| SCK / BCLK | GPIO 26 | SCK on **both** microphones |
| WS / LRCLK | GPIO 25 | WS on **both** microphones |
| SD / DATA | GPIO 33 | SD on **both** microphones |
| 3V3 | 3V3 | VDD on both microphones |
| GND | GND | GND on both microphones |

Then, and this is the part that decides which microphone is which channel:

| Microphone | `L/R` pin | Becomes |
|---|---|---|
| Mic 1 | tied to **GND** | LEFT = **channel 0** |
| Mic 2 | tied to **3V3** | RIGHT = **channel 1** |

Mount the two microphones **0.135 m apart, centre to centre**, in a straight
line, both facing the same way. Measure the spacing rather than trusting the
mounting — it goes directly into every angle the sensor computes. Mic 1 goes on
the side you want `+90°` to point toward.

If your spacing differs, say so explicitly:

```python
from acoustic_array.geometry import linear_array
array = AcousticArray.hardware(array=linear_array(2, spacing=0.18))
```

### What to flash

`firmware/esp32_mic/esp32_mic.ino`, with the Arduino IDE, board "ESP32 Dev
Module", using arduino-esp32 core 3.x.

The sketch has one build switch at the top:

- `#define START_IN_DIAG 1` — **bring-up build.** Text diagnostics only, port at
  115200. Use this first: open the Serial Monitor at 115200 and confirm the
  acquisition rates, that `i2s failures` reads 0, and that tapping mic 1 moves
  only `ch0 rms`. `<< CH0 SILENT` or `<< BOTH CHANNELS SILENT` is a wiring or
  power problem, not a software one.
- `#define START_IN_DIAG 0` — **streaming build.** Binary packets at 921600.
  This is what the package reads. Close the Serial Monitor before running
  anything else; Windows COM ports are exclusive.

Find the port with `python tools/detect_device.py`.

---

## What the output means

```json
{
  "timestamp": 1.024,
  "event_type": "POSSIBLE_SPEECH",
  "direction_degrees": -19.8,
  "confidence": 0.74,
  "localization_confidence": 0.86,
  "angular_resolution_degrees": 4.83,
  "duration": 0.512,
  "channel_rms": [0.041, 0.041],
  "source_kind": "synthetic",
  "reason": ""
}
```

| field | meaning |
|---|---|
| `timestamp` | seconds from the start of the stream, at the event's first sample |
| `event_type` | `SILENCE`, `SOUND_DETECTED`, `POSSIBLE_SPEECH` or `POSSIBLE_WHISPER` |
| `direction_degrees` | the bearing, or `None` when the sensor declines to answer |
| `confidence` | how sure the classifier is of the **event type** |
| `localization_confidence` | how sure the correlation is of the **direction** — these are different things and both can be low |
| `angular_resolution_degrees` | the array's own precision limit at this bearing. An error smaller than this number is not meaningful |
| `duration` | seconds the event lasted |
| `channel_rms` | per-channel level. Wildly unequal values mean a weak or dead microphone |
| `source_kind` | `hardware` or `synthetic` — **display this**; a demo that looks the same either way is a trap |
| `reason` | why there is no direction. Empty when there is one |

`localization_confidence` is peak-to-sidelobe ratio, not signal level. Clean
audio scores around 0.83; heavy noise around 0.29; uncorrelated channels around
0.12. The default threshold is 0.30, just above the garbage floor.

`angular_resolution_degrees` is worth taking seriously. At 0.135 m spacing and
the 16 kHz transmitted rate it is about **4.6° at broadside**, degrading toward
the array axis. It is a property of the geometry and the sample rate, not of
the software, and no amount of processing improves it.

---

## Audio format

The ESP32 acquires at 48 kHz but transmits at **16 kHz**. Those are different
numbers on purpose: an 8N1 link at 921600 baud carries 92,160 B/s, and
continuous 48 kHz stereo int16 needs 192,000 B/s, so it physically will not
fit. The ESP32 decimates 3:1 behind a 48-tap anti-aliasing FIR.

`AudioFrame.sample_rate` is therefore **16000**. The filter is not optional:
dropping every third sample instead costs roughly 6° of mean bearing error
against 0.7° with it, because aliasing corrupts exactly the inter-channel phase
the direction estimate depends on.

## The link is loss-free but occasionally corrupt

Measured over three 300 s runs — 56,252 packets, ~468 Mbit — the USB serial
link lost **zero** bytes and corrupted **two single bytes** (BER ≈ 4×10⁻⁹), on
the unprotected ESP32 → CP2102 UART hop. So the receiver:

- validates the header CRC **and** the payload CRC on every packet;
- **drops** any packet failing either, whole — never repaired, never
  interpolated;
- **counts** header and payload failures separately;
- exposes those counts via `array.link_diagnostics()`.

A dropped packet is 16 ms of missing audio. When that lands mid-frame, the
partial frame is **discarded** rather than spliced across the gap — a false
splice is a phase discontinuity, and phase is the entire signal here. Expect to
lose roughly one 64 ms frame every 7.5 minutes. Watch
`link_diagnostics()["packets_dropped_total"]`: a rising count means the link is
degrading.

---

## If you mount a camera with this

The package itself knows nothing about cameras — `acoustic_camera` handles that
— but one constraint is physics and belongs here, next to the array.

**The camera must sit at the array centre.** A lateral offset introduces
parallax: the angle to a source differs between the two viewpoints by
`atan(offset / range)`. Correcting that needs the range, and **this array has
no range**. So the error is *uncorrectable in principle*. It cannot be
calibrated out, and a fixed `azimuth_offset_degrees` will not absorb it either,
because parallax changes with distance and a fixed angle does not.

Parallax error, in degrees:

| offset | at 1 m | at 2 m | at 3 m | at 4 m |
|---|---|---|---|---|
| 2 cm | 1.15 | 0.57 | 0.38 | 0.29 |
| 5 cm | 2.86 | 1.43 | 0.95 | 0.72 |
| 10 cm | 5.71 | 2.86 | 1.91 | 1.43 |
| 20 cm | 11.31 | 5.71 | 3.81 | 2.86 |

Against the array's own 4.55° resolution, parallax drops below the resolution
limit only beyond 0.25 m (2 cm offset), 0.63 m (5 cm), 1.26 m (10 cm) and
2.51 m (20 cm). To keep it under the resolution limit for everything past 1 m,
mount within **8 cm** of the array centre. At 20 cm it dominates the
measurement for anyone standing close, and the overlay will be visibly wrong.

Record the real offset in `config/camera.yaml` as `lateral_offset_m`; the tools
warn when it is large enough to matter rather than assuming it is zero.

---

## Layout

```
config.py     transport contract; works with no config file present
device.py     find the board on a serial port by USB VID/PID
packets.py    the byte-level packet contract, CRC, resync decoder
sources.py    SyntheticAudioSource and ESP32AudioSource
receiver.py   background thread, bounded queue, drops oldest under backpressure
frame.py      AudioFrame: (num_samples, num_channels) float32
analysis.py   RMS, per-channel level, spectral features, WAV I/O
gcc_phat.py   time-delay estimation
doa.py        time delay -> bearing
geometry.py   the microphone array. No rooms, no seats
events.py     SILENCE / SOUND_DETECTED / POSSIBLE_SPEECH / POSSIBLE_WHISPER
api.py        AcousticArray and AcousticEvent - the only surface you need
```

No configuration file is required. `AcousticArray.synthetic()` and
`AcousticArray.hardware()` both work against built-in defaults matching the
as-built hardware.
