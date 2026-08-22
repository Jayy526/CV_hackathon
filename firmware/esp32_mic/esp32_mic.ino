// Heimdall — ESP32 continuous stereo audio transport
//
//   2x INMP441 -> I2S 48 kHz 32-bit stereo -> int16 -> anti-aliased 3:1 FIR
//              -> continuous 16 kHz stereo int16 -> USB serial -> laptop
//
// This is the PROJECT firmware. mic_check_1.ino is the hardware diagnostic and
// is deliberately left untouched; its I2S initialisation is reused verbatim
// below because it is the configuration that was validated on real hardware.
//
// Board:  ESP32-WROOM-32 DevKit (classic ESP32), "ESP32 Dev Module"
// Core:   arduino-esp32 3.x   (uses the i2s_std driver)
//
// WHY 16 kHz ON THE WIRE, WHEN WE ACQUIRE AT 48 kHz
// -------------------------------------------------
// An 8N1 serial link carries baud/10 bytes/sec, so 921600 baud gives 92,160
// B/s. Continuous stereo int16 costs rate * 2 ch * 2 B:
//     48 kHz -> 192,000 B/s   208% of the link, impossible
//     24 kHz ->  96,000 B/s   104%, impossible
//     16 kHz ->  64,000 B/s    70%, fits -- this is what we send
// 115200 baud carries 11,520 B/s and tops out near 2.4 kHz stereo: fine for
// hardware diagnostics, useless for continuous audio.
//
// 921600 baud is NOT yet proven on this board. The diagnostics mode below
// exists to measure it. Treat any throughput claim as unverified until then.
//
// WHY THE FIR IS NOT OPTIONAL
// ---------------------------
// Measured through the laptop's own DOA pipeline at 0.135 m mic spacing, with
// reverberation and noise:
//     48 kHz native                   0.21 deg mean bearing error
//     16 kHz, this anti-aliased FIR   0.71 deg
//     16 kHz, taking every 3rd sample 6.10 deg mean, 30 deg worst case
// Aliasing folds high-frequency energy back into the band at a frequency-
// dependent phase shift, and inter-channel phase is precisely what GCC-PHAT
// measures. Dropping every third sample is not decimation.

#include <driver/i2s_std.h>
#include <atomic>

#if ESP_ARDUINO_VERSION_MAJOR < 3
#error "Install esp32 board package version 3.x - this sketch uses the i2s_std driver."
#endif

// ---------------------------------------------------------------------------
// configuration - must match config/audio.yaml on the laptop
// ---------------------------------------------------------------------------
#define PIN_SCK          26        // both mics, SCK / BCLK
#define PIN_WS           25        // both mics, WS / LRCLK
#define PIN_SD           33        // both mics, SD / data

#define ACQUIRE_RATE     48000     // audio.sample_rate
#define TRANSMIT_RATE    16000     // transport.transmit_sample_rate
#define DECIMATION       3         // ACQUIRE_RATE / TRANSMIT_RATE, exact
#define NUM_CHANNELS     2         // ch0 = mic 1 = LEFT, ch1 = mic 2 = RIGHT

#define BAUD_RATE        921600    // transport.baud_rate
#define SAMPLES_PER_PKT  256       // transport.samples_per_packet, PER CHANNEL
#define PAYLOAD_BYTES    (SAMPLES_PER_PKT * NUM_CHANNELS * 2)   // 1024
#define HEADER_BYTES     16        // transport HEADER_BYTES
#define PACKET_BYTES     (HEADER_BYTES + PAYLOAD_BYTES)         // 1040

#define PROTOCOL_VERSION 1
#define MAGIC_0          0xA5
#define MAGIC_1          0x5A

// Build mode. This picks the BAUD RATE OF THE PORT as well as the boot mode,
// because the two cannot be chosen independently on one UART.
//
//   1 = bring-up build.  Text diagnostics only, port at DIAG_BAUD. Streaming is
//       compiled out, so nothing binary can ever reach the port. Use for the
//       Test 1 in CONTEXT.md §14. DIAG_BAUD is 115200 - the rate
//       mic_check_1.ino proved on this board - so "can the board talk to me at
//       all" is answered WITHOUT also betting on 921600, which is not proven
//       here and is the documented risk in §11.
//   0 = streaming build.  Port at BAUD_RATE (the transport contract) in BOTH
//       modes, so §14 Tests 2 and 3 work unchanged: stream, then press 'd' and
//       read the counters over the same port at the same rate.
#define START_IN_DIAG    0

// Only ever used by a bring-up build. Not part of the transport contract.
#define DIAG_BAUD        115200

#if START_IN_DIAG
  #define PORT_BAUD      DIAG_BAUD
#else
  #define PORT_BAUD      BAUD_RATE
#endif

// Mirrors TransportConfig.__post_init__ in heimdall/audio/config.py, which
// rejects a bad combination rather than clamping it. Checked at compile time
// here so the two ends cannot silently disagree.
static_assert(ACQUIRE_RATE % TRANSMIT_RATE == 0,
              "ACQUIRE_RATE must be an exact integer multiple of TRANSMIT_RATE.");
static_assert(ACQUIRE_RATE / TRANSMIT_RATE == DECIMATION,
              "DECIMATION must equal ACQUIRE_RATE / TRANSMIT_RATE.");
static_assert(DECIMATION == 3,
              "The 48-tap FIR below is designed for the 8 kHz fold point of a "
              "3:1 decimation. A different factor needs a different filter.");
#if !START_IN_DIAG
static_assert((long)TRANSMIT_RATE * NUM_CHANNELS * 2 * 100
                <= (long)(BAUD_RATE / 10) * 85,
              "Wire rate exceeds 85% of the link. Lower TRANSMIT_RATE or raise "
              "BAUD_RATE; a UART driven near capacity drops bytes silently.");
#endif

#define I2S_BLOCK        512       // I2S frames per read
#define RING_FRAMES      4096      // 16 kHz stereo frames of slack = 256 ms

// ---------------------------------------------------------------------------
// packet header, 16 bytes, little-endian
// ---------------------------------------------------------------------------
//   0     magic 0xA5
//   1     magic 0x5A
//   2     protocol version
//   3     flags: bit0 ring overrun, bit1 I2S read failure (since last packet)
//   4-7   sequence number, uint32, wraps
//   8-9   sample count PER CHANNEL, uint16
//   10-11 payload length in bytes, uint16
//   12-13 CRC-16/CCITT-FALSE over header bytes 0..11
//   14-15 CRC-16/CCITT-FALSE over the payload
//
// Sample rate and channel count are NOT in the header: they are the contract in
// config/audio.yaml, shared by both ends, and the version byte covers changes.
// The header is fixed-size so a receiver that loses sync can resynchronise by
// scanning for the magic and validating the header CRC, with nothing
// variable-length to parse first.

// ---------------------------------------------------------------------------
// anti-aliasing FIR: 48 taps, Kaiser(7.5), -6 dB at 5.6 kHz, Q15
// ---------------------------------------------------------------------------
// Designed for the 8 kHz fold point of the 16 kHz output: >71 dB attenuation
// everywhere above 8 kHz, -0.26 dB at 4 kHz. Symmetric, so it is linear phase -
// it delays both channels by exactly (NTAPS-1)/2 samples and therefore does not
// disturb the inter-channel delay GCC-PHAT measures.
//
// Coefficients sum to 32768 (unity gain in Q15). Worst-case |accumulator| is
// sum(|h|) * 32767 = 1,583,480,832, which fits int32 with 26% to spare, so the
// inner loop can stay in 32-bit arithmetic.
#define NTAPS 48
static const int16_t FIR[NTAPS] = {
      -2,     -3,      0,     10,     26,     34,     16,    -40,
    -113,   -154,   -100,     75,    313,    472,    382,    -40,
    -683,  -1221,  -1214,   -319,   1501,   3873,   6108,   7463,
    7463,   6108,   3873,   1501,   -319,  -1214,  -1221,   -683,
     -40,    382,    472,    313,     75,   -100,   -154,   -113,
     -40,     16,     34,     26,     10,      0,     -3,     -2,
};

// ---------------------------------------------------------------------------
// state - all fixed-size, no heap, no String
// ---------------------------------------------------------------------------
static i2s_chan_handle_t rx = nullptr;

static int32_t  rawBlock[I2S_BLOCK * NUM_CHANNELS];

// FIR history, stored twice so the inner loop needs no wrap test.
static int16_t  hist0[NTAPS * 2];
static int16_t  hist1[NTAPS * 2];
static int      histPos  = 0;
static int      decPhase = 0;

// Single-producer / single-consumer ring of decimated stereo frames.
// std::atomic rather than volatile: volatile orders nothing between the data
// write and the index publish, which is exactly the guarantee an SPSC ring
// needs across two cores. Relaxed loads are used where only eventual
// visibility matters.
static int16_t ring[RING_FRAMES * NUM_CHANNELS];
static std::atomic<uint32_t> ringHead{0};   // producer (acquisition task)
static std::atomic<uint32_t> ringTail{0};   // consumer (loop)

static uint8_t  packet[PACKET_BYTES];
static uint16_t crcTable[256];

// Counters. 32-bit on purpose: 64-bit atomics on Xtensa fall back to a
// library lock, and a sample counter at 48 kHz takes ~24.8 hours to wrap,
// which is far longer than any run these diagnose.
static std::atomic<uint32_t> cntI2sFailures{0};
static std::atomic<uint32_t> cntRingOverruns{0};
static std::atomic<uint32_t> cntFramesDropped{0};
static std::atomic<uint32_t> cntSamplesAcq{0};   // per channel, at 48 kHz
static std::atomic<uint32_t> cntSamplesOut{0};   // per channel, at 16 kHz
static uint32_t cntPackets     = 0;              // consumer-only, no sharing
static uint32_t cntBytesSent   = 0;
static uint32_t cntShortWrites = 0;
static uint32_t sequence       = 0;
static uint32_t runStartMs     = 0;

static std::atomic<bool> flagOverrun{false};
static std::atomic<bool> flagI2sFail{false};

// live level, diagnostics only
static std::atomic<int32_t> lastRms0{0};
static std::atomic<int32_t> lastRms1{0};

static inline void bump(std::atomic<uint32_t>& c, uint32_t n = 1) {
  c.fetch_add(n, std::memory_order_relaxed);
}

enum Mode { MODE_STREAM, MODE_DIAG };
static Mode mode = START_IN_DIAG ? MODE_DIAG : MODE_STREAM;

// ---------------------------------------------------------------------------
// CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)
// ---------------------------------------------------------------------------
static void crcInit() {
  for (int i = 0; i < 256; ++i) {
    uint16_t c = (uint16_t)(i << 8);
    for (int b = 0; b < 8; ++b) c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
    crcTable[i] = c;
  }
}

static uint16_t crc16(const uint8_t* data, size_t len) {
  uint16_t c = 0xFFFF;
  for (size_t i = 0; i < len; ++i) c = (uint16_t)((c << 8) ^ crcTable[((c >> 8) ^ data[i]) & 0xFF]);
  return c;
}

// ---------------------------------------------------------------------------
// I2S - reused verbatim from the validated mic_check_1.ino
// ---------------------------------------------------------------------------
static bool startI2S() {
  i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chan.dma_desc_num  = 8;
  chan.dma_frame_num = I2S_BLOCK;
  if (i2s_new_channel(&chan, nullptr, &rx) != ESP_OK) return false;

  i2s_std_config_t cfg = {
    .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(ACQUIRE_RATE),
    // INMP441 is 24-bit left-justified inside a 32-bit slot.
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                  I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = {
      .mclk = I2S_GPIO_UNUSED,
      .bclk = (gpio_num_t)PIN_SCK,
      .ws   = (gpio_num_t)PIN_WS,
      .dout = I2S_GPIO_UNUSED,
      .din  = (gpio_num_t)PIN_SD,
      .invert_flags = { false, false, false },
    },
  };
  if (i2s_channel_init_std_mode(rx, &cfg) != ESP_OK) return false;
  if (i2s_channel_enable(rx) != ESP_OK) return false;
  return true;
}

// ---------------------------------------------------------------------------
// ring buffer
// ---------------------------------------------------------------------------
static inline uint32_t ringCount() {
  // Unsigned wrap is intentional: head - tail stays correct across overflow.
  return ringHead.load(std::memory_order_acquire)
       - ringTail.load(std::memory_order_acquire);
}

static inline bool ringPush(int16_t a, int16_t b) {
  const uint32_t head = ringHead.load(std::memory_order_relaxed);
  if (head - ringTail.load(std::memory_order_acquire) >= RING_FRAMES) return false;
  const uint32_t i = (head % RING_FRAMES) * NUM_CHANNELS;
  ring[i]     = a;                          // ch0 = mic 1
  ring[i + 1] = b;                          // ch1 = mic 2
  // Release: the samples above are visible before the index that publishes them.
  ringHead.store(head + 1, std::memory_order_release);
  return true;
}

// ---------------------------------------------------------------------------
// acquisition task: I2S -> int16 -> FIR -> decimate -> ring
// Pinned to core 0 so a blocking Serial.write on core 1 can never stall it.
// ---------------------------------------------------------------------------
static void acquireTask(void*) {
  for (;;) {
    size_t got = 0;
    if (i2s_channel_read(rx, rawBlock, sizeof(rawBlock), &got, pdMS_TO_TICKS(200)) != ESP_OK
        || got == 0) {
      bump(cntI2sFailures);
      flagI2sFail.store(true, std::memory_order_relaxed);
      continue;
    }

    const size_t frames = got / sizeof(int32_t) / NUM_CHANNELS;
    int64_t sq0 = 0, sq1 = 0;

    for (size_t i = 0; i < frames; ++i) {
      // Validated narrowing: the INMP441's 24 bits sit at the top of the
      // 32-bit slot, so >> 16 keeps the most significant 16. Channel order is
      // untouched: slot 0 is mic 1, slot 1 is mic 2.
      const int16_t a = (int16_t)(rawBlock[i * NUM_CHANNELS + 0] >> 16);
      const int16_t b = (int16_t)(rawBlock[i * NUM_CHANNELS + 1] >> 16);

      sq0 += (int32_t)a * a;
      sq1 += (int32_t)b * b;

      hist0[histPos] = a; hist0[histPos + NTAPS] = a;
      hist1[histPos] = b; hist1[histPos + NTAPS] = b;
      if (++histPos == NTAPS) histPos = 0;

      // Polyphase in effect: the FIR is only evaluated on output samples, so
      // the cost is NTAPS MACs per OUTPUT sample, not per input sample.
      if (++decPhase == DECIMATION) {
        decPhase = 0;

        int32_t acc0 = 0, acc1 = 0;
        const int16_t* w0 = &hist0[histPos];
        const int16_t* w1 = &hist1[histPos];
        for (int k = 0; k < NTAPS; ++k) {
          acc0 += (int32_t)FIR[k] * (int32_t)w0[k];
          acc1 += (int32_t)FIR[k] * (int32_t)w1[k];
        }
        acc0 >>= 15;
        acc1 >>= 15;
        if (acc0 >  32767) acc0 =  32767;
        if (acc0 < -32768) acc0 = -32768;
        if (acc1 >  32767) acc1 =  32767;
        if (acc1 < -32768) acc1 = -32768;

        if (ringPush((int16_t)acc0, (int16_t)acc1)) {
          bump(cntSamplesOut);
        } else {
          // Consumer is behind: the wire cannot keep up. Count it loudly
          // rather than silently corrupting the timeline.
          bump(cntRingOverruns);
          bump(cntFramesDropped);
          flagOverrun.store(true, std::memory_order_relaxed);
        }
      }
    }

    bump(cntSamplesAcq, (uint32_t)frames);
    if (frames) {
      lastRms0.store((int32_t)sqrt((double)sq0 / frames), std::memory_order_relaxed);
      lastRms1.store((int32_t)sqrt((double)sq1 / frames), std::memory_order_relaxed);
    }
  }
}

// ---------------------------------------------------------------------------
// packet emission
// ---------------------------------------------------------------------------
static inline void put16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static inline void put32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static void sendPacket() {
  uint8_t* payload = packet + HEADER_BYTES;

  // Copy 256 stereo frames out of the ring, handling the wrap.
  uint32_t remaining = SAMPLES_PER_PKT;
  uint8_t* dst  = payload;
  uint32_t tail = ringTail.load(std::memory_order_relaxed);
  while (remaining) {
    const uint32_t offset = tail % RING_FRAMES;
    uint32_t chunk = RING_FRAMES - offset;
    if (chunk > remaining) chunk = remaining;
    memcpy(dst, &ring[offset * NUM_CHANNELS], chunk * NUM_CHANNELS * sizeof(int16_t));
    dst += chunk * NUM_CHANNELS * sizeof(int16_t);
    tail += chunk;
    remaining -= chunk;
  }
  // Release the space only once the whole payload has been copied out.
  ringTail.store(tail, std::memory_order_release);

  uint8_t flags = 0;
  if (flagOverrun.exchange(false, std::memory_order_relaxed)) flags |= 0x01;
  if (flagI2sFail.exchange(false, std::memory_order_relaxed)) flags |= 0x02;

  packet[0] = MAGIC_0;
  packet[1] = MAGIC_1;
  packet[2] = PROTOCOL_VERSION;
  packet[3] = flags;
  put32(&packet[4], sequence);
  put16(&packet[8], SAMPLES_PER_PKT);
  put16(&packet[10], PAYLOAD_BYTES);
  put16(&packet[12], crc16(packet, 12));
  put16(&packet[14], crc16(payload, PAYLOAD_BYTES));

  const size_t written = Serial.write(packet, PACKET_BYTES);
  if (written != PACKET_BYTES) cntShortWrites++;

  sequence++;
  cntPackets++;
  cntBytesSent += (uint32_t)written;
}

// ---------------------------------------------------------------------------
// diagnostics - text only, and ONLY in MODE_DIAG, so the binary stream is
// never contaminated. Switching to diagnostics reports the counters
// accumulated during the streaming run, which is how sustained throughput is
// measured: stream for a minute, then press 'd'.
// ---------------------------------------------------------------------------
static void printDiagnostics() {
  const float    seconds = (millis() - runStartMs) / 1000.0f;
  const float    wireRate = seconds > 0 ? cntBytesSent / seconds : 0.0f;
  const float    pktRate  = seconds > 0 ? cntPackets / seconds : 0.0f;
  const uint32_t acq  = cntSamplesAcq.load(std::memory_order_relaxed);
  const uint32_t out  = cntSamplesOut.load(std::memory_order_relaxed);
  const int32_t  rms0 = lastRms0.load(std::memory_order_relaxed);
  const int32_t  rms1 = lastRms1.load(std::memory_order_relaxed);

  Serial.println();
  Serial.println(F("--- heimdall esp32_mic diagnostics ---"));
  Serial.printf("elapsed          : %.2f s\n", seconds);
  Serial.printf("acquired         : %lu samples/ch  (%.0f Hz actual, expect %d)\n",
                (unsigned long)acq, seconds > 0 ? acq / seconds : 0.0f, ACQUIRE_RATE);
  Serial.printf("decimated        : %lu samples/ch  (%.0f Hz actual, expect %d)\n",
                (unsigned long)out, seconds > 0 ? out / seconds : 0.0f, TRANSMIT_RATE);
  Serial.printf("packets sent     : %lu  (%.2f/s, expect %.2f/s)\n",
                (unsigned long)cntPackets, pktRate,
                (float)TRANSMIT_RATE / SAMPLES_PER_PKT);
  Serial.printf("bytes sent       : %lu  (%.0f B/s, expect ~65000 B/s)\n",
                (unsigned long)cntBytesSent, wireRate);
  Serial.printf("link utilisation : %.1f%% of %d B/s at %d baud\n",
                100.0f * wireRate / (PORT_BAUD / 10.0f), PORT_BAUD / 10, PORT_BAUD);
  Serial.printf("i2s failures     : %lu\n",
                (unsigned long)cntI2sFailures.load(std::memory_order_relaxed));
  Serial.printf("ring overruns    : %lu  (frames dropped %lu)\n",
                (unsigned long)cntRingOverruns.load(std::memory_order_relaxed),
                (unsigned long)cntFramesDropped.load(std::memory_order_relaxed));
  Serial.printf("short writes     : %lu\n", (unsigned long)cntShortWrites);
  Serial.printf("ring occupancy   : %lu / %d frames\n",
                (unsigned long)ringCount(), RING_FRAMES);
  Serial.printf("levels           : ch0 rms %ld   ch1 rms %ld\n",
                (long)rms0, (long)rms1);
  if (rms0 < 3 && rms1 < 3)  Serial.println(F("  << BOTH CHANNELS SILENT"));
  else if (rms0 < 3)         Serial.println(F("  << CH0 SILENT (mic 1)"));
  else if (rms1 < 3)         Serial.println(F("  << CH1 SILENT (mic 2)"));
  Serial.println(F("keys: s = stream (binary)   d = diagnostics   r = reset counters"));
}

static void resetCounters() {
  cntI2sFailures.store(0, std::memory_order_relaxed);
  cntRingOverruns.store(0, std::memory_order_relaxed);
  cntFramesDropped.store(0, std::memory_order_relaxed);
  cntSamplesAcq.store(0, std::memory_order_relaxed);
  cntSamplesOut.store(0, std::memory_order_relaxed);
  cntPackets = cntBytesSent = cntShortWrites = 0;
  runStartMs = millis();
}

// ---------------------------------------------------------------------------
static void handleCommands() {
  while (Serial.available() > 0) {
    switch (Serial.read()) {
      case 's': case 'S':
#if START_IN_DIAG
        // Refused, not attempted. At DIAG_BAUD the link cannot carry the
        // stream, and emitting binary anyway would only look like corruption.
        Serial.println(F("refused: bring-up build (START_IN_DIAG 1). "
                         "Set START_IN_DIAG 0 and re-upload to stream."));
#else
        if (mode != MODE_STREAM) { mode = MODE_STREAM; resetCounters(); }
#endif
        break;
      case 'd': case 'D':
        mode = MODE_DIAG;
        break;
      case 'r': case 'R':
        resetCounters();
        break;
      default:
        break;                              // ignore anything else
    }
  }
}

void setup() {
  Serial.setRxBufferSize(256);
  // Large TX buffer so a 1040-byte packet is handed to the UART driver in one
  // go. Serial.write still blocks when the buffer is full, which is why
  // acquisition runs on the other core.
  Serial.setTxBufferSize(8192);
  Serial.begin(PORT_BAUD);
  delay(600);

  crcInit();

  if (!startI2S()) {
    // I2S failure is fatal and must be visible even in streaming mode, so it is
    // reported as text and no binary stream is ever started.
    mode = MODE_DIAG;
    for (;;) {
      Serial.println(F("FAIL: i2s did not start (firmware/pin problem, not wiring)."));
      delay(1000);
    }
  }

  resetCounters();

  if (mode == MODE_DIAG) {
    Serial.println();
    Serial.println(F("=== heimdall esp32_mic ==="));
    Serial.printf("port     : %d baud  (%s build)\n",
                  PORT_BAUD, START_IN_DIAG ? "bring-up, diagnostics only"
                                           : "streaming");
    Serial.printf("i2s ready: %d Hz, %d ch, 32-bit; transmitting %d Hz int16\n",
                  ACQUIRE_RATE, NUM_CHANNELS, TRANSMIT_RATE);
    Serial.println(F("keys: s = stream (binary)   d = diagnostics   r = reset counters"));
  }

  // Acquisition on core 0, priority above the Arduino loop. Stack sized for the
  // FIR working set; nothing here allocates.
  xTaskCreatePinnedToCore(acquireTask, "heimdall-i2s", 8192, nullptr, 5, nullptr, 0);
}

void loop() {
  handleCommands();

  if (mode == MODE_STREAM) {
    if (ringCount() >= SAMPLES_PER_PKT) {
      sendPacket();
    } else {
      // Nothing ready yet: one packet is 16 ms of audio, so a short yield here
      // costs nothing and keeps the idle task fed.
      vTaskDelay(1);
    }
    return;
  }

  // MODE_DIAG: text only, no binary output at all. The ring is drained so it
  // does not sit permanently full and report overruns that are an artefact of
  // diagnostics rather than of the link.
  ringTail.store(ringHead.load(std::memory_order_acquire), std::memory_order_release);
  static uint32_t lastPrint = 0;
  if (millis() - lastPrint >= 1000) {
    lastPrint = millis();
    printDiagnostics();
  }
  vTaskDelay(10);
}
