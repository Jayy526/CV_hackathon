// Heimdall — MIC CHECK
//
// A single self-contained sketch for validating the hardware over USB.
// No WiFi, no Python, no other files. Open it, flash it, watch the Serial
// Monitor at 115200 baud.
//
// It answers, in order:
//   1. Does I2S start at all?           -> "i2s ready"
//   2. Is each microphone alive?        -> live RMS bars per channel
//   3. Are the L/R straps correct?      -> warns if the channels are identical
//   4. Do the two mics hear a clap at
//      measurably different times?      -> prints TDOA in samples + bearing
//
// Board:  ESP32-WROOM-32 DevKit (classic ESP32)
// Core:   arduino-esp32 3.x   (Tools > Board > Boards Manager > "esp32")
// Board setting: "ESP32 Dev Module"

#include <driver/i2s_std.h>

#if ESP_ARDUINO_VERSION_MAJOR < 3
#error "Install esp32 board package version 3.x — this sketch uses the i2s_std driver."
#endif

// --------------------------------------------------------------------------
// configuration — must match the wiring
// --------------------------------------------------------------------------
#define PIN_SCK        26      // both mics, SCK / BCLK
#define PIN_WS         25      // both mics, WS / LRCLK
#define PIN_SD         33      // both mics, SD / data

#define SAMPLE_RATE    48000
#define MIC_SPACING_M  0.135f  // measured centre-to-centre, in metres
#define SPEED_OF_SOUND 343.0f

#define BLOCK          512     // samples per channel per read
#define HISTORY        1024    // samples per channel kept for the clap analysis
#define MAX_LAG        28      // > spacing/c * rate  (0.135 m -> ~19 samples)
#define CLAP_THRESHOLD 4000    // int16 peak that counts as a transient

// --------------------------------------------------------------------------
static i2s_chan_handle_t rx = nullptr;
static int32_t  raw[BLOCK * 2];
static int16_t  hist0[HISTORY], hist1[HISTORY];
static uint32_t histPos = 0;
static uint32_t lastReport = 0;
static uint32_t lastClap = 0;

static bool startI2S() {
  i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chan.dma_desc_num  = 8;
  chan.dma_frame_num = BLOCK;
  if (i2s_new_channel(&chan, nullptr, &rx) != ESP_OK) return false;

  i2s_std_config_t cfg = {
    .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
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

static void bar(int32_t rms) {
  // rough log bar, 0..24 characters
  int n = 0;
  int32_t v = rms;
  while (v > 8 && n < 24) { v = (v * 3) / 4; n++; }
  Serial.print('[');
  for (int i = 0; i < 24; ++i) Serial.print(i < n ? '#' : ' ');
  Serial.print(']');
}

// Brute-force cross-correlation. Positive lag means channel 1 arrives LATER.
static int bestLag(int32_t* peakOut) {
  int64_t best = INT64_MIN; int bestL = 0;
  for (int l = -MAX_LAG; l <= MAX_LAG; ++l) {
    int64_t acc = 0;
    for (int n = MAX_LAG; n < HISTORY - MAX_LAG; ++n) {
      int i0 = (histPos + n) % HISTORY;
      int i1 = (histPos + n + l + HISTORY) % HISTORY;
      acc += (int32_t)hist0[i0] * (int32_t)hist1[i1];
    }
    if (acc > best) { best = acc; bestL = l; }
  }
  *peakOut = (int32_t)(best >> 20);
  return bestL;
}

void setup() {
  Serial.begin(115200);
  delay(600);
  Serial.println();
  Serial.println("=== heimdall mic check ===");
  Serial.printf("pins: SCK=%d  WS=%d  SD=%d\n", PIN_SCK, PIN_WS, PIN_SD);
  Serial.printf("rate: %d Hz, 2 channels, spacing %.3f m\n",
                SAMPLE_RATE, MIC_SPACING_M);

  if (!startI2S()) {
    Serial.println("FAIL: i2s did not start.");
    Serial.println("  This is a firmware/pin problem, not a wiring one.");
    while (true) delay(1000);
  }
  Serial.println("i2s ready");
  Serial.println();
  Serial.println("Tap mic 1, then mic 2. Each should move ITS OWN bar only.");
  Serial.println("Then clap once, a metre away, off to one side.");
  Serial.println();
  delay(300);
}

void loop() {
  size_t got = 0;
  if (i2s_channel_read(rx, raw, sizeof(raw), &got, pdMS_TO_TICKS(300)) != ESP_OK
      || got == 0) {
    Serial.println("FAIL: no data from i2s.");
    Serial.println("  Check SCK/WS/SD wiring and that both mics have 3V3 + GND.");
    delay(500);
    return;
  }

  const size_t n = got / sizeof(int32_t) / 2;
  int64_t sq0 = 0, sq1 = 0, sum0 = 0, sum1 = 0;
  int16_t pk0 = 0, pk1 = 0;
  uint32_t identical = 0;

  for (size_t i = 0; i < n; ++i) {
    int16_t a = (int16_t)(raw[i * 2 + 0] >> 16);
    int16_t b = (int16_t)(raw[i * 2 + 1] >> 16);

    hist0[histPos] = a; hist1[histPos] = b;
    histPos = (histPos + 1) % HISTORY;

    sq0 += (int32_t)a * a;  sq1 += (int32_t)b * b;
    sum0 += a;              sum1 += b;
    if (abs(a) > pk0) pk0 = abs(a);
    if (abs(b) > pk1) pk1 = abs(b);
    if (a == b) identical++;
  }

  const int32_t rms0 = (int32_t)sqrt((double)sq0 / n);
  const int32_t rms1 = (int32_t)sqrt((double)sq1 / n);
  const int32_t dc0  = (int32_t)(sum0 / (int64_t)n);
  const int32_t dc1  = (int32_t)(sum1 / (int64_t)n);

  // ---- clap detected: measure the delay between the two channels ----------
  if ((pk0 > CLAP_THRESHOLD || pk1 > CLAP_THRESHOLD) &&
      millis() - lastClap > 700) {
    lastClap = millis();
    int32_t strength = 0;
    const int lag = bestLag(&strength);
    const float tdoa_us = lag * 1e6f / SAMPLE_RATE;
    float s = lag * (SPEED_OF_SOUND / SAMPLE_RATE) / MIC_SPACING_M;
    s = constrain(s, -1.0f, 1.0f);
    Serial.println();
    Serial.printf(">> CLAP  lag=%+d samples  (%+.0f us)  bearing=%+.0f deg\n",
                  lag, tdoa_us, degrees(asinf(s)));
    if (abs(lag) >= MAX_LAG)
      Serial.println("   (at the search limit — probably noise, not a clap)");
    Serial.println();
    return;
  }

  // ---- periodic level report ---------------------------------------------
  if (millis() - lastReport < 250) return;
  lastReport = millis();

  Serial.print("ch0 "); bar(rms0);
  Serial.printf(" rms%6d pk%6d   ", rms0, pk0);
  Serial.print("ch1 "); bar(rms1);
  Serial.printf(" rms%6d pk%6d", rms1, pk1);

  if (rms0 < 3 && rms1 < 3)      Serial.print("   << BOTH SILENT");
  else if (rms0 < 3)             Serial.print("   << CH0 SILENT (mic 1)");
  else if (rms1 < 3)             Serial.print("   << CH1 SILENT (mic 2)");
  else if (identical > n - 4)    Serial.print("   << CHANNELS IDENTICAL");
  if (abs(dc0) > 2000 || abs(dc1) > 2000) Serial.print("   << LARGE DC OFFSET");
  Serial.println();
}
