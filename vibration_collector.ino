#include <Wire.h>
#include <math.h>

// --- Pin Definitions ---
#define SDA_PIN        8
#define SCL_PIN        9
#define BUZZER_PIN     4

// Built-in WS2812 RGB LED on ESP32-S3 DevKit (usually GPIO 48)
#ifdef RGB_BUILTIN
  #define RGB_PIN RGB_BUILTIN
#else
  #define RGB_PIN 48
#endif

// --- MPU9250 / MPU6050 Registers ---
#define MPU_ADDR       0x68
#define PWR_MGMT_1     0x6B
#define ACCEL_XOUT_H   0x3B

// --- Sampling Parameters ---
#define FS             100          // 100 Hz sampling rate
#define WINDOW_SIZE    100          // 100 samples = 1000 ms window
#define SAMPLE_DELAY   10           // 10 ms delay (1/100 Hz)

// Buffers for 1-second window
float ax_buf[WINDOW_SIZE];
float ay_buf[WINDOW_SIZE];
float az_buf[WINDOW_SIZE];
float amag_buf[WINDOW_SIZE];

// Built-in LED Helper using ESP32 Arduino Core
void setRGB(uint8_t r, uint8_t g, uint8_t b) {
  #if defined(neopixelWrite)
    neopixelWrite(RGB_PIN, r, g, b);
  #elif defined(rgbLedWrite)
    rgbLedWrite(RGB_PIN, r, g, b);
  #endif
}

void beep(int duration_ms) {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(duration_ms);
  digitalWrite(BUZZER_PIN, LOW);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000); // Wait for USB CDC connection

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // Initialize RGB LED (Blue: Initializing)
  setRGB(0, 0, 30);

  // Initialize I2C
  Wire.begin(SDA_PIN, SCL_PIN, 400000); // 400 kHz Fast-Mode

  // Wake up MPU9250 (Clear sleep mode bit 6 in PWR_MGMT_1)
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0x00);
  if (Wire.endTransmission() != 0) {
    Serial.println("ERR: MPU9250 connection failed! Check wiring.");
    setRGB(50, 0, 0); // Solid Red on error
    while (1) delay(100);
  }

  // Self-test complete (Green flash + beep)
  setRGB(0, 50, 0);
  beep(100);
  delay(500);
  setRGB(0, 0, 0); // Off

  Serial.println("READY_FOR_ACQUISITION");
}

void loop() {
  unsigned long window_start = millis();

  // 1. Collect 100 samples over 1000 ms
  for (int i = 0; i < WINDOW_SIZE; i++) {
    unsigned long sample_start = millis();

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(ACCEL_XOUT_H);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6, true);

    if (Wire.available() >= 6) {
      int16_t raw_ax = (Wire.read() << 8) | Wire.read();
      int16_t raw_ay = (Wire.read() << 8) | Wire.read();
      int16_t raw_az = (Wire.read() << 8) | Wire.read();

      // Convert raw 16-bit to 'g' (±2g scale -> sensitivity: 16384 LSB/g)
      ax_buf[i] = (float)raw_ax / 16384.0f;
      ay_buf[i] = (float)raw_ay / 16384.0f;
      az_buf[i] = (float)raw_az / 16384.0f;
      amag_buf[i] = sqrtf(ax_buf[i]*ax_buf[i] + ay_buf[i]*ay_buf[i] + az_buf[i]*az_buf[i]);
    }

    // Maintain precise 10ms (100Hz) timing
    while (millis() - sample_start < SAMPLE_DELAY);
  }

  // 2. Feature Extraction (P09 Requirements)
  float sum_sq_x = 0, sum_sq_y = 0, sum_sq_z = 0, sum_sq_mag = 0;
  float peak_acc = 0.0f;
  float mean_mag = 0.0f;

  for (int i = 0; i < WINDOW_SIZE; i++) {
    sum_sq_x += ax_buf[i] * ax_buf[i];
    sum_sq_y += ay_buf[i] * ay_buf[i];
    sum_sq_z += az_buf[i] * az_buf[i];
    sum_sq_mag += amag_buf[i] * amag_buf[i];
    mean_mag += amag_buf[i];

    if (amag_buf[i] > peak_acc) {
      peak_acc = amag_buf[i];
    }
  }

  mean_mag /= WINDOW_SIZE;
  float rms_x = sqrtf(sum_sq_x / WINDOW_SIZE);
  float rms_y = sqrtf(sum_sq_y / WINDOW_SIZE);
  float rms_z = sqrtf(sum_sq_z / WINDOW_SIZE);
  float energie = sum_sq_mag / WINDOW_SIZE;

  // 3. Dominant Frequency Estimation via Zero-Crossing Rate (ZCR)
  // Clean, deterministic on MCU for dominant vibration peak
  int zero_crossings = 0;
  for (int i = 1; i < WINDOW_SIZE; i++) {
    if ((amag_buf[i] - mean_mag) * (amag_buf[i - 1] - mean_mag) < 0) {
      zero_crossings++;
    }
  }
  // Dominant frequency approximation: (Zero Crossings / 2) / (Time = 1.0 s)
  float frequence_dominante = (float)zero_crossings / 2.0f;

  // 4. Output Formatted Line to Serial (timestamp, features...)
  Serial.print(window_start);
  Serial.print(",");
  Serial.print(rms_x, 4);
  Serial.print(",");
  Serial.print(rms_y, 4);
  Serial.print(",");
  Serial.print(rms_z, 4);
  Serial.print(",");
  Serial.print(peak_acc, 4);
  Serial.print(",");
  Serial.print(frequence_dominante, 2);
  Serial.print(",");
  Serial.println(energie, 4);

  // Quick blink indicating a window was sent
  setRGB(10, 10, 10);
  delay(20);
  setRGB(0, 0, 0);
}