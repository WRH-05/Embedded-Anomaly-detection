#include <Wire.h>
#include <math.h>

// --- Pin Definitions ---
#define SDA_PIN        8
#define SCL_PIN        9
#define BUZZER_PIN     4

// Built-in WS2812 RGB LED (Usually GPIO 48 on ESP32-S3 DevKit)
#ifdef RGB_BUILTIN
  #define RGB_PIN RGB_BUILTIN
#else
  #define RGB_PIN 48
#endif

// --- MPU9250 / MPU6050 Registers ---
#define MPU_ADDR       0x68
#define PWR_MGMT_1     0x6B
#define ACCEL_XOUT_H   0x3B

// --- Sampling Configuration ---
#define FS             100         // 100 Hz
#define WINDOW_SIZE    100         // 100 samples = 1000 ms window
#define SAMPLE_DELAY   10          // 10 ms interval

float ax_buf[WINDOW_SIZE];
float ay_buf[WINDOW_SIZE];
float az_buf[WINDOW_SIZE];
float amag_buf[WINDOW_SIZE];

void setRGB(uint8_t r, uint8_t g, uint8_t b) {
  #if defined(neopixelWrite)
    neopixelWrite(RGB_PIN, r, g, b);
  #elif defined(rgbLedWrite)
    rgbLedWrite(RGB_PIN, r, g, b);
  #endif
}

void beep(int ms) {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(ms);
  digitalWrite(BUZZER_PIN, LOW);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  setRGB(0, 0, 30); // Blue: Booting

  Wire.begin(SDA_PIN, SCL_PIN, 400000);

  // Wake up MPU9250
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0x00);
  if (Wire.endTransmission() != 0) {
    Serial.println("ERR: MPU connection failed!");
    setRGB(50, 0, 0); // Solid Red on error
    while (1) delay(100);
  }

  // Ready signal
  setRGB(0, 50, 0); // Green
  beep(80);
  delay(300);
  setRGB(0, 0, 0);

  Serial.println("READY_FOR_ACQUISITION");
}

void loop() {
  unsigned long window_timestamp = millis();

  // 1. Collect 100 samples over exactly 1 second (100 Hz)
  for (int i = 0; i < WINDOW_SIZE; i++) {
    unsigned long sample_start = millis();

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(ACCEL_XOUT_H);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6, true);

    if (Wire.available() >= 6) {
      int16_t raw_x = (Wire.read() << 8) | Wire.read();
      int16_t raw_y = (Wire.read() << 8) | Wire.read();
      int16_t raw_z = (Wire.read() << 8) | Wire.read();

      // Convert to 'g' (±2g scale -> 16384 LSB/g)
      ax_buf[i] = (float)raw_x / 16384.0f;
      ay_buf[i] = (float)raw_y / 16384.0f;
      az_buf[i] = (float)raw_z / 16384.0f;
      amag_buf[i] = sqrtf(ax_buf[i]*ax_buf[i] + ay_buf[i]*ay_buf[i] + az_buf[i]*az_buf[i]);
    }

    while (millis() - sample_start < SAMPLE_DELAY);
  }

  // 2. Feature Extraction
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

  // 3. Dominant Frequency via Zero-Crossing Rate (ZCR)
  int zero_crossings = 0;
  for (int i = 1; i < WINDOW_SIZE; i++) {
    float prev = amag_buf[i - 1] - mean_mag;
    float curr = amag_buf[i] - mean_mag;
    if ((prev < 0.0f && curr >= 0.0f) || (prev > 0.0f && curr <= 0.0f)) {
      zero_crossings++;
    }
  }

  // Guard against negative numbers: strictly positive physical frequency
  float frequence_dominante = (float)zero_crossings / 2.0f;
  if (frequence_dominante < 0.0f) {
    frequence_dominante = fabsf(frequence_dominante);
  }

  // 4. Output: timestamp + 6 features (comma-separated)
  Serial.print(window_timestamp);
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

  // Visual blink for output
  setRGB(15, 15, 15);
  delay(20);
  setRGB(0, 0, 0);
}