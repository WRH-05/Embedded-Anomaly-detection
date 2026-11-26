#include <Arduino.h>
#include <Wire.h>
#include <math.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model_data.h"
#include "scaler_params.h"

constexpr uint8_t kSdaPin = 8;
constexpr uint8_t kSclPin = 9;
constexpr uint8_t kBuzzerPin = 4;
constexpr uint8_t kRgbPin = 48;
constexpr uint8_t kMpuAddress = 0x68;
constexpr uint8_t kPowerManagementRegister = 0x6B;
constexpr uint8_t kAccelConfigRegister = 0x1C;
constexpr uint8_t kAccelOutputRegister = 0x3B;
constexpr uint16_t kWindowSize = 100;
constexpr uint32_t kSamplePeriodUs = 10000;
constexpr uint32_t kBuzzerPeriodMs = 1000;
constexpr uint32_t kBuzzerOnMs = 180;
constexpr int kTensorArenaSize = 16 * 1024;
constexpr int kBenchmarkRuns = 100;
constexpr float kActiveCurrentMa = 110.0f;
constexpr float kSleepCurrentMa = 5.0f;
constexpr float kBatteryCapacityMah = 1000.0f;
constexpr float kBatteryVoltage = 3.7f;

static float axBuffer[kWindowSize];
static float ayBuffer[kWindowSize];
static float azBuffer[kWindowSize];
static float magnitudeBuffer[kWindowSize];
static uint8_t writeIndex = 0;
static uint16_t samplesInWindow = 0;
static uint32_t nextSampleUs = 0;
static uint32_t lastAlarmCycleMs = 0;
static bool alarmToneOn = false;
static size_t freeHeapBeforeAllocation = 0;
static size_t freeHeapAfterAllocation = 0;

alignas(16) static uint8_t tensorArena[kTensorArenaSize];
static tflite::MicroMutableOpResolver<2> resolver;
static const tflite::Model* model = tflite::GetModel(g_model);
static tflite::MicroInterpreter interpreter(
    model, resolver, tensorArena, kTensorArenaSize);
static TfLiteTensor* inputTensor = nullptr;
static TfLiteTensor* outputTensor = nullptr;

static const char* const kClassNames[] = {
    "CALME",
    "VIBRATION_NORMALE",
    "VIBRATION_ANORMALE",
};

struct WindowFeatures {
  float values[6];
};

static void setRgb(uint8_t red, uint8_t green, uint8_t blue) {
  neopixelWrite(kRgbPin, red, green, blue);
}

static void setAlarm(bool enabled) {
  if (!enabled) {
    noTone(kBuzzerPin);
    alarmToneOn = false;
    return;
  }

  const uint32_t now = millis();
  const uint32_t cyclePosition = now % kBuzzerPeriodMs;
  const bool shouldSound = cyclePosition < kBuzzerOnMs;
  if (shouldSound != alarmToneOn) {
    if (shouldSound) {
      tone(kBuzzerPin, 2000);
    } else {
      noTone(kBuzzerPin);
    }
    alarmToneOn = shouldSound;
  }
}

static bool readAcceleration(float* ax, float* ay, float* az) {
  Wire.beginTransmission(kMpuAddress);
  Wire.write(kAccelOutputRegister);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }
  if (Wire.requestFrom(kMpuAddress, static_cast<uint8_t>(6),
                       static_cast<uint8_t>(true)) != 6) {
    while (Wire.available()) {
      Wire.read();
    }
    return false;
  }

  const int16_t rawX =
      static_cast<int16_t>((static_cast<uint16_t>(Wire.read()) << 8) |
                           static_cast<uint16_t>(Wire.read()));
  const int16_t rawY =
      static_cast<int16_t>((static_cast<uint16_t>(Wire.read()) << 8) |
                           static_cast<uint16_t>(Wire.read()));
  const int16_t rawZ =
      static_cast<int16_t>((static_cast<uint16_t>(Wire.read()) << 8) |
                           static_cast<uint16_t>(Wire.read()));

  // The MPU9250 is configured for +/-2 g, or 16384 LSB/g.
  *ax = static_cast<float>(rawX) / 16384.0f;
  *ay = static_cast<float>(rawY) / 16384.0f;
  *az = static_cast<float>(rawZ) / 16384.0f;
  return true;
}

static WindowFeatures extractFeatures() {
  WindowFeatures features = {};
  float sumSquareX = 0.0f;
  float sumSquareY = 0.0f;
  float sumSquareZ = 0.0f;
  float sumSquareMagnitude = 0.0f;
  float sumMagnitude = 0.0f;
  float peakAcceleration = 0.0f;

  for (uint16_t offset = 0; offset < kWindowSize; ++offset) {
    const uint16_t index = (writeIndex + offset) % kWindowSize;
    const float ax = axBuffer[index];
    const float ay = ayBuffer[index];
    const float az = azBuffer[index];
    const float magnitude = sqrtf(ax * ax + ay * ay + az * az);
    magnitudeBuffer[offset] = magnitude;
    sumSquareX += ax * ax;
    sumSquareY += ay * ay;
    sumSquareZ += az * az;
    sumSquareMagnitude += magnitude * magnitude;
    sumMagnitude += magnitude;
    if (magnitude > peakAcceleration) {
      peakAcceleration = magnitude;
    }
  }

  const float meanMagnitude = sumMagnitude / static_cast<float>(kWindowSize);
  uint16_t zeroCrossings = 0;
  for (uint16_t index = 1; index < kWindowSize; ++index) {
    const float previous = magnitudeBuffer[index - 1] - meanMagnitude;
    const float current = magnitudeBuffer[index] - meanMagnitude;
    if ((previous < 0.0f && current >= 0.0f) ||
        (previous > 0.0f && current <= 0.0f)) {
      ++zeroCrossings;
    }
  }

  features.values[0] = sqrtf(sumSquareX / kWindowSize);
  features.values[1] = sqrtf(sumSquareY / kWindowSize);
  features.values[2] = sqrtf(sumSquareZ / kWindowSize);
  features.values[3] = peakAcceleration;
  features.values[4] = fabsf((static_cast<float>(zeroCrossings) / 2.0f) / 1.0f);
  features.values[5] = sumSquareMagnitude / kWindowSize;
  return features;
}

static bool invokeModel(const float normalizedFeatures[6], float* probabilities,
                        uint32_t* latencyUs) {
  for (int feature = 0; feature < 6; ++feature) {
    const float quantizedValue =
        normalizedFeatures[feature] / inputTensor->params.scale +
        inputTensor->params.zero_point;
    const long rounded = lroundf(quantizedValue);
    const long bounded = rounded < -128 ? -128 : (rounded > 127 ? 127 : rounded);
    inputTensor->data.int8[feature] = static_cast<int8_t>(bounded);
  }

  const uint32_t startUs = micros();
  if (interpreter.Invoke() != kTfLiteOk) {
    return false;
  }
  *latencyUs = micros() - startUs;

  if (probabilities != nullptr) {
    for (int category = 0; category < 3; ++category) {
      probabilities[category] =
          (static_cast<int32_t>(outputTensor->data.int8[category]) -
           outputTensor->params.zero_point) *
          outputTensor->params.scale;
    }
  }
  return true;
}

static void printBenchmark() {
  const float benchmarkInput[6] = {0.15f, -0.10f, 0.05f, 0.20f, -0.25f, 0.12f};
  uint32_t minLatencyUs = UINT32_MAX;
  uint32_t maxLatencyUs = 0;
  uint64_t latencySumUs = 0;
  float probabilities[3];

  Serial.println();
  Serial.println("=== TABLE 4.2: ESP32-S3 TFLite Micro Benchmark ===");
  for (int run = 0; run < kBenchmarkRuns; ++run) {
    uint32_t latencyUs = 0;
    if (!invokeModel(benchmarkInput, probabilities, &latencyUs)) {
      Serial.println("ERROR: Benchmark inference failed.");
      while (true) {
        delay(1000);
      }
    }
    if (latencyUs < minLatencyUs) {
      minLatencyUs = latencyUs;
    }
    if (latencyUs > maxLatencyUs) {
      maxLatencyUs = latencyUs;
    }
    latencySumUs += latencyUs;
  }

  const double meanLatencyUs =
      static_cast<double>(latencySumUs) / kBenchmarkRuns;
  const double throughput = meanLatencyUs > 0.0 ? 1000000.0 / meanLatencyUs : 0.0;
  const size_t arenaUsedBytes = interpreter.arena_used_bytes();
  const float inferenceSeconds = static_cast<float>(meanLatencyUs) / 1000000.0f;
  const float periodsSeconds[] = {0.5f, 5.0f};

  Serial.println("Metric,Value,Unit");
  Serial.printf("Inference runs,%d,count\n", kBenchmarkRuns);
  Serial.printf("Minimum inference latency,%lu,us\n",
                static_cast<unsigned long>(minLatencyUs));
  Serial.printf("Maximum inference latency,%lu,us\n",
                static_cast<unsigned long>(maxLatencyUs));
  Serial.printf("Mean inference latency,%.2f,us\n", meanLatencyUs);
  Serial.printf("Jitter,%lu,us\n",
                static_cast<unsigned long>(maxLatencyUs - minLatencyUs));
  Serial.printf("Throughput,%.2f,inferences/s\n", throughput);
  Serial.printf("Tensor arena allocated,%d,bytes\n", kTensorArenaSize);
  Serial.printf("Tensor arena used,%u,bytes\n",
                static_cast<unsigned int>(arenaUsedBytes));
  Serial.printf("Free heap before AllocateTensors,%u,bytes\n",
                static_cast<unsigned int>(freeHeapBeforeAllocation));
  Serial.printf("Free heap after AllocateTensors,%u,bytes\n",
                static_cast<unsigned int>(freeHeapAfterAllocation));

  for (int scenario = 0; scenario < 2; ++scenario) {
    const float dutyCycle =
        fminf(inferenceSeconds / periodsSeconds[scenario], 1.0f);
    const float averageCurrentMa =
        kSleepCurrentMa + (kActiveCurrentMa - kSleepCurrentMa) * dutyCycle;
    const float autonomyHours = kBatteryCapacityMah / averageCurrentMa;
    const float autonomyDays = autonomyHours / 24.0f;
    const float batteryEnergyWh =
        (kBatteryCapacityMah / 1000.0f) * kBatteryVoltage;
    Serial.printf("Scenario %c inference interval,%.1f,s\n",
                  scenario == 0 ? 'A' : 'B', periodsSeconds[scenario]);
    Serial.printf("Scenario %c estimated average current,%.3f,mA\n",
                  scenario == 0 ? 'A' : 'B', averageCurrentMa);
    Serial.printf("Scenario %c estimated autonomy,%.2f,h (%.2f days)\n",
                  scenario == 0 ? 'A' : 'B', autonomyHours, autonomyDays);
    Serial.printf("Scenario %c battery energy,%.2f,Wh\n",
                  scenario == 0 ? 'A' : 'B', batteryEnergyWh);
  }
  Serial.println(
      "Autonomy estimate assumes only inference time draws active current; "
      "sensor/CPU idle current is modeled as sleep current.");
  Serial.println("=== END TABLE 4.2 ===");
}

static void applyPrediction(const float probabilities[3], uint32_t latencyUs) {
  int predictedClass = 0;
  for (int category = 1; category < 3; ++category) {
    if (probabilities[category] > probabilities[predictedClass]) {
      predictedClass = category;
    }
  }
  const float confidence = probabilities[predictedClass] * 100.0f;

  if (predictedClass == 2) {
    setRgb(60, 0, 0);
    setAlarm(true);
  } else if (predictedClass == 1) {
    setRgb(0, 20, 50);
    setAlarm(false);
  } else {
    setRgb(0, 50, 0);
    setAlarm(false);
  }
  Serial.printf("Classe: %s, Confiance: %.2f%%, Latence: %lu us\n",
                kClassNames[predictedClass], confidence,
                static_cast<unsigned long>(latencyUs));
}

void setup() {
  Serial.begin(115200);
  const uint32_t serialWaitStart = millis();
  while (!Serial && millis() - serialWaitStart < 3000) {
    delay(10);
  }

  pinMode(kBuzzerPin, OUTPUT);
  digitalWrite(kBuzzerPin, LOW);
  setRgb(0, 0, 30);

  if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Invalid TFLite model or schema version mismatch.");
    setRgb(60, 0, 0);
    while (true) {
      delay(1000);
    }
  }
  if (resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk) {
    Serial.println("ERROR: Could not register required TFLite operators.");
    while (true) {
      delay(1000);
    }
  }

  freeHeapBeforeAllocation = ESP.getFreeHeap();
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: Tensor allocation failed; increase arena or inspect model.");
    setRgb(60, 0, 0);
    while (true) {
      delay(1000);
    }
  }
  freeHeapAfterAllocation = ESP.getFreeHeap();

  inputTensor = interpreter.input(0);
  outputTensor = interpreter.output(0);
  if (inputTensor == nullptr || outputTensor == nullptr ||
      inputTensor->type != kTfLiteInt8 || outputTensor->type != kTfLiteInt8 ||
      inputTensor->bytes < 6 ||
      outputTensor->bytes < 3 ||
      inputTensor->params.scale <= 0.0f || outputTensor->params.scale <= 0.0f) {
    Serial.println("ERROR: Model tensor types, shapes, or quantization are invalid.");
    setRgb(60, 0, 0);
    while (true) {
      delay(1000);
    }
  }

  Wire.begin(kSdaPin, kSclPin, 400000);
  Wire.setTimeOut(20);
  Wire.beginTransmission(kMpuAddress);
  Wire.write(kPowerManagementRegister);
  Wire.write(0x00);
  if (Wire.endTransmission() != 0) {
    Serial.println("ERROR: MPU9250 did not acknowledge wake-up.");
    setRgb(60, 0, 0);
    while (true) {
      delay(1000);
    }
  }
  Wire.beginTransmission(kMpuAddress);
  Wire.write(kAccelConfigRegister);
  Wire.write(0x00);
  if (Wire.endTransmission() != 0) {
    Serial.println("ERROR: Could not configure MPU9250 +/-2 g range.");
    setRgb(60, 0, 0);
    while (true) {
      delay(1000);
    }
  }

  printBenchmark();
  setRgb(0, 50, 0);
  Serial.println("READY_FOR_INFERENCE");
  nextSampleUs = micros();
}

void loop() {
  const int32_t remainingUs = static_cast<int32_t>(nextSampleUs - micros());
  if (remainingUs > 0) {
    delayMicroseconds(static_cast<uint32_t>(remainingUs));
  }

  float ax;
  float ay;
  float az;
  if (!readAcceleration(&ax, &ay, &az)) {
    Serial.println("ERROR: MPU9250 accelerometer read failed; sample skipped.");
    setRgb(60, 0, 0);
    setAlarm(false);
    nextSampleUs = micros() + kSamplePeriodUs;
    return;
  }

  axBuffer[writeIndex] = ax;
  ayBuffer[writeIndex] = ay;
  azBuffer[writeIndex] = az;
  writeIndex = (writeIndex + 1) % kWindowSize;
  if (samplesInWindow < kWindowSize) {
    ++samplesInWindow;
  }

  if (samplesInWindow == kWindowSize) {
    const WindowFeatures features = extractFeatures();
    float normalizedFeatures[6];
    for (int feature = 0; feature < 6; ++feature) {
      normalizedFeatures[feature] =
          (features.values[feature] - SCALER_MEAN[feature]) /
          SCALER_STD[feature];
    }

    float probabilities[3];
    uint32_t latencyUs = 0;
    if (!invokeModel(normalizedFeatures, probabilities, &latencyUs)) {
      Serial.println("ERROR: TFLite inference failed.");
      setRgb(60, 0, 0);
      setAlarm(false);
    } else {
      applyPrediction(probabilities, latencyUs);
    }
  }

  nextSampleUs += kSamplePeriodUs;
  if (static_cast<int32_t>(micros() - nextSampleUs) > 0) {
    nextSampleUs = micros();
  }
}
