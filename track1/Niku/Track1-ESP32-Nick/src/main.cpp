
#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

// =================================
// TESA TOP GUN 2026 — TRACK 1
// =================================

// Replace XX with your team number
const char* TEAM_CODE = "TGR-01";

// ESP32-S3 pins
const int CHARGE_PIN = 4;
const int COMP_PIN   = 5;
const int BUTTON_PIN = 6;

// Built-in RGB LED
// Original DevKitC-1: GPIO48
// DevKitC-1 v1.1: GPIO38
const int RGB_PIN = 48;

Adafruit_NeoPixel led(
    1, RGB_PIN, NEO_GRB + NEO_KHZ800
);

// Measurement settings
const int SAMPLES = 512;

// Current calibration
const float SLOPE = 0.52807f;
const float OFFSET = 7.51012f;

// Interrupt variables
volatile bool detected = false;
volatile uint32_t stopTime = 0;

// =================================
// Comparator interrupt
// =================================

void IRAM_ATTR comparatorISR() {
    if (!detected) {
        stopTime = micros();
        detected = true;
    }
}

// =================================
// Single measurement
// =================================

bool measureOnce(float &elapsed) {

    // Discharge capacitor
    digitalWrite(CHARGE_PIN, LOW);
    delayMicroseconds(1000);

    if (digitalRead(COMP_PIN) != HIGH) {
        return false;
    }

    // Reset interrupt data
    noInterrupts();
    detected = false;
    stopTime = 0;
    interrupts();

    // Begin charging
    uint32_t startTime = micros();
    digitalWrite(CHARGE_PIN, HIGH);

    // Wait for comparator
    while (!detected &&
           (uint32_t)(micros() - startTime) < 500) {
    }

    // Capture result
    noInterrupts();
    bool success = detected;
    uint32_t endTime = stopTime;
    interrupts();

    digitalWrite(CHARGE_PIN, LOW);

    if (!success) {
        return false;
    }

    elapsed = (float)(uint32_t)(
        endTime - startTime
    );

    return true;
}

// =================================
// Calculate capacitance
// =================================

float measureCapacitance() {

    float totalTime = 0;
    int validSamples = 0;

    for (int i = 0; i < SAMPLES; i++) {
        float elapsed;

        if (measureOnce(elapsed)) {
            totalTime += elapsed;
            validSamples++;
        }
    }

    // Reject incomplete measurement batches
    if (validSamples < SAMPLES) {
        return 0.0f;
    }

    float avgTime = totalTime / validSamples;

    float capacitance =
        (avgTime - OFFSET) / SLOPE;

    return max(0.0f, capacitance);
}

// =================================
// RGB LED control
// =================================

void setLED(bool on) {

    if (on) {
        led.setPixelColor(
            0, led.Color(0, 30, 0)
        );
    } else {
        led.clear();
    }

    led.show();
}

// =================================
// Wait for stable button release
// =================================

void waitForRelease() {

    while (true) {

        while (digitalRead(BUTTON_PIN) == LOW) {
            delay(1);
        }

        delay(30);

        if (digitalRead(BUTTON_PIN) == HIGH) {
            return;
        }
    }
}

// =================================
// Setup
// =================================

void setup() {

    Serial.begin(9600);

    pinMode(CHARGE_PIN, OUTPUT);
    digitalWrite(CHARGE_PIN, LOW);

    pinMode(COMP_PIN, INPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    led.begin();
    setLED(false);

    attachInterrupt(
        digitalPinToInterrupt(COMP_PIN),
        comparatorISR,
        FALLING
    );
}

// =================================
// Main loop
// =================================

void loop() {

    // Wait for button press
    if (digitalRead(BUTTON_PIN) != LOW) {
        delay(1);
        return;
    }

    // Debounce press
    delay(30);

    if (digitalRead(BUTTON_PIN) != LOW) {
        return;
    }

    // LED on for 200 ms
    setLED(true);
    delay(200);
    setLED(false);

    // Require one complete press-release
    waitForRelease();

    // Take measurement
    float capacitance = measureCapacitance();

    // Official competition output
    Serial.println(TEAM_CODE);

    Serial.print("Capacitance = ");
    Serial.print(capacitance, 2);
    Serial.println(" pF");

    Serial.println();
}
