/*
 * FINAL DATASET COLLECTION — ESP32
 * Sensors: MQ3 + MQ6 + MQ138 + MAX30102 + DHT22
 * MQ sensors powered by 5V with voltage dividers
 */

#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "DHT.h"

// Pins
#define MQ3_PIN   34
#define MQ6_PIN   35
#define MQ138_PIN 32
#define DHT_PIN   23
#define DHT_TYPE  DHT22

// MQ constants
const float RL           = 10000.0;
const float MQ_VCC       = 5.0;
const float ESP_ADC_VREF = 3.3;
const float DIV_RATIO    = 1.5;

// Objects
MAX30105 sensor;
DHT dht(DHT_PIN, DHT_TYPE);

// State
bool gasWarmupDone  = false;
bool fingerDetected = false;
bool pulseMeasuring = false;
bool breathPhase    = false;
bool sessionDone    = false;
bool waitingEnter   = false;

unsigned long gasStart   = 0;
unsigned long fingerTime = 0;
unsigned long lastPrint  = 0;

int personNumber = 1;

// MAX30102
float sumBPM = 0, sumSpO2 = 0, sumIR = 0, sumRed = 0;
int pulseCount = 0;

const byte RATE_SIZE = 6;
byte rates[RATE_SIZE];
byte rateSpot = 0;
byte validRates = 0;
long lastBeat = 0;
float beatAvg = 0;
float spo2Avg = 95.0;

float dcIR = 0, dcRed = 0;
const float alpha = 0.95;

// MQ data
float mq3_min = 9999999.0, mq3_sum = 0;
float mq6_min = 9999999.0, mq6_sum = 0;
float mq138_min = 9999999.0, mq138_sum = 0;

int mq3_count = 0;
int mq6_count = 0;
int mq138_count = 0;

// DHT22
float roomTemp = 0;
float roomHum  = 0;

// Read MQ sensor Rs
float readRs(int pin) {
  long sum = 0;

  for (int i = 0; i < 10; i++) {
    sum += analogRead(pin);
    delay(5);
  }

  float adc = sum / 10.0;

  float Vadc = adc * (ESP_ADC_VREF / 4095.0);
  float Vout = Vadc * DIV_RATIO;

  if (Vout < 0.05 || Vout >= MQ_VCC) return -1;

  return RL * (MQ_VCC - Vout) / Vout;
}

// Reset for next person
void resetSession() {
  fingerDetected = false;
  pulseMeasuring = false;
  breathPhase    = false;
  sessionDone    = false;
  waitingEnter   = false;

  sumBPM = sumSpO2 = sumIR = sumRed = 0;
  pulseCount = 0;

  mq3_min = 9999999.0;
  mq6_min = 9999999.0;
  mq138_min = 9999999.0;

  mq3_sum = mq6_sum = mq138_sum = 0;
  mq3_count = mq6_count = mq138_count = 0;

  roomTemp = 0;
  roomHum = 0;

  rateSpot = 0;
  validRates = 0;
lastBeat = 0;
  beatAvg = 0;
  spo2Avg = 95.0;
  dcIR = 0;
  dcRed = 0;

  for (byte i = 0; i < RATE_SIZE; i++) {
    rates[i] = 0;
  }
}

// Process MAX30102
void processPulse(long ir, long red) {
  dcIR  = alpha * dcIR  + (1 - alpha) * (float)ir;
  dcRed = alpha * dcRed + (1 - alpha) * (float)red;

  float acIR  = ir  - dcIR;
  float acRed = red - dcRed;

  if (checkForBeat(ir)) {
    long delta = millis() - lastBeat;
    lastBeat = millis();

    if (delta > 400 && delta < 1800) {
      float bpm = 60000.0 / delta;

      if (bpm > 50 && bpm < 150) {
        rates[rateSpot++] = (byte)bpm;
        rateSpot %= RATE_SIZE;

        if (validRates < RATE_SIZE) validRates++;

beatAvg = 0;
for (byte i = 0; i < validRates; i++) {
  beatAvg += rates[i];
}
beatAvg /= validRates;
      }
    }
  }

  // Safe SpO2 calculation
  if (dcIR > 0 && dcRed > 0 && abs(acIR) > 1) {
    float ratio = (acRed / dcRed) / (acIR / dcIR);
    float spo2 = -45.060 * ratio * ratio + 30.354 * ratio + 94.845;

   if (spo2 > 85 && spo2 < 102) {
  spo2Avg = spo2Avg * 0.85 + spo2 * 0.15;  // ✅ smoothing
}
  }

  if (pulseMeasuring) {

  if (beatAvg >= 50 && beatAvg <= 150) {
    sumBPM += beatAvg;
  }

    sumSpO2 += spo2Avg;
    sumIR   += ir;
    sumRed  += red;
    pulseCount++;
  }
}

void setup() {
  Serial.begin(115200);

  analogSetAttenuation(ADC_11db);
  analogSetWidth(12);

  dht.begin();
  delay(2000);

  Serial.println("=============================================");
  Serial.println(" FINAL DATASET COLLECTION — ESP32");
  Serial.println(" MQ3 + MQ6 + MQ138 + MAX30102 + DHT22");
  Serial.println(" MQ sensors: 5V + voltage divider");
  Serial.println("=============================================");
  Serial.println("Gas sensors warming up — 5 minutes.");
  Serial.println("For final dataset, 20–30 minutes is better.");
  Serial.println("---------------------------------------------");

  gasStart = millis();

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found. Check wiring.");
    while (1);
  }

  sensor.setup(30, 4, 2, 200, 411, 4096);
}

void loop() {

  // Gas warmup
  if (!gasWarmupDone) {
    if (millis() - lastPrint >= 30000) {
      lastPrint = millis();

      unsigned long elapsed = millis() - gasStart;
      unsigned long rem = 300000UL > elapsed ? (300000UL - elapsed) / 1000 : 0;

      Serial.print("Warmup remaining: ");
      Serial.print(rem / 60);
      Serial.print("m ");
      Serial.print(rem % 60);
      Serial.println("s");
    }

    if (millis() - gasStart >= 300000UL) {
      gasWarmupDone = true;
      Serial.println("=============================================");
      Serial.println("WARMUP DONE.");
      Serial.println("Place finger on MAX30102 to begin.");
      Serial.println("=============================================");
    }

    return;
  }

  long ir  = sensor.getIR();
  long red = sensor.getRed();

  unsigned long t = fingerDetected ? millis() - fingerTime : 0;

  // Finger detected
  if (!fingerDetected && ir > 50000) {
    fingerDetected = true;
    fingerTime = millis();

    Serial.println("---------------------------------------------");
    Serial.print("Person ");
    Serial.print(personNumber);
    Serial.println(" — Finger detected.");
    Serial.println("0–25s: stabilization. Do NOT breathe yet.");
    Serial.println("---------------------------------------------");
  }

  // Finger removed
  if (ir < 50000 && fingerDetected) {
    if (!sessionDone) {
      Serial.println("WARNING: Finger removed. Session reset.");
    }
    resetSession();
    return;
  }

  if (!fingerDetected) return;

  // 0–25s stabilization
  if (t < 25000) {
    processPulse(ir, red);

    if (millis() - lastPrint > 5000) {
      lastPrint = millis();
      Serial.print("Stabilizing: ");
      Serial.print(t / 1000);
      Serial.println("/25s");
    }

    return;
  }

  // 25–50s MAX30102 measuring
  if (t >= 25000 && t < 50000) {
    if (!pulseMeasuring) {
      pulseMeasuring = true;
      Serial.println("---------------------------------------------");
      Serial.println("25–50s: MAX30102 measuring.");
      Serial.println("40–50s: breathe into MQ3 + MQ6 + MQ138.");
      Serial.println("---------------------------------------------");
    }

    // 40–50s breath phase
    if (t >= 40000) {
      if (!breathPhase) {
        breathPhase = true;
        Serial.println("=============================================");
        Serial.println("BREATHE NOW into MQ3 + MQ6 + MQ138");
        Serial.println("10 seconds continuously.");
        Serial.println("=============================================");
      }

      float rs3   = readRs(MQ3_PIN);
      float rs6   = readRs(MQ6_PIN);
      float rs138 = readRs(MQ138_PIN);

      if (rs3 > 0) {
        if (rs3 < mq3_min) mq3_min = rs3;
        mq3_sum += rs3;
        mq3_count++;
      }

      if (rs6 > 0) {
        if (rs6 < mq6_min) mq6_min = rs6;
        mq6_sum += rs6;
        mq6_count++;
      }

      if (rs138 > 0) {
        if (rs138 < mq138_min) mq138_min = rs138;
        mq138_sum += rs138;
        mq138_count++;
      }

      if (millis() - lastPrint > 2000) {
        lastPrint = millis();
        Serial.print("Breath ");
        Serial.print((t - 40000) / 1000);
        Serial.print("/10s | MQ3=");
        Serial.print(rs3, 0);
        Serial.print(" | MQ6=");
        Serial.print(rs6, 0);
        Serial.print(" | MQ138=");
        Serial.println(rs138, 0);
      }

    } else {
      if (millis() - lastPrint > 3000) {
        lastPrint = millis();
        Serial.print("t=");
        Serial.print(t / 1000);
        Serial.print("s | BPM=");
        Serial.print(beatAvg, 0);
        Serial.print(" | SpO2=");
        Serial.print(spo2Avg, 1);
        Serial.print(" | IR=");
        Serial.print(ir);
        Serial.print(" | RED=");
        Serial.println(red);
      }
    }

    processPulse(ir, red);
    return;
  }

  // Session complete
  if (!sessionDone) {
    sessionDone = true;
    waitingEnter = true;

    roomTemp = dht.readTemperature();
    roomHum  = dht.readHumidity();

    if (isnan(roomTemp) || isnan(roomHum)) {
      Serial.println("WARNING: DHT22 read failed.");
      roomTemp = 0;
      roomHum = 0;
    }

    float avgBPM   = pulseCount > 0 ? sumBPM / pulseCount : 0;
    float avgSpO2  = pulseCount > 0 ? sumSpO2 / pulseCount : 0;
    float avgIR    = pulseCount > 0 ? sumIR / pulseCount : 0;
    float avgRed   = pulseCount > 0 ? sumRed / pulseCount : 0;

    float avgMQ3   = mq3_count > 0 ? mq3_sum / mq3_count : 0;
    float avgMQ6   = mq6_count > 0 ? mq6_sum / mq6_count : 0;
    float avgMQ138 = mq138_count > 0 ? mq138_sum / mq138_count : 0;

    Serial.println("=============================================");
    Serial.print("PERSON ");
    Serial.print(personNumber);
    Serial.println(" — SESSION COMPLETE");
    Serial.println("=============================================");

    Serial.println("--- MAX30102 avg over 25s ---");
    Serial.print("IR:             "); Serial.println(avgIR, 0);
    Serial.print("RED:            "); Serial.println(avgRed, 0);
    Serial.print("BPM:            "); Serial.println(avgBPM, 1);
    Serial.print("SpO2:           "); Serial.print(avgSpO2, 1); Serial.println(" %");

    Serial.println("--- Gas sensors avg over 10s ---");
    Serial.print("MQ3 MinRs:      "); Serial.print(mq3_min, 1); Serial.println(" ohm");
    Serial.print("MQ3 AvgRs:      "); Serial.print(avgMQ3, 1); Serial.println(" ohm");
    Serial.print("MQ6 MinRs:      "); Serial.print(mq6_min, 1); Serial.println(" ohm");
    Serial.print("MQ6 AvgRs:      "); Serial.print(avgMQ6, 1); Serial.println(" ohm");
    Serial.print("MQ138 MinRs:    "); Serial.print(mq138_min, 1); Serial.println(" ohm");
    Serial.print("MQ138 AvgRs:    "); Serial.print(avgMQ138, 1); Serial.println(" ohm");

    Serial.println("--- DHT22 ---");
    Serial.print("Temperature:    "); Serial.print(roomTemp, 1); Serial.println(" C");
    Serial.print("Humidity:       "); Serial.print(roomHum, 1); Serial.println(" %");

    Serial.println("---------------------------------------------");
    Serial.println("CSV ROW:");
    Serial.println("IR,RED,BPM,SpO2,MQ3_MinRs,MQ3_AvgRs,MQ6_MinRs,MQ6_AvgRs,MQ138_MinRs,MQ138_AvgRs,Temperature_C,Humidity_percent,Glucose_mg_dL");

    Serial.print(avgIR, 0);     Serial.print(",");
    Serial.print(avgRed, 0);    Serial.print(",");
    Serial.print(avgBPM, 1);    Serial.print(",");
    Serial.print(avgSpO2, 1);   Serial.print(",");
    Serial.print(mq3_min, 1);   Serial.print(",");
    Serial.print(avgMQ3, 1);    Serial.print(",");
    Serial.print(mq6_min, 1);   Serial.print(",");
    Serial.print(avgMQ6, 1);    Serial.print(",");
    Serial.print(mq138_min, 1); Serial.print(",");
    Serial.print(avgMQ138, 1);  Serial.print(",");
    Serial.print(roomTemp, 1);  Serial.print(",");
    Serial.print(roomHum, 1);   Serial.print(",");
    Serial.println("___GLUCOSE___");

    Serial.println("---------------------------------------------");
    Serial.println("Write glucose from glucometer, then press ENTER.");
    Serial.println("=============================================");
  }

  // Next person
  if (waitingEnter && Serial.available()) {
    while (Serial.available() > 0) Serial.read();

    Serial.println("Sensor recovery — 30 seconds...");
    delay(30000);

    resetSession();
    personNumber++;

    Serial.println("=============================================");
    Serial.print("READY FOR PERSON ");
    Serial.println(personNumber);
    Serial.println("Place finger on MAX30102.");
    Serial.println("=============================================");
  }
}