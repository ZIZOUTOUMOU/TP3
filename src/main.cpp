#include "DHT.h"
#include <math.h>

#define DHTPIN 2
#define DHTTYPE DHT22
#define LED_PIN 8
#define BUZZER_PIN 9
DHT dht(DHTPIN, DHTTYPE);

// ---- Parameters exported from the trained model (Temperature, Humidity) ----
const int N_FEATURES = 2;
const float MEAN[N_FEATURES]    = {15.953115f, 48.574302f};
const float STD[N_FEATURES]     = {14.358038f, 8.821188f};
const float WEIGHTS[N_FEATURES] = {-0.208676f, 0.967978f};
const float BIAS = 0.969380f;
// ----------------------------------------------------------------------------

float standardize(float x_raw, int idx)
{
  return (x_raw - MEAN[idx]) / STD[idx];
}

float sigmoid(float z)
{
  return 1.0f / (1.0f + exp(-z));
}

float predict(float features[])
{
  float z = BIAS;
  for (int i = 0; i < N_FEATURES; i++)
  {
    z += WEIGHTS[i] * features[i];
  }
  return sigmoid(z);
}

void setup()
{
  Serial.begin(9600);
  Serial.println(F("Logistic Regression on Arduino Uno"));
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  dht.begin();
}

void loop()
{
  delay(2000);

  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (isnan(h) || isnan(t))
  {
    Serial.println(F("Failed to read from DHT sensor!"));
    return;
  }

  // Feature 0 -> Temperature, Feature 1 -> Humidity
  float x_scaled[N_FEATURES];
  x_scaled[0] = standardize(t, 0);
  x_scaled[1] = standardize(h, 1);

  float y_pred = predict(x_scaled);
  bool fire = (y_pred >= 0.5f);

  // Alarm outputs
  if (fire)
  {
    digitalWrite(LED_PIN, HIGH);
    tone(BUZZER_PIN, 1000);   // 1 kHz beep until the next reading
  }
  else
  {
    digitalWrite(LED_PIN, LOW);
    noTone(BUZZER_PIN);
  }

  Serial.print(F("Temperature: "));
  Serial.print(t);
  Serial.print(F(" C  Humidity: "));
  Serial.print(h);
  Serial.println(F(" %"));

  Serial.print(F("Predicted Probability: "));
  Serial.println(y_pred, 4);

  Serial.print(F("Predicted Class: "));
  Serial.println(fire ? 1 : 0);
  Serial.println();
}