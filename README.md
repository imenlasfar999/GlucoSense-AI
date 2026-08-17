# GlucoSense-AI
AI-enhanced non-invasive glucose prediction prototype using embedded sensors, machine learning, and a Streamlit dashboard.
# GlucoSense AI

GlucoSense AI is an embedded AI-based research prototype for non-invasive glucose prediction using breath biomarkers, physiological signals, environmental data, and machine learning.

The system combines ESP32-based sensor acquisition, Python serial data logging, an XGBoost regression model, a Streamlit dashboard, and an AI-assisted interpretation report using Gemini.

## Project Objective

The objective of this project is to investigate whether glucose levels can be estimated non-invasively using breath sensor responses, physiological signals, environmental parameters, and machine learning.

This project is developed for academic research and educational purposes. It is not a certified medical device and does not replace a clinical glucometer or professional medical advice.

## Main Components

- ESP32 WROOM-32
- MQ138 gas sensor
- MQ3 gas sensor
- MQ6 gas sensor
- MAX30102 PPG sensor
- DHT22 temperature and humidity sensor
- Python
- XGBoost
- Streamlit
- Gemini AI Agent

## Project Structure

```text
GlucoSense-AI/
├── hardware_esp32/
│   └── esp32_sensor_acquisition.ino
├── data_logger/
│   └── serial_data_logger.py
├── dashboard/
│   └── app.py
├── model/
│   ├── xgboost_glucose_model.pkl
│   └── model_features.pkl
├── sample_data/
│   └── latest_sensor_data_sample.csv
├── reports/
│   └── ai_agent_report_example.pdf
├── requirements.txt
└── README.md
