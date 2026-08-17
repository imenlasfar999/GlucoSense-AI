import serial
import pandas as pd
import time

PORT = "COM3"
BAUD = 115200
OUTPUT_FILE = "latest_sensor_data.csv"

print("Starting ESP32 serial logger...")
print("Make sure Arduino Serial Monitor is CLOSED.")
print(f"Listening on {PORT}...")

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

header = None

while True:
    line = ser.readline().decode(errors="ignore").strip()

    if not line:
        continue

    print(line)

    if line.startswith("IR,RED,BPM"):
        header = line.split(",")

    elif header is not None and line[0].isdigit():
        values = line.split(",")

        if len(values) == len(header):
            df = pd.DataFrame([values], columns=header)
            df = df.replace("___GLUCOSE___", 0)

            df.to_csv(OUTPUT_FILE, index=False)

            print("\n✅ New sensor row saved to latest_sensor_data.csv\n")