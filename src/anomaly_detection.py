# src/anomaly_detection.py
import pandas as pd
import joblib

def detect_anomalies(df):
    model = joblib.load('model.pkl')
    features = df[['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets']]
    predictions = model.predict(features)
    anomalies = df[predictions == 1]  # Assuming 1 indicates an anomaly
    return anomalies

if __name__ == "__main__":
    df = pd.read_csv("data/processed_data.csv")
    anomalies = detect_anomalies(df)
    print(anomalies)
