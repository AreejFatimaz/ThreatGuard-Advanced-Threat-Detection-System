# src/main.py
from data_preprocessing import preprocess_data
from feature_extraction import extract_features
from model_training import train_model
from anomaly_detection import detect_anomalies

def main():
    # Load and preprocess data
    df = preprocess_data("data/CICIDS_2017.csv")
    df.to_csv("data/processed_data.csv", index=False)
    
    # Extract features
    features, labels = extract_features(df)
    
    # Train model
    train_model(features, labels)
    
    # Detect anomalies
    anomalies = detect_anomalies(df)
    print(anomalies)

if __name__ == "__main__":
    main()

