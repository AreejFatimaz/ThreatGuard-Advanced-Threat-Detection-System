# src/model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    print(f"Model Accuracy: {model.score(X_test, y_test)}")

if __name__ == "__main__":
    features = pd.read_csv("data/features.csv")
    labels = pd.read_csv("data/labels.csv")
    train_model(features, labels)
