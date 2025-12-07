# test_api.py
# Script to test the Flask prediction API and local model loading
# Usage:
#   python test_api.py --api   # to test the running Flask API
#   python test_api.py --local # to test local model/scaler loading

import argparse
import json
import sys

# For API testing
import requests
# For local testing
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import load_model
import joblib

API_URL = 'http://127.0.0.1:5000/predict'
MODEL_PATH = 'model.h5'
SCALER_PATH = 'scaler.pkl'


def test_api():
    """
    Send a sample request to the Flask API and print the response.
    """
    # Load sample data
    data = load_breast_cancer()
    X_sample = data.data[0].tolist()  # first sample
    payload = {'features': X_sample}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        print("API Response:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error calling API at {API_URL}:", e)


def test_local():
    """
    Load the model and scaler locally and run prediction on a sample.
    """
    # Load model and scaler
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Load sample data
    data = load_breast_cancer()
    X_sample = data.data[1].reshape(1, -1)  # second sample
    y_true = data.target[1]

    # Scale and predict
    X_scaled = scaler.transform(X_sample)
    prob = model.predict(X_scaled)[0][0]
    pred = int(prob > 0.5)

    print("Local Test:")
    print(f"True label: {y_true}")
    print(f"Predicted: {pred}, Probability: {prob:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the breast cancer model via API or locally.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--api', action='store_true', help='Test the running Flask API')
    group.add_argument('--local', action='store_true', help='Test local model loading')
    args = parser.parse_args()

    if args.api:
        test_api()
    elif args.local:
        test_local()
