from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")
feature_cols = joblib.load("features.pkl")

def prepare_features(df):
    df['Amount_Log'] = np.log1p(df['Amount'])
    df['V1_V2'] = df['V1'] * df['V2']
    df['V3_V4'] = df['V3'] * df['V4']
    df['V5_V6'] = df['V5'] * df['V6']
    df['V7_V8'] = df['V7'] * df['V8']
    df['V9_V10'] = df['V9'] * df['V10']
    df['magnitude'] = np.sqrt((df[['V1','V2','V3','V4','V5']]**2).sum(axis=1))
    return df[feature_cols]

@app.route("/")
def home():
    return "Fraud Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    df = pd.DataFrame([data])

    features = prepare_features(df)
    scaled = scaler.transform(features)

    prob = model.predict_proba(scaled)[0][1]

    prediction = "FRAUD" if prob > threshold else "NORMAL"

    return jsonify({
        "prediction": prediction,
        "probability": float(prob)
    })
