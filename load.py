import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask import Flask, request, jsonify

# Load the trained model and preprocessing tools
model_file_path = "XGBoost_Anomaly_Model.pkl"
scaler_file_path = "scaler.pkl"
label_encoder_file_path = "label_encoder.pkl"

xgb_model = joblib.load(model_file_path)
scaler = joblib.load(scaler_file_path)
label_encoders = joblib.load(label_encoder_file_path)  # This contains LabelEncoders for categorical columns

print("✅ Model and preprocessing tools loaded successfully!")

# Initialize Flask API
app = Flask(__name__)

# Define categorical columns that need encoding
categorical_cols = ["HTTP Method", "API Endpoint", "User-Agent"]

def preprocess_input(data):
    """ Preprocess incoming API request data """
    df = pd.DataFrame([data])

    # Encode categorical variables using pre-fitted LabelEncoders
    for col in categorical_cols:
        if col in df:
            df[col] = label_encoders[col].transform(df[col])

    # Select the correct feature columns
    feature_columns = [
        "Rate Limiting", "Endpoint Entropy", "HTTP Method", "API Endpoint",
        "HTTP Status", "User-Agent", "Token Used"
    ]
    
    # Normalize numeric values
    df_scaled = scaler.transform(df[feature_columns])

    return df_scaled

@app.route('/predict', methods=['POST'])
def predict():
    """ API endpoint to classify an API request as legitimate or misuse """
    data = request.json

    try:
        processed_data = preprocess_input(data)  # Preprocess input

        # Make prediction
        predicted_class = xgb_model.predict(processed_data)[0]

        # Convert back to human-readable label
        predicted_label = "Potential Misuse" if predicted_class == 1 else "Legitimate"

        return jsonify({"prediction": predicted_label})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
