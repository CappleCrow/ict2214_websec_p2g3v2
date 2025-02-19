import requests
import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier

# Load the trained models and preprocessing tools
xgb_model_path = "XGBoost_Anomaly_Model.pkl"
rf_model_path = "random_forest_api_key_model.pkl"
scaler_file_path = "scaler.pkl"
label_encoder_file_path = "label_encoder.pkl"

xgb_model = pickle.load(open(xgb_model_path, "rb"))
scaler = joblib.load(scaler_file_path)
label_encoders = joblib.load(label_encoder_file_path)
rf_model = joblib.load(rf_model_path)

# Flask API Gateway
app = Flask(__name__)

# Define categorical columns
categorical_cols = ["HTTP Method", "API Endpoint", "User-Agent", "Generalized API Endpoint", "Time of Day"]

# OpenAI API URL
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def preprocess_input(data):
    """ Preprocess incoming API request data """
    df = pd.DataFrame([data])

    # Encode categorical variables using pre-fitted LabelEncoders
    for col in categorical_cols:
        if col in df and col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])

    # Select the correct feature columns
    feature_columns = [
        "Rate Limiting", "Endpoint Entropy", "HTTP Method", "API Endpoint",
        "HTTP Status", "User-Agent", "Token Used", "Generalized API Endpoint", "Method_POST", "Time of Day"
    ]
    
    # Normalize numeric values
    df_scaled = scaler.transform(df[feature_columns])
    return df_scaled

def check_api_key(api_key):
    """ Check if the provided API key is valid using the Random Forest model """
    from keyrecognition import test_api_key  # Import function from keyrecognition.py
    return test_api_key(api_key, rf_model)

@app.route('/validate_openai_request', methods=['POST'])
def validate_openai_request():
    """ API Gateway Endpoint to analyze OpenAI requests """
    data = request.json  # Incoming OpenAI API request payload
    
    try:
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        key_validity = check_api_key(api_key)
        
        if key_validity != "Valid OpenAI":
            return jsonify({"status": "blocked", "reason": "Invalid API Key"}), 403
        
        # Extract request metadata for classification
        request_metadata = {
            "Rate Limiting": int(request.headers.get("x-ratelimit-remaining-requests", 100)),
            "Endpoint Entropy": np.random.uniform(0.1, 1.0),
            "HTTP Method": request.method,
            "API Endpoint": "/v1/chat/completions",
            "HTTP Status": 200,
            "User-Agent": request.headers.get("User-Agent", "Unknown"),
            "Token Used": np.random.randint(1, 1000),
            "Generalized API Endpoint": "/v1/chat",
            "Method_POST": 1 if request.method == "POST" else 0,
            "Time of Day": "Afternoon"
        }
        
        # Preprocess input for the AI model
        processed_data = preprocess_input(request_metadata)
        
        # Make prediction using the AI model
        predicted_class = xgb_model.predict(processed_data)[0]
        predicted_label = "Potential Misuse" if predicted_class == 1 else "Legitimate"
        
        # If request is suspicious, block it
        if predicted_label == "Potential Misuse":
            return jsonify({"status": "blocked", "reason": "Suspicious activity detected by AI model."}), 403
        
        # ✅ Otherwise, forward the request to OpenAI
        openai_headers = {
            "Authorization": request.headers.get("Authorization"),
            "Content-Type": "application/json",
            "User-Agent": request.headers.get("User-Agent", "Unknown"),
        }
        
        openai_response = requests.post(
            OPENAI_API_URL,
            headers=openai_headers,
            json=data
        )
        
        # Handle potential non-JSON responses
        if openai_response.status_code == 200:
            try:
                return jsonify(openai_response.json())
            except ValueError:
                return jsonify({"error": "OpenAI returned a non-JSON response", "response_text": openai_response.text}), 500
        else:
            return jsonify({"error": "OpenAI API returned an error", "status_code": openai_response.status_code, "response_text": openai_response.text}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
