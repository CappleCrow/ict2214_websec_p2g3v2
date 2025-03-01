import requests
import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb
from flask import Flask, request, jsonify
from keyrecognition import test_api_key  # Import updated function
from pathlib import Path
import tiktoken
import re
import datetime

MODEL_DIR = Path(__file__).parent

model_paths = {
    'xgboost': str(MODEL_DIR / 'XGBoost_Anomaly_Model.pkl'),
    'scaler': MODEL_DIR / 'scaler.pkl',
    'label_encoder': MODEL_DIR / 'label_encoder.pkl',
    'rf_model': MODEL_DIR / 'random_forest_api_key_model_v1.5.0.pkl',
    'api_keys': MODEL_DIR / 'api_keys_dataset.csv'  # Add the path to the API keys dataset
}

# Load the trained models and preprocessing tools
xgb_model = pickle.load(open(model_paths['xgboost'], 'rb'))
scaler = joblib.load(model_paths['scaler'])
label_encoders = joblib.load(model_paths['label_encoder'])
rf_model = joblib.load(model_paths['rf_model'])

# Load the API key dataset (assuming it's in CSV format)
api_key_data = pd.read_csv(model_paths['api_keys'])
api_key_dict = dict(zip(api_key_data['key'], api_key_data['label']))

# Flask API Gateway
app = Flask(__name__)

# Define categorical columns
categorical_cols = ["HTTP Method", "API Endpoint", "User-Agent", "Time of Day"]

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
        "HTTP Status", "User-Agent", "Token Used", "Method_POST", "Time of Day"
    ]
    
    # Normalize numeric values
    df_scaled = scaler.transform(df[feature_columns])
    return df_scaled

def calculate_tokens(messages):
    """ Calculate total tokens per message """
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        num_tokens = 0

        for message in messages:
            num_tokens += 4 

            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name": # if there is name role is omitted
                    num_tokens -= 1
        
        num_tokens += 2
        return num_tokens
    except Exception as e:
        print(f"Cannot calculate tokens: {str(e)}")
        return 0

def validate_api_key(api_key):
    """ Validate the API key using the loaded model and dataset """
    if api_key in api_key_dict:
        return api_key_dict[api_key]
    else:
        return "Invalid API Key"
    
def categorize_time_of_day():
    """Categorizes the time of day based on the current server time"""
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        return "Morning"
    elif 12 <= current_hour < 18:
        return "Afternoon"
    else:
        return "Night"

@app.route('/validate_openai_request', methods=['POST'])
def validate_openai_request():
    """ API Gateway Endpoint to analyze OpenAI requests """
    data = request.json
    
    try:
        # Check API Key Validity
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        key_validity = validate_api_key(api_key)
        
        if key_validity != "Valid OpenAI":
            return jsonify({"status": "blocked", "reason": "Invalid API Key"}), 403
        
        messages = data.get("messages", [])
        total_tokens = calculate_tokens(messages)

        # Determine Time of Day based on server's system time
        time_of_day = categorize_time_of_day()

        # Extract request metadata for classification
        request_metadata = {
            "Rate Limiting": int(request.headers.get("x-ratelimit-remaining-requests", 100)),
            "Endpoint Entropy": np.random.uniform(0.1, 1.0),
            "HTTP Method": request.method,
            "API Endpoint": "/v1/chat/completions",  # Legit endpoint
            "HTTP Status": 200,
            "User-Agent": request.headers.get("User-Agent", "Unknown"),
            "Token Used": data.get("max_tokens", 0),
            "Method_POST": 1 if request.method == "POST" else 0,
            "Time of Day": time_of_day
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
            return jsonify(openai_response.json())
        else:
            return jsonify({
                "error": "OpenAI API returned an error",
                "status_code": openai_response.status_code,
                "response_text": openai_response.text
            }), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
