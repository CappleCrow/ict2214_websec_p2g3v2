from flask import Flask, render_template, request, session, jsonify
import os
import requests
from reportlab.pdfgen import canvas
import anthropic  # Import the Anthropic package
import cohere     # Import the Cohere package
import asyncio
import fastapi_poe as fp
import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb
from pathlib import Path
import tiktoken
import datetime
import re
from faker import Faker
import random

# -------------------- Model and Preprocessing Setup --------------------
MODEL_DIR = Path(__file__).parent
model_paths = {
    'xgboost': str(MODEL_DIR / 'XGBoost_Anomaly_Model.pkl'),
    'scaler': MODEL_DIR / 'scaler.pkl',
    'label_encoder': MODEL_DIR / 'label_encoder.pkl',
    'rf_model': MODEL_DIR / 'random_forest_api_key_model_v1.5.0.pkl'
}

# Load anomaly detection models and preprocessing tools
xgb_model = pickle.load(open(model_paths['xgboost'], 'rb'))
scaler = joblib.load(model_paths['scaler'])
label_encoders = joblib.load(model_paths['label_encoder'])
rf_model = joblib.load(model_paths['rf_model'])

# -------------------- Functions for Request Metadata --------------------
categorical_cols = ["HTTP Method", "API Endpoint", "User-Agent", "Time of Day"]

def preprocess_input(data):
    """Preprocess API request metadata into a scaled feature array for anomaly detection."""
    df = pd.DataFrame([data])
    for col in categorical_cols:
        if col in df and col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
    feature_columns = [
        "Rate Limiting", "Endpoint Entropy", "HTTP Method", "API Endpoint",
        "HTTP Status", "User-Agent", "Token Used", "Method_POST", "Time of Day"
    ]
    df_scaled = scaler.transform(df[feature_columns])
    return df_scaled

def calculate_tokens(messages):
    """Calculate total tokens for a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens -= 1
        num_tokens += 2
        return num_tokens
    except Exception as e:
        print(f"Cannot calculate tokens: {str(e)}")
        return 0

def categorize_time_of_day():
    """Categorize the time of day based on the current server time."""
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        return "Morning"
    elif 12 <= current_hour < 18:
        return "Afternoon"
    else:
        return "Night"

# -------------------- API Key Feature Extraction and Validation --------------------
def extract_api_key_features(api_key):
    """
    Extract 10 features from the API key:
      1. length: Length of the key.
      2. starts_with_sk_proj: 1 if key starts with 'sk-proj-', else 0.
      3. starts_with_ant: 1 if key starts with 'sk-ant-api03-', else 0.
      4. digit_ratio: Number of digits divided by length.
      5. uppercase_ratio: Number of uppercase letters divided by length.
      6. non_alphanumeric_count: Count of characters that are not alphanumeric and not '-' or '_'.
      7. matches_openai_format: 1 if key matches OpenAI format, else 0.
      8. matches_anthropic_format: 1 if key matches Anthropic format, else 0.
      9. matches_cohere_format: 1 if key matches Cohere format, else 0.
      10. matches_poe_format: 1 if key matches Poe format, else 0.
    Returns a pandas DataFrame with the features in the same order as during model training.
    """
    length = len(api_key)
    starts_with_sk_proj = int(api_key.startswith('sk-proj-'))
    starts_with_ant = int(api_key.startswith('sk-ant-api03-'))
    digit_ratio = sum(c.isdigit() for c in api_key) / length if length > 0 else 0
    uppercase_ratio = sum(c.isupper() for c in api_key) / length if length > 0 else 0
    non_alphanumeric_count = sum((not c.isalnum()) and (c not in "-_") for c in api_key)
    matches_openai_format = int(bool(re.match(r'^sk-proj-([A-Za-z0-9]{10,20}-){4}[A-Za-z0-9]{10,20}$', api_key)))
    matches_anthropic_format = int(bool(re.match(r'^sk-ant-api03-[A-Za-z0-9_-]{95}$', api_key)))
    matches_cohere_format = int(bool(re.match(r'^[A-Za-z0-9]{40}$', api_key)))
    matches_poe_format = int(bool(re.match(r'^[A-Za-z0-9_]{43}$', api_key)))
    features = {
        'length': length,
        'starts_with_sk_proj': starts_with_sk_proj,
        'starts_with_ant': starts_with_ant,
        'digit_ratio': digit_ratio,
        'uppercase_ratio': uppercase_ratio,
        'non_alphanumeric_count': non_alphanumeric_count,
        'matches_openai_format': matches_openai_format,
        'matches_anthropic_format': matches_anthropic_format,
        'matches_cohere_format': matches_cohere_format,
        'matches_poe_format': matches_poe_format,
    }
    columns_order = ['length', 'starts_with_sk_proj', 'starts_with_ant',
                     'digit_ratio', 'uppercase_ratio', 'non_alphanumeric_count',
                     'matches_openai_format', 'matches_anthropic_format', 'matches_cohere_format',
                     'matches_poe_format']
    return pd.DataFrame([features])[columns_order]

def test_api_key(key, model):
    """
    Extract features from the key and use the provided model to predict its label.
    Returns the prediction (e.g., 1 for OpenAI, 2 for Cohere, 3 for Anthropic, 4 for Poe).
    """
    input_features = extract_api_key_features(key)
    prediction = model.predict(input_features)[0]
    return prediction
# -------------------- Logging Setup --------------------
def log_api_request(api_key, request_metadata):
    """Log the API request details to a CSV file using actual IP and User-Agent."""
    log_file = "api_requests_log.csv"
    # Use actual IP from the request
    actual_ip = request.remote_addr or "N/A"
    # Use actual User-Agent from the request headers
    actual_user_agent = request.headers.get("User-Agent", "Unknown")
    new_time_of_access = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    masked_api_key = api_key[:8] + "*****" if api_key else "N/A"
    
    log_data = {
        "Rate Limiting": request_metadata.get("Rate Limiting"),
        "Endpoint Entropy": request_metadata.get("Endpoint Entropy"),
        "HTTP Method": request_metadata.get("HTTP Method"),
        "API Endpoint": request_metadata.get("API Endpoint"),
        "HTTP Status": request_metadata.get("HTTP Status"),
        "User-Agent": actual_user_agent,
        "Token Used": request_metadata.get("Token Used"),
        "Method_POST": request_metadata.get("Method_POST"),
        "Time of Day": request_metadata.get("Time of Day"),
        "API_Key": masked_api_key,
        "New Time of Access": new_time_of_access,
        "IP": actual_ip
    }
    df = pd.DataFrame([log_data])
    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False)
    else:
        df.to_csv(log_file, mode='a', header=False, index=False)

# -------------------- Flask App Setup --------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"

def call_cohere_api(api_key, messages):
    co = cohere.ClientV2(api_key=api_key)
    filtered_messages = [msg for msg in messages if msg.get("content", "").strip() != ""]
    res = co.chat(model="command-r-plus-08-2024", messages=filtered_messages)
    return "".join([item.text for item in res.message.content if item.type == "text"]).strip()

def call_anthropic_api(api_key, messages):
    client = anthropic.Anthropic(api_key=api_key)
    response_message = client.messages.create(model=ANTHROPIC_MODEL, max_tokens=1024, messages=messages)
    return response_message

async def call_poe_api(api_key, messages):
    mapped_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "assistant":
            role = "bot"
        mapped_messages.append({"role": role, "content": msg.get("content", "")})
    poe_messages = [fp.ProtocolMessage(role=msg["role"], content=msg["content"]) for msg in mapped_messages]
    full_response = ""
    try:
        async for partial in fp.get_bot_response(messages=poe_messages, bot_name="gpt-4o-mini", api_key=api_key):
            full_response += getattr(partial, "text", str(partial))
    except Exception as e:
        raise Exception(f"POE API request failed: {str(e)}")
    return full_response

def prepare_poe_messages(messages):
    mapped_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "assistant":
            role = "bot"
        mapped_messages.append({"role": role, "content": msg.get("content", "")})
    return mapped_messages

# -------------------- Routes for HTML-based Interface --------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if "conversation" not in session:
        session["conversation"] = []
    response_text = ""
    error_message = ""
    if request.method == "POST" and "clear_chat" not in request.form:
        api_key = request.form.get("api_key")
        user_message = request.form.get("user_message")
        provider = request.form.get("provider", "openai")
        if not api_key or not user_message:
            error_message = "API Key and Message are required!"
            return render_template("validate_api_request.html", response_text="", error_message=error_message, conversation=[])
        session["conversation"].append({"role": "user", "content": user_message})
        session.modified = True
        # Validate API Key using the model-based function
        prediction = test_api_key(api_key, rf_model)
        if prediction not in ["Valid OpenAI", "Valid Cohere", "Valid Anthropic", "Valid Poe"]:
            error_message = "Invalid API Key"
            return render_template("validate_api_request.html", response_text="", error_message=error_message, conversation=[])
        endpoint_mapping = {
            "openai": "/v1/chat/completions",
            "poe": "/api/message",
            "cohere": "/v1/generate"
        }
        selected_endpoint = endpoint_mapping.get(provider.lower(), "/v1/chat/completions")
        if provider.lower() == "cohere":
            try:
                response_text = call_cohere_api(api_key, session["conversation"])
            except Exception as e:
                error_message = f"Cohere API request failed: {str(e)}"
        elif provider.lower() == "anthropic":
            try:
                response_text = call_anthropic_api(api_key, session["conversation"]).content.strip()
            except Exception as e:
                error_message = f"Anthropic API request failed: {str(e)}"
        elif provider.lower() == "poe":
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response_text = loop.run_until_complete(call_poe_api(api_key, prepare_poe_messages(session["conversation"])))
                loop.close()
            except Exception as e:
                error_message = f"POE API request failed: {str(e)}"
        else:
            try:
                total_tokens = calculate_tokens(session["conversation"])
                time_of_day = categorize_time_of_day()
                request_metadata = {
                    "Rate Limiting": int(request.headers.get("x-ratelimit-remaining-requests", 100)),
                    "Endpoint Entropy": 0.5,
                    "HTTP Method": request.method,
                    "API Endpoint": selected_endpoint,
                    "HTTP Status": 200,
                    "User-Agent": request.headers.get("User-Agent", "Unknown"),
                    "Token Used": total_tokens,
                    "Method_POST": 1 if request.method.upper() == "POST" else 0,
                    "Time of Day": time_of_day
                }
                # Log the API request
                log_api_request(api_key, request_metadata)
                processed_data = preprocess_input(request_metadata)
                predicted_class = xgb_model.predict(processed_data)[0]
                if predicted_class == 1:
                    error_message = "🚨 Suspicious activity detected. Request blocked."
                    return render_template("validate_api_request.html", response_text="", error_message=error_message, conversation=[])
                payload = {"model": "gpt-4o-mini", "messages": session["conversation"], "temperature": 0.7, "max_tokens": 100}
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                response = requests.post(OPENAI_API_URL, json=payload, headers=headers)
                response_data = response.json()
                if "error" in response_data:
                    error_message = response_data["error"]["message"]
                else:
                    response_text = response_data["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                error_message = f"API request failed: {str(e)}"
        session["conversation"].append({"role": "assistant", "content": response_text})
        session.modified = True
    return render_template("validate_api_request.html", response_text=response_text, error_message=error_message, conversation=session["conversation"])

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("conversation", None)
    return jsonify({"message": "Chat history cleared!"})

if __name__ == "__main__":
    app.run(debug=True)