from flask import Flask, render_template, request, session, jsonify, send_file
import os
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, Spacer
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Line
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.utils import simpleSplit
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
import tempfile
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

# -------------------- Helper function for generating PDF --------------------
def get_downloads_folder():
    """
    Get a writable folder path for storing reports.
    Tries the user's Downloads folder, but if that doesn't exist or isn't writable,
    falls back to a temporary directory.
    """
    home = Path.home()
    downloads = home / "Downloads"
    try:
        downloads.mkdir(parents=True, exist_ok=True)
        # Test writing a temporary file to ensure it's writable.
        test_file = downloads / "test.txt"
        with test_file.open("w") as f:
            f.write("test")
        test_file.unlink()  # Remove test file
        return downloads
    except Exception as e:
        # Fallback to a temporary folder on Linux/Unix systems (Azure App Service on Linux allows /tmp)
        temp_folder = Path(tempfile.gettempdir())
        temp_folder.mkdir(parents=True, exist_ok=True)
        return temp_folder

def generate_pdf_report(api_key, request_metadata, file_name="suspicious_activity_report.pdf"):
    """
    Generate a professionally designed PDF report for suspicious activity detected in API requests.
    The report includes detailed information, charts, and formatted sections.
    """
    # Determine the downloads folder path
    downloads_folder = get_downloads_folder()
    report_file_path = downloads_folder / file_name

    # Check if the report file already exists and update the file name accordingly
    if report_file_path.exists():
        base = file_name.rsplit('.', 1)[0]  # 'suspicious_activity_report'
        ext = file_name.rsplit('.', 1)[1] if '.' in file_name else ""
        counter = 1
        new_file_name = f"{base}({counter}).{ext}" if ext else f"{base}({counter})"
        report_file_path = downloads_folder / new_file_name
        while report_file_path.exists():
            counter += 1
            new_file_name = f"{base}({counter}).{ext}" if ext else f"{base}({counter})"
            report_file_path = downloads_folder / new_file_name

    # Create canvas with letter size
    c = canvas.Canvas(str(report_file_path), pagesize=letter)
    width, height = letter
    
    # -------- Header Section --------
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.rect(0, height - 2*inch, width, 2*inch, fill=True, stroke=False)
    
    c.setFillColor(colors.HexColor('#FF0000'))
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1*inch, height - 1*inch, "   SECURITY ALERT")
    
    c.setFont("Helvetica-Bold", 30)
    c.drawString(0.5*inch, height - 1*inch, "⚠️")
    
    c.setFillColor(colors.HexColor('#333333'))
    c.setFont("Helvetica", 10)
    current_time = datetime.datetime.now().strftime("%B %d, %Y %H:%M:%S")
    c.drawString(width - 3*inch, height - 0.6*inch, f"Generated: {current_time}")
    
    c.setFillColor(colors.HexColor('#0066CC'))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, height - 1.4*inch, "Suspicious API Request Activity Report")
    
    c.setStrokeColor(colors.HexColor('#0066CC'))
    c.setLineWidth(2)
    c.line(0.5*inch, height - 1.8*inch, width - 0.5*inch, height - 1.8*inch)
    
    # -------- Request Information Section --------
    y_position = height - 2.5*inch
    c.setFillColor(colors.HexColor('#0066CC'))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.5*inch, y_position, "Request Information")
    y_position -= 0.4*inch
    
    c.setFillColor(colors.HexColor('#FF0000'))
    c.setFont("Courier-Bold", 12)
    masked_key = f"{api_key[:8]}{'*' * 15}"
    c.drawString(0.5*inch, y_position, f"API Key: {masked_key}")
    y_position -= 0.4*inch
    
    left_column_x = 0.5*inch
    right_column_x = 4*inch
    
    c.setFillColor(colors.HexColor('#333333'))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left_column_x, y_position, "Timestamp:")
    c.setFont("Helvetica", 11)
    c.drawString(left_column_x + 1*inch, y_position, datetime.datetime.now().strftime("%d/%m/%Y %H:%M"))
    y_position -= 0.3*inch
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left_column_x, y_position, "HTTP Method:")
    c.setFont("Helvetica", 11)
    c.drawString(left_column_x + 1*inch, y_position, str(request_metadata['HTTP Method']))
    y_position -= 0.3*inch
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left_column_x, y_position, "API Endpoint:")
    c.setFont("Helvetica", 11)
    endpoint_text = str(request_metadata['API Endpoint'])
    c.drawString(left_column_x + 1*inch, y_position, endpoint_text)
    y_position -= 0.3*inch
    
    y_position = height - 3.3*inch
    c.setFont("Helvetica-Bold", 11)
    c.drawString(right_column_x, y_position, "Rate Limiting:")
    c.setFont("Helvetica", 11)
    c.drawString(right_column_x + 1*inch, y_position, str(request_metadata['Rate Limiting']))
    y_position -= 0.3*inch
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(right_column_x, y_position, "Time of Day:")
    c.setFont("Helvetica", 11)
    c.drawString(right_column_x + 1*inch, y_position, str(request_metadata['Time of Day']))
    y_position -= 0.3*inch
    
    # -------- User Agent Information --------
    y_position = height - 4.5*inch
    c.setFillColor(colors.HexColor('#0066CC'))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.5*inch, y_position, "User Agent Details")
    y_position -= 0.4*inch
    
    c.setFillColor(colors.lightgrey)
    ua_box_height = 0.8*inch
    c.rect(0.5*inch, y_position - ua_box_height, width - 1*inch, ua_box_height, fill=True, stroke=False)
    
    c.setFillColor(colors.HexColor('#333333'))
    c.setFont("Courier", 10)
    ua_text = request_metadata['User-Agent']
    ua_lines = simpleSplit(ua_text, "Courier", 10, width - 1.2*inch)
    for i, line in enumerate(ua_lines):
        c.drawString(0.6*inch, y_position - 0.2*inch - (i * 0.2*inch), line)
    
    y_position -= ua_box_height + 0.3*inch
    
    # -------- Warning Message Section --------
    y_position -= 0.3*inch
    c.setFillColor(colors.pink)
    warning_box_height = 1*inch
    c.rect(0.5*inch, y_position - warning_box_height, width - 1*inch, warning_box_height, fill=True, stroke=False)
    
    c.setStrokeColor(colors.HexColor('#FF0000'))
    c.setLineWidth(2)
    c.rect(0.5*inch, y_position - warning_box_height, width - 1*inch, warning_box_height, fill=False, stroke=True)
    
    c.setFillColor(colors.darkred)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(width/2 - 2.5*inch, y_position - 0.4*inch, "⚠️ SUSPICIOUS ACTIVITY DETECTED")
    
    c.setFont("Helvetica", 12)
    c.drawString(width/2 - 2*inch, y_position - 0.7*inch, "This request has been blocked for security reasons.")
    
    y_position -= warning_box_height + 0.5*inch
    
    # -------- Potential Risk Factors --------
    c.setFillColor(colors.HexColor('#0066CC'))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.5*inch, y_position, "Potential Risk Factors")
    y_position -= 0.4*inch
    
    risk_factors = [
        "Unusual token usage pattern",
        "Suspicious API endpoint access",
        "Abnormal request rate",
        "Potential unauthorized access attempt"
    ]
    
    c.setFillColor(colors.HexColor('#333333'))
    c.setFont("Helvetica", 11)
    for factor in risk_factors:
        c.drawString(0.7*inch, y_position, "•")
        c.drawString(1*inch, y_position, factor)
        y_position -= 0.25*inch
    
    # -------- Footer --------
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.rect(0, 0.5*inch, width, 0.5*inch, fill=True, stroke=False)
    
    c.setFillColor(colors.HexColor('#333333'))
    c.setFont("Helvetica", 8)
    footer_text = "This report was automatically generated by the API Security Protection System. " 
    footer_text += "For more information, please contact your system administrator."
    c.drawString(0.5*inch, 0.7*inch, footer_text)
    
    c.drawString(width - 1*inch, 0.7*inch, "Page 1 of 1")
    
    # Save the PDF and return the report file path
    c.save()
    
    print(f"✅ Enhanced security report generated at: {report_file_path}")
    return report_file_path

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
        
        # Validate if API Key and Message are provided
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

        # Set the selected endpoint before checking for anomalies
        endpoint_mapping = {
            "openai": "/v1/chat/completions",
            "poe": "/api/message",
            "cohere ai": "/v1/generate"
        }
        selected_endpoint = endpoint_mapping.get(provider.lower(), "/v1/chat/completions")

        # Check for anomaly before proceeding to make the API request
        total_tokens = calculate_tokens(session["conversation"])
        time_of_day = categorize_time_of_day()
        request_metadata = {
            "Rate Limiting": int(request.headers.get("x-ratelimit-remaining-requests", 100)),
            "Endpoint Entropy": 0.5,
            "HTTP Method": request.method,
            "API Endpoint": "/v1/chat/completions",
            "HTTP Status": 200,
            "User-Agent": request.headers.get("User-Agent", "Unknown"),
            "Token Used": total_tokens,
            "Method_POST": 1 if request.method.upper() == "POST" else 0,
            "Time of Day": time_of_day

        }

        # Log the API request
        log_api_request(api_key, request_metadata)
        
        # Preprocess the input and check for anomalies
        processed_data = preprocess_input(request_metadata)
        predicted_class = xgb_model.predict(processed_data)[0]
        
        if predicted_class == 1:  # Anomaly detected
            error_message = "🚨 Suspicious activity detected. Request blocked."
            # Generate the PDF report for suspicious activity
            report_file_path = generate_pdf_report(api_key, request_metadata)
            print(f"PDF Report generated at: {report_file_path}")
            return render_template("validate_api_request.html", response_text="", error_message=error_message, conversation=[], pdf_report_path=report_file_path)
        
        # Proceed to API calls if no anomaly detected
        if provider.lower() == "cohere ai":
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

@app.route("/view_report")
def view_report():
    file_name = request.args.get("file")
    downloads_folder = get_downloads_folder()
    file_path = downloads_folder / file_name
    if file_path.exists():
        # Serve the file inline so it opens in the browser (new tab)
        return send_file(str(file_path), mimetype="application/pdf", as_attachment=False)
    else:
        return "File not found", 404

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("conversation", None)
    return jsonify({"message": "Chat history cleared!"})

if __name__ == "__main__":
    app.run(debug=True)