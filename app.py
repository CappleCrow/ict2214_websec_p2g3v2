from flask import Flask, render_template, request, jsonify
import random
import numpy as np
from faker import Faker
from datetime import datetime
import pandas as pd

app = Flask(__name__)
fake = Faker()

# List of API Endpoints
API_ENDPOINTS = [
    "/data",
    "/api/product",
    "/api/user",
    "/api/admin",
    "/v1/chat/completions"
]

# List of User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/537.36",
    "PostmanRuntime/7.29.0",
    "Python-requests/2.25.1"
]

# Time of Day Categories
TIME_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/send_request', methods=['POST'])
def send_request():
    api_key = request.form.get("api_key")  # Get API key from input field

    if not api_key:
        return jsonify({"error": "API key is required"})

    # Simulated values
    fake_ip = fake.ipv4()
    user_agent = random.choice(USER_AGENTS)
    http_method = "POST"
    tokens_used = random.randint(50, 500)
    rate_limit_remaining = random.randint(50, 200)  # Simulating rate limit
    endpoint_entropy = np.random.uniform(0.1, 1.0)
    endpoint = random.choice(API_ENDPOINTS)
    time_of_day = random.choice(TIME_OF_DAY)
    classification_label = random.choice([0, 1])
    new_time_of_access = datetime.now().strftime("%d/%m/%Y %H:%M")  # Format: 19/2/2025 1:54
    method_post = 1  # HTTP method is POST
    http_status = random.choice([200, 301, 404, 500])  # Random HTTP status

    # Request headers
    headers = {
        "X-Forwarded-For": fake_ip,
        "User-Agent": user_agent,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Log data in the requested format
    log_data = {
        "Rate Limiting": rate_limit_remaining,
        "Endpoint Entropy": round(endpoint_entropy, 8),
        "HTTP Method": http_method,
        "API Endpoint": endpoint,
        "HTTP Status": http_status,
        "User-Agent": user_agent,
        "Token Used": tokens_used,
        "Method_POST": method_post,
        "Time of Day": time_of_day,
        "Classification Label": classification_label,
        "New Time of Access": new_time_of_access,
        "API_Key": api_key[:8] + "*****"  # Mask the API key
    }

    # Save the log to a CSV file
    df = pd.DataFrame([log_data])
    df.to_csv("api_requests_log.csv", mode='a', header=False, index=False)

    return jsonify({"log": log_data, "message": "API request logged successfully."})

if __name__ == '__main__':
    app.run(debug=True)
