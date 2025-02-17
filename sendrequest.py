import requests

# Define the API endpoint
api_url = "http://127.0.0.1:5000/predict"

# Simulated API request payload
test_request = {
    "Rate Limiting": 10,
    "Endpoint Entropy": 0.2,
    "HTTP Method": "POST",  # This will be encoded
    "API Endpoint": "/v1/auth/login",  # This will be encoded
    "HTTP Status": 200,
    "User-Agent": "cURL/7.79.1",  # This will be encoded
    "Token Used": 50
}


try:
    response = requests.post(api_url, json=test_request)
    print("Raw Response Text:", response.text)  # Debugging: Print the response content

    # Ensure the response contains valid JSON before parsing
    if response.headers.get("Content-Type") == "application/json":
        response_data = response.json()
        print("🚀 API Gateway Response:", response_data)
    else:
        print("❌ API did not return JSON, got:", response.text)

except requests.exceptions.RequestException as e:
    print("❌ API request failed:", e)
