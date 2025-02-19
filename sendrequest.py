# import requests

# # Define the API endpoint
# api_url = "http://127.0.0.1:5000/predict"

# # Simulated API request payload
# test_request = {
#     "Rate Limiting": 30,  # Low rate limiting (not triggering any limits)
#     "Endpoint Entropy": 0.8,  # Higher entropy means diverse and normal API use
#     "HTTP Method": "POST",  # Common, low-risk method
#     "API Endpoint": "/v1/chat/completions",  # A normal data-fetching endpoint
#     "HTTP Status": 200,  # Standard successful response
#     "User-Agent": "PostmanRuntime/7.29.0",  # Common browser user-agent
#     "Token Used": 20,  # Low usage, indicating normal API interaction
#     "Generalized API Endpoint": "/v1/chat",  # More abstracted API path
#     "Method_POST": 1,  # Encoded as 1 for POST requests
#     "Time of Day": "Afternoon"  # New time-of-day category
# }

# try:
#     response = requests.post(api_url, json=test_request)
#     print("Raw Response Text:", response.text)  # Debugging: Print the response content

#     # Ensure the response contains valid JSON before parsing
#     if response.headers.get("Content-Type") == "application/json":
#         response_data = response.json()
#         print("🚀 API Gateway Response:", response_data)
#     else:
#         print("❌ API did not return JSON, got:", response.text)

# except requests.exceptions.RequestException as e:
#     print("❌ API request failed:", e)


import requests

gateway_url = "http://127.0.0.1:5001/validate_openai_request"

# OpenAI API request payload
test_payload = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "write a haiku about web security"}],
    "temperature": 0.7,
    "max_tokens": 100
}

# API headers
headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
    "User-Agent": "PostmanRuntime/7.29.0",
    "x-ratelimit-remaining-requests": "198"
}

response = requests.post(gateway_url, json=test_payload, headers=headers)

print("🚀 API Gateway Response:", response.json())
