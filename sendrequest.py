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
    "User-Agent": "curl/7.64.1",
    "x-ratelimit-remaining-requests": "198"
}

response = requests.post(gateway_url, json=test_payload, headers=headers)

print("🚀 API Gateway Response:", response.json())
