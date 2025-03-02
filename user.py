from flask import Flask, render_template, request, session, jsonify
import os
import requests
from reportlab.pdfgen import canvas
import anthropic  # Import the Anthropic package
import cohere  # Import the Cohere package
import asyncio
import fastapi_poe as fp
from load import validate_api_key, calculate_tokens, preprocess_input, categorize_time_of_day, xgb_model  # Import necessary functions and models

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"

def call_cohere_api(api_key, messages):
    co = cohere.ClientV2(api_key=api_key)
    res = co.chat(model="command-r-plus-08-2024", messages=messages)
    return "".join([item.text for item in res.message.content if item.type == "text"]).strip()

def call_anthropic_api(api_key, messages):
    client = anthropic.Anthropic(api_key=api_key)
    response_message = client.messages.create(model=ANTHROPIC_MODEL, max_tokens=1024, messages=messages)
    return response_message

async def call_poe_api(api_key, messages):
    poe_messages = [fp.ProtocolMessage(role=msg.get("role", "user"), content=msg.get("content", "")) for msg in messages]
    full_response = ""
    try:
        async for partial in fp.get_bot_response(messages=poe_messages, bot_name="gpt-4o-mini", api_key=api_key):
            full_response += getattr(partial, "text", str(partial))
    except Exception as e:
        raise Exception(f"POE API request failed: {str(e)}")
    return full_response

def prepare_poe_messages(messages):
    return [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in messages]

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

        # Validate API Key
        key_validity = validate_api_key(api_key)
        if key_validity != "Valid OpenAI":
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
                    "Method_POST": 1,
                    "Time of Day": time_of_day
                }
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

# -------------------- Dedicated JSON API Endpoint --------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    # Expect a JSON payload from the client
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "Missing JSON payload"}), 400

    api_key = payload.get("api_key")
    user_message = payload.get("user_message")
    provider = payload.get("provider", "openai")
    conversation = payload.get("conversation", [])

    if not api_key or not user_message:
        return jsonify({"error": "API Key and user_message are required"}), 400

    # Append the user's new message to the conversation history
    conversation.append({"role": "user", "content": user_message})

    # Dynamically select the API endpoint based on provider
    endpoint_mapping = {
        "openai": "/v1/chat/completions",
        "poe": "/api/message",
        "cohere": "/v1/generate",
        "anthropic": "/api/anthropic"  # Example; adjust as needed
    }
    selected_endpoint = endpoint_mapping.get(provider.lower(), "/v1/chat/completions")

    # Validate API key and compute some metadata (as in your original logic)
    total_tokens = calculate_tokens(conversation)
    time_of_day = categorize_time_of_day()
    request_metadata = {
        "Rate Limiting": int(request.headers.get("x-ratelimit-remaining-requests", 100)),
        "Endpoint Entropy": 0.5,
        "HTTP Method": request.method,
        "API Endpoint": selected_endpoint,  # Dynamic endpoint here
        "HTTP Status": 200,
        "User-Agent": request.headers.get("User-Agent", "Unknown"),
        "Token Used": total_tokens,
        "Method_POST": 1,
        "Time of Day": time_of_day
    }
    processed_data = preprocess_input(request_metadata)
    predicted_class = xgb_model.predict(processed_data)[0]
    if predicted_class == 1:
        return jsonify({"error": "🚨 Suspicious activity detected. Request blocked."}), 403

    # Process the request according to the provider
    try:
        if provider.lower() == "cohere":
            response_text = call_cohere_api(api_key, conversation)
        elif provider.lower() == "poe":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_text = loop.run_until_complete(call_poe_api(api_key, prepare_poe_messages(conversation)))
            loop.close()
        elif provider.lower() == "anthropic":
            response_text = call_anthropic_api(api_key, conversation).content.strip()
        else:  # Default to OpenAI
            payload_data = {
                "model": "gpt-4o-mini",
                "messages": conversation,
                "temperature": 0.7,
                "max_tokens": 100
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = requests.post(OPENAI_API_URL, json=payload_data, headers=headers)
            response_data = response.json()
            if "error" in response_data:
                return jsonify({"error": response_data["error"]["message"]}), 400
            response_text = response_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Append the assistant's reply to the conversation
    conversation.append({"role": "assistant", "content": response_text})
    return jsonify({"conversation": conversation, "response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
