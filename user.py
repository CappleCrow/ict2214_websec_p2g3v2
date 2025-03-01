from flask import Flask, render_template, request, session, jsonify
import os
import requests
from reportlab.pdfgen import canvas
from openai import OpenAI  # For DeepSeek API calls
import anthropic  # Import the Anthropi­c package
import cohere  # Import the Cohere package

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"  

def call_cohere_api(api_key, messages):
    import cohere
    co = cohere.ClientV2(api_key=api_key)
    res = co.chat(
        model="command-r-plus-08-2024",  
        messages=messages
    )

    response_text = ""
    for item in res.message.content:
        if item.type == "text":
            response_text += item.text  

   
    response_text = response_text.strip()
    return response_text



# Updated function for Anthropi­c API using the official client
def call_anthropic_api(api_key, messages):
    client = anthropic.Anthropic(api_key=api_key)
    # Call the API using the conversation history as the messages list
    response_message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        messages=messages
    )
    return response_message

@app.route("/", methods=["GET", "POST"])
def home():
    if "conversation" not in session:
        session["conversation"] = [] 

        print("Debug: session conversation:", session["conversation"])

    response_text = ""
    error_message = ""

    if request.method == "POST" and "clear_chat" not in request.form:
        api_key = request.form.get("api_key")
        user_message = request.form.get("user_message")
        provider = request.form.get("provider", "openai")  # Default to OpenAI

        if not api_key or not user_message:
            error_message = "API Key and Message are required!"
            return render_template("validate_api_request.html", response_text="", error_message=error_message, conversation=[])

        
        session["conversation"].append({"role": "user", "content": user_message})
        session.modified = True

        if provider.lower() == "cohere":
            try:
                response_text = call_cohere_api(api_key, session["conversation"])
                session["conversation"].append({"role": "assistant", "content": response_text})
                session.modified = True
            except Exception as e:
                error_message = f"Cohere API request failed: {str(e)}"

        elif provider.lower() == "anthropic":
            try:
                response_message = call_anthropic_api(api_key, session["conversation"])
                response_text = response_message.content.strip()
                session["conversation"].append({"role": "assistant", "content": response_text})
                session.modified = True
            except Exception as e:
                error_message = f"Anthropic API request failed: {str(e)}"
        else:
            # OpenAI API request payload
            payload = {
                "model": "gpt-4o-mini",
                "messages": session["conversation"],
                "temperature": 0.7,
                "max_tokens": 100
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            try:
                response = requests.post(OPENAI_API_URL, json=payload, headers=headers)
                response_data = response.json()
                if "error" in response_data:
                    error_message = response_data["error"]["message"]
                    return render_template("validate_api_request.html", response_text="", error_message=error_message, conversation=[])
                if "status" in response_data and response_data["status"] == "blocked":
                    generate_pdf_report(response_data)
                    error_message = "🚨 Suspicious activity detected. A report has been generated."
                    return render_template("validate_api_request.html", response_text="", error_message=error_message, conversation=[])
                response_text = response_data["choices"][0]["message"]["content"].strip()
                session["conversation"].append({"role": "assistant", "content": response_text})
                session.modified = True
            except requests.exceptions.RequestException as e:
                error_message = f"API request failed: {str(e)}"

    return render_template("validate_api_request.html", response_text=response_text, error_message=error_message, conversation=session["conversation"])

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("conversation", None)
    return jsonify({"message": "Chat history cleared!"})

def generate_pdf_report(response_data):
    filename = "report.pdf"
    document_title = "Suspicious Activity Detected"
    title = "API Protection Intelligence Report"
    textlines = [response_data.get("reason", "No reason provided"), response_data.get("status", "No status")]
    pdf = canvas.Canvas(filename)
    pdf.setTitle(document_title)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(100, 800, title)
    text = pdf.beginText(100, 750)
    text.setFont("Courier", 12)
    for line in textlines:
        text.textLine(line)
    pdf.drawText(text)
    pdf.save()
    print("✅ Report generated successfully as", filename)

if __name__ == "__main__":
    app.run(debug=True)
