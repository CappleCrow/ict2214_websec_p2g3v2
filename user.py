from flask import Flask, render_template, request, session, jsonify
import requests
import os
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

@app.route("/", methods=["GET", "POST"])
def home():
    if "conversation" not in session:
        session["conversation"] = []  # Store chat history in session

    response_text = ""
    error_message = ""

    if request.method == "POST" and "clear_chat" not in request.form:  # Process user message, not chat clearing
        api_key = request.form.get("api_key")
        user_message = request.form.get("user_message")

        if not api_key or not user_message:
            error_message = "API Key and Message are required!"
            return render_template("validate_openai_request.html", response_text="", error_message=error_message, conversation=[])

        # Append user message to session history
        session["conversation"].append({"role": "user", "content": user_message})
        session.modified = True  # Ensure session updates persist

        # OpenAI API request payload
        payload = {
            "model": "gpt-4o-mini",
            "messages": session["conversation"],  # Pass full conversation
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

            # Handle OpenAI API errors
            if "error" in response_data:
                error_message = response_data["error"]["message"]
                return render_template("validate_openai_request.html", response_text="", error_message=error_message, conversation=[])

            # If API request is blocked, generate a PDF report
            if "status" in response_data and response_data["status"] == "blocked":
                generate_pdf_report(response_data)
                error_message = "🚨 Suspicious activity detected. A report has been generated."
                return render_template("validate_openai_request.html", response_text="", error_message=error_message, conversation=[])

            response_text = response_data["choices"][0]["message"]["content"].strip()

            # Append OpenAI response to conversation history
            session["conversation"].append({"role": "assistant", "content": response_text})
            session.modified = True

        except requests.exceptions.RequestException as e:
            error_message = f"API request failed: {str(e)}"

    return render_template("validate_openai_request.html", response_text=response_text, error_message=error_message, conversation=session["conversation"])

@app.route("/clear", methods=["POST"])
def clear_chat():
    session.pop("conversation", None)  # Clear chat history
    return jsonify({"message": "Chat history cleared!"})  # Only return JSON, no API call

def generate_pdf_report(response_data):
    filename = "report.pdf"
    document_title = "Suspicious Activity Detected"
    title = "API Protection Intelligence Report"
    subtitle = "test123"
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
