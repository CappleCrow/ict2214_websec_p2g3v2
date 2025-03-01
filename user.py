from flask import Flask, render_template, request, session, jsonify
import os
import requests
from reportlab.pdfgen import canvas
from openai import OpenAI  # For DeepSeek API calls
import anthropic  # Import the Anthropi­c package
import cohere  # Import the Cohere package
import asyncio
import fastapi_poe as fp

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

async def call_poe_api(api_key, messages):
    #convert messages to format expected by POE
    poe_messages = []

    for message in messages:
        role = message.get("role", "user")
        if role == "assistant":
            role == "bot"
        elif role not in ["user", "bot", "system"]:
            role = "user"
        poe_messages.append(fp.ProtocolMessage(
            role=role,
            content=message.get("content", "")))
    
    #collect the response from the POE API
    full_response = ""
    try:
        async for partial in fp.get_bot_response(
            messages=poe_messages,
            bot_name="gpt-4o-mini",
            api_key=api_key
        ):
            # extract text content from PartialResponse object
            if hasattr(partial, "text"):
                full_response += partial.text
            elif isinstance(partial, str):
                full_response += partial
            else:
                full_response += str(partial)
            #full_response += partial.message.content #maybe partial.message.content
    except Exception as e:
        print(f"POE API request from async failed: {str(e)}")
        print("Messages being sent")
        for i, msg in enumerate(poe_messages):
            print(f"message{i+1}: {msg.role}: {msg.content[:30]}...")
        raise Exception(f"POE API request from async failed: {str(e)}")
    
    return full_response

def prepare_poe_messages(messages):
    """Convert regular convo history to compatible format"""
    poe_messages = []
    for message in messages:
        role = message.get("role", "user")
        if role == "assistant":
            poe_role = "bot"
        elif role == "system":
            poe_role = "system"
        else:
            poe_role = "user"
        poe_messages.append({"role": poe_role, "content": message.get("content", "")})
    return poe_messages

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
        elif provider.lower() == "poe":
            try:
                # create new asyncio event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                #use copy of convo with proper formatting
                poe_convo = prepare_poe_messages(session["conversation"])
                # run async function in event loop
                response_text = loop.run_until_complete(call_poe_api(api_key, poe_convo))
                loop.close()

                session["conversation"].append({"role": "assistant", "content": response_text})
                session.modified = True
            except Exception as e:
                error_message = f"POE API request failed: {str(e)}"
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
