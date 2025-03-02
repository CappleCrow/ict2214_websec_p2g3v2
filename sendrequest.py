import requests
from reportlab.pdfgen import canvas 
from reportlab.pdfbase.ttfonts import TTFont 
from reportlab.pdfbase import pdfmetrics 
from reportlab.lib import colors 
import cohere
import asyncio
import fastapi_poe as fp

# Update the gateway URL to point to your dedicated JSON endpoint
gateway_url = "http://127.0.0.1:5000/api/chat"

def call_cohere_api(api_key, messages):
    co = cohere.ClientV2(api_key=api_key)
    res = co.chat(model="command-r-plus-08-2024", messages=messages)
    return "".join([item.text for item in res.message.content if item.type == "text"]).strip()

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

def send_request(provider, api_key, messages):
    if provider.lower() == "cohere":
        return {"choices": [{"message": {"content": call_cohere_api(api_key, messages)}}]}
    elif provider.lower() == "poe":
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response_text = loop.run_until_complete(call_poe_api(api_key, prepare_poe_messages(messages)))
        loop.close()
        return {"choices": [{"message": {"content": response_text}}]}
    else:
        # For OpenAI (default), send to our dedicated JSON API endpoint.
        payload = {
            "api_key": api_key,
            "user_message": messages[0]["content"],  # Assuming the first message is the current query
            "provider": provider,
            "conversation": messages
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(gateway_url, json=payload, headers=headers)
        
        # Debug: Print the raw response text if JSON decoding fails.
        try:
            return response.json()
        except Exception as e:
            print("Error decoding JSON. Response text:")
            print(response.text)
            raise

# Example OpenAI request using the dedicated JSON API endpoint
response_data = send_request(
    "openai",
    "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",
    [{"role": "user", "content": "write a haiku about web security"}]
)

if ('status' in response_data and 
    response_data['status'] == 'blocked' and 
    response_data.get('reason') == 'Suspicious activity detected by AI model.'):
    
    filename = 'report.pdf'
    documentitle = 'Suspicious Activity Detected'
    title = 'API Protection Intelligence Report'
    subtitle = 'test123'
    textlines = [response_data['reason'], response_data['status']]

    pdf = canvas.Canvas(filename)
    pdf.setTitle(documentitle)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(100, 800, title)
    text = pdf.beginText(100, 750)
    text.setFont("Courier", 12)
    
    for line in textlines:
        text.textLine(line)
    pdf.drawText(text)
    pdf.save()
    print("✅ Report generated successfully as", filename)
else:
    print("Message Content Response:", response_data)