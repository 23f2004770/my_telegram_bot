import os
import json
from typing import Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
import httpx
from google import genai
from google.genai import types

app = FastAPI()

# Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PUBLIC_HOST_URL = os.getenv("PUBLIC_HOST_URL")  # e.g. https://my-data-analyst-bot.onrender.com
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Ensure static directory and run.jsonl log file exist
os.makedirs("static", exist_ok=True)
LOG_FILE_PATH = "static/run.jsonl"
if not os.path.exists(LOG_FILE_PATH):
    open(LOG_FILE_PATH, "w").close()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Multi-turn conversation store: { chat_id: [messages] }
CHAT_HISTORIES: Dict[int, list] = {}

def append_to_log(chat_id: int, user_prompt: str, answer: Any):
    """Appends execution trace to public JSONL log file."""
    log_entry = {
        "chat_id": chat_id,
        "prompt": user_prompt,
        "status": "completed",
        "answer": answer
    }
    with open(LOG_FILE_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

async def solve_data_question(chat_id: int, conversation_history: list) -> tuple[Any, list]:
    """Uses Gemini 2.5 Flash to solve data analysis questions and return parsed output."""
    steps = []
    latest_message = conversation_history[-1]
    steps.append({"step": "receive_message", "content": latest_message})

    if not client:
        return {"error": "GEMINI_API_KEY environment variable is missing"}, steps

    # System instruction guiding Gemini to output exact requested JSON format
    prompt = f"""
You are an expert Data Analyst AI agent.
Solve the following data analysis task or question submitted by the user.

User History:
{json.dumps(conversation_history, indent=2)}

Latest Task:
{latest_message}

CRITICAL REQUIREMENT:
The user prompt specifies an exact output JSON structure (e.g. key-value pairs under 'answer').
Compute or extract the answer and respond ONLY with a valid JSON object.
"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        
        raw_output = response.text
        steps.append({"step": "llm_response", "content": raw_output})
        
        parsed_json = json.loads(raw_output)
        # If model wrapped it inside "answer", extract it to match exact requested shape
        if isinstance(parsed_json, dict) and "answer" in parsed_json:
            answer = parsed_json["answer"]
        else:
            answer = parsed_json
            
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        answer = {"error": str(e)}

    return answer, steps

@app.post("/webhook")
async def telegram_webhook(request: Request):
    try:
        payload = await request.json()
        
        if "message" not in payload or "text" not in payload["message"]:
            return Response(status_code=200)

        message = payload["message"]
        chat_id = message["chat"]["id"]
        text = message["text"]

        # Maintain multi-turn message history per chat
        if chat_id not in CHAT_HISTORIES:
            CHAT_HISTORIES[chat_id] = []
        CHAT_HISTORIES[chat_id].append(text)

        # Handle simple bot command
        if text.strip() == "/start":
            reply_text = "Bot is online and ready for data analysis tasks!"
        else:
            # Solve data analysis problem
            answer, steps = await solve_data_question(chat_id, CHAT_HISTORIES[chat_id])

            # Write to JSONL log
            append_to_log(chat_id, text, answer)
            
            # Construct log URL for evaluator download
            log_url = f"{PUBLIC_HOST_URL}/run.jsonl"

            # Strict required response payload
            final_payload = {
                "answer": answer,
                "log_url": log_url
            }
            reply_text = json.dumps(final_payload)

        # Send HTTP POST reply to Telegram API
        async with httpx.AsyncClient() as http_client:
            await http_client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": reply_text
                },
                timeout=30.0
            )

    except Exception as e:
        print(f"Error handling Telegram update: {e}")

    return Response(status_code=200)

@app.get("/run.jsonl")
async def get_log():
    """Endpoint serving raw JSONL log file for evaluator wget access."""
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as f:
            content = f.read()
        return Response(content=content, media_type="application/x-ndjson")
    return Response(content="", media_type="application/x-ndjson")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Bot is running"}
