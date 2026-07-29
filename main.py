import os
import json
import re
from typing import Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
import httpx
from openai import OpenAI

app = FastAPI()

# Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PUBLIC_HOST_URL = os.getenv("PUBLIC_HOST_URL")  # e.g., https://your-app.onrender.com
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client (or OpenRouter/Gemini API)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Ensure static directory exists for hosting run.jsonl
os.makedirs("static", exist_ok=True)
LOG_FILE_PATH = "static/run.jsonl"
if not os.path.exists(LOG_FILE_PATH):
    open(LOG_FILE_PATH, "w").close()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Multi-turn chat history store: { chat_id: [messages] }
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
    """Uses LLM to solve data analysis questions and return parsed output."""
    steps = []
    latest_message = conversation_history[-1]
    steps.append({"step": "receive_message", "content": latest_message})

    # System prompt directing LLM to solve the data task and extract requested JSON structure
    system_prompt = (
        "You are an expert Data Analyst AI agent. "
        "Solve the data analysis question asked by the user. "
        "The user prompt explicitly specifies the required output JSON structure under 'answer'. "
        "Extract or compute the precise answer and respond ONLY with a valid JSON object matching the requested schema."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in conversation_history:
        messages.append({"role": "user", "content": msg})

    if client:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-4o / gemini
            messages=messages,
            response_format={"type": "json_object"}
        )
        raw_output = response.choices[0].message.content
        steps.append({"step": "llm_response", "content": raw_output})
        
        try:
            parsed_json = json.loads(raw_output)
            # If LLM wrapped it inside "answer", extract it, else take the whole object
            answer = parsed_json.get("answer", parsed_json)
        except Exception:
            answer = raw_output
    else:
        # Fallback dummy answer if API key is missing
        answer = {"result": "No LLM API key configured"}

    return answer, steps

@app.post("/webhook")
async def telegram_webhook(request: Request):
    payload = await request.json()
    
    if "message" not in payload or "text" not in payload["message"]:
        return Response(status_code=200)

    message = payload["message"]
    chat_id = message["chat"]["id"]
    text = message["text"]

    # Maintain multi-turn history
    if chat_id not in CHAT_HISTORIES:
        CHAT_HISTORIES[chat_id] = []
    CHAT_HISTORIES[chat_id].append(text)

    # Solve the problem
    answer, steps = await solve_data_question(chat_id, CHAT_HISTORIES[chat_id])

    # Log to JSONL file
    append_to_log(chat_id, text, answer)
    log_url = f"{PUBLIC_HOST_URL}/static/run.jsonl"

    # Strict grading output payload
    final_payload = {
        "answer": answer,
        "log_url": log_url
    }

    # Send message back to Telegram
    async with httpx.AsyncClient() as http_client:
        await http_client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": json.dumps(final_payload)
            },
            timeout=30.0
        )

    return Response(status_code=200)

@app.get("/run.jsonl")
async def get_log():
    """Endpoint serving raw JSONL log file for wget validation."""
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as f:
            content = f.read()
        return Response(content=content, media_type="application/x-ndjson")
    return Response(content="", media_type="application/x-ndjson")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Bot is running"}
