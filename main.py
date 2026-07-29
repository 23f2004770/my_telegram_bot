import os
import json
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
import httpx
from google import genai
from google.genai import types

app = FastAPI()

# Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PUBLIC_HOST_URL = os.getenv("PUBLIC_HOST_URL")  # e.g., https://my-data-analyst-bot.onrender.com
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

# Candidate models to attempt in order if one fails or hits rate limits
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

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
    """Uses Gemini API with automatic model fallback and retries to solve data tasks."""
    steps = []
    latest_message = conversation_history[-1]
    steps.append({"step": "receive_message", "content": latest_message})

    if not client:
        return {"error": "GEMINI_API_KEY environment variable is missing"}, steps

    # Prompt forcing model to generate JSON output adhering to user request
    prompt = f"""
You are an expert Data Analyst AI agent.
Solve the following data analysis task or question submitted by the user.

Conversation History:
{json.dumps(conversation_history, indent=2)}

Latest Task:
{latest_message}

CRITICAL REQUIREMENT:
The user prompt specifies an exact output JSON structure (e.g. key-value pairs under 'answer').
Compute or extract the answer and respond ONLY with a valid JSON object matching the requested schema.
Do NOT surround your output with markdown code blocks like ```json ... ```. Output raw JSON only.
"""

    answer = None

    # Try each model candidate with automatic retries on 429 quota limits
    for model_name in MODEL_CANDIDATES:
        for attempt in range(2):  # Try twice per model
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                
                raw_output = response.text
                steps.append({"step": "llm_response", "model": model_name, "content": raw_output})
                
                parsed_json = json.loads(raw_output)
                
                # Extract inner answer if model wrapped it in an outer {"answer": ...} key
                if isinstance(parsed_json, dict) and "answer" in parsed_json:
                    answer = parsed_json["answer"]
                else:
                    answer = parsed_json
                
                return answer, steps  # Success!

            except Exception as e:
                err_msg = str(e)
                print(f"Error on model {model_name} (Attempt {attempt+1}): {err_msg}")
                
                # If rate-limited (429), pause briefly before retrying
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    await asyncio.sleep(4)
                    continue
                # If model not found (404), break loop to move to next model candidate
                elif "404" in err_msg or "NOT_FOUND" in err_msg:
                    break

    # Final fallback if all API calls failed
    if answer is None:
        answer = {"error": "All AI model attempts timed out or exceeded quota. Please retry."}

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

        # Maintain multi-turn history per chat
        if chat_id not in CHAT_HISTORIES:
            CHAT_HISTORIES[chat_id] = []
        CHAT_HISTORIES[chat_id].append(text)

        # Handle start command
        if text.strip() == "/start":
            reply_text = "Bot is online and ready for data analysis tasks!"
        else:
            # Solve data analysis problem
            answer, steps = await solve_data_question(chat_id, CHAT_HISTORIES[chat_id])

            # Append to log file
            append_to_log(chat_id, text, answer)
            
            # Construct log URL for evaluator download
            log_url = f"{PUBLIC_HOST_URL}/run.jsonl"

            # Required final response JSON schema
            final_payload = {
                "answer": answer,
                "log_url": log_url
            }
            reply_text = json.dumps(final_payload)

        # Send response message back to Telegram API
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
        print(f"Error processing Telegram update: {e}")

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
