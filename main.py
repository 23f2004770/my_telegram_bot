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

# Primary model candidates to try
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
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

def clean_and_parse_json(text: str) -> Any:
    """Helper function to clean markdown fences and parse valid JSON."""
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # If wrapped in markdown code block or extra prose, extract content inside brackets
        start_obj, end_obj = raw.find("{"), raw.rfind("}")
        start_arr, end_arr = raw.find("["), raw.rfind("]")
        
        start = start_obj if (start_obj != -1 and (start_arr == -1 or start_obj < start_arr)) else start_arr
        end = end_obj if (end_obj != -1 and (end_arr == -1 or end_obj > end_arr)) else end_arr

        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        raise

async def solve_data_question(chat_id: int, conversation_history: list) -> tuple[Any, list]:
    """Uses Gemini API with robust fallbacks and retries to solve data tasks."""
    steps = []
    latest_message = conversation_history[-1]
    steps.append({"step": "receive_message", "content": latest_message})

    if not client:
        return {"error": "GEMINI_API_KEY environment variable is missing"}, steps

    system_instruction = (
        "You are a careful data analyst. The user's LAST message asks a data-analysis "
        "question and specifies an exact JSON structure to reply with. Work out the "
        "real answer (use any public data you know, e.g. MOSPI statistics, general "
        "world knowledge, or arithmetic on numbers given in the message). "
        "Reply ONLY with the exact required JSON object and absolutely nothing else."
    )

    prompt = f"""Conversation History:
{json.dumps(conversation_history[-6:], indent=2)}

Task:
{latest_message}
"""

    answer = None

    # Try each model candidate with automatic retries on quota/rate limits
    for model_name in MODEL_CANDIDATES:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                    ),
                )
                
                raw_output = response.text
                steps.append({"step": "llm_response", "model": model_name, "content": raw_output})
                
                parsed_json = clean_and_parse_json(raw_output)
                
                # Unwrap inner "answer" if model nested it
                if isinstance(parsed_json, dict) and "answer" in parsed_json and len(parsed_json) == 1:
                    answer = parsed_json["answer"]
                else:
                    answer = parsed_json
                
                return answer, steps  # Success!

            except Exception as e:
                err_msg = str(e)
                print(f"Error on model {model_name} (Attempt {attempt+1}): {err_msg}")
                
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    await asyncio.sleep(2)
                    continue
                elif "404" in err_msg or "NOT_FOUND" in err_msg:
                    break

    if answer is None:
        answer = "Unable to process the data question at this time. Please retry."

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

        # Maintain multi-turn conversation history
        if chat_id not in CHAT_HISTORIES:
            CHAT_HISTORIES[chat_id] = []
        CHAT_HISTORIES[chat_id].append(text)

        # Handle start command
        if text.strip() == "/start":
            reply_text = "Bot is online and ready for data analysis tasks!"
        else:
            # Solve data analysis question
            answer, steps = await solve_data_question(chat_id, CHAT_HISTORIES[chat_id])

            # Append to public JSONL log
            append_to_log(chat_id, text, answer)
            
            # Construct log URL
            log_url = f"{PUBLIC_HOST_URL.rstrip('/')}/run.jsonl"

            # Construct final payload ensuring required format with log_url
            if isinstance(answer, dict):
                final_payload = dict(answer)
                final_payload["log_url"] = log_url
            else:
                final_payload = {
                    "answer": answer,
                    "log_url": log_url
                }

            reply_text = json.dumps(final_payload)

        # Send response back to Telegram API
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
    """Endpoint serving raw JSONL log file for wget/evaluator access."""
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as f:
            content = f.read()
        return Response(content=content, media_type="application/x-ndjson")
    return Response(content="", media_type="application/x-ndjson")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Bot is running"}
