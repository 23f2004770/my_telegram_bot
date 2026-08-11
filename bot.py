import json
import time
import os
import threading
from fastapi import FastAPI, Response
import uvicorn
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# --- Environment Variables ---
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]
LOG_URL = os.environ.get("LOG_URL", "https://your-render-app.onrender.com/logs")
PORT = int(os.environ.get("PORT", 10000))

# --- FastAPI setup for serving log file ---
web_app = FastAPI()
LOG_FILE = "run.jsonl"

@web_app.get("/logs")
def get_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            content = f.read()
        return Response(content=content, media_type="application/json", headers={"Content-Disposition": "attachment; filename=run.jsonl"})
    return {"error": "Log file not found yet."}

def run_web():
    uvicorn.run(web_app, host="0.0.0.0", port=PORT)

# --- Telegram Bot Setup ---
client = OpenAI(base_url="https://aipipe.org/openai/v1", api_key=AIPIPE_TOKEN)
conversation_history = {}

def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_exit = {"type": "incoming", "chat_id": chat_id, "text": user_text}
    log_event(log_exit)

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    system_prompt = (
        "You are a careful data analyst. The user's LAST message asks a data-analysis "
        "question and tells you exactly what JSON shape to reply with. Work out the "
        "real answer. Reply with ONLY that exact JSON object and absolutely nothing else — no "
        "explanation, no markdown, no code fences, just the raw JSON."
    )
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": system_prompt}] + history[-6:],
    )
    reply_text = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply_text})

    try:
        parsed = json.loads(reply_text)
    except json.JSONDecodeError:
        start, end = reply_text.find("{"), reply_text.rfind("}")
        parsed = json.loads(reply_text[start:end + 1])
    
    # Ensure parsed is a dictionary before item assignment
    if isinstance(parsed, dict):
        parsed["log_url"] = LOG_URL
        final_reply = json.dumps(parsed)
    else:
        final_reply = json.dumps({"result": parsed, "log_url": LOG_URL})

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)

def main():
    # Start FastAPI server in a background thread so Render detects an open port
    t = threading.Thread(target=run_web, daemon=True)
    t.start()

    # Start Telegram Bot
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot and Web Server are running...")
    app.run_polling()

if __name__ == "__main__":
    main()
