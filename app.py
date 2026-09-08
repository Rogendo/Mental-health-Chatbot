import os
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


app = Flask(__name__)

PRIMARY_MODEL = os.getenv("PSYCHAI_MODEL", "gpt-5.2")
FAST_MODEL = os.getenv("PSYCHAI_FAST_MODEL", "gpt-5-nano")
MAX_HISTORY_MESSAGES = 10

CRISIS_TERMS = (
    "kill myself",
    "suicide",
    "suicidal",
    "end my life",
    "self harm",
    "self-harm",
    "hurt myself",
    "overdose",
    "can't go on",
    "cant go on",
)

SYSTEM_PROMPT = """
You are psychAI, a warm mental wellbeing companion for reflection, coping skills,
and practical next steps. You are not a doctor or therapist and you do not diagnose.

Your style:
- Start by acknowledging the user's feeling in plain language.
- Ask at most one gentle follow-up question when useful.
- Offer grounded, low-risk techniques such as breathing, journaling, reframing,
  sleep hygiene, planning, and reaching out to trusted support.
- Keep responses concise enough for a chat UI.
- Encourage professional help for severe, persistent, or worsening symptoms.
- If the user may be in immediate danger or mentions self-harm, urge them to
  contact local emergency services or a crisis hotline now and to reach a trusted
  person nearby. Do not provide harmful instructions.
"""


def get_client():
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAI()


def is_crisis_message(message):
    normalized = message.lower()
    return any(term in normalized for term in CRISIS_TERMS)


def crisis_response():
    return (
        "I am really sorry you are carrying this right now. If you might hurt "
        "yourself or feel unable to stay safe, please contact emergency services "
        "now or call/text 988 in the U.S. and Canada. If you are elsewhere, use "
        "your local crisis line or emergency number. If possible, move away from "
        "anything you could use to harm yourself and reach out to someone you "
        "trust who can stay with you. You do not have to handle the next few "
        "minutes alone."
    )


def fallback_response(message, mood=None):
    if is_crisis_message(message):
        return crisis_response()

    mood_line = f" Since you marked your mood as {mood}, " if mood else " "
    return (
        f"I hear you.{mood_line}let's make this moment a little more manageable. "
        "Try naming one feeling, one body sensation, and one small thing you can "
        "do in the next ten minutes. A simple reset is: breathe in for 4, hold for "
        "2, breathe out for 6, repeat five times. What feels like the main weight "
        "on your mind right now?"
    )


def build_messages(message, history, mood):
    messages = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
    if mood:
        messages.append(
            {
                "role": "system",
                "content": f"The user selected their current mood as: {mood}.",
            }
        )

    for item in history[-MAX_HISTORY_MESSAGES:]:
        role = item.get("role")
        content = item.get("content", "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:1200]})

    messages.append({"role": "user", "content": message})
    return messages


def ask_openai(message, history=None, mood=None):
    history = history or []
    if is_crisis_message(message):
        return crisis_response(), "crisis-protocol"

    client = get_client()
    if client is None:
        return fallback_response(message, mood), "local-fallback"

    response = client.responses.create(
        model=PRIMARY_MODEL,
        input=build_messages(message, history, mood),
        max_output_tokens=450,
    )
    return response.output_text.strip(), PRIMARY_MODEL


@app.route("/")
def home():
    return render_template(
        "index.html",
        primary_model=PRIMARY_MODEL,
        fast_model=FAST_MODEL,
        has_api_key=bool(os.getenv("OPENAI_API_KEY")),
    )


@app.route("/api/config")
def config():
    return jsonify(
        {
            "primaryModel": PRIMARY_MODEL,
            "fastModel": FAST_MODEL,
            "apiReady": bool(get_client()),
            "generatedAt": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    history = payload.get("history") or []
    mood = (payload.get("mood") or "").strip()

    if not message:
        return jsonify({"error": "Message is required."}), 400

    try:
        reply, model_used = ask_openai(message, history, mood)
    except Exception as exc:
        app.logger.exception("OpenAI response failed")
        reply = fallback_response(message, mood)
        model_used = f"local-fallback ({exc.__class__.__name__})"

    return jsonify(
        {
            "reply": reply,
            "model": model_used,
            "crisis": is_crisis_message(message),
        }
    )


@app.route("/get")
def get_bot_response():
    user_input = (request.args.get("msg") or "").strip()
    if not user_input:
        return ""
    reply, _model_used = ask_openai(user_input)
    return reply


if __name__ == "__main__":
    app.run(debug=True)
