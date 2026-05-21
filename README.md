# psychAI

psychAI is a Flask-based mental wellbeing platform with GPT-powered supportive chat, mood check-ins, quick grounding prompts, a breathing reset, and crisis resource surfacing.

It is designed for reflection and coping support, not diagnosis or emergency care. If someone may be in immediate danger, they should contact local emergency services or a crisis line right away.

## Model choice

The app defaults to `gpt-5.2` for the main wellbeing coach because OpenAI's current model catalog lists GPT-5.2 as a featured frontier model recommended for demanding production work. It also defines `gpt-5-nano` as the fast, low-cost model slot for lightweight classification or routing work.

You can override either model without code changes:

```powershell
$env:PSYCHAI_MODEL="gpt-5.2"
$env:PSYCHAI_FAST_MODEL="gpt-5-nano"
```

## Run locally

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY="your_api_key_here"
flask --app app --debug run
```

Then open `http://127.0.0.1:5000`.

If `OPENAI_API_KEY` is not set, psychAI still runs with a local fallback response so the UI can be developed and reviewed.

## Key routes

- `GET /` renders the psychAI platform.
- `POST /api/chat` returns a GPT-backed wellbeing response.
- `GET /api/config` returns model and API readiness metadata.
- `GET /get?msg=...` preserves the older chatbot route for compatibility.
