"""
app.py — Flask web server for the personal voice assistant.

Routes:
  GET  /                  → Serve index.html
  GET  /health            → Model load status
  POST /api/transcribe    → Audio → transcript (Whisper STT)
  POST /api/chat          → Transcript + history → reply + audio (Groq LLM + RAG + Kokoro TTS)
"""

import subprocess
import sys

from flask import Flask, jsonify, render_template, request
from groq import Groq

import audio_utils
import models
import rag
from config import (
    FLASK_DEBUG,
    FLASK_PORT,
    FLASK_SECRET_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_TOKENS,
    RAG_ENABLED,
    SYSTEM_PROMPT,
    TTS_SAMPLE_RATE,
)

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit

# Groq client — initialized once, reused across requests
groq_client = Groq(api_key=GROQ_API_KEY)

_VALID_ROLES    = {"user", "assistant"}
_MAX_HISTORY    = 40   # 20 turns × 2 messages each
_MAX_MSG_CHARS  = 4000  # per-message content cap


def _sanitize_history(raw):
    """
    Validate conversation history from the client.

    Rejects entries with unknown roles (blocks system-prompt injection),
    enforces a message cap (blocks token-cost abuse), and ensures each
    content field is a plain string.
    """
    if not isinstance(raw, list):
        return []
    clean = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role    = item.get("role")
        content = item.get("content")
        if role not in _VALID_ROLES or not isinstance(content, str):
            continue
        clean.append({"role": role, "content": content[:_MAX_MSG_CHARS]})
    return clean[-_MAX_HISTORY:]  # keep only the most recent turns


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": models.models_loaded,
        "device": models.DEVICE,
    })


# ---------------------------------------------------------------------------
# STT endpoint — audio blob → transcript
# ---------------------------------------------------------------------------

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request", "transcript": ""}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return jsonify({"error": "Empty audio file", "transcript": ""}), 400

    if not models.models_loaded:
        return jsonify({"error": "Models still loading, please wait", "transcript": ""}), 503

    try:
        # Convert browser audio (WebM/Opus) → 16kHz mono float32 numpy
        content_type = audio_file.content_type or ""
        float32_audio = audio_utils.webm_to_wav_numpy(audio_bytes, content_type)

        # Run Whisper STT
        result = models.whisper_pipe(
            {"array": float32_audio, "sampling_rate": 16000},
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
        transcript = result["text"].strip()
    except RuntimeError as e:
        return jsonify({"error": str(e), "transcript": ""}), 500
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {e}", "transcript": ""}), 500

    return jsonify({"transcript": transcript})


# ---------------------------------------------------------------------------
# Chat endpoint — transcript + history → LLM reply + TTS audio
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body", "reply": "", "audio_b64": ""}), 400

    user_message = (data.get("message") or "").strip()
    history = _sanitize_history(data.get("history"))  # validated {role, content} list

    if not user_message:
        return jsonify({"error": "Empty message", "reply": "", "audio_b64": ""}), 400

    if not models.models_loaded:
        return jsonify({"error": "Models still loading", "reply": "", "audio_b64": ""}), 503

    # --- RAG: single web search → context string + source links ---
    import datetime
    today = datetime.date.today().strftime("%B %d, %Y")

    sources = []
    if RAG_ENABLED:
        context, sources = rag.search(user_message)
        if context:
            augmented_system = (
                SYSTEM_PROMPT
                + f"\n\nToday's date is {today}."
                + "\n\n"
                + context
                + "\n\n"
                + "IMPORTANT: The web search results above contain current, real-time information. "
                + "You MUST base your answer on these results rather than your training knowledge, "
                + "especially for anything time-sensitive (prices, events, news, versions, etc.). "
                + "Cite the information naturally in your spoken response. "
                + "Only fall back to your training knowledge if the search results are clearly unrelated to the question."
            )
        else:
            augmented_system = SYSTEM_PROMPT + f"\n\nToday's date is {today}."
    else:
        augmented_system = SYSTEM_PROMPT + f"\n\nToday's date is {today}."

    # Build Groq messages — augmented system prompt first, then history, then new user turn
    messages = (
        [{"role": "system", "content": augmented_system}]
        + history
        + [{"role": "user", "content": user_message}]
    )

    # --- LLM call (Groq) ---
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
        )
        reply_text = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": f"Groq API error: {e}", "reply": "", "audio_b64": ""}), 502

    # --- TTS synthesis (Kokoro) ---
    audio_b64 = ""
    tts_error = None
    try:
        audio_array = audio_utils.synthesize_long_text(reply_text)
        audio_b64 = audio_utils.numpy_to_wav_b64(audio_array, sample_rate=TTS_SAMPLE_RATE)
    except Exception as e:
        tts_error = str(e)
        # Return the text reply even if TTS fails — better than total failure

    response_body = {
        "reply": reply_text,
        "audio_b64": audio_b64,
        "audio_format": "wav",
        "sources": sources,
    }
    if tts_error:
        response_body["tts_error"] = tts_error

    return jsonify(response_body)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _check_ffmpeg():
    """Warn early if ffmpeg is missing (pydub will fail silently later)."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise FileNotFoundError
        print("[startup] ffmpeg found.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(
            "\n[startup] WARNING: ffmpeg not found on PATH!\n"
            "  Audio transcription will fail.\n"
            "  Install with:  sudo apt install ffmpeg\n",
            file=sys.stderr,
        )


if __name__ == "__main__":
    _check_ffmpeg()
    print("[startup] Loading HuggingFace models (this may take 1–2 minutes)...")
    models.load_all_models()
    print(f"[startup] Starting Flask on http://0.0.0.0:{FLASK_PORT}")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=FLASK_DEBUG)
