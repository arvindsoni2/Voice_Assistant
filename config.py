import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-prod")
FLASK_DEBUG      = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
FLASK_PORT       = int(os.environ.get("FLASK_PORT", "5000"))

GROQ_MODEL    = "llama-3.3-70b-versatile"  # Free tier, ~100-200ms latency, high quality
MAX_TOKENS    = 512   # Keep replies concise — TTS latency grows with length

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep your replies concise and conversational — "
    "ideally 1-3 sentences. Avoid using markdown, bullet points, code blocks, or any "
    "special formatting since your responses will be spoken aloud. Speak naturally."
)

WHISPER_SAMPLE_RATE = 16000  # Hz — Whisper requires 16kHz input
TTS_SAMPLE_RATE     = 24000  # Hz — Kokoro outputs 24kHz

# Kokoro TTS voice settings
# Voice options: af_heart, af_sky, af_bella, am_adam, am_michael (American)
#                bm_george, bm_lewis, bf_emma, bf_isabella (British)
TTS_VOICE = "af_heart"  # Warm American female voice
TTS_SPEED = 1.0         # 0.5–2.0; 1.0 = normal speed

# RAG — real-time web search via DuckDuckGo (no API key needed)
RAG_ENABLED     = True  # Set False to skip web search (faster, offline-friendly)
RAG_MAX_RESULTS = 3     # Number of search results to retrieve per query
