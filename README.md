# Voice Assistant

A web-based personal voice assistant with push-to-talk recording, real-time speech-to-text, LLM-powered responses with live web search (RAG), and natural text-to-speech playback — all in the browser.

## Demo

1. Hold the microphone button and speak
2. Release to send — your speech is transcribed instantly
3. The assistant searches the web for up-to-date context, then answers
4. The reply is spoken aloud using a natural AI voice

## Architecture

```
Browser (PTT)
    │
    ▼ audio/webm (MediaRecorder)
POST /api/transcribe
    │  pydub + ffmpeg converts WebM → 16kHz mono
    │  Whisper (openai/whisper-small) → transcript
    ▼
POST /api/chat
    │  DuckDuckGo web search (RAG) → top 3 results injected as context
    │  Groq API (llama-3.3-70b-versatile) → reply text
    │  Kokoro-82M TTS → 24kHz WAV → base64
    ▼
Browser plays audio + shows chat bubble + source links
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **STT** | `openai/whisper-small` (HuggingFace Transformers) |
| **LLM** | Groq API — `llama-3.3-70b-versatile` (free tier) |
| **RAG** | DuckDuckGo web search via `duckduckgo-search` |
| **TTS** | `hexgrad/Kokoro-82M` via `kokoro` package |
| **Server** | Flask 3 |
| **Frontend** | Vanilla JS + MediaRecorder API |

## Features

- **Push-to-talk** — hold the button to record, release to process
- **Real-time RAG** — every query searches DuckDuckGo for current information
- **Multi-turn memory** — conversation history maintained throughout the session
- **Source links** — web sources shown below each assistant reply
- **Mobile-friendly** — touch events supported alongside mouse
- **Graceful TTS failure** — text reply shown even if audio synthesis fails

## Prerequisites

- Python 3.10+
- `ffmpeg` installed on the system
- A free [Groq API key](https://console.groq.com) (no credit card needed)

```bash
sudo apt install ffmpeg   # Ubuntu/Debian
# or
brew install ffmpeg       # macOS
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/arvindsoni2/Voice_Assistant.git
cd Voice_Assistant

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies (CPU-only PyTorch)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Configure environment
cp .env.example .env
# Edit .env and add your Groq API key:
#   GROQ_API_KEY=gsk_...

# 5. Run the server
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

> **First startup:** Models download automatically (~500MB total for Whisper + Kokoro). This takes 1–3 minutes. Subsequent starts are instant (cached in `~/.cache/huggingface/`).

## Configuration

All settings are in [`config.py`](config.py):

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_VOICE` | `af_heart` | Kokoro voice (see voices below) |
| `TTS_SPEED` | `1.0` | Speech speed (0.5–2.0) |
| `RAG_ENABLED` | `True` | Enable/disable web search |
| `RAG_MAX_RESULTS` | `3` | Number of search results per query |
| `MAX_TOKENS` | `512` | Max LLM reply length |

### Available Voices

| Voice | Style |
|-------|-------|
| `af_heart` | American female, warm (default) |
| `af_sky` | American female, clear |
| `af_bella` | American female, expressive |
| `am_adam` | American male |
| `am_michael` | American male, deep |
| `bm_george` | British male, professional |
| `bm_lewis` | British male |
| `bf_emma` | British female |
| `bf_isabella` | British female, warm |

## Project Structure

```
Voice_Assistant/
├── app.py              # Flask server — routes & orchestration
├── models.py           # HuggingFace model loading (Whisper + Kokoro)
├── audio_utils.py      # Audio conversion & TTS synthesis
├── rag.py              # DuckDuckGo web search (RAG)
├── config.py           # Environment variables & constants
├── templates/
│   └── index.html      # Single-page web UI
├── static/
│   ├── style.css       # Chat bubbles, PTT button, animations
│   └── app.js          # MediaRecorder, fetch pipeline, audio playback
├── requirements.txt
├── .env.example        # Environment template
└── .gitignore
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve the web UI |
| `GET` | `/health` | `{"models_loaded": true/false}` |
| `POST` | `/api/transcribe` | `multipart/form-data` audio → `{"transcript": "..."}` |
| `POST` | `/api/chat` | `{"message": "...", "history": [...]}` → `{"reply": "...", "audio_b64": "...", "sources": [...]}` |

## GPU Support

For faster inference on CUDA 12.1, replace the PyTorch lines in `requirements.txt`:

```
torch==2.3.1+cu121
torchaudio==2.3.1+cu121
--extra-index-url https://download.pytorch.org/whl/cu121
```

## Getting a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com) and sign up (no credit card required)
2. Navigate to **API Keys** → **Create API Key**
3. Add it to `.env` as `GROQ_API_KEY=gsk_...`

Free tier limits: **14,400 requests/day**, **30 req/min** — more than enough for personal use.

## License

MIT
