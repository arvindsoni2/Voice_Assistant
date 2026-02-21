"""
models.py — HuggingFace model loading.

All models are loaded once at startup via load_all_models().
Never import and call inference functions here directly — use audio_utils.py
which imports the globals populated by load_all_models().
"""

import torch
from transformers import pipeline
from kokoro import KPipeline
from config import TTS_VOICE

# Module-level globals populated by load_all_models()
whisper_pipe     = None
kokoro_pipeline  = None
DEVICE           = None
models_loaded    = False


def load_all_models():
    global whisper_pipe, kokoro_pipeline, DEVICE, models_loaded

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[models] Using device: {DEVICE}")

    # ------------------------------------------------------------------
    # Whisper STT (openai/whisper-small — 244MB, fast, multilingual)
    # ------------------------------------------------------------------
    print("[models] Loading Whisper STT (openai/whisper-small)...")
    whisper_pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
        device=0 if DEVICE == "cuda" else -1,
        chunk_length_s=30,        # Handle recordings longer than 30s
        return_timestamps=True,   # Required for chunked inference
    )
    print("[models] Whisper STT loaded.")

    # ------------------------------------------------------------------
    # Kokoro TTS (hexgrad/Kokoro-82M)
    # lang_code 'a' = American English (af_*, am_* voices)
    # lang_code 'b' = British English  (bf_*, bm_* voices)
    # ------------------------------------------------------------------
    print(f"[models] Loading Kokoro TTS (voice: {TTS_VOICE})...")
    kokoro_pipeline = KPipeline(lang_code=TTS_VOICE[0])
    print("[models] Kokoro TTS loaded.")

    models_loaded = True
    print("[models] All models ready.")
