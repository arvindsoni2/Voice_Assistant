"""
audio_utils.py — Audio conversion and TTS synthesis helpers.

Functions:
  webm_to_wav_numpy(bytes)     → np.ndarray (16kHz float32, for Whisper)
  synthesize_long_text(text)   → np.ndarray (24kHz float32 audio, via Kokoro)
  numpy_to_wav_b64(array, sr)  → str (base64-encoded WAV, for JSON transport)
"""

import io
import wave
import base64

import numpy as np
from pydub import AudioSegment


# ---------------------------------------------------------------------------
# STT helper — WebM/Opus bytes (from browser) → 16kHz mono float32 numpy
# ---------------------------------------------------------------------------

def webm_to_wav_numpy(audio_bytes: bytes, content_type: str = "") -> np.ndarray:
    """
    Convert raw audio bytes from the browser (usually WebM/Opus) to a
    16 kHz mono float32 numpy array suitable for the Whisper pipeline.

    Requires ffmpeg installed on the system (used internally by pydub).
    """
    # Determine format hint for pydub; let ffmpeg auto-detect if unsure
    if "webm" in content_type:
        fmt = "webm"
    elif "mp4" in content_type or "mpeg" in content_type:
        fmt = "mp4"
    else:
        fmt = None  # pydub/ffmpeg will auto-detect

    try:
        if fmt:
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        else:
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode audio (is ffmpeg installed?): {e}"
        ) from e

    # Normalize to mono, 16 kHz, 16-bit PCM
    seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    # Convert raw PCM bytes → int16 → float32 in [-1.0, 1.0]
    pcm_bytes = seg.raw_data
    int16_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int16_array.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# TTS helper — text → 24kHz float32 numpy  (Kokoro TTS)
# ---------------------------------------------------------------------------

def synthesize_long_text(text: str) -> np.ndarray:
    """
    Synthesize speech using Kokoro TTS.
    Returns a 24kHz float32 mono numpy array.
    Kokoro handles long text natively — no manual sentence splitting needed.
    """
    import models  # lazy import to avoid circular dependency at module load
    from config import TTS_VOICE, TTS_SPEED

    chunks = []
    for _, _, audio in models.kokoro_pipeline(text, voice=TTS_VOICE, speed=TTS_SPEED):
        chunks.append(audio)

    if not chunks:
        return np.zeros(int(0.5 * 24000), dtype=np.float32)

    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Transport helper — numpy audio → base64-encoded WAV string
# ---------------------------------------------------------------------------

def numpy_to_wav_b64(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """
    Encode a float32 numpy audio array as a base64 WAV string suitable
    for embedding in a JSON response and playing in the browser.
    """
    # Clamp to prevent clipping artifacts
    audio = np.clip(audio, -1.0, 1.0)
    int16_array = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(int16_array.tobytes())

    return base64.b64encode(buf.getvalue()).decode("utf-8")
