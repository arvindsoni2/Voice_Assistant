/**
 * app.js — Voice Assistant frontend logic
 *
 * Flow:
 *   PTT hold → MediaRecorder captures audio →
 *   POST /api/transcribe → show user bubble →
 *   POST /api/chat → show assistant bubble + play audio
 */

'use strict';

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const chatContainer = document.getElementById('chat-container');
const statusEl      = document.getElementById('status');
const pttButton     = document.getElementById('ptt-button');
const clearBtn      = document.getElementById('clear-btn');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let mediaRecorder      = null;
let recordedChunks     = [];
let isRecording        = false;
let currentAudio       = null;   // HTMLAudioElement currently playing
let conversationHistory = [];    // [{role: "user"|"assistant", content: "..."}]
let processingLock     = false;  // Prevent overlapping requests
let thinkingBubble     = null;   // Reference to animated "thinking" bubble row

// ---------------------------------------------------------------------------
// Status helper
// ---------------------------------------------------------------------------
function setStatus(text, state = 'idle') {
  statusEl.textContent = text;
  statusEl.className = `status ${state}`;
}

// ---------------------------------------------------------------------------
// Chat bubble helpers
// ---------------------------------------------------------------------------
function removeWelcome() {
  const welcome = chatContainer.querySelector('.welcome');
  if (welcome) welcome.remove();
}

function appendBubble(text, role) {
  removeWelcome();
  const row = document.createElement('div');
  row.classList.add('bubble-row', role);

  const label = document.createElement('div');
  label.classList.add('bubble-label');
  label.textContent = role === 'user' ? 'You' : 'Assistant';

  const bubble = document.createElement('div');
  bubble.classList.add('bubble', role);
  bubble.textContent = text;

  row.appendChild(label);
  row.appendChild(bubble);
  chatContainer.appendChild(row);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return row;
}

function showThinkingBubble() {
  removeWelcome();
  const row = document.createElement('div');
  row.classList.add('bubble-row', 'assistant');

  const label = document.createElement('div');
  label.classList.add('bubble-label');
  label.textContent = 'Assistant';

  const bubble = document.createElement('div');
  bubble.classList.add('bubble', 'assistant', 'thinking-bubble');
  bubble.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';

  row.appendChild(label);
  row.appendChild(bubble);
  chatContainer.appendChild(row);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  thinkingBubble = row;
  return row;
}

function removeThinkingBubble() {
  if (thinkingBubble) {
    thinkingBubble.remove();
    thinkingBubble = null;
  }
}

// ---------------------------------------------------------------------------
// Audio playback
// ---------------------------------------------------------------------------
function playBase64Wav(b64String, onEnd) {
  // Stop any audio currently playing
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  const binary = atob(b64String);
  const bytes  = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  const blob = new Blob([bytes], { type: 'audio/wav' });
  const url  = URL.createObjectURL(blob);

  currentAudio = new Audio(url);
  currentAudio.addEventListener('ended', () => {
    URL.revokeObjectURL(url);
    currentAudio = null;
    if (onEnd) onEnd();
  });
  currentAudio.addEventListener('error', (e) => {
    console.error('Audio playback error:', e);
    URL.revokeObjectURL(url);
    currentAudio = null;
    if (onEnd) onEnd();
  });

  currentAudio.play().catch((err) => {
    console.error('Audio play() rejected:', err);
    if (onEnd) onEnd();
  });
}

// ---------------------------------------------------------------------------
// MediaRecorder — recording
// ---------------------------------------------------------------------------
async function startRecording() {
  if (isRecording || processingLock) return;

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setStatus('Microphone access denied — check browser settings', 'error');
    return;
  }

  recordedChunks = [];

  // Choose best supported MIME type (WebM preferred; Safari needs mp4)
  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus'
    : MediaRecorder.isTypeSupported('audio/webm')
      ? 'audio/webm'
      : 'audio/mp4';

  try {
    mediaRecorder = new MediaRecorder(stream, { mimeType });
  } catch {
    mediaRecorder = new MediaRecorder(stream);  // fallback: browser default
  }

  mediaRecorder.addEventListener('dataavailable', (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  });

  mediaRecorder.addEventListener('stop', () => {
    // Release microphone indicator in browser immediately
    stream.getTracks().forEach((t) => t.stop());
    handleRecordingStop();
  });

  mediaRecorder.start(100);  // Collect in 100 ms chunks
  isRecording = true;
  pttButton.classList.add('recording');
  pttButton.setAttribute('aria-pressed', 'true');
  setStatus('Recording…', 'recording');
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  if (mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  isRecording = false;
  pttButton.classList.remove('recording');
  pttButton.setAttribute('aria-pressed', 'false');
}

// ---------------------------------------------------------------------------
// Core processing pipeline
// ---------------------------------------------------------------------------
async function handleRecordingStop() {
  if (!recordedChunks.length) {
    setStatus('Hold to speak', 'idle');
    return;
  }

  processingLock = true;
  pttButton.classList.add('disabled');

  const mimeType  = recordedChunks[0]?.type || 'audio/webm';
  const audioBlob = new Blob(recordedChunks, { type: mimeType });
  recordedChunks  = [];

  // --- Step 1: Transcribe ---
  setStatus('Transcribing…', 'processing');

  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');

  let transcript;
  try {
    const res  = await fetch('/api/transcribe', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Transcription failed');
    transcript = (data.transcript || '').trim();
  } catch (err) {
    setStatus(`Error: ${err.message}`, 'error');
    unlock();
    return;
  }

  if (!transcript) {
    setStatus("Didn't catch that — try again", 'error');
    setTimeout(() => setStatus('Hold to speak', 'idle'), 2500);
    unlock();
    return;
  }

  // Show user message immediately
  appendBubble(transcript, 'user');

  // --- Step 2: Chat (LLM + RAG + TTS) ---
  setStatus('Searching web & thinking…', 'processing');
  showThinkingBubble();

  // Declare outside try so they're accessible after the block
  let replyText = '';
  let audioB64  = '';
  let sources   = [];

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: transcript,
        history: conversationHistory,   // prior completed turns only
      }),
    });
    const data = await res.json();
    if (!res.ok || (!data.reply && data.error)) throw new Error(data.error || 'Chat failed');
    replyText = (data.reply   || '').trim();
    audioB64  =  data.audio_b64 || '';
    sources   =  data.sources   || [];
    if (data.tts_error) console.warn('TTS error (text reply available):', data.tts_error);
  } catch (err) {
    removeThinkingBubble();
    setStatus(`Error: ${err.message}`, 'error');
    setTimeout(() => setStatus('Hold to speak', 'idle'), 3000);
    unlock();
    return;
  }

  removeThinkingBubble();

  if (replyText) {
    appendBubble(replyText, 'assistant');
    // Update history AFTER full successful round-trip
    conversationHistory.push({ role: 'user',      content: transcript });
    conversationHistory.push({ role: 'assistant', content: replyText  });
    // Show source links if web search was used
    if (sources.length) {
      appendSources(sources);
    }
  }

  // --- Step 3: Play audio ---
  if (audioB64) {
    setStatus('Speaking…', 'speaking');
    playBase64Wav(audioB64, () => {
      setStatus('Hold to speak', 'idle');
      unlock();
    });
  } else {
    setStatus('Hold to speak', 'idle');
    unlock();
  }
}

function unlock() {
  processingLock = false;
  pttButton.classList.remove('disabled');
}

// ---------------------------------------------------------------------------
// Source links — shown below assistant bubble when RAG retrieves results
// ---------------------------------------------------------------------------
function appendSources(sources) {
  const div = document.createElement('div');
  div.className = 'sources';
  div.innerHTML = sources
    .filter(s => s.link)
    .map(s => `<a href="${s.link}" target="_blank" rel="noopener noreferrer">${s.title || s.link}</a>`)
    .join('');
  if (div.children.length) {
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
}

// ---------------------------------------------------------------------------
// PTT button event listeners
// ---------------------------------------------------------------------------

// Mouse
pttButton.addEventListener('mousedown',  (e) => { e.preventDefault(); startRecording(); });
pttButton.addEventListener('mouseup',    (e) => { e.preventDefault(); stopRecording();  });
pttButton.addEventListener('mouseleave', (e) => { if (isRecording) stopRecording();     });

// Touch (mobile)
pttButton.addEventListener('touchstart', (e) => {
  e.preventDefault();
  startRecording();
}, { passive: false });

pttButton.addEventListener('touchend', (e) => {
  e.preventDefault();
  stopRecording();
}, { passive: false });

pttButton.addEventListener('touchcancel', (e) => {
  e.preventDefault();
  if (isRecording) stopRecording();
}, { passive: false });

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Clear button
// ---------------------------------------------------------------------------
clearBtn.addEventListener('click', () => {
  // Stop any current playback
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  conversationHistory = [];

  // Clear all chat bubbles and restore welcome screen
  chatContainer.innerHTML = `
    <div class="welcome">
      <div class="welcome-icon">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" y1="19" x2="12" y2="23"/>
          <line x1="8" y1="23" x2="16" y2="23"/>
        </svg>
      </div>
      <h2>Hello! I'm your voice assistant.</h2>
      <p>Hold the button below and speak. I'll listen, think, and talk back.</p>
    </div>
  `;
  setStatus('Hold to speak', 'idle');
});

// ---------------------------------------------------------------------------
// Startup — verify models are ready
// ---------------------------------------------------------------------------
(async function checkHealth() {
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    if (!data.models_loaded) {
      setStatus('Loading models, please wait…', 'processing');
      // Poll every 3 seconds until models are loaded
      setTimeout(checkHealth, 3000);
    } else {
      setStatus('Hold to speak', 'idle');
    }
  } catch {
    // Server might not be up yet — retry
    setTimeout(checkHealth, 3000);
  }
})();
