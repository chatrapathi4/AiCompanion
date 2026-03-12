"""
settings.py
-----------
Centralised configuration for the AI Companion.

All tuneable parameters live here so they can be adjusted in one place
without touching module internals.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq LLM ─────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "256"))
MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_HISTORY", "20"))

# ── Emotion Detection ────────────────────────────────────────────────
CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
EMOTION_ANALYSIS_INTERVAL: float = 1.0          # seconds between FER runs
EMOTION_REFRESH_TURNS: int = 3                   # re-detect every N turns
EMOTION_CONFIDENCE_THRESHOLD: float = 0.25       # ignore below this score

# ── Voice I/O ────────────────────────────────────────────────────────
TTS_RATE: int = int(os.getenv("TTS_RATE", "160"))
TTS_VOLUME: float = float(os.getenv("TTS_VOLUME", "1.0"))
STT_ENERGY_THRESHOLD: int = 300
STT_PAUSE_THRESHOLD: float = 1.0
STT_LISTEN_TIMEOUT: int = 10
STT_PHRASE_TIME_LIMIT: int = 30

# ── Session Logging ──────────────────────────────────────────────────
SESSION_LOG_DIR: str = os.getenv("SESSION_LOG_DIR", "sessions")
ENABLE_SESSION_LOGGING: bool = os.getenv("ENABLE_SESSION_LOGGING", "true").lower() == "true"

# ── Text Sentiment ───────────────────────────────────────────────────
SENTIMENT_WEIGHT: float = 0.3   # how much text sentiment influences combined mood
EMOTION_WEIGHT: float = 0.7    # how much facial emotion influences combined mood

# ── Conversation ─────────────────────────────────────────────────────
EXIT_KEYWORDS: set = {"goodbye", "bye", "exit", "quit", "stop", "see you", "i'm done"}
