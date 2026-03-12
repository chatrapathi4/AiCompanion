"""
session_logger.py
-----------------
Persists conversation sessions as structured JSON files.

Each session records:
- Timestamps for every turn
- User and AI messages
- Detected emotions and sentiment
- A summary generated at the end of the session

This makes the project data-driven and auditable — great for a portfolio.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional
from logger_setup import get_logger
import settings

log = get_logger(__name__)


class SessionLogger:
    """Records a conversation session to a JSON file."""

    def __init__(self, session_dir: str = settings.SESSION_LOG_DIR):
        self._session_dir = session_dir
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._start_time = datetime.now(timezone.utc).isoformat()
        self._turns: list[dict] = []
        self._emotion_history: list[str] = []

        if settings.ENABLE_SESSION_LOGGING:
            os.makedirs(self._session_dir, exist_ok=True)
            log.info("Session logger active — id=%s", self._session_id)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def emotion_history(self) -> list[str]:
        return list(self._emotion_history)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def log_turn(
        self,
        user_text: str,
        ai_reply: str,
        facial_emotion: str = "neutral",
        text_sentiment: Optional[dict] = None,
        combined_mood: str = "neutral",
    ) -> None:
        """Record one conversation turn."""
        turn = {
            "turn": len(self._turns) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user_text,
            "ai": ai_reply,
            "facial_emotion": facial_emotion,
            "text_sentiment": text_sentiment,
            "combined_mood": combined_mood,
        }
        self._turns.append(turn)
        self._emotion_history.append(facial_emotion)
        log.debug("Turn %d recorded", turn["turn"])

    def log_emotion(self, emotion: str) -> None:
        """Record an emotion observation (from the background thread)."""
        self._emotion_history.append(emotion)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def build_summary(self) -> dict:
        """Build an end-of-session analytics summary."""
        end_time = datetime.now(timezone.utc).isoformat()
        emotion_counts: dict[str, int] = {}
        for e in self._emotion_history:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"

        return {
            "session_id": self._session_id,
            "start": self._start_time,
            "end": end_time,
            "total_turns": len(self._turns),
            "emotion_distribution": emotion_counts,
            "dominant_emotion": dominant,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Optional[str]:
        """Write the session to disk as JSON. Returns the file path or None."""
        if not settings.ENABLE_SESSION_LOGGING:
            return None

        summary = self.build_summary()
        data = {
            "summary": summary,
            "turns": self._turns,
        }

        filepath = os.path.join(self._session_dir, f"session_{self._session_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log.info("Session saved → %s", filepath)
        return filepath
