"""
voice_output.py
---------------
Offline text-to-speech using pyttsx3.

Converts AI-generated text responses into spoken audio
so the companion can talk to the user.
"""

import pyttsx3
import logging
from logger_setup import get_logger
import settings

# Suppress comtypes logging noise (used by pyttsx3 on Windows)
logging.getLogger('comtypes').setLevel(logging.WARNING)

log = get_logger(__name__)


class VoiceOutput:
    """Offline text-to-speech wrapper around pyttsx3."""

    def __init__(self, rate: int = settings.TTS_RATE, volume: float = settings.TTS_VOLUME, voice_index: int = 0):
        self._rate = rate
        self._volume = volume
        self._voice_index = voice_index
        self._engine = None
        self._init_engine()

    def _init_engine(self) -> None:
        """Create and configure a fresh pyttsx3 engine instance."""
        try:
            # Stop and discard the previous engine to release SAPI5 COM resources
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:
                    pass
                self._engine = None

            # Force-clear pyttsx3's cached engine registry.
            # Without this, pyttsx3.init() returns the same stale instance,
            # causing runAndWait() to silently skip playback on Windows.
            try:
                pyttsx3.engine.Engine._activeEngines.clear()
            except AttributeError:
                pass

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
            voices = self._engine.getProperty("voices")
            if voices and self._voice_index < len(voices):
                self._engine.setProperty("voice", voices[self._voice_index].id)
            log.info("TTS engine configured (rate=%d, volume=%.1f).", self._rate, self._volume)
        except Exception as exc:
            log.error("Error during engine init: %s", exc)
            self._engine = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Convert *text* to speech and play it through the speakers."""
        if not text or not text.strip():
            return

        log.info("Speaking: %s", text)

        # Re-init the engine each time to avoid the pyttsx3 hang-on-second-call
        # bug that is common on Windows.
        self._init_engine()

        if self._engine is None:
            log.error("Engine is None. Speech cannot be played.")
            return

        try:
            self._engine.say(text)
            self._engine.runAndWait()
            log.debug("Speech finished.")
        except Exception as exc:
            log.error("Error during speech: %s", exc)
            self._engine = None

    def set_rate(self, rate: int) -> None:
        """Change the speech rate dynamically."""
        self._engine.setProperty("rate", rate)

    def set_volume(self, volume: float) -> None:
        """Change the volume dynamically (0.0 – 1.0)."""
        self._engine.setProperty("volume", max(0.0, min(1.0, volume)))

    def list_voices(self) -> list:
        """Return a list of available system voices (id, name, languages)."""
        voices = self._engine.getProperty("voices")
        info = []
        for v in voices:
            info.append({"id": v.id, "name": v.name, "languages": v.languages})
        return info
