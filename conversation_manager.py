"""
conversation_manager.py
-----------------------
Orchestrates the full conversation loop:

  Camera → Emotion Detection → AI speaks → User responds (mic) →
  Speech-to-Text → Send to LLM (with emotion + sentiment) → AI speaks → repeat

Gracefully handles missing hardware by falling back to text-only mode.
Integrates sentiment analysis, session logging, and emotion analytics.
"""

import threading
import time
import queue
from typing import Optional
from emotion_detector import EmotionDetector
from voice_output import VoiceOutput
from voice_input import VoiceInput
from groq_chat import GroqChat
from sentiment_analyzer import SentimentAnalyzer
from session_logger import SessionLogger
from emotion_analytics import EmotionAnalytics
from logger_setup import get_logger
import settings

log = get_logger(__name__)


# Fallback opening messages if the LLM API is unreachable at startup
FALLBACK_OPENERS = {
    "happy":    "You look really happy today! What's making you smile?",
    "sad":      "You look a little sad. Do you want to talk about what's bothering you?",
    "angry":    "You seem a bit frustrated. Want to vent? I'm here to listen.",
    "surprise": "Oh, you look surprised! Did something unexpected happen?",
    "fear":     "You seem a little anxious. Take a deep breath — I'm right here with you.",
    "disgust":  "Something seems to be bothering you. Want to talk about it?",
    "neutral":  "Hey there! How's your day going so far?",
}


class ConversationManager:
    """Orchestrates emotion detection, voice I/O, LLM chat, sentiment, and analytics."""

    def __init__(
        self,
        use_camera: bool = True,
        use_voice: bool = True,
        show_camera: bool = True,
        mic_index: Optional[int] = None,
        cam_index: int = 0,
    ):
        self._use_camera = use_camera
        self._use_voice = use_voice
        self._show_camera = show_camera
        self._mic_index = mic_index
        self._cam_index = cam_index
        self._current_emotion = "neutral"
        self._turn_count = 0

        # Module instances (created in start())
        self._emotion_detector: Optional[EmotionDetector] = None
        self._voice_out: Optional[VoiceOutput] = None
        self._voice_in: Optional[VoiceInput] = None
        self._chat: Optional[GroqChat] = None

        # New modules
        self._sentiment = SentimentAnalyzer()
        self._session_logger = SessionLogger()
        self._analytics = EmotionAnalytics()

        # Threading
        self._stop_event = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._voice_queue: queue.Queue = queue.Queue()
        self._listening = False
        self._last_analysis_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise all modules and run the main conversation loop."""
        self._init_modules()

        try:
            # AI speaks first (opening message based on initial emotion snapshot)
            if self._use_camera:
                self._take_emotion_snapshot()

            opening = self._generate_opening()
            self._output(opening)

            # Enter the conversation loop
            # (main thread handles camera preview; voice listens in bg thread)
            self._conversation_loop()

        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_modules(self) -> None:
        """Create and initialise each subsystem."""

        # ── Camera / Emotion ──
        if self._use_camera:
            try:
                self._emotion_detector = EmotionDetector(camera_index=self._cam_index)
                self._emotion_detector.start_camera()
            except RuntimeError as exc:
                log.warning("Camera unavailable: %s — continuing without emotion detection.", exc)
                self._use_camera = False

        # ── Voice Output (TTS) ──
        if self._use_voice:
            try:
                self._voice_out = VoiceOutput()
            except RuntimeError as exc:
                log.warning("TTS unavailable: %s — falling back to text output.", exc)
                self._use_voice = False

        # ── Voice Input (STT) ──
        if self._use_voice:
            try:
                self._voice_in = VoiceInput(device_index=self._mic_index)
            except RuntimeError as exc:
                log.warning("Microphone unavailable: %s — falling back to text input.", exc)
                self._use_voice = False

        # ── LLM Chat ──
        self._chat = GroqChat()

    # ------------------------------------------------------------------
    # Core conversation loop
    # ------------------------------------------------------------------

    def _conversation_loop(self) -> None:
        """Main loop: listen → analyse → send to LLM → speak → log → repeat.

        If a camera is active the main thread keeps refreshing the preview
        (OpenCV requires this on Windows).  Voice input runs in a background
        thread and posts results to ``self._voice_queue``.
        """
        while True:
            # --- kick off voice / text input -------------------------
            user_text = self._get_input_async()

            if user_text is None:
                self._output("I didn't catch that. Could you try again?")
                continue

            # Check for exit intent
            if self._wants_to_exit(user_text):
                farewell = "It was really nice talking to you. Take care and see you next time!"
                self._output(farewell)
                break

            # Analyse text sentiment and combine with facial emotion
            sentiment_result = self._sentiment.analyze(user_text)
            combined = self._sentiment.combined_mood(
                self._current_emotion, user_text,
                emotion_weight=settings.EMOTION_WEIGHT,
                sentiment_weight=settings.SENTIMENT_WEIGHT,
            )
            self._analytics.record(combined)

            # Get AI response with combined mood context
            reply = self._chat.get_response(user_text, combined)

            # Speak the response
            self._output(reply)

            # Log this turn
            self._session_logger.log_turn(
                user_text=user_text,
                ai_reply=reply,
                facial_emotion=self._current_emotion,
                text_sentiment=sentiment_result,
                combined_mood=combined,
            )
            self._turn_count += 1

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _get_input_async(self) -> str | None:
        """Get user input while keeping the camera preview alive.

        If voice mode is active, starts a background thread for listening
        and pumps camera frames on the main thread until speech arrives.
        If text mode, just calls input() directly (camera still pumped).
        """
        if self._use_voice and self._voice_in is not None:
            # Start listening in a background thread
            self._listening = True
            listener = threading.Thread(target=self._listen_worker, daemon=True)
            listener.start()

            # While waiting, keep pumping the camera preview on main thread
            while self._listening:
                self._pump_camera_frame()
                try:
                    result = self._voice_queue.get_nowait()
                    return result
                except queue.Empty:
                    pass
            # One final check
            try:
                return self._voice_queue.get_nowait()
            except queue.Empty:
                return None
        else:
            # Text input — just read from stdin
            try:
                text = input("\nYou: ").strip()
                return text if text else None
            except EOFError:
                return None

    def _listen_worker(self) -> None:
        """Background thread: listen for speech and post result to queue."""
        try:
            text = self._voice_in.listen()
            self._voice_queue.put(text)
        except Exception as exc:
            log.warning("Voice listen error: %s", exc)
            self._voice_queue.put(None)
        finally:
            self._listening = False

    def _pump_camera_frame(self) -> None:
        """Capture one camera frame, run throttled analysis, and show preview.

        Called from the main thread so cv2.imshow works on Windows.
        """
        if not self._use_camera or self._emotion_detector is None:
            # No camera — just sleep briefly so we don't busy-loop
            time.sleep(0.05)
            return

        try:
            frame = self._emotion_detector.capture_frame()

            # Throttled FER analysis (heavy, ~1 s interval)
            now = time.time()
            if now - self._last_analysis_time >= settings.EMOTION_ANALYSIS_INTERVAL:
                threading.Thread(
                    target=self._run_single_analysis,
                    args=(frame,),
                    daemon=True,
                ).start()
                self._last_analysis_time = now

            # Show preview (main thread — works on Windows)
            if self._show_camera:
                self._emotion_detector.show_preview(
                    frame=frame,
                    label=self._current_emotion,
                    duration_ms=1,
                )
        except Exception as exc:
            log.warning("Camera frame error: %s", exc)
            time.sleep(0.1)

    def _get_input(self) -> str | None:
        """Get user input from microphone or keyboard (blocking, no camera)."""
        if self._use_voice and self._voice_in is not None:
            return self._voice_in.listen()
        else:
            try:
                text = input("\nYou: ").strip()
                return text if text else None
            except EOFError:
                return None

    def _output(self, text: str) -> None:
        """Output text via TTS or print to console."""
        print(f"\nAI Companion: {text}")
        if self._use_voice and self._voice_out is not None:
            self._voice_out.speak(text)

    def _take_emotion_snapshot(self) -> None:
        """Capture a single frame and detect emotion (used at startup)."""
        if not self._use_camera or self._emotion_detector is None:
            return
        try:
            frame = self._emotion_detector.capture_frame()
            emotion, confidence = self._emotion_detector.detect_emotion(frame=frame)
            if confidence >= settings.EMOTION_CONFIDENCE_THRESHOLD:
                self._current_emotion = emotion
                self._analytics.record(emotion)
                self._session_logger.log_emotion(emotion)
            log.info("Initial emotion snapshot: %s (%.2f)", emotion, confidence)
        except Exception as exc:
            log.warning("Initial emotion snapshot failed: %s", exc)

    def _run_single_analysis(self, frame) -> None:
        """Helper to run one FER analysis turn without blocking the preview."""
        if self._emotion_detector is None:
            return
        try:
            emotion, confidence = self._emotion_detector.detect_emotion(frame=frame)
            if confidence >= settings.EMOTION_CONFIDENCE_THRESHOLD:
                self._current_emotion = emotion
                self._analytics.record(emotion)
                self._session_logger.log_emotion(emotion)
        except Exception:
            pass

    def _generate_opening(self) -> str:
        """Generate an opening message using the LLM, with a fallback."""
        try:
            return self._chat.get_opening_message(self._current_emotion)
        except Exception:
            return FALLBACK_OPENERS.get(self._current_emotion, FALLBACK_OPENERS["neutral"])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _wants_to_exit(text: str) -> bool:
        """Check whether the user wants to end the conversation."""
        lower = text.lower().strip()
        return any(kw in lower for kw in settings.EXIT_KEYWORDS)

    def _shutdown(self) -> None:
        """Release all resources, print analytics, and save the session."""
        self._stop_event.set()

        if self._emotion_detector is not None:
            self._emotion_detector.release()

        # Print emotion analytics report
        self._analytics.print_report()

        # Save session transcript
        filepath = self._session_logger.save()
        if filepath:
            print(f"\n  💾  Session transcript saved → {filepath}")

        log.info("Session ended. Goodbye!")
