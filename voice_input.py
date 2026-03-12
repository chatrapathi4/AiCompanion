"""
voice_input.py
--------------
Speech-to-text using the SpeechRecognition library.

Captures audio from the microphone, adjusts for ambient noise,
and transcribes it to text using Google's free speech recognition API.
"""

import speech_recognition as sr
from logger_setup import get_logger
import settings

log = get_logger(__name__)


class VoiceInput:
    """Microphone-based speech-to-text using SpeechRecognition."""

    def __init__(self, device_index: int = None, energy_threshold: int = settings.STT_ENERGY_THRESHOLD, pause_threshold: float = settings.STT_PAUSE_THRESHOLD):
        """
        Initialise the speech recogniser.

        Args:
            device_index:     Index of the microphone to use (None for default).
            energy_threshold: Minimum audio energy to consider as speech.
            pause_threshold:  Seconds of silence before a phrase is considered complete.
        """
        self._device_index = device_index
        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = energy_threshold
        self._recognizer.pause_threshold = pause_threshold
        self._recognizer.dynamic_energy_threshold = True

        # Verify that a microphone is available
        mic_list = sr.Microphone.list_microphone_names()
        if not mic_list:
            raise RuntimeError(
                "No microphone detected. Please connect a microphone and try again."
            )
        log.info("Microphone ready. Available devices: %d", len(mic_list))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def listen(self, timeout: int = settings.STT_LISTEN_TIMEOUT, phrase_time_limit: int = settings.STT_PHRASE_TIME_LIMIT) -> str | None:
        """Listen to the microphone and return transcribed text.

        Args:
            timeout:           Max seconds to wait for speech to begin.
            phrase_time_limit: Max seconds of speech to capture.

        Returns:
            The transcribed text, or None if recognition failed.
        """
        try:
            with sr.Microphone(device_index=self._device_index) as source:
                # Brief ambient-noise calibration
                log.debug("Adjusting for ambient noise...")
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)

                log.info("Listening... (speak now)")
                audio = self._recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )

            # Transcribe using Google's free Speech Recognition API
            text = self._recognizer.recognize_google(audio)
            log.info("Transcribed: %s", text)
            return text

        except sr.WaitTimeoutError:
            log.warning("No speech detected within the timeout period.")
            return None

        except sr.UnknownValueError:
            log.warning("Could not understand the audio. Please try again.")
            return None

        except sr.RequestError as exc:
            log.error("Google Speech Recognition service error: %s", exc)
            return None

        except OSError as exc:
            log.error("Microphone error: %s", exc)
            return None
