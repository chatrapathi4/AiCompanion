"""
vision_analyzer.py
------------------
Vision analysis using LLaMA 4 Scout multimodal model via Groq.

Analyzes camera frames to describe objects, read text, and provide
context-aware responses like Google Gemini Live.
"""

import base64
import os
import time
from typing import Optional, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from groq import Groq
import settings
from logger_setup import get_logger

log = get_logger(__name__)

# Vision model (LLaMA 4 Scout with multimodal capabilities)
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Cooldown between proactive observations (seconds)
PROACTIVE_COOLDOWN = 15


class VisionAnalyzer:
    """Analyzes images using LLaMA 4 Scout vision model."""

    def __init__(self):
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        
        self._client = Groq(api_key=api_key)
        self._last_proactive_time = 0
        self._last_scene_hash = ""
        log.info("VisionAnalyzer initialized with model: %s", VISION_MODEL)

    def analyze_image(
        self,
        image_base64: str,
        user_question: Optional[str] = None,
        emotion: str = "neutral",
        lang: str = "en",
    ) -> str:
        """Analyze an image and return a description or answer.

        Args:
            image_base64: Base64-encoded image (with or without data URL prefix).
            user_question: Optional question about the image.
            emotion: User's current facial emotion.
            lang: Language code for the response.

        Returns:
            AI's response about the image content.
        """
        # Strip data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        # Build the prompt based on whether there's a question
        if user_question:
            prompt = (
                f"The user is showing you something and asking: '{user_question}'\n"
                f"Their facial expression shows they are feeling {emotion}.\n"
                f"Look at the image carefully and answer their question helpfully. "
                f"Be concise (2-3 sentences). Respond in {lang}."
            )
        else:
            prompt = (
                f"Briefly describe what you see in this image (1-2 sentences). "
                f"Focus on the main subject or any interesting details. "
                f"Respond in {lang}."
            )

        try:
            response = self._client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=256,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            log.error("Vision API error: %s", exc)
            return ""

    def should_proactive_observe(self) -> bool:
        """Check if enough time has passed for a proactive observation."""
        now = time.time()
        if now - self._last_proactive_time >= PROACTIVE_COOLDOWN:
            return True
        return False

    def proactive_observe(
        self,
        image_base64: str,
        emotion: str = "neutral",
        lang: str = "en",
    ) -> Tuple[bool, str]:
        """Proactively observe the scene and comment if something interesting.

        Returns:
            (should_speak, message) — whether to speak and what to say.
        """
        if not self.should_proactive_observe():
            return (False, "")

        self._last_proactive_time = time.time()

        # Strip data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        prompt = (
            f"You are an AI companion observing the user through their camera. "
            f"The user's facial expression shows they are feeling: {emotion}.\n\n"
            f"Look at this image. If you notice something interesting worth commenting on "
            f"(like the user holding an object, reading something, doing an activity, "
            f"or if their emotional state suggests they might need support), "
            f"make a brief, friendly observation or ask a caring question.\n\n"
            f"If nothing notable is happening (just a person sitting normally), "
            f"respond with exactly 'NOTHING_NOTABLE'.\n\n"
            f"Keep your response to 1-2 sentences. Be warm and conversational. "
            f"Respond in the user's language (code: {lang})."
        )

        try:
            response = self._client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=150,
                temperature=0.8,
            )
            result = response.choices[0].message.content.strip()
            
            if "NOTHING_NOTABLE" in result.upper():
                return (False, "")
            
            return (True, result)
        except Exception as exc:
            log.error("Proactive vision error: %s", exc)
            return (False, "")
