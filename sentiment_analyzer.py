"""
sentiment_analyzer.py
---------------------
Text-based sentiment analysis using TextBlob.

Complements the facial emotion detector by analysing what the user *says*
(not just how they look).  The ``combined_mood`` helper merges both signals
into a single label so the LLM can receive a richer emotional context.
"""

from textblob import TextBlob
from logger_setup import get_logger

log = get_logger(__name__)

# Mapping from (polarity_bucket, subjectivity_bucket) → mood label
_POLARITY_MAP = {
    "very_positive": "happy",
    "positive":      "happy",
    "neutral":       "neutral",
    "negative":      "sad",
    "very_negative": "sad",
}

# Priority ranking for mood labels (higher = more intense)
_MOOD_PRIORITY = {
    "happy": 3, "surprise": 3,
    "neutral": 0,
    "sad": 2, "fear": 2, "angry": 2, "disgust": 2,
}


class SentimentAnalyzer:
    """Lightweight text sentiment analyser powered by TextBlob."""

    @staticmethod
    def analyze(text: str) -> dict:
        """Analyse sentiment of *text*.

        Returns:
            dict with keys: polarity (-1..1), subjectivity (0..1), label.
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.5:
            label = "very_positive"
        elif polarity > 0.1:
            label = "positive"
        elif polarity >= -0.1:
            label = "neutral"
        elif polarity >= -0.5:
            label = "negative"
        else:
            label = "very_negative"

        mood = _POLARITY_MAP[label]
        log.debug("Sentiment: polarity=%.2f subjectivity=%.2f label=%s mood=%s",
                  polarity, subjectivity, label, mood)

        return {
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "label": label,
            "mood": mood,
        }

    @staticmethod
    def combined_mood(
        facial_emotion: str,
        text: str,
        emotion_weight: float = 0.7,
        sentiment_weight: float = 0.3,
    ) -> str:
        """Merge facial emotion and text sentiment into a single mood label.

        Uses a simple priority-weighted approach:
        - If both agree, return the shared mood.
        - If they disagree, favour the signal with higher weight × priority.

        Returns:
            A mood label string (e.g. "happy", "sad", "neutral").
        """
        result = SentimentAnalyzer.analyze(text)
        text_mood = result["mood"]

        if facial_emotion == text_mood:
            return facial_emotion

        face_score = _MOOD_PRIORITY.get(facial_emotion, 0) * emotion_weight
        text_score = _MOOD_PRIORITY.get(text_mood, 0) * sentiment_weight

        chosen = facial_emotion if face_score >= text_score else text_mood
        log.debug("Combined mood: face=%s(%.1f) text=%s(%.1f) → %s",
                  facial_emotion, face_score, text_mood, text_score, chosen)
        return chosen
