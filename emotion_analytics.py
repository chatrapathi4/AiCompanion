"""
emotion_analytics.py
--------------------
Tracks and visualises emotion data across a conversation session.

Provides a live running summary and a pretty end-of-session report
printed to the console.
"""

from collections import Counter
from logger_setup import get_logger

log = get_logger(__name__)

# Bar-chart characters
_BAR = "█"
_BAR_WIDTH = 20

# Emotion → display colour (ANSI escape codes for terminal)
_COLOURS = {
    "happy":    "\033[92m",  # green
    "sad":      "\033[94m",  # blue
    "angry":    "\033[91m",  # red
    "surprise": "\033[93m",  # yellow
    "fear":     "\033[95m",  # magenta
    "disgust":  "\033[33m",  # dark yellow
    "neutral":  "\033[37m",  # white
}
_RESET = "\033[0m"


class EmotionAnalytics:
    """Accumulates emotion observations and produces reports."""

    def __init__(self):
        self._records: list[str] = []

    def record(self, emotion: str) -> None:
        """Record a single emotion observation."""
        self._records.append(emotion)

    @property
    def total(self) -> int:
        return len(self._records)

    def distribution(self) -> dict[str, float]:
        """Return emotion distribution as percentages."""
        if not self._records:
            return {}
        counts = Counter(self._records)
        total = len(self._records)
        return {e: round(c / total * 100, 1) for e, c in counts.most_common()}

    def dominant(self) -> str:
        """Return the most frequent emotion."""
        if not self._records:
            return "neutral"
        return Counter(self._records).most_common(1)[0][0]

    def print_report(self) -> None:
        """Print a colourful end-of-session emotion report to the console."""
        dist = self.distribution()
        if not dist:
            print("\n  No emotion data recorded.\n")
            return

        print("\n" + "=" * 54)
        print("   📊  Emotion Analytics — Session Summary")
        print("=" * 54)
        print(f"   Total observations : {self.total}")
        print(f"   Dominant emotion   : {self.dominant().capitalize()}")
        print("-" * 54)

        for emotion, pct in dist.items():
            bar_len = int(pct / 100 * _BAR_WIDTH)
            colour = _COLOURS.get(emotion, "")
            bar = colour + _BAR * bar_len + _RESET
            print(f"   {emotion:<10s} {bar} {pct:5.1f}%")

        print("=" * 54 + "\n")
        log.info("Emotion report printed (dominant=%s)", self.dominant())
