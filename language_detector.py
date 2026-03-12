"""
language_detector.py
--------------------
Detects the language of user input text using script detection + langdetect.

Returns an ISO 639-1 code (e.g. 'en', 'hi', 'te') so the LLM
can respond in the same language and the frontend can set TTS/STT locale.
"""

import re
from langdetect import detect, LangDetectException
from logger_setup import get_logger

log = get_logger(__name__)

# Languages we actively support for TTS voice matching
SUPPORTED_LANGUAGES = {
    "en", "hi", "te", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "ur",
    "es", "fr", "de", "ja", "ko", "zh-cn", "zh-tw", "ar", "pt", "ru", "it",
}

# langdetect returns 'zh-cn' / 'zh-tw'; normalise to primary subtag for simple cases
_NORMALIZE = {"zh-cn": "zh", "zh-tw": "zh"}

# Unicode script ranges for reliable detection of native scripts
_SCRIPT_PATTERNS = {
    "te": re.compile(r"[\u0C00-\u0C7F]"),  # Telugu
    "hi": re.compile(r"[\u0900-\u097F]"),  # Devanagari (Hindi)
    "ta": re.compile(r"[\u0B80-\u0BFF]"),  # Tamil
    "kn": re.compile(r"[\u0C80-\u0CFF]"),  # Kannada
    "ml": re.compile(r"[\u0D00-\u0D7F]"),  # Malayalam
    "bn": re.compile(r"[\u0980-\u09FF]"),  # Bengali
    "gu": re.compile(r"[\u0A80-\u0AFF]"),  # Gujarati
    "pa": re.compile(r"[\u0A00-\u0A7F]"),  # Gurmukhi (Punjabi)
    "ar": re.compile(r"[\u0600-\u06FF]"),  # Arabic
    "ja": re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]"),  # Japanese
    "ko": re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]"),  # Korean
    "zh": re.compile(r"[\u4E00-\u9FFF]"),  # Chinese
    "ru": re.compile(r"[\u0400-\u04FF]"),  # Cyrillic (Russian)
}


def _detect_by_script(text: str) -> str | None:
    """Detect language by Unicode script (more reliable for native scripts)."""
    for lang, pattern in _SCRIPT_PATTERNS.items():
        if pattern.search(text):
            return lang
    return None


def detect_language(text: str) -> str:
    """Detect the language of *text*.

    Uses script-based detection first (reliable for native scripts),
    then falls back to langdetect for Latin-script languages.

    Returns:
        ISO 639-1 language code (e.g. ``'te'``, ``'hi'``, ``'en'``).
        Falls back to ``'en'`` when detection fails or text is too short.
    """
    if not text or len(text.strip()) < 3:
        return "en"

    # First: try script-based detection (very reliable for native scripts)
    script_lang = _detect_by_script(text)
    if script_lang:
        log.debug("Detected by script: %s for text: %.40s…", script_lang, text)
        return script_lang

    # Fallback: use langdetect for Latin-script languages
    try:
        code = detect(text)
        code = _NORMALIZE.get(code, code)
        # For very short ASCII-only text, langdetect is unreliable —
        # default to English to avoid false positives (e.g. 'cy', 'af')
        if code not in SUPPORTED_LANGUAGES and text.isascii():
            code = "en"
        log.debug("Detected by langdetect: %s for text: %.40s…", code, text)
        return code
    except LangDetectException:
        return "en"
