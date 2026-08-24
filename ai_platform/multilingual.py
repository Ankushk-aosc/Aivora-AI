"""Multilingual AI (spec Part 17) - language detection only.

Real language detection via langdetect (a pure-Python port of Google's
language-detection library, no network/model download needed).
Translation is deliberately NOT implemented: the only realistic local
option (argostranslate) pulls in ~35 packages including spacy and
onnxruntime and downloads per-language model files at runtime - too
heavy a footprint to add reliably on a machine with ~1GB free RAM, and
adding it without testing it actually works here would risk exactly the
kind of unverified claim this registry exists to prevent. Reported as
PARTIAL, not silently upgraded to TESTED.
"""

from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect_langs

# Deterministic results across runs (langdetect is otherwise seeded
# randomly per-process, which would make outputs non-reproducible).
DetectorFactory.seed = 0

_LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
}


@dataclass
class LanguageDetection:
    text: str
    language_code: str
    language_name: str
    confidence: float
    all_candidates: list


def detect_language(text: str) -> LanguageDetection:
    if not text or not text.strip():
        raise ValueError("Cannot detect language of empty text")
    try:
        candidates = detect_langs(text)
    except LangDetectException as e:
        raise ValueError(f"Language detection failed: {e}")

    top = candidates[0]
    return LanguageDetection(
        text=text, language_code=top.lang,
        language_name=_LANGUAGE_NAMES.get(top.lang, top.lang),
        confidence=round(top.prob, 4),
        all_candidates=[{"lang": c.lang, "prob": round(c.prob, 4)} for c in candidates],
    )
