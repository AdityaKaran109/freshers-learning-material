"""Shared Devanagari text normalization for training and evaluation."""

import re
import unicodedata


_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_PUNCT = re.compile(
    r"""[।॥\.\,\!\?\;\:"'\u201c\u201d\u2018\u2019\(\)\[\]\{\}<>@#\$%\^&\*_\+=~`\|/\\\u2014\u2013\-]"""
)
_TAGS = re.compile(r"\[[^\]]*\]|\([^\)]*\)|<[^>]*>")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str | None) -> str:
    """Produce the canonical transcript representation used for WER and CER."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFC", str(text))
    normalized = _TAGS.sub(" ", normalized)
    normalized = normalized.translate(_DIGITS)
    normalized = _PUNCT.sub(" ", normalized)
    normalized = normalized.lower()
    return _WHITESPACE.sub(" ", normalized).strip()


def is_usable(text: str | None, min_chars: int = 2, min_words: int = 1) -> bool:
    """Return whether a normalized transcript is suitable for ASR training."""
    normalized = normalize(text)
    if len(normalized) < min_chars or len(normalized.split()) < min_words:
        return False
    return bool(re.search(r"[\u0900-\u097F a-z]", normalized))
"""
Devanagari text normalization for ASR training and evaluation.

CRITICAL: training and evaluation must use the EXACT same normalizer, and you
must state in your model card which normalization you used. WER numbers are
not comparable across different normalizers -- this is the single most common
way people accidentally publish inflated results.
"""

import re
import unicodedata

# Devanagari digits -> ASCII digits
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Punctuation to strip. Includes danda (।) and double danda (॥).
_PUNCT = re.compile(r"[।॥\.\,\!\?\;\:\"\'“”‘’\(\)\[\]\{\}<>@#\$%\^&\*_\+=~`\|/\\—–\-]")

_WS = re.compile(r"\s+")

# Filler / annotation markers that appear in spontaneous-speech transcripts.
# Inspect your own data before trusting this list -- add whatever you find.
_TAGS = re.compile(r"\[[^\]]*\]|\([^\)]*\)|<[^>]*>")


def normalize(text: str) -> str:
    """Canonical normalizer. Conservative: does not touch nukta or matras,
    since those changes alter actual words."""
    if text is None:
        return ""
    t = str(text)
    t = unicodedata.normalize("NFC", t)
    t = _TAGS.sub(" ", t)          # drop [noise], (laughs), <unk> style tags
    t = t.translate(_DIGITS)
    t = _PUNCT.sub(" ", t)
    t = t.lower()                  # only affects Latin chars in code-mixed text
    t = _WS.sub(" ", t).strip()
    return t


def is_usable(text: str, min_chars: int = 2, min_words: int = 1) -> bool:
    """Filter out empty / junk transcripts after normalization."""
    t = normalize(text)
    if len(t) < min_chars:
        return False
    if len(t.split()) < min_words:
        return False
    # Reject transcripts that are entirely digits or Latin punctuation noise
    if not re.search(r"[ऀ-ॿ a-z]", t):
        return False
    return True


if __name__ == "__main__":
    samples = [
        "हम   बाज़ार गइनी। [noise] ठीक बा ?",
        "मेरे पास १५ रुपये हैं, OK?",
        "   ",
    ]
    for s in samples:
        print(repr(s), "->", repr(normalize(s)), "| usable:", is_usable(s))
