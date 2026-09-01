from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "has", "have", "he", "her", "his", "i", "in", "is", "it", "its",
    "of", "on", "or", "she", "that", "the", "their", "there", "they", "this",
    "to", "was", "were", "will", "with", "you", "your",
}


def words(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def content_words(text: str) -> set[str]:
    return {token for token in words(text) if token not in STOPWORDS and len(token) > 2}


def sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text.strip()) if part.strip()]


def jaccard(left: str, right: str) -> float:
    left_words = content_words(left)
    right_words = content_words(right)
    union = left_words | right_words
    if not union:
        return 0.0
    return len(left_words & right_words) / len(union)
