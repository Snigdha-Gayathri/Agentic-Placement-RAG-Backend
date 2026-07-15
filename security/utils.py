"""Utility helpers for text normalization, scoring, and redaction."""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from collections import Counter

from .constants import SECRET_PATTERNS


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\x00", "")
    normalized = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def remove_control_characters(text: str) -> str:
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def looks_like_base64(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 24 or len(compact) % 4 != 0:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        return False
    try:
        base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


def looks_like_hex_blob(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return len(compact) >= 24 and len(compact) % 2 == 0 and bool(re.fullmatch(r"[0-9a-fA-F]+", compact))


def repeated_token_ratio(text: str) -> float:
    tokens = [t.lower() for t in re.findall(r"\b\w+\b", text)]
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    most_common = counts.most_common(1)[0][1]
    return most_common / len(tokens)


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(re.findall(r"\b\w+\b", a.lower()))
    sb = set(re.findall(r"\b\w+\b", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def redact_secrets(text: str) -> str:
    out = text
    for pattern in SECRET_PATTERNS:
        out = re.sub(pattern, "[REDACTED]", out)
    return out
