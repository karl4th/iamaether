"""Pure-python text normalization shared by source validation and forced alignment.

No torch/torchaudio dependency so this module is importable (and testable)
without any heavy ML packages installed.
"""
from __future__ import annotations

import re
import unicodedata

_LATIN1_LETTER_RANGE = r"À-ÖØ-öø-ÿ"
"""Latin-1 Supplement letters (À..ÿ), skipping × (U+00D7) and ÷ (U+00F7),
which sit inside that block but are math symbols, not letters. Covers common
English loanwords: sauté, café, jalapeño, naïve, résumé, etc."""

_WORD_RE = re.compile(rf"[A-Za-z{_LATIN1_LETTER_RANGE}']+")

_ALLOWED_EXTRA_UNICODE = {
    "‘",  # left single quote
    "’",  # right single quote / apostrophe
    "“",  # left double quote
    "”",  # right double quote
    "—",  # em dash
    "–",  # en dash
    "…",  # ellipsis
}

_LATIN1_ACCENTED_LETTERS = frozenset(
    chr(c) for c in (*range(0xC0, 0xD7), *range(0xD8, 0xF7), *range(0xF8, 0x100))
)

MAX_UTTERANCE_CHARS = 400


def tokenize_words(text: str) -> list[str]:
    """Extract alignment-relevant word tokens, stripping punctuation.

    Preserves original casing and apostrophes so a produced word can be
    traced back to the source text by case-insensitive substring search.
    """
    return [w for w in _WORD_RE.findall(text) if w.strip("'")]


def has_control_characters(text: str) -> bool:
    for ch in text:
        if ch in ("\n", "\t", "\r"):
            return True
        category = unicodedata.category(ch)
        if category.startswith("C"):  # Cc, Cf, Co, Cs
            return True
    return False


def has_unsupported_unicode(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if 0x20 <= code <= 0x7E:
            continue
        if ch in _ALLOWED_EXTRA_UNICODE or ch in _LATIN1_ACCENTED_LETTERS:
            continue
        return True
    return False


_URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)


def has_url(text: str) -> bool:
    return bool(_URL_RE.search(text))


_MARKDOWN_PATTERNS = (
    re.compile(r"\*\*[^*]+\*\*"),
    re.compile(r"(?<!\w)_[^_]+_(?!\w)"),
    re.compile(r"`[^`]+`"),
    re.compile(r"^\s*#{1,6}\s"),
    re.compile(r"^\s*[-*]\s"),
    re.compile(r"\[[^\]]+\]\([^)]+\)"),
)


def has_markdown(text: str) -> bool:
    return any(p.search(text) for p in _MARKDOWN_PATTERNS)


def is_excessively_long(text: str, max_chars: int = MAX_UTTERANCE_CHARS) -> bool:
    return len(text) > max_chars


def normalize_word_for_mms_alignment(word: str) -> str:
    """Transliterate a word to the plain a-z(+apostrophe) inventory MMS_FA expects.

    Accented Latin letters are decomposed and their combining diacritic is
    dropped (e.g. "Sauté" -> "saute"), so a legitimate accented English
    loanword can still be forced-aligned. The original spelling is preserved
    separately wherever the word is stored (see ``tokenize_words``); only the
    aligner's input is transliterated.
    """
    decomposed = unicodedata.normalize("NFKD", word.lower())
    return "".join(ch for ch in decomposed if (ch.isalpha() and ord(ch) < 128) or ch == "'")
