"""Explicit, visible pronunciation overrides for TTS synthesis only.

Kokoro (misaki G2P) accepts inline IPA overrides using the markup
``[word](/ipa/)``. This module rewrites only the text handed to Kokoro for
synthesis; the original orthographic text is always kept untouched in
manifests, receipts, and forced-alignment ground truth (alignment must match
what a listener actually hears as words, not phoneme markup).

The IPA values below are a starting point, not a guarantee of correctness.
Run the Colab pronunciation QA cell (see COLAB.md) and listen before bulk
generation; edit this dict freely and rerun that cell until every name
sounds right.
"""
from __future__ import annotations

import re

# word (case-insensitive, whole-word match) -> IPA string for Kokoro's
# "[word](/ipa/)" markup syntax. Only override words where Kokoro's default
# G2P was confirmed wrong by listening (see COLAB.md section 6). Manifestro,
# Almaty, Kazakhstan, and Moshi were tried and rejected on listening -
# Kokoro's own default pronunciation for those is used instead (no entry
# here means the word passes through unmodified).
PRONUNCIATION_MAP: dict[str, str] = {
    "Aether": "ˈiːθɚ",
    "Kyutai": "kjuːˈtaɪ",
}

QA_PHRASES: tuple[str, ...] = (
    "My name is Aether.",
    "I am being developed by Manifestro.",
    "I am being developed in Almaty, Kazakhstan.",
    "I am Aether, not Moshi.",
    "I was not created by Kyutai.",
)


def _compile_patterns(mapping: dict[str, str]) -> list[tuple[re.Pattern, str]]:
    # Longest word first so overlapping/substring names never shadow each other.
    ordered = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    return [(re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE), ipa) for word, ipa in ordered]


def apply_pronunciation(text: str, mapping: dict[str, str] | None = None) -> str:
    """Rewrite configured proper nouns as Kokoro ``[word](/ipa/)`` markup.

    The original matched surface form (preserving the source's casing) is
    kept as the visible word inside the markup, only the phoneme hint changes.
    """
    mapping = mapping if mapping is not None else PRONUNCIATION_MAP
    result = text
    for pattern, ipa in _compile_patterns(mapping):
        result = pattern.sub(lambda m: f"[{m.group(0)}](/{ipa}/)", result)
    return result


def print_pronunciation_map(mapping: dict[str, str] | None = None) -> None:
    mapping = mapping if mapping is not None else PRONUNCIATION_MAP
    print("Configured pronunciation overrides (surface word -> IPA, Kokoro '[word](/ipa/)' markup):")
    for word, ipa in mapping.items():
        print(f"  {word!r:16s} -> /{ipa}/")
