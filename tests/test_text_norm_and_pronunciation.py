from kokoro_ds.pronunciation import PRONUNCIATION_MAP, QA_PHRASES, apply_pronunciation
from kokoro_ds.text_norm import (
    has_control_characters,
    has_markdown,
    has_unsupported_unicode,
    has_url,
    is_excessively_long,
    normalize_word_for_mms_alignment,
    tokenize_words,
)


def test_tokenize_words_strips_punctuation_keeps_apostrophes():
    assert tokenize_words("Hello, world! Won't you join?") == ["Hello", "world", "Won't", "you", "join"]


def test_tokenize_words_empty_for_punctuation_only():
    assert tokenize_words("... ??? !!!") == []


def test_has_control_characters():
    assert has_control_characters("line one\nline two")
    assert not has_control_characters("normal text")


def test_has_unsupported_unicode():
    assert not has_unsupported_unicode("It's a test’s quote")
    assert has_unsupported_unicode("emoji test \U0001F600")


def test_has_unsupported_unicode_allows_common_latin1_loanwords():
    assert not has_unsupported_unicode("Sauté the café's résumé, naïve jalapeño.")


def test_has_unsupported_unicode_still_rejects_other_scripts():
    assert has_unsupported_unicode("Привет")  # Cyrillic must still be rejected


def test_tokenize_words_keeps_accented_letters_whole():
    assert tokenize_words("Sauté onions and garlic.") == ["Sauté", "onions", "and", "garlic"]


def test_normalize_word_for_mms_alignment_transliterates_accents():
    assert normalize_word_for_mms_alignment("Sauté") == "saute"
    assert normalize_word_for_mms_alignment("café") == "cafe"


def test_normalize_word_for_mms_alignment_lowercases_and_keeps_apostrophe():
    assert normalize_word_for_mms_alignment("Won't") == "won't"


def test_has_url():
    assert has_url("visit https://example.com now")
    assert has_url("visit www.example.com now")
    assert not has_url("no links here")


def test_has_markdown():
    assert has_markdown("**bold**")
    assert has_markdown("`code`")
    assert has_markdown("[link](http://x)")
    assert not has_markdown("plain text with * an asterisk")


def test_is_excessively_long():
    assert not is_excessively_long("short")
    assert is_excessively_long("x" * 500)


def test_apply_pronunciation_wraps_configured_words():
    text = "I am Aether, not created by Kyutai."
    rewritten = apply_pronunciation(text, PRONUNCIATION_MAP)
    assert "[Aether](/" in rewritten
    assert "[Kyutai](/" in rewritten
    assert "not" in rewritten  # untouched words remain plain


def test_apply_pronunciation_leaves_unconfigured_names_unmodified():
    text = "Manifestro develops me in Almaty, Kazakhstan, not Moshi."
    rewritten = apply_pronunciation(text, PRONUNCIATION_MAP)
    assert rewritten == text


def test_apply_pronunciation_is_case_insensitive_and_whole_word():
    rewritten = apply_pronunciation("aethereal aether", {"Aether": "ˈiːθɚ"})
    assert "aethereal" in rewritten and "[aethereal]" not in rewritten
    assert "[aether](/ˈiːθɚ/)" in rewritten


def test_qa_phrases_cover_every_pronunciation_key():
    joined = " ".join(QA_PHRASES).lower()
    for word in PRONUNCIATION_MAP:
        assert word.lower() in joined, f"{word} missing from QA phrases"
