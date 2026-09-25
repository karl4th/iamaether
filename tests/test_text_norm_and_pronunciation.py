from kokoro_ds.pronunciation import PRONUNCIATION_MAP, QA_PHRASES, apply_pronunciation
from kokoro_ds.text_norm import (
    has_control_characters,
    has_markdown,
    has_unsupported_unicode,
    has_url,
    is_excessively_long,
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
    text = "I am Aether, not Moshi."
    rewritten = apply_pronunciation(text, PRONUNCIATION_MAP)
    assert "[Aether](/" in rewritten
    assert "[Moshi](/" in rewritten
    assert "not" in rewritten  # untouched words remain plain


def test_apply_pronunciation_is_case_insensitive_and_whole_word():
    rewritten = apply_pronunciation("aethereal aether", {"Aether": "ˈiːθɚ"})
    assert "aethereal" in rewritten and "[aethereal]" not in rewritten
    assert "[aether](/ˈiːθɚ/)" in rewritten


def test_qa_phrases_cover_every_pronunciation_key():
    joined = " ".join(QA_PHRASES).lower()
    for word in PRONUNCIATION_MAP:
        assert word.lower() in joined, f"{word} missing from QA phrases"
