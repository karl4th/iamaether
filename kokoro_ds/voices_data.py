"""Voice pool definitions and disjointness invariants.

All voice IDs follow the Kokoro v1 naming convention: a two-letter prefix
encodes language + gender (``af`` = American female, ``am`` = American male,
``bf`` = British female, ``bm`` = British male). The prefix selects which
Kokoro language pipeline ("a" for American English, "b" for British English)
must be used to synthesize that voice.
"""
from __future__ import annotations

SAMPLE_RATE = 24_000
CHANNELS = 2

ASSISTANT_VOICE = "af_heart"
ASSISTANT_SPEED = 1.0
ASSISTANT_LANG_CODE = "a"

TRAIN_USER_VOICES: tuple[str, ...] = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "am_adam",
    "am_echo",
    "am_fenrir",
    "am_michael",
    "am_onyx",
    "am_puck",
    "bf_emma",
    "bf_isabella",
    "bm_fable",
    "bm_george",
    "bm_lewis",
)

VALIDATION_USER_VOICES: tuple[str, ...] = (
    "af_sarah",
    "am_eric",
    "bf_alice",
    "bm_daniel",
)

TEST_USER_VOICES: tuple[str, ...] = (
    "af_river",
    "af_sky",
    "am_liam",
    "am_santa",
    "bf_lily",
)

VOICE_POOLS_BY_SPLIT: dict[str, tuple[str, ...]] = {
    "train": TRAIN_USER_VOICES,
    "validation": VALIDATION_USER_VOICES,
    "test": TEST_USER_VOICES,
}

USER_SPEED_MIN = 0.92
USER_SPEED_MAX = 1.08

TURN_PAUSE_MIN_SECONDS = 0.25
TURN_PAUSE_MAX_SECONDS = 0.75


def lang_code_for_voice(voice_id: str) -> str:
    """Return the Kokoro pipeline language code ("a" or "b") for a voice id."""
    prefix = voice_id.split("_", 1)[0]
    if prefix.startswith("a"):
        return "a"
    if prefix.startswith("b"):
        return "b"
    raise ValueError(f"Cannot determine language pipeline for voice {voice_id!r}")


def assert_voice_pools_valid(available_voices: set[str] | None = None) -> None:
    """Validate the disjointness and membership invariants required by spec.

    Raises ``ValueError`` with an actionable message on any violation.
    """
    pools = {
        "train": set(TRAIN_USER_VOICES),
        "validation": set(VALIDATION_USER_VOICES),
        "test": set(TEST_USER_VOICES),
    }

    names = list(pools.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = pools[a] & pools[b]
            if overlap:
                raise ValueError(
                    f"User voice pools for {a!r} and {b!r} are not disjoint: {sorted(overlap)}"
                )

    for split, pool in pools.items():
        if ASSISTANT_VOICE in pool:
            raise ValueError(
                f"Assistant voice {ASSISTANT_VOICE!r} must not appear in the {split!r} user pool"
            )
        if len(pool) != len(set(pool)):
            raise ValueError(f"Duplicate voice ids inside the {split!r} user pool")

    if available_voices is not None:
        configured = {ASSISTANT_VOICE, *TRAIN_USER_VOICES, *VALIDATION_USER_VOICES, *TEST_USER_VOICES}
        missing = configured - available_voices
        if missing:
            raise ValueError(
                f"The following configured voices are not available in the installed "
                f"Kokoro voice pack: {sorted(missing)}"
            )
