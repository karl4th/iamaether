"""Static generation configuration.

Values here are intentionally not CLI-configurable unless the spec calls for
it, so that a given (dialogue id, config version) pair is always
reproducible. Bump ``CONFIG_VERSION`` whenever a value below changes in a way
that would alter previously generated audio, so checksums correctly detect
staleness during resume.
"""
from __future__ import annotations

from dataclasses import dataclass

from .voices_data import (
    ASSISTANT_SPEED,
    ASSISTANT_VOICE,
    TURN_PAUSE_MAX_SECONDS,
    TURN_PAUSE_MIN_SECONDS,
    USER_SPEED_MAX,
    USER_SPEED_MIN,
)

CONFIG_VERSION = "aether-kokoro-v1"

FADE_SECONDS = 0.015
"""Fade-in/out applied to every synthesized utterance to avoid clicks."""

INTER_CHUNK_PAUSE_SECONDS = 0.12
"""Pause inserted between concatenated Kokoro generator chunks within one utterance."""

MIN_UTTERANCE_SECONDS = 0.05
"""Utterances shorter than this after synthesis are treated as a synthesis failure."""

MAX_DIALOGUE_SECONDS = 600.0
"""Sanity ceiling; a longer assembled dialogue fails the record instead of silently succeeding."""

CLIPPING_PEAK_THRESHOLD = 0.999
"""Absolute sample value at/above this is considered clipping."""

ALIGNMENT_MIN_CONFIDENCE = 0.55
"""Assistant utterances whose mean alignment confidence is below this are flagged for review."""

SPEAKER_MAIN_LABEL = "SPEAKER_MAIN"


@dataclass(frozen=True)
class GenerationConfig:
    config_version: str = CONFIG_VERSION
    sample_rate: int = 24_000
    channels: int = 2
    assistant_voice: str = ASSISTANT_VOICE
    assistant_speed: float = ASSISTANT_SPEED
    user_speed_min: float = USER_SPEED_MIN
    user_speed_max: float = USER_SPEED_MAX
    pause_min_seconds: float = TURN_PAUSE_MIN_SECONDS
    pause_max_seconds: float = TURN_PAUSE_MAX_SECONDS
    fade_seconds: float = FADE_SECONDS
    inter_chunk_pause_seconds: float = INTER_CHUNK_PAUSE_SECONDS
    clipping_peak_threshold: float = CLIPPING_PEAK_THRESHOLD
    alignment_min_confidence: float = ALIGNMENT_MIN_CONFIDENCE

    def as_dict(self) -> dict:
        return {
            "config_version": self.config_version,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "assistant_voice": self.assistant_voice,
            "assistant_speed": self.assistant_speed,
            "user_speed_min": self.user_speed_min,
            "user_speed_max": self.user_speed_max,
            "pause_min_seconds": self.pause_min_seconds,
            "pause_max_seconds": self.pause_max_seconds,
            "fade_seconds": self.fade_seconds,
            "inter_chunk_pause_seconds": self.inter_chunk_pause_seconds,
            "clipping_peak_threshold": self.clipping_peak_threshold,
            "alignment_min_confidence": self.alignment_min_confidence,
        }


DEFAULT_CONFIG = GenerationConfig()
