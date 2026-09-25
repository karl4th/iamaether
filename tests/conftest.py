import numpy as np
import pytest

from kokoro_ds.schema import SourceRecord, Turn


def make_record(record_id: str, category: str = "general_conversation", n_turns: int = 4) -> SourceRecord:
    turns = []
    for i in range(n_turns):
        speaker = "user" if i % 2 == 0 else "assistant"
        turns.append(Turn(speaker=speaker, text=f"This is {speaker} turn number {i}."))
    return SourceRecord(id=record_id, category=category, turns=tuple(turns), line_no=1)


def sine_wave(seconds: float, sample_rate: int = 24_000, freq: float = 220.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def stub_synth_fn():
    def _synth(text: str, voice: str, speed: float, lang_code: str):
        seconds = max(0.2, len(text) / 15.0 / speed)
        return [sine_wave(seconds)]

    return _synth


@pytest.fixture
def stub_align_fn():
    def _align(samples: np.ndarray, sample_rate: int, text: str):
        from kokoro_ds.text_norm import tokenize_words

        words = tokenize_words(text)
        duration = samples.shape[0] / float(sample_rate)
        n = len(words)
        step = duration / n
        return [(w, i * step, (i + 1) * step - 1e-4, 0.9) for i, w in enumerate(words)]

    return _align
