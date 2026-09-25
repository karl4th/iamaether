"""Forced alignment using torchaudio's pinned MMS_FA bundle.

Only imported by the real Colab entrypoint, never by :mod:`kokoro_ds.pipeline`
or a unit test — see the module docstring in ``kokoro_backend.py`` for why.

Produces genuine word-level timestamps from the isolated assistant waveform
and its known transcript. Never falls back to proportional/word-length
timestamps: any alignment failure must raise, so the calling pipeline fails
that record instead of writing a fake alignment.
"""
from __future__ import annotations

import numpy as np

from .text_norm import normalize_word_for_mms_alignment, tokenize_words

try:
    import torch
    import torchaudio
except ImportError as e:  # pragma: no cover - exercised only on Colab
    raise ImportError(
        "alignment_engine requires 'torch' and 'torchaudio' (pinned to 2.6.0) to be "
        "installed in the active uv environment. See COLAB.md."
    ) from e


class ForcedAligner:
    """Loads the MMS_FA bundle once and keeps it resident on the GPU."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        bundle = torchaudio.pipelines.MMS_FA
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model(with_star=False).to(device)
        self.model.eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

    def align(self, samples: np.ndarray, sample_rate: int, transcript: str) -> list[tuple[str, float, float, float]]:
        """Return ``[(original_word, start_seconds, end_seconds, confidence), ...]``.

        Raises on any failure (empty transcript, model/tokenizer mismatch,
        non-finite scores) rather than returning a partial or approximate
        result.
        """
        original_words = tokenize_words(transcript)
        if not original_words:
            raise ValueError(f"transcript normalizes to zero alignable words: {transcript!r}")

        normalized_words = [normalize_word_for_mms_alignment(w) for w in original_words]
        if any(not w for w in normalized_words):
            raise ValueError(f"one or more words normalized to empty string: {original_words!r}")

        waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        waveform = waveform.to(self.device)

        with torch.inference_mode():
            emission, _ = self.model(waveform)

        token_spans_per_word = self.aligner(emission[0], self.tokenizer(normalized_words))
        if len(token_spans_per_word) != len(original_words):
            raise RuntimeError(
                f"aligner returned {len(token_spans_per_word)} word spans for "
                f"{len(original_words)} input words"
            )

        num_frames = emission.shape[1]
        ratio = waveform.shape[1] / num_frames / self.sample_rate

        results: list[tuple[str, float, float, float]] = []
        for original_word, spans in zip(original_words, token_spans_per_word):
            if not spans:
                raise RuntimeError(f"aligner returned an empty span list for word {original_word!r}")
            start_frame = min(s.start for s in spans)
            end_frame = max(s.end for s in spans)
            scores = [s.score for s in spans]
            confidence = float(np.exp(np.mean(np.log(np.clip(scores, 1e-6, 1.0)))))
            start = float(start_frame * ratio)
            end = float(end_frame * ratio)
            if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
                raise RuntimeError(f"invalid alignment timestamps for word {original_word!r}: [{start}, {end}]")
            results.append((original_word, start, end, confidence))

        return results

    def model_identity(self) -> dict:
        return {
            "alignment_bundle": "torchaudio.pipelines.MMS_FA",
            "alignment_sample_rate": self.sample_rate,
            "torch_version": torch.__version__,
            "torchaudio_version": torchaudio.__version__,
        }
