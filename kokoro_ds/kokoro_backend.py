"""Kokoro TTS backend.

Only imported by the real Colab entrypoint (``generate_kokoro_dataset.py``),
never by :mod:`kokoro_ds.pipeline` or any unit test, so the rest of this
package stays importable without torch/kokoro installed.
"""
from __future__ import annotations

import numpy as np

from .voices_data import ASSISTANT_LANG_CODE, ASSISTANT_VOICE, lang_code_for_voice, VOICE_POOLS_BY_SPLIT

try:
    import torch
    from kokoro import KPipeline
except ImportError as e:  # pragma: no cover - exercised only on Colab
    raise ImportError(
        "kokoro_backend requires 'torch' and 'kokoro' to be installed in the active "
        "uv environment. See COLAB.md for the exact install cells."
    ) from e


def _extract_audio(item) -> np.ndarray:
    """Normalize one Kokoro generator yield across kokoro package versions to a mono float32 array."""
    audio = getattr(item, "audio", None)
    if audio is None and isinstance(item, (tuple, list)) and len(item) >= 3:
        audio = item[2]
    if audio is None:
        raise RuntimeError(f"Could not extract audio from Kokoro output item: {item!r}")
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().to("cpu").float().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class KokoroBackend:
    """Lazily-constructed, reused-per-process Kokoro pipelines, one per language code."""

    def __init__(self, device: str = "cuda", repo_id: str | None = None):
        self.device = device
        self.repo_id = repo_id
        self._pipelines: dict[str, "KPipeline"] = {}

    def _pipeline_for(self, lang_code: str) -> "KPipeline":
        if lang_code not in self._pipelines:
            kwargs = {"lang_code": lang_code, "device": self.device}
            if self.repo_id:
                kwargs["repo_id"] = self.repo_id
            self._pipelines[lang_code] = KPipeline(**kwargs)
        return self._pipelines[lang_code]

    def synthesize(self, text: str, voice: str, speed: float, lang_code: str) -> list[np.ndarray]:
        """Return every chunk Kokoro produced for ``text``, in order. Never truncates."""
        pipeline = self._pipeline_for(lang_code)
        generator = pipeline(text, voice=voice, speed=speed)
        chunks = [_extract_audio(item) for item in generator]
        if not chunks:
            raise RuntimeError(f"Kokoro produced zero audio chunks for text={text!r} voice={voice!r}")
        return chunks

    def model_identity(self) -> dict:
        try:
            import kokoro

            version = getattr(kokoro, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            version = "unknown"
        return {"kokoro_version": version, "repo_id": self.repo_id or "default", "device": self.device}


def assert_all_voices_available(backend: "KokoroBackend") -> None:
    """Synthesize a one-word probe with every configured voice; raise with the full failure list.

    This is the "every configured voice is available" startup assertion from
    the spec, implemented functionally since Kokoro does not expose a static
    voice-pack manifest API.
    """
    all_voices = {ASSISTANT_VOICE: ASSISTANT_LANG_CODE}
    for pool in VOICE_POOLS_BY_SPLIT.values():
        for voice in pool:
            all_voices[voice] = lang_code_for_voice(voice)

    failures: dict[str, str] = {}
    for voice, lang_code in sorted(all_voices.items()):
        try:
            chunks = backend.synthesize("test", voice, 1.0, lang_code)
            if not chunks:
                failures[voice] = "produced zero audio chunks"
        except Exception as e:  # noqa: BLE001
            failures[voice] = str(e)

    if failures:
        raise RuntimeError(f"The following configured voices failed a synthesis probe: {failures}")
