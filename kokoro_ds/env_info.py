"""Runtime environment fingerprinting for provenance/receipt metadata.

Every lookup degrades to ``"unavailable"`` instead of raising, so this module
can be imported and called from both the generator (where torch is expected)
and the validator (where it may not be installed).
"""
from __future__ import annotations

import sys


def python_version() -> str:
    return sys.version.split()[0]


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch_version": "unavailable", "cuda_available": False, "cuda_version": "unavailable", "gpu_name": "unavailable"}
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda or "unavailable",
        "gpu_name": "unavailable",
    }
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            pass
    return info


def torchaudio_version() -> str:
    try:
        import torchaudio

        return torchaudio.__version__
    except ImportError:
        return "unavailable"


def package_versions(package_names: tuple[str, ...] = ("kokoro", "numpy", "soundfile", "orjson", "tqdm")) -> dict[str, str]:
    versions = {}
    for name in package_names:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "unavailable"
    return versions


def gpu_memory_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return None


def collect_environment(kokoro_model_identity: dict, alignment_model_identity: dict) -> dict:
    t = torch_info()
    return {
        "python_version": python_version(),
        "torch_version": t["torch_version"],
        "cuda_available": t["cuda_available"],
        "cuda_version": t["cuda_version"],
        "gpu_name": t["gpu_name"],
        "torchaudio_version": torchaudio_version(),
        "package_versions": package_versions(),
        "kokoro_model_identity": kokoro_model_identity,
        "alignment_model_identity": alignment_model_identity,
    }
