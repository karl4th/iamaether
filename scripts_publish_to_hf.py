"""Publish a generated dataset to a private Hugging Face Hub dataset repo.

Reads the token from Colab Secrets (HF_TOKEN) first, falling back to the
HF_TOKEN env var, then to any cached `huggingface-cli login` token. The
token is never printed or written to any file.

Usage (Colab):
    !uv pip install --python .venv/bin/python -U huggingface_hub
    !.venv/bin/python scripts_publish_to_hf.py \
        --output-root /content/drive/MyDrive/deepseek_batches/audio/aether_kokoro_v1 \
        --repo-id Manifestro/i-am-aether-v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUTPUT_ROOT = "/content/drive/MyDrive/deepseek_batches/audio/aether_kokoro_v1"
DEFAULT_REPO_ID = "Manifestro/i-am-aether-v1"

IGNORE_PATTERNS = [
    ".venv/*",
    "**/__pycache__/*",
    "**/.pytest_cache/*",
    ".*.tmp-*",
    "**/.*.tmp-*",
]


def get_token() -> str:
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
        if token:
            return token
    except Exception:  # noqa: BLE001 - not in Colab, or secret not set
        pass

    import os

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    from huggingface_hub import HfFolder

    token = HfFolder.get_token()
    if token:
        return token

    raise RuntimeError(
        "No HF token found. Set it in Colab Secrets as HF_TOKEN, export HF_TOKEN, "
        "or run `huggingface-cli login` first."
    )


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_readme(output_root: Path, repo_id: str, private: bool) -> str:
    gen_summary = _read_json(output_root / "reports" / "generation_summary.json") or {}
    qa_summary = _read_json(output_root / "reports" / "qa_summary.json") or {}
    provenance = _read_json(output_root / "reports" / "provenance.json") or {}

    by_split = gen_summary.get("by_split") or {
        split: {"total": qa_summary.get("per_split", {}).get(split, {}).get("total_found")}
        for split in ("train", "validation", "test")
    }

    lines = [
        "---",
        "license: unknown",
        f"private: {'true' if private else 'false'}",
        "tags:",
        "  - audio",
        "  - text-to-speech",
        "  - voice-assistant",
        "  - moshi",
        "---",
        "",
        "# i-am-aether-v1",
        "",
        "Kokoro-synthesized stereo dialogue audio for fine-tuning the Aether voice "
        "assistant (Kyutai Moshi architecture). Left channel = assistant (Aether, "
        "voice `af_heart`), right channel = user (per-split disjoint voice pool). "
        "24kHz, PCM16, stereo WAV per dialogue, with word-level forced-alignment "
        "timestamps for the assistant channel.",
        "",
        "## Splits",
        "",
    ]
    for split, info in by_split.items():
        lines.append(f"- **{split}**: {info}")

    lines += [
        "",
        "## Generation provenance",
        "",
        f"- seed: `{gen_summary.get('seed', provenance.get('seed', 'unknown'))}`",
        f"- total audio hours: {gen_summary.get('total_audio_hours', 'unknown')}",
        f"- kokoro: {provenance.get('kokoro_model_identity', 'unknown')}",
        f"- alignment: {provenance.get('alignment_model_identity', 'unknown')}",
        f"- torch: {provenance.get('torch_version', 'unknown')}",
        "",
        "See `reports/qa_summary.json`, `reports/generation_summary.json`, and "
        "`reports/provenance.json` in this repo for full detail.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Publish a generated Kokoro/Aether dataset to the HF Hub.")
    p.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    p.add_argument("--public", action="store_true", help="Create/keep the repo public instead of private.")
    p.add_argument("--no-readme", action="store_true", help="Do not generate/overwrite README.md before upload.")
    p.add_argument("--commit-message", type=str, default="Publish generated dataset")
    p.add_argument("--legacy-upload", action="store_true", help="Use upload_folder instead of upload_large_folder.")
    args = p.parse_args(argv)

    if not args.output_root.exists():
        raise SystemExit(f"--output-root does not exist: {args.output_root}")

    token = get_token()

    from huggingface_hub import HfApi, create_repo

    private = not args.public
    create_repo(repo_id=args.repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    print(f"Repo ready: https://huggingface.co/datasets/{args.repo_id} (private={private})")

    if not args.no_readme:
        readme_path = args.output_root / "README.md"
        readme_path.write_text(build_readme(args.output_root, args.repo_id, private), encoding="utf-8")
        print(f"Wrote {readme_path}")

    api = HfApi(token=token)

    if args.legacy_upload or not hasattr(api, "upload_large_folder"):
        print("Uploading with upload_folder (non-resumable)...")
        api.upload_folder(
            folder_path=str(args.output_root),
            repo_id=args.repo_id,
            repo_type="dataset",
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=args.commit_message,
        )
    else:
        print("Uploading with upload_large_folder (resumable, recommended for thousands of files)...")
        api.upload_large_folder(
            folder_path=str(args.output_root),
            repo_id=args.repo_id,
            repo_type="dataset",
            ignore_patterns=IGNORE_PATTERNS,
        )

    print(f"\nDone: https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
