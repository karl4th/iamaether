"""Colab cell: listen to every dialogue generated in a smoke-test/output run.

Paste into a Colab cell (or run as a script for a text-only summary).
Adjust OUTPUT_ROOT / SPLITS / PLAY_CHANNELS as needed.
"""
import sys
from pathlib import Path

REPO_ROOT = "/content/iamaether"
OUTPUT_ROOT = Path("/content/drive/MyDrive/deepseek_batches/audio/qa_test_50")
SPLITS = ["train", "validation", "test"]
PLAY_CHANNELS = False  # True also plays left/right channels separately per dialogue

sys.path.insert(0, REPO_ROOT)
from kokoro_ds.manifest import read_json  # noqa: E402
from kokoro_ds.wav_io import read_wav  # noqa: E402

try:
    from IPython.display import Audio, display

    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False


def iter_dialogues():
    for split in SPLITS:
        audio_dir = OUTPUT_ROOT / split / "audio"
        if not audio_dir.exists():
            continue
        for wav_path in sorted(audio_dir.glob("*.wav")):
            json_path = wav_path.with_suffix(".json")
            if json_path.exists():
                yield split, wav_path, json_path


total = 0
for split, wav_path, json_path in iter_dialogues():
    total += 1
    companion = read_json(json_path)
    info = read_wav(wav_path)

    print(f"\n{'=' * 70}")
    print(f"[{split}] {companion['id']}  ({companion['category']})")
    print(f"  assistant={companion['assistant_voice']}  user={companion['user_voice']}")
    print(f"  duration={companion['duration_seconds']:.2f}s  "
          f"alignment_conf={companion['alignment_confidence']['mean']:.3f}"
          f"{'  [LOW CONFIDENCE]' if companion['alignment_confidence']['flagged_low_confidence'] else ''}")
    for t in companion["turns"]:
        print(f"    turn[{t['index']}] {t['speaker']:9s} {t['text']!r}")

    if IN_NOTEBOOK:
        display(Audio(info.data.T, rate=info.sample_rate))
        if PLAY_CHANNELS:
            print("  left (Aether):")
            display(Audio(info.data[:, 0], rate=info.sample_rate))
            print("  right (user):")
            display(Audio(info.data[:, 1], rate=info.sample_rate))

print(f"\n{'=' * 70}\nTotal dialogues: {total}")
