# Aether Kokoro Dataset — Colab A100 Runbook

This is a manual runbook, not an auto-run script. Run each cell yourself, in
order, on an **A100 GPU runtime**. Nothing here starts full generation on its
own — you decide when to move from the smoke test to the full run.

Repo layout referenced below (already created in this PR):

```
generate_kokoro_dataset.py     # entrypoint: generation
validate_generated_dataset.py  # entrypoint: QA validation
kokoro_ds/                     # all the actual logic (unit tested)
tests/                         # pytest suite, no GPU/model deps
pyproject.toml
```

You'll need this repo checked out somewhere Colab can see it (e.g. cloned
into `/content/iamaether`, or synced via Drive). Every cell below assumes
`REPO=/content/iamaether` — adjust if different.

---

## 0. Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Confirm the source data is where it's expected:

```python
!ls -la /content/drive/MyDrive/deepseek_batches
```

You should see `train.jsonl`, `validation.jsonl`, `test.jsonl` (2700 / 150 /
150 records) and possibly raw `batch_*.jsonl` files. The tooling below reads
**only** the three authoritative split files when they exist — the raw
batches are never touched, so nothing gets duplicated.

---

## 1. Install a pinned `uv`, Python 3.12, and a persistent venv

Do **not** use Colab's system Python and do **not** use Python 3.13.

```bash
%%bash
# Pinned to a known-good uv release. Bump this deliberately, not implicitly —
# check https://github.com/astral-sh/uv/releases for a newer version first.
curl -LsSf https://astral.sh/uv/0.12.13/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

```bash
%%bash
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
```

Create the venv **on Drive** so it survives runtime restarts:

```bash
%%bash
export PATH="$HOME/.local/bin:$PATH"
cd /content/drive/MyDrive/deepseek_batches
uv venv --python 3.12 .venv
.venv/bin/python --version
```

From here on, every command uses `.venv/bin/python` explicitly. Do not rely
on `source .venv/bin/activate` persisting between Colab cells — it will not.

```python
VENV_PY = "/content/drive/MyDrive/deepseek_batches/.venv/bin/python"
```

---

## 2. Install espeak-ng (required by Kokoro's phonemizer)

```bash
%%bash
apt-get -qq update && apt-get -qq install -y espeak-ng
espeak-ng --version
```

---

## 3. Install pinned Python dependencies through `uv` (never raw `pip`)

This does **not** touch the Colab CUDA driver — it only installs
CUDA-enabled PyTorch *wheels* that match it. Check the driver's CUDA version
first:

```bash
!nvidia-smi
```

Pick the `cu1xx` wheel index matching (or just below) the driver's CUDA
version — `cu124` below is correct for driver CUDA 12.4+; use `cu121` if
`nvidia-smi` reports an older 12.1–12.3 driver.

```bash
%%bash
export PATH="$HOME/.local/bin:$PATH"
VENV=/content/drive/MyDrive/deepseek_batches/.venv

uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchaudio==2.6.0

uv pip install --python "$VENV/bin/python" \
  kokoro==0.9.4 soundfile numpy tqdm orjson pytest
```

Verify the runtime is actually correct before doing anything else:

```bash
%%bash
VENV=/content/drive/MyDrive/deepseek_batches/.venv
"$VENV/bin/python" - <<'PY'
import sys
import torch

assert sys.version_info[:2] == (3, 12), sys.version_info
assert torch.cuda.is_available(), "CUDA not available — check the Colab runtime type is set to GPU (A100)"
print("Python:", sys.version)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
```

If this cell fails, stop — fix the runtime/environment before continuing.

---

## 4. Run the local unit test suite (no GPU / no model download required)

This exercises deterministic voice/speed/pause assignment, stereo assembly,
manifest writing, resume/integrity verification, and alignment-format
validation — all pure-python/numpy, all fast.

```bash
%%bash
VENV=/content/drive/MyDrive/deepseek_batches/.venv
cd /content/iamaether
"$VENV/bin/python" -m pytest -q
```

All tests should pass before you generate a single second of audio.

---

## 5. Source validation only (no synthesis)

```bash
%%bash
VENV=/content/drive/MyDrive/deepseek_batches/.venv
cd /content/iamaether
"$VENV/bin/python" generate_kokoro_dataset.py \
  --source-root /content/drive/MyDrive/deepseek_batches \
  --validate-only
```

This checks all 3,000 dialogues: exact split counts (2700 / 150 / 150), JSON
shape, exact keys, alternating roles starting with `user` and ending with
`assistant`, 2–8 turns, allowed categories, no URLs/Markdown/control
characters/unsupported Unicode, no excessively long utterances, no empty
forced-alignment transcripts, globally unique ids, and no split overlap. It
fails fast with the exact offending record ids — fix the source data (never
this tooling) if it fails.

---

## 6. Pronunciation preview — listen before you commit to anything

Pronunciation overrides live in `kokoro_ds/pronunciation.py` as one visible
dict (`PRONUNCIATION_MAP`), using Kokoro's `[word](/ipa/)` markup. **Edit
that file and rerun this cell** until every name sounds right — do this
before the smoke test, not after.

```python
import sys
sys.path.insert(0, "/content/iamaether")

from kokoro_ds.pronunciation import PRONUNCIATION_MAP, QA_PHRASES, apply_pronunciation, print_pronunciation_map
from kokoro_ds.kokoro_backend import KokoroBackend
from kokoro_ds.audio_ops import concat_kokoro_chunks
from kokoro_ds.voices_data import ASSISTANT_VOICE, ASSISTANT_LANG_CODE
from IPython.display import Audio, display

print_pronunciation_map(PRONUNCIATION_MAP)

backend = KokoroBackend(device="cuda")
for phrase in QA_PHRASES:
    text = apply_pronunciation(phrase, PRONUNCIATION_MAP)
    print(f"\n{phrase!r} -> synthesized as: {text!r}")
    chunks = backend.synthesize(text, ASSISTANT_VOICE, 1.0, ASSISTANT_LANG_CODE)
    audio = concat_kokoro_chunks(chunks, sample_rate=24_000, inter_chunk_pause_seconds=0.12)
    display(Audio(audio, rate=24_000))
```

Reload `kokoro_ds.pronunciation` (or restart the runtime) after editing the
map, then rerun this cell. Repeat until satisfied.

---

## 7. Dry-run duration and disk estimate (no synthesis)

```bash
%%bash
VENV=/content/drive/MyDrive/deepseek_batches/.venv
cd /content/iamaether
"$VENV/bin/python" generate_kokoro_dataset.py \
  --source-root /content/drive/MyDrive/deepseek_batches \
  --output-root /content/drive/MyDrive/deepseek_batches/audio/aether_kokoro_v1 \
  --dry-run
```

---

## 8. Stratified 50-dialogue smoke test

This is **not** "the first 50 rows" — it deterministically selects dialogues
so the subset covers every category and every configured user voice
whenever the source data makes that mathematically possible, spread
proportionally across train/validation/test. Any coverage gap is printed as
a warning, never hidden.

```bash
%%bash
VENV=/content/drive/MyDrive/deepseek_batches/.venv
cd /content/iamaether
"$VENV/bin/python" generate_kokoro_dataset.py \
  --source-root /content/drive/MyDrive/deepseek_batches \
  --output-root /content/drive/MyDrive/deepseek_batches/audio/qa_test_50 \
  --smoke-count 50 \
  --seed 0
```

Exit code `0` means every selected dialogue generated cleanly. Read the
printed warnings (if any) about uncoverable categories/voices — that only
happens if the real dataset is smaller/more skewed than expected.

---

## 9. Full QA validation of the smoke test output

```bash
%%bash
VENV=/content/drive/MyDrive/deepseek_batches/.venv
cd /content/iamaether
"$VENV/bin/python" validate_generated_dataset.py \
  --output-root /content/drive/MyDrive/deepseek_batches/audio/qa_test_50 \
  --source-root /content/drive/MyDrive/deepseek_batches
```

This writes `reports/qa_summary.json` under the output root and prints a
short summary: file counts, hours by split/category, voice distribution,
duration/RMS/peak/alignment-confidence percentiles, and any invalid or
partially-written records. **A nonzero exit code means do not proceed to
full generation** — fix the cause first.

---

## 10. Listening cell — actually listen to one generated dialogue

```python
import sys, json
sys.path.insert(0, "/content/iamaether")

from pathlib import Path
from kokoro_ds.wav_io import read_wav
from kokoro_ds.manifest import read_json
from IPython.display import Audio, display

SPLIT_DIR = Path("/content/drive/MyDrive/deepseek_batches/audio/qa_test_50/train")
dialogue_id = sorted(p.stem for p in (SPLIT_DIR / "audio").glob("*.wav"))[0]  # pick any id you want to inspect

companion = read_json(SPLIT_DIR / "audio" / f"{dialogue_id}.json")
print("id:", dialogue_id)
print("category:", companion["category"])
print("assistant_voice/speed:", companion["assistant_voice"], companion["assistant_speed"])
print("user_voice:", companion["user_voice"])
for t in companion["turns"]:
    print(f"  turn[{t['index']}] {t['speaker']:9s} voice={t['voice']:10s} speed={t['speed']:.3f} "
          f"pause_before={t['pause_before_seconds']:.3f}  text={t['text']!r}")
print("duration_seconds:", companion["duration_seconds"])
print("rms:", companion["rms"], "peak:", companion["peak"])
print("alignments:")
for word, span, speaker in companion["alignments"]:
    print(f"  {word:15s} [{span[0]:6.3f}, {span[1]:6.3f}]  {speaker}")

info = read_wav(SPLIT_DIR / "audio" / f"{dialogue_id}.wav")
print("\nFull stereo dialogue:")
display(Audio(info.data.T, rate=info.sample_rate))
print("Left channel only (Aether):")
display(Audio(info.data[:, 0], rate=info.sample_rate))
print("Right channel only (user):")
display(Audio(info.data[:, 1], rate=info.sample_rate))
```

Listen to a handful of ids across categories and both channels before
trusting the pipeline with the full 3,000-dialogue run.

---

## 11. Full generation — only after every step above is green

```bash
%%bash
VENV=/content/drive/MyDrive/deepseek_batches/.venv
cd /content/iamaether
"$VENV/bin/python" generate_kokoro_dataset.py \
  --source-root /content/drive/MyDrive/deepseek_batches \
  --output-root /content/drive/MyDrive/deepseek_batches/audio/aether_kokoro_v1 \
  --splits train validation test \
  --seed 0 \
  --resume
```

`--resume` is safe to pass on the very first run too — there is nothing to
resume yet, and it costs nothing. Colab will very likely disconnect at least
once during a 3,000-dialogue A100 run; **always** rerun this exact command
(same `--seed`) to continue. Each existing record is re-verified (WAV
opens, correct format, alignments valid, receipt checksums match) before
being skipped — a corrupted or stale record is regenerated automatically,
never silently accepted.

Exit code `0` means zero unresolved failures. A nonzero exit code means
check `receipts/failed.jsonl` under the output root.

Debug exactly one dialogue at any time with:

```bash
%%bash
VENV=/content/drive/MyDrive/deepseek_batches/.venv
cd /content/iamaether
"$VENV/bin/python" generate_kokoro_dataset.py \
  --source-root /content/drive/MyDrive/deepseek_batches \
  --output-root /content/drive/MyDrive/deepseek_batches/audio/aether_kokoro_v1 \
  --record-id batch_0001_0001 \
  --seed 0
```

---

## 12. Final QA validation of the full dataset

```bash
%%bash
VENV=/content/drive/MyDrive/deepseek_batches/.venv
cd /content/iamaether
"$VENV/bin/python" validate_generated_dataset.py \
  --output-root /content/drive/MyDrive/deepseek_batches/audio/aether_kokoro_v1 \
  --source-root /content/drive/MyDrive/deepseek_batches
```

A clean run of this validator is required before treating the dataset as
ready for Moshi fine-tuning — a finished `generate_kokoro_dataset.py` run by
itself is not sufficient proof of validity.

## What this tooling cannot verify for you

- That the Kokoro pronunciation markup in `pronunciation.py` is actually
  correct — only your own ears in step 6 can confirm that.
- Forced-alignment *accuracy* beyond structural validity (monotonic,
  in-bounds, nonempty) — spot-check the printed word timestamps against the
  audio in the listening cell, especially any record flagged
  `flagged_low_confidence` in `reports/qa_summary.json`.
- Real A100 throughput/ETA and Colab session stability over a multi-hour
  run — the smoke test only proves correctness, not full-run duration.
- Whether the full 3,000-record dataset introduces categories not present
  in `batch_0001.jsonl` (see the assumption noted in `kokoro_ds/schema.py`,
  `DEFAULT_ALLOWED_CATEGORIES`) — if source validation rejects a legitimate
  new category, extend that set rather than disabling the check.
