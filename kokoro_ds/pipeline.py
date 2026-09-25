"""Per-dialogue generation orchestration.

Deliberately free of any torch/Kokoro import: synthesis and forced alignment
are injected as callables (``SynthFn`` / ``AlignFn``), so this module -- the
part with the actual assembly, checksum, and resume logic -- is fully
unit-testable with lightweight stub callables and no GPU or model download.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .alignment_schema import validate_alignment_list
from .audio_ops import (
    TurnAudio,
    apply_fade,
    assemble_stereo,
    concat_kokoro_chunks,
    is_clipping,
    peak_amplitude,
    prevent_overflow,
    rms_amplitude,
)
from .checksums import generation_config_checksum, sha256_file, source_dialogue_checksum
from .config import GenerationConfig, SPEAKER_MAIN_LABEL, MAX_DIALOGUE_SECONDS, MIN_UTTERANCE_SECONDS
from .determinism import assign_turn_pause_seconds, assign_user_turn_speed, assign_user_voice
from .manifest import atomic_write_json
from .pronunciation import PRONUNCIATION_MAP, apply_pronunciation
from .receipts import ReceiptStore, json_path_for, wav_path_for
from .schema import SourceRecord
from .voices_data import ASSISTANT_LANG_CODE, lang_code_for_voice
from .wav_io import write_stereo_wav_atomic

# (text_to_speak, voice_id, speed, lang_code) -> list of raw mono float32 chunks at config.sample_rate
SynthFn = Callable[[str, str, float, str], list[np.ndarray]]

# (mono float32 samples, sample_rate, original_transcript) -> [(word, start_s, end_s, confidence), ...]
AlignFn = Callable[[np.ndarray, int, str], list[tuple[str, float, float, float]]]


@dataclass
class GenerationOutcome:
    success: bool
    record_id: str
    reasons: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    relative_wav_path: str | None = None
    receipt: dict | None = None


def _record_to_source_dict(record: SourceRecord) -> dict:
    return {
        "id": record.id,
        "category": record.category,
        "turns": [{"speaker": t.speaker, "text": t.text} for t in record.turns],
    }


def generate_one_record(
    record: SourceRecord,
    split: str,
    split_dir: Path,
    config: GenerationConfig,
    global_seed: int,
    synth_fn: SynthFn,
    align_fn: AlignFn,
    receipt_store: ReceiptStore,
    environment: dict,
    pronunciation_map: dict[str, str] | None = None,
    keep_intermediate: bool = False,
) -> GenerationOutcome:
    reasons: list[str] = []
    pronunciation_map = pronunciation_map if pronunciation_map is not None else PRONUNCIATION_MAP
    user_voice = assign_user_voice(record.id, split, global_seed)

    turn_audios: list[TurnAudio] = []
    turn_meta: list[dict] = []
    debug_dir = split_dir / "audio" / "_debug" if keep_intermediate else None

    for idx, turn in enumerate(record.turns):
        if turn.speaker == "assistant":
            voice, speed, lang = config.assistant_voice, config.assistant_speed, ASSISTANT_LANG_CODE
        else:
            voice = user_voice
            speed = assign_user_turn_speed(record.id, idx, global_seed, config.user_speed_min, config.user_speed_max)
            lang = lang_code_for_voice(user_voice)

        tts_text = apply_pronunciation(turn.text, pronunciation_map)
        try:
            chunks = synth_fn(tts_text, voice, speed, lang)
            raw = concat_kokoro_chunks(chunks, config.sample_rate, config.inter_chunk_pause_seconds)
        except Exception as e:  # noqa: BLE001
            return GenerationOutcome(False, record.id, [f"turn[{idx}] synthesis failed: {e}"])

        if raw.shape[0] < int(MIN_UTTERANCE_SECONDS * config.sample_rate):
            return GenerationOutcome(False, record.id, [f"turn[{idx}] synthesized utterance is empty/too short"])

        faded = apply_fade(raw, config.sample_rate, config.fade_seconds)

        if debug_dir is not None:
            try:
                write_stereo_wav_atomic(
                    debug_dir / f"{record.id}_turn{idx:02d}_{turn.speaker}.wav",
                    np.stack([faded, faded], axis=0),
                    config.sample_rate,
                )
            except Exception:  # noqa: BLE001 - debug artifacts are best-effort
                pass

        pause_before = assign_turn_pause_seconds(record.id, idx, global_seed, config.pause_min_seconds, config.pause_max_seconds)
        turn_audios.append(TurnAudio(speaker=turn.speaker, samples=faded, pause_before_seconds=pause_before))
        turn_meta.append(
            {
                "index": idx,
                "speaker": turn.speaker,
                "text": turn.text,
                "voice": voice,
                "speed": round(speed, 4),
                "pause_before_seconds": round(pause_before, 4),
            }
        )

    stereo, placements = assemble_stereo(turn_audios, config.sample_rate)
    duration_seconds = stereo.shape[1] / float(config.sample_rate)
    if duration_seconds > MAX_DIALOGUE_SECONDS:
        return GenerationOutcome(False, record.id, [f"assembled duration {duration_seconds:.1f}s exceeds sanity ceiling"])

    left = prevent_overflow(stereo[0])
    right = prevent_overflow(stereo[1])
    stereo = np.stack([left, right], axis=0)
    if is_clipping(left, config.clipping_peak_threshold) or is_clipping(right, config.clipping_peak_threshold):
        return GenerationOutcome(False, record.id, ["clipping detected in assembled stereo audio"])

    alignments: list[list] = []
    confidences: list[float] = []
    for placement, turn, meta in zip(placements, record.turns, turn_meta):
        if turn.speaker != "assistant":
            continue
        utter_samples = turn_audios[placement.index].samples
        try:
            words = align_fn(utter_samples, config.sample_rate, turn.text)
        except Exception as e:  # noqa: BLE001
            return GenerationOutcome(False, record.id, [f"forced alignment failed for turn[{placement.index}]: {e}"])
        if not words:
            return GenerationOutcome(False, record.id, [f"forced alignment returned no words for turn[{placement.index}]"])
        offset = placement.start_sample / float(config.sample_rate)
        for word, start, end, confidence in words:
            alignments.append([word, [round(start + offset, 4), round(end + offset, 4)], SPEAKER_MAIN_LABEL])
            confidences.append(confidence)

    align_problems = validate_alignment_list(alignments, duration_seconds)
    if align_problems:
        return GenerationOutcome(False, record.id, [f"alignment validation failed: {p}" for p in align_problems])

    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    min_confidence = float(np.min(confidences)) if confidences else 0.0
    flagged_low_confidence = mean_confidence < config.alignment_min_confidence

    wav_path = wav_path_for(split_dir, record.id)
    json_path = json_path_for(split_dir, record.id)
    write_stereo_wav_atomic(wav_path, stereo, config.sample_rate)

    source_dict = _record_to_source_dict(record)
    source_checksum = source_dialogue_checksum(source_dict)
    config_checksum = generation_config_checksum(config.as_dict(), split, user_voice)
    wav_checksum = sha256_file(wav_path)

    companion = {
        "id": record.id,
        "split": split,
        "category": record.category,
        "assistant_voice": config.assistant_voice,
        "assistant_speed": config.assistant_speed,
        "user_voice": user_voice,
        "turns": turn_meta,
        "sample_rate": config.sample_rate,
        "channels": config.channels,
        "duration_seconds": round(duration_seconds, 4),
        "rms": {"left": rms_amplitude(left), "right": rms_amplitude(right)},
        "peak": {"left": peak_amplitude(left), "right": peak_amplitude(right)},
        "alignments": alignments,
        "alignment_confidence": {
            "mean": round(mean_confidence, 4),
            "min": round(min_confidence, 4),
            "flagged_low_confidence": flagged_low_confidence,
        },
        "config": config.as_dict(),
        "seed": global_seed,
        "checksums": {
            "source_dialogue": source_checksum,
            "generation_config": config_checksum,
            "wav": wav_checksum,
        },
        "environment": environment,
    }
    atomic_write_json(json_path, companion)
    json_checksum = sha256_file(json_path)

    relative_wav_path = f"audio/{record.id}.wav"
    receipt = {
        "id": record.id,
        "split": split,
        "category": record.category,
        "source_checksum": source_checksum,
        "config_checksum": config_checksum,
        "wav_checksum": wav_checksum,
        "json_checksum": json_checksum,
        "duration_seconds": round(duration_seconds, 4),
        "alignment_confidence_mean": round(mean_confidence, 4),
        "flagged_low_confidence": flagged_low_confidence,
        "user_voice": user_voice,
        "assistant_voice": config.assistant_voice,
    }
    receipt_store.mark_completed(receipt)

    return GenerationOutcome(
        success=True,
        record_id=record.id,
        reasons=[],
        duration_seconds=duration_seconds,
        relative_wav_path=relative_wav_path,
        receipt=receipt,
    )
