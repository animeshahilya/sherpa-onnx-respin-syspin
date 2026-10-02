#!/usr/bin/env python3
"""Shared no-retrain synthesis helper for sherpa-onnx-respin-syspin.

Improves quality without retraining:
  1. Text frontend: Hindi schwa/numeral/punctuation normalizer for
     Devanagari langs (hi/mr/bho/hne/mai/mag/ne/san/brx/doi), passthrough
     for others (bn/te/kn/gu/en/asm/mal/pan/tam).
  2. Sentence-by-sentence synthesis with 350ms breathing pauses
     (prevents attention diffusion / rushed words on long text).
  3. Leading/trailing silence trim + peak normalization to 0.90
     (consistent loudness, no clipping).

Mirrors generate_rasa_samples.synthesize_voice but adds trim + frontend
dispatch so generate_samples.py (SYSPIN) gets the same treatment.
"""

import re

import numpy as np

try:
    from hindi_frontend import frontend as hindi_frontend_fn
except ImportError:  # pragma: no cover - helper import
    def hindi_frontend_fn(text: str) -> str:
        return text

# Devanagari-script lang codes that benefit from the Hindi frontend.
# Rules are Hindi-tuned but harmless for mr/bho/hne/mai/mag/ne/san/brx/doi
# (same script, halant + danda coverage); other scripts pass through.
DEVANAGARI_LANGS = {"hi", "mr", "bho", "hne", "mai", "mag", "ne", "san", "brx", "doi"}

PAUSE_SECONDS = 0.35
PEAK_TARGET = 0.90


def frontend_for_lang(text: str, lang: str) -> str:
    """Apply Hindi frontend for Devanagari langs, passthrough otherwise."""
    if lang in DEVANAGARI_LANGS:
        return hindi_frontend_fn(text)
    return text


def split_sentences(text: str):
    """Split on danda / CJK-normalized '.' / newlines, keep non-empty."""
    parts = [s.strip() for s in re.split(r"[.।\n]+", text) if s.strip()]
    return parts


def trim_silence(wav: np.ndarray, sample_rate: int, threshold: float = 0.02, margin_ms: float = 50.0) -> np.ndarray:
    """Trim leading/trailing near-silence. No-op on empty/short input."""
    if wav.size == 0:
        return wav
    above = np.where(np.abs(wav) > threshold)[0]
    if above.size == 0:
        return wav
    margin = int(sample_rate * margin_ms / 1000.0)
    start = max(0, int(above[0]) - margin)
    end = min(wav.size, int(above[-1]) + margin + 1)
    return wav[start:end]


def peak_normalize(wav: np.ndarray, target: float = PEAK_TARGET) -> np.ndarray:
    mx = float(np.max(np.abs(wav))) if wav.size else 0.0
    if mx > 0.01:
        wav = wav * (target / mx)
    return wav


def synthesize_voice(tts, sid: int, text: str, lang: str = "",
                      length_scale: float = 1.0, sample_rate: int = 22050) -> np.ndarray:
    """Frontend -> per-sentence generate -> 350ms pauses -> trim -> norm."""
    text = frontend_for_lang(text, lang)
    sents = split_sentences(text)
    pause = np.zeros(int(sample_rate * PAUSE_SECONDS), dtype=np.float32)
    parts = []
    for idx, s in enumerate(sents):
        clean = s.rstrip(".?!:;, ")
        if not clean:
            continue
        w = np.array(
            tts.generate(clean, sid=sid, speed=1.0 / length_scale).samples,
            dtype=np.float32,
        )
        w = trim_silence(w, sample_rate)
        parts.append(w)
        if idx < len(sents) - 1:
            parts.append(pause)
    if not parts:
        return np.zeros(sample_rate, dtype=np.float32)
    full = np.concatenate(parts)
    return peak_normalize(full)
