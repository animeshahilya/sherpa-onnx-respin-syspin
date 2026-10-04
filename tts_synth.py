#!/usr/bin/env python3
"""Shared no-retrain synthesis helper for sherpa-onnx-respin-syspin.

Improves quality without retraining:
  1. Text frontend per language: Hindi schwa/numeral/punctuation rules
     (lang-parameterized: full for hi/hne, final-only for mr/bho/mai/mag)
     for Devanagari langs; numeral/punctuation frontends + experimental
     Bengali apocope for bn/te/kn/gu/en (indic_frontend.py).
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
    def hindi_frontend_fn(text: str, lang: str = "hi") -> str:
        return text

# Devanagari-script lang codes with Hindi-tuned schwa rules. mr/bho/mai/mag/
# brx/doi/ne use final-only mode, san disables deletion entirely (classical
# pronunciation keeps all schwas); hne follows Hindi. See SCHWA_MODE.
DEVANAGARI_LANGS = {"hi", "mr", "bho", "hne", "mai", "mag", "ne", "san", "brx", "doi"}
# Langs with dedicated numeral/punctuation normalization (indic_frontend.py).
# NOTE: san appears in BOTH sets on purpose -- indic handles its numerals,
# hindi handles (skips) its schwa. brx/doi/ne numerals fall back to Hindi
# words (documented approximation; native tables pending).
INDIC_FRONTEND_LANGS = {"bn", "te", "kn", "gu", "en", "asm", "pan", "tam", "mal", "san"}

PAUSE_SECONDS = 0.35
CLAUSE_PAUSE_SECONDS = 0.18
PEAK_TARGET = 0.90
# Broadcast-style target: consistent loudness across voices (item 4).
# Final peak ceiling keeps a safety margin below clipping.
TARGET_LUFS = -16.0
PEAK_CEILING = 0.95


def frontend_for_lang(text: str, lang: str, apocope: bool = True) -> str:
    """Two-stage frontend: (1) numeral/punctuation normalization -- indic
    tables where available, else Hindi normalize (Devanagari digits share
    the script); (2) schwa handling -- Hindi rules lang-parameterized
    (san: none), Bengali apocope if enabled. Passthrough otherwise."""
    if lang in INDIC_FRONTEND_LANGS:
        try:
            from indic_frontend import normalize as indic_normalize
            text = indic_normalize(text, lang)
        except (ImportError, KeyError):
            pass
    elif lang in DEVANAGARI_LANGS:
        try:
            from hindi_frontend import normalize as hindi_normalize
            text = hindi_normalize(text)
        except ImportError:
            pass
    if lang in DEVANAGARI_LANGS:
        try:
            return hindi_frontend_fn(text, lang)
        except TypeError:  # older single-arg frontend
            return hindi_frontend_fn(text)
    if lang == "bn" and apocope:
        try:
            from indic_frontend import delete_schwa_bn
            words = []
            for tok in text.split(" "):
                if re.fullmatch(r"[A-Za-z]+", tok or ""):
                    words.append(tok)
                else:
                    i = len(tok)
                    while i > 0 and tok[i - 1] in ".,?!:;,—–'\"()":
                        i -= 1
                    core, tail = tok[:i], tok[i:]
                    words.append(delete_schwa_bn(core) + tail if core else tok)
            return " ".join(w for w in words if w)
        except ImportError:
            return text
    return text


def split_sentences(text: str):
    """Split on danda / CJK-normalized '.' / newlines, keep non-empty."""
    parts = [s.strip() for s in re.split(r"[.।\n]+", text) if s.strip()]
    return parts


def split_clauses(sentence: str):
    """Split a sentence at mid-sentence pauses (comma/semicolon/colon/dash).

    Returns [(clause_text, pause_after_seconds)]. Commas get a short
    180ms beat so lists and subordinate clauses breathe; the final clause
    takes no clause pause (the sentence pause follows instead)."""
    bits = [b.strip(" ") for b in re.split(r"([,;:—–])", sentence)]
    clauses, buf = [], ""
    for b in bits:
        if b in (",", ";", ":", "—", "–"):
            buf = (buf + " ").strip()
            if buf:
                clauses.append((buf, CLAUSE_PAUSE_SECONDS))
            buf = ""
        elif b:
            buf = (buf + " " + b).strip()
    if buf:
        clauses.append((buf, 0.0))
    return [(c, p) for c, p in clauses if c]


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


def loudness_normalize(wav: np.ndarray, sample_rate: int,
                       target_lufs: float = TARGET_LUFS,
                       peak_ceiling: float = PEAK_CEILING) -> np.ndarray:
    """ITU-R BS.1770 loudness normalize with peak ceiling.

    Falls back to peak_normalize when pyloudnorm is unavailable. The ceiling
    cap means hot mixes keep headroom instead of clipping: gain is limited
    so the final peak never exceeds peak_ceiling."""
    if wav.size == 0:
        return wav
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sample_rate)
        loud = meter.integrated_loudness(wav.astype(np.float64))
        if loud <= -70.0:  # silence / degenerate: peak fallback
            return peak_normalize(wav, peak_ceiling)
        gain = 10.0 ** ((target_lufs - loud) / 20.0)
        peak = float(np.max(np.abs(wav)))
        if peak * gain > peak_ceiling:
            gain = peak_ceiling / peak if peak > 0 else 1.0
        return (wav * gain).astype(np.float32)
    except ImportError:
        return peak_normalize(wav, peak_ceiling)


# NOTE on determinism (verified 2026-10-02, sherpa-onnx 1.13.8, hi-female):
# generation is BIT-IDENTICAL across sessions for a fixed config, and
# noise_scale (0.1/0.667/1.5) / noise_scale_w (0.1/1.5) are NO-OPs on these
# exports (identical length/peak/energy). Only length_scale (via speed=1/x)
# changes the output. So there is no seed to control and no per-run
# variance to average out; tune_inference.py still supports repeats for
# forward-compatibility if a future sherpa restores stochastic sampling.


def synthesize_voice(tts, sid: int, text: str, lang: str = "",
                      length_scale: float = 1.0, sample_rate: int = 22050,
                      apocope: bool = True) -> np.ndarray:
    """Frontend -> per-sentence generate (with clause pauses) -> trim -> LUFS norm."""
    text = frontend_for_lang(text, lang, apocope=apocope)
    sents = split_sentences(text)
    pause = np.zeros(int(sample_rate * PAUSE_SECONDS), dtype=np.float32)
    parts = []
    for idx, s in enumerate(sents):
        for cdx, (clause, cpause) in enumerate(split_clauses(s.rstrip(".?!:;, "))):
            if not clause:
                continue
            w = np.array(
                tts.generate(clause, sid=sid, speed=1.0 / length_scale).samples,
                dtype=np.float32,
            )
            w = trim_silence(w, sample_rate)
            parts.append(w)
            if cpause > 0:
                parts.append(np.zeros(int(sample_rate * cpause), dtype=np.float32))
        if idx < len(sents) - 1:
            parts.append(pause)
    if not parts:
        return np.zeros(sample_rate, dtype=np.float32)
    full = np.concatenate(parts)
    return loudness_normalize(full, sample_rate)
