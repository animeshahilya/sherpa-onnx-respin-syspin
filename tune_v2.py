#!/usr/bin/env python3
"""Tune syspin-v2 / rasa-v2 models (noise scales are LIVE here, unlike v1).

Design notes (all verified 2026-10-04):
- v2 outputs are NON-deterministic across runs in one session (fresh
  sessions are deterministic per config), so repeats=2 with median.
- Pace is applied ONLY via speed=1/length (model-config length_scale stays
  1.0) to avoid double-counting; noise dims go in the model config.
- Grid: noise {0.5,0.8} x noise_w {0.6,1.0} x length {0.95,1.0,1.05}.
- Results go to tune_results_v2/ (never clobbers v1 results).

usage: python tune_v2.py --voices hi-female hi-male
       python tune_v2.py --engine rasa-v2 --sids 7 16
"""
import argparse
import itertools
import json
import os
import re
import statistics
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
RES = os.path.join(BASE, "tune_results_v2")
WAVS = os.path.join(RES, "wav")
os.makedirs(WAVS, exist_ok=True)

from tune_inference import (EVAL_TEXTS, RASA_WHISPER, get_asr, norm_text,  # noqa: E402
                            proxy_mos, resample16k)
from tts_synth import synthesize_voice  # noqa: E402

V2D = os.path.join(BASE, "release_assets_v2")
V1D = os.path.join(BASE, "release_assets_fp16")

GRID = {"noise_scale": [0.5, 0.8], "noise_w": [0.6, 1.0],
        "length_scale": [0.95, 1.0, 1.05]}
# The upstream default point (0.667, 1.0, 1.0) is ALWAYS evaluated too:
# v1's frozen values were exactly these, so this is the parity anchor.
DEFAULT_CFG = {"noise_scale": 0.667, "noise_w": 1.0, "length_scale": 1.0}

SYS_VOICES = {"hi-female": "hi", "hi-male": "hi"}
RASA_SIDS_TUNE = [2, 7, 8, 9, 10, 12, 14, 16, 17, 18]
from tune_inference import RASA_SIDS as _RASA_SIDS
from generate_rasa_samples import RASA_TEXTS as _RT
RASA_EVAL = {lang: _RT[lang]["text"].split(".")[0].strip()
             for _, _, lang in _RASA_SIDS}


def load_syspin_v2(noise_scale, noise_w):
    import sherpa_onnx
    m = os.path.join(V2D, "vits-syspin-%s-model.onnx" % CUR_VOICE)
    t = os.path.join(V1D, "vits-syspin-%s-tokens.txt" % CUR_VOICE)
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=m, tokens=t, noise_scale=noise_scale,
                noise_scale_w=noise_w, length_scale=1.0),
            provider="cpu", num_threads=4))
    return sherpa_onnx.OfflineTts(cfg)


_RASA_TTS = None


def load_rasa_v2():
    """Character-frontend rasa model WITH per-speaker styles (built locally
    via build_rasa_styles.py from v2.0.0; verified: all-ALEXA == original,
    sid 0 unchanged, non-ALEXA sids switch). The rasa-v2 release assets are
    Piper-frontend graphs and cannot run here (wrong input names)."""
    global _RASA_TTS
    if _RASA_TTS is None:
        import sherpa_onnx
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=os.path.join(V2D, "vits-rasa-13-styles-model.onnx"),
                    tokens=os.path.join(BASE, "vits-rasa-13", "tokens.txt"),
                    noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0),
                provider="cpu", num_threads=4))
        _RASA_TTS = sherpa_onnx.OfflineTts(cfg)
    return _RASA_TTS


CUR_VOICE = "hi-female"


def score_one(tts, sid, lang, text, cfg, whisper_map=None):
    from jiwer import wer
    ref = norm_text(text)
    wers = []
    for _ in range(2):
        w = synthesize_voice(tts, sid=sid, text=text, lang=lang,
                             length_scale=cfg["length_scale"],
                             sample_rate=tts.sample_rate)
        segs, _ = get_asr().transcribe(
            resample16k(w, tts.sample_rate),
            language=(whisper_map or {}).get(lang, "hi"),
            beam_size=1, initial_prompt=text[:40])
        hyp = norm_text(" ".join(s.text for s in segs).strip())
        if re_arabic(hyp):
            continue
        try:
            wers.append(wer(ref, hyp))
        except ValueError:
            wers.append(1.0)
    return wers


def median_or_none(xs):
    return round(float(statistics.median(xs)), 4) if xs else None


def re_arabic(hyp):
    return bool(re.search(r"[\u0600-\u06FF]", hyp))


def run_syspin(voice):
    global CUR_VOICE
    CUR_VOICE = voice  # e.g. "hi-female": vits-syspin-<voice>-model.onnx
    lang = SYS_VOICES[voice]
    texts = EVAL_TEXTS[lang]
    passage = " ".join(texts)
    rows = []
    cfgs = [dict(zip(GRID.keys(), vals))
            for vals in itertools.product(*(GRID[k] for k in GRID))]
    if DEFAULT_CFG not in cfgs:
        cfgs.append(dict(DEFAULT_CFG))
    for cfg in cfgs:
        tts = load_syspin_v2(cfg["noise_scale"], cfg["noise_w"])
        s = score_one(tts, 0, lang, passage, cfg)
        med = round(float(statistics.median(s)), 4) if s else None
        rows.append({"cfg": cfg, "wer_runs": [round(float(x), 4) for x in s],
                     "wer": med})
        print("  %s -> WER %s" % (cfg, med), flush=True)
    rows.sort(key=lambda r: (r["wer"] if r["wer"] is not None else 9.0,
                             abs(r["cfg"]["length_scale"] - 1.0)))
    win = rows[0]
    # save winner audio
    import soundfile as sf
    tts = load_syspin_v2(win["cfg"]["noise_scale"], win["cfg"]["noise_w"])
    w = synthesize_voice(tts, sid=0, text=passage, lang=lang,
                         length_scale=win["cfg"]["length_scale"],
                         sample_rate=tts.sample_rate)
    sf.write(os.path.join(WAVS, "syspin-v2-%s.wav" % voice), w, tts.sample_rate)
    json.dump({"voice": voice, "rows": rows, "winner": win},
              open(os.path.join(RES, "syspin-v2-%s.json" % voice), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("syspin-v2-%s winner=%s WER=%s" % (voice, win["cfg"], win["wer"]), flush=True)


def run_rasa(sids):
    from tune_inference import RASA_SIDS
    want = {s: (v, l) for s, v, l in RASA_SIDS if s in sids}
    tts = load_rasa_v2()
    for sid in sorted(want):
        vid, lang = want[sid]
        text = RASA_EVAL[lang]
        rows = []
        for ls in GRID["length_scale"]:
            cfg = {"length_scale": ls}
            s = score_one(tts, sid, lang, text, cfg, whisper_map=RASA_WHISPER)
            med = round(float(statistics.median(s)), 4) if s else None
            rows.append({"cfg": cfg, "wer_runs": [round(float(x), 4) for x in s],
                         "wer": med})
            print("  sid %d length %s -> WER %s" % (sid, ls, med), flush=True)
        rows.sort(key=lambda r: (r["wer"] if r["wer"] is not None else 9.0,
                                 abs(r["cfg"]["length_scale"] - 1.0)))
        win = rows[0]
        import soundfile as sf
        w = synthesize_voice(tts, sid=sid, text=text, lang=lang,
                             length_scale=win["cfg"]["length_scale"],
                             sample_rate=tts.sample_rate)
        sf.write(os.path.join(WAVS, "rasa-v2-sid%d.wav" % sid), w, tts.sample_rate)
        json.dump({"voice": vid, "sid": sid, "lang": lang, "rows": rows, "winner": win,
                   "model": "vits-rasa-13-styles-model.onnx (local build from v2.0.0 + upstream style table)"},
                  open(os.path.join(RES, "rasa-v2-sid%d.json" % sid), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("rasa-v2 sid %d winner=%s WER=%s" % (sid, win["cfg"], win["wer"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="*", default=[])
    ap.add_argument("--engine", choices=("syspin-v2", "rasa-v2"), default="syspin-v2")
    ap.add_argument("--sids", nargs="*", type=int, default=[])
    args = ap.parse_args()
    t0 = time.time()
    if args.engine == "syspin-v2":
        for v in (args.voices or ["hi-female", "hi-male"]):
            run_syspin(v)
    else:
        run_rasa(args.sids or RASA_SIDS_TUNE)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
