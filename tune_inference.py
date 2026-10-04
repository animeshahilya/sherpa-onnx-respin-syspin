#!/usr/bin/env python3
"""Per-voice inference tuning for SYSPIN voices (item 3) + eval scorecards (item 5).

Grid-searches length_scale per voice (verified 2026-10-02: noise_scale and
noise_scale_w are NO-OPs in sherpa-onnx 1.13.8 for these exports, generation
is bit-deterministic) and scores each config with ASR-WER (faster-whisper
large-v3 -- base proved too weak at 0.867 vs large 0.133 on identical audio;
relative comparison within a voice) plus a no-reference quality proxy.
Winners can be written back into voices.json `recommended` with --apply
(backup voices.json.bak first).

Honest limits (read before quoting numbers):
  * large-v3 WER is still HIGH-ish in absolute terms for Indic langs and for
    bho/hne/mai/mag we decode with the Hindi model. Only *relative*
    differences within one voice mean anything.
  * proxy_mos is a signal heuristic (clipping/silence/dynamics/tilt), NOT a
    listening test. It breaks ties; WER picks the winner.
  * repeats/shortlist machinery is kept for forward-compatibility; with
    current deterministic output repeats add no information.

Usage:
  python tune_inference.py --voices vits-syspin-hi-female --quick   # smoke test
  python tune_inference.py --langs hi bn                             # full grid, subset
  python tune_inference.py --all                                     # all 22 (slow, ~1h+)
  python tune_inference.py --probe-variance --voices vits-syspin-hi-female
  python tune_inference.py --ablate-frontend --voices vits-syspin-bn-female vits-syspin-bn-male
  python tune_inference.py --apply                                   # write winners to voices.json
  python eval_scorecard.py                                            # scorecards/*.json + SCOREBOARD.md
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
RES = os.path.join(BASE, "tune_results")
WAVS = os.path.join(RES, "wav")
os.makedirs(WAVS, exist_ok=True)

from tts_synth import synthesize_voice  # noqa: E402

EVAL_TEXTS = {
    "hi": ["नमस्ते आप कैसे हैं।", "आज मौसम बहुत अच्छा है।", "मुझे यह किताब बहुत पसंद आई।"],
    "en": ["Hello, how are you today?", "The weather is very nice today.", "I really enjoyed reading this book."],
    "bn": ["নমস্কার, আপনি কেমন আছেন।", "আজ আবহাওয়া খুব ভালো।", "এই বইটা আমার খুব ভালো লেগেছে।"],
    "te": ["నమస్కారం, మీరు ఎలా ఉన్నారు.", "ఈరోజు వాతావరణం చాలా బాగుంది.", "ఈ పుస్తకం నాకు చాలా నచ్చింది."],
    "kn": ["ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ.", "ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.", "ಈ ಪುಸ್ತಕ ನನಗೆ ತುಂಬಾ ಇಷ್ಟವಾಯಿತು."],
    "mr": ["नमस्कार, तुम्ही कसे आहात.", "आज हवामान खूप छान आहे.", "मला हे पुस्तक खूप आवडले."],
    "gu": ["નમસ્તે, તમે કેમ છો.", "આજે હવામાન ખૂબ સરસ છે.", "મને આ પુસ્તક ખૂબ ગમ્યું."],
    "bho": ["नमस्कार, रउआ कइसे बानी।", "आज मौसम बहुत बढ़िया बा।", "हमरा ई किताब बहुत नीक लागल।"],
    "hne": ["नमस्कार, आप मन कइसे हव।", "आज मौसम बहुत अच्छा हे।", "मोला ई किताब बहुत पसंद आईस।"],
    "mai": ["नमस्कार, अहाँ कना छी।", "आइ मौसम बहुत नीक अछि।", "हमरा ई किताब बहुत नीक लागल।"],
    "mag": ["नमस्कार, रउआ कइसे हई।", "आज मौसम बहुत बढ़िया हे।", "हमरा ई किताब बहुत नीक लागल।"],
}
# bho/hne/mai/mag have no whisper model: decode with Hindi (documented limit).
WHISPER_LANG = {"hi": "hi", "en": "en", "bn": "bn", "te": "te", "kn": "kn",
                "mr": "mr", "gu": "gu", "bho": "hi", "hne": "hi",
                "mai": "hi", "mag": "hi"}

# VERIFIED 2026-10-02 on hi-female (direct probe, bit-identical outputs):
# noise_scale (0.1/0.667/1.5) and noise_scale_w (0.1/1.5) are NO-OPs in
# sherpa-onnx 1.13.8 for these exports, and generation is fully
# deterministic across sessions. Only length_scale moves the output.
GRID_FULL = {"length_scale": [0.9, 0.95, 1.0, 1.05, 1.1]}
# noise_scale_w fixed at the repo default 0.8 (see above: no-op).
GRID_QUICK = {"length_scale": [1.0, 1.05]}

_asr = None


def get_asr():
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel
        # large-v3, NOT base: verified on hi-female passage (same audio) --
        # base WER 0.867 (short-vowel confusions: namaste->naasti), large
        # WER 0.133. Base cannot discriminate configs; large can.
        _asr = WhisperModel("large-v3", device="cpu", compute_type="int8")
    return _asr


def norm_text(s):
    s = re.sub(r"[।॥.,?!:;—–\-'\"()“”‘’]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def load_tts(voice_id, noise_scale, noise_scale_w, length_scale):
    import sherpa_onnx
    m = os.path.join(BASE, "release_assets_fp16", voice_id + "-model.onnx")
    t = os.path.join(BASE, "release_assets_fp16", voice_id + "-tokens.txt")
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=m, tokens=t, noise_scale=noise_scale,
                noise_scale_w=noise_scale_w, length_scale=length_scale),
            provider="cpu", num_threads=4))
    return sherpa_onnx.OfflineTts(cfg)


def proxy_mos(wav, sr):
    """No-reference quality proxy 1..5 (NOT a listening test).

    Penalizes: clipping, dead silence, crushed dynamics, harsh spectral
    tilt. Rewards: healthy RMS band. Calibrated so the repo default config
    on hi-female lands ~3.5-4.0; use only for tie-breaks."""
    x = np.asarray(wav, dtype=np.float64)
    if x.size == 0:
        return 1.0
    score = 5.0
    clip = float(np.mean(np.abs(x) > 0.99))
    score -= min(2.0, clip * 40.0)
    frame = 2048
    if x.size > frame:
        e = np.array([np.sqrt(np.mean(x[i:i + frame] ** 2) + 1e-12)
                      for i in range(0, len(x) - frame, frame // 2)])
        silence = float(np.mean(e < 10 ** (-40 / 20)))
        score -= min(1.5, silence * 3.0)
        dyn = 20 * np.log10((np.percentile(e, 95) + 1e-9) / (np.percentile(e, 5) + 1e-9))
        if dyn < 12:
            score -= (12 - dyn) * 0.08
    if x.size >= 4096:
        spec = np.abs(np.fft.rfft(x[:sr * 2] * np.hanning(min(len(x), sr * 2))))
        n = len(spec)
        tilt = float(np.mean(spec[int(n * 0.6):]) / (np.mean(spec[:int(n * 0.6)]) + 1e-12))
        if tilt > 0.6:
            score -= min(1.0, (tilt - 0.6) * 2.0)
    return round(max(1.0, min(5.0, score)), 2)


def transcribe(wav16k, lang, prompt=None):
    """faster-whisper base. Two gotchas handled:
    1. Short clips hallucinate -> callers use paragraph-level audio.
    2. Hindi sometimes decodes in Urdu script despite language='hi'
       (verified) -> Devanagari initial_prompt usually steers it right;
       residual Arabic-block output is flagged unreliable (None) so a
       script flip can't win a tuning comparison.
    Returns (hyp, reliable)."""
    kwargs = dict(language=WHISPER_LANG[lang], beam_size=1)
    if prompt:
        kwargs["initial_prompt"] = prompt
    segs, _ = get_asr().transcribe(wav16k, **kwargs)
    hyp = " ".join(s.text for s in segs).strip()
    if lang in ("hi", "bho", "hne", "mai", "mag", "mr", "bn") and re.search(
            r"[\u0600-\u06FF]", hyp):
        return hyp, False
    return hyp, True


def resample16k(wav, sr):
    import math
    from scipy.signal import resample_poly
    x = np.asarray(wav, dtype=np.float64)
    if sr == 16000:
        return x.astype(np.float32)
    g = math.gcd(sr, 16000)
    return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


def score_config(tts, voice_id, lang, cfg, texts, repeats=1, apocope=True):
    """One paragraph-level synthesis per repeat (the real chunking path):
    short clips truncate/hallucinate in whisper, ~5s+ passages transcribe
    reliably (verified: 1.4s clip -> truncated hyp, 5.4s concat -> full)."""
    from jiwer import wer
    passage = " ".join(texts)
    ref = norm_text(passage)
    wer_runs, mos, rates, clips = [], [], [], []
    for rep in range(repeats):
        w = synthesize_voice(tts, sid=0, text=passage, lang=lang,
                             length_scale=cfg["length_scale"],
                             sample_rate=tts.sample_rate, apocope=apocope)
        hyp, reliable = transcribe(resample16k(w, tts.sample_rate), lang, prompt=texts[0])
        if reliable:
            try:
                wer_runs.append(wer(ref, norm_text(hyp)))
            except ValueError:
                wer_runs.append(1.0)
        mos.append(proxy_mos(w, tts.sample_rate))
        rates.append(len(ref) / max(0.1, len(w) / tts.sample_rate))
        clips.append(float(np.mean(np.abs(np.asarray(w)) > 0.99)))
    return {"wer_runs": [round(float(x), 4) for x in wer_runs],
            "wer": round(float(statistics.median(wer_runs)), 4) if wer_runs else None,
            "proxy_mos": round(float(statistics.median(mos)), 2),
            "chars_per_sec": round(float(statistics.median(rates)), 2),
            "clip_frac": round(float(statistics.median(clips)), 5)}


def lang_of(voice_id):
    m = re.match(r"vits-syspin-([a-z]+)-(fe)?male", voice_id)
    return m.group(1) if m else "hi"


def tune_voice(voice_id, grid, repeats=1, ablation=None, shortlist_extra=0):
    """Phase 1: every grid config x `repeats`. Phase 2: top-3 configs get
    `shortlist_extra` more repeats and the winner is picked by median over
    all runs. (With current deterministic output the extra runs just confirm
    stability; kept for forward-compatibility.) Ablation (bn apocope)
    doubles the grid, not the repeats."""
    lang = lang_of(voice_id)
    texts = EVAL_TEXTS[lang]
    rows = []
    keys = list(grid.keys())
    apo_values = [True, False] if (ablation == "apocope" and lang == "bn") else [True]
    for vals in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, vals))
        cfg.setdefault("noise_scale", 0.667)
        cfg.setdefault("noise_scale_w", 0.8)
        for apo in apo_values:
            tts = load_tts(voice_id, cfg["noise_scale"], cfg["noise_scale_w"], cfg["length_scale"])
            s = score_config(tts, voice_id, lang, cfg, texts, repeats, apocope=apo)
            s.update(cfg)
            if len(apo_values) > 1:
                s["apocope"] = apo
            rows.append(s)
            label = {k: cfg[k] for k in keys}
            if len(apo_values) > 1:
                label["apocope"] = apo
            print("  %s -> WER %s mos %s cps %s" % (
                label, s["wer"], s["proxy_mos"], s["chars_per_sec"]), flush=True)
    # winner: min WER (None = unreliable, sorts last), tie-break closest to
    # repo defaults, then higher proxy_mos
    def keyfn(r):
        return (r["wer"] if r["wer"] is not None else 9.0,
                abs(r["noise_scale"] - 0.667) + abs(r["noise_scale_w"] - 0.8)
                + abs(r["length_scale"] - 1.0) * 2.0,
                -r["proxy_mos"])
    rows.sort(key=keyfn)
    if shortlist_extra > 0:
        for s in rows[:3]:
            cfg = {k: s[k] for k in ("noise_scale", "noise_scale_w", "length_scale")}
            tts = load_tts(voice_id, cfg["noise_scale"], cfg["noise_scale_w"], cfg["length_scale"])
            extra = score_config(tts, voice_id, lang, cfg, texts, shortlist_extra,
                                 apocope=s.get("apocope", True))
            s["wer_runs"].extend(extra["wer_runs"])
            if s["wer_runs"]:
                s["wer"] = round(float(statistics.median(s["wer_runs"])), 4)
            print("  shortlist %s -> WER %s over %d runs" % (
                ({k: cfg[k] for k in keys} |
                 ({"apocope": s["apocope"]} if "apocope" in s else {})),
                s["wer"], len(s["wer_runs"])), flush=True)
    rows.sort(key=keyfn)
    return {"voice": voice_id, "lang": lang, "rows": rows, "winner": rows[0],
            "asr": "faster-whisper large-v3 (int8, beam 1); bho/hne/mai/mag decoded as hi"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="*", default=None)
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--shortlist-extra", type=int, default=0)
    ap.add_argument("--redo", action="store_true",
                    help="re-tune voices that already have tune_results/<voice>.json")
    ap.add_argument("--probe-variance", action="store_true")
    ap.add_argument("--ablate-frontend", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.apply:
        import shutil
        applied = 0
        vpath = os.path.join(BASE, "voices.json")
        shutil.copy2(vpath, vpath + ".bak")
        data = json.load(open(vpath, encoding="utf-8"))
        for f in sorted(os.listdir(RES)):
            if not f.endswith(".json") or f == "tuning_summary.json":
                continue
            r = json.load(open(os.path.join(RES, f), encoding="utf-8"))
            w = r["winner"]
            for entry in data:
                for v in entry["voices"]:
                    if v["id"] == r["voice"]:
                        v["recommended"] = {"noise_scale": w["noise_scale"],
                                            "noise_scale_w": w.get("noise_scale_w", 0.8),
                                            "length_scale": w["length_scale"],
                                            "speed": 1.0}
                        applied += 1
        json.dump(data, open(vpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("applied %d winners (backup voices.json.bak)" % applied)
        return

    grid = GRID_QUICK if args.quick else GRID_FULL
    if args.voices:
        voices = args.voices
    else:
        allv = []
        for lang in (args.langs or list(EVAL_TEXTS)):
            allv += [f"vits-syspin-{lang}-female", f"vits-syspin-{lang}-male"]
        voices = allv if (args.langs or args.all) else allv[:1]

    if args.probe_variance:
        v = voices[0]
        lang = lang_of(v)
        tts = load_tts(v, 0.667, 0.8, 1.0)
        ws = [score_config(tts, v, lang,
                           {"noise_scale": 0.667, "noise_scale_w": 0.8,
                            "length_scale": 1.0}, EVAL_TEXTS[lang])["wer"]
              for _ in range(5)]
        ws = [w for w in ws if w is not None]
        print("variance probe %s WERs=%s median=%s spread=%s"
              % (v, ws, round(statistics.median(ws), 3) if ws else None,
                 round(max(ws) - min(ws), 3) if ws else None))
        return

    summary = {}
    try:
        summary = json.load(open(os.path.join(RES, "tuning_summary.json"), encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        pass
    for v in voices:
        if not args.redo and os.path.exists(os.path.join(RES, v + ".json")):
            print("tuning %s ... exists, skipping (--redo to redo)" % v, flush=True)
            r = json.load(open(os.path.join(RES, v + ".json"), encoding="utf-8"))
            summary[v] = {"winner": {k: r["winner"][k] for k in grid},
                          "wer": r["winner"]["wer"], "proxy_mos": r["winner"]["proxy_mos"]}
            continue
        print("tuning %s ..." % v, flush=True)
        t0 = time.time()
        r = tune_voice(v, grid, args.repeats,
                       ablation="apocope" if args.ablate_frontend else None,
                       shortlist_extra=args.shortlist_extra)
        json.dump(r, open(os.path.join(RES, v + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        # save winner audio for the scorecard
        import soundfile as sf
        tts = load_tts(v, r["winner"]["noise_scale"], r["winner"].get("noise_scale_w", 0.8),
                       r["winner"]["length_scale"])
        w = synthesize_voice(tts, sid=0, text=" ".join(EVAL_TEXTS[r["lang"]]),
                             lang=r["lang"],
                             length_scale=r["winner"]["length_scale"],
                             sample_rate=tts.sample_rate)
        sf.write(os.path.join(WAVS, "%s_passage.wav" % v), w, tts.sample_rate)
        print("%s winner=%s WER=%s (%.0fs)" % (
            v, {k: r["winner"][k] for k in grid}, r["winner"]["wer"], time.time() - t0),
            flush=True)
        summary[v] = {"winner": {k: r["winner"][k] for k in grid},
                      "wer": r["winner"]["wer"], "proxy_mos": r["winner"]["proxy_mos"]}
    json.dump(summary, open(os.path.join(RES, "tuning_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("summary -> tune_results/tuning_summary.json")


if __name__ == "__main__":
    main()
