#!/usr/bin/env python3
"""Eval scorecards for SYSPIN voices (item 5).

Reads tune_results/<voice>.json + winner passage wavs, writes
scorecards/<voice>.json and a SCOREBOARD.md table with per-voice metrics
at the tuned config: WER (large-v3, relative within voice), proxy MOS,
chars/sec, clip fraction, duration, plus the full length_scale sweep.

Also re-verifies each winner by re-transcribing its saved wav (guards
against stale results).
"""

import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
SC = os.path.join(BASE, "scorecards")
os.makedirs(SC, exist_ok=True)

from tune_inference import (EVAL_TEXTS, RASA_SIDS, RASA_WHISPER, WHISPER_LANG,  # noqa: E402
                             get_asr, norm_text, proxy_mos, resample16k)


def score_rasa():
    """Rasa scorecards: tune_results/rasa-sid<N>.json + *_eval.wav.
    Judge-fail sids (no sweep rows) get a card with the keep-1.0 note."""
    import soundfile as sf
    from jiwer import wer
    from generate_rasa_samples import RASA_TEXTS
    resdir = os.path.join(BASE, "tune_results")
    rows = []
    for sid, vid, lang in RASA_SIDS:
        r = json.load(open(os.path.join(resdir, "rasa-sid%d.json" % sid), encoding="utf-8"))
        w = r["winner"]
        # identical prompt to tuning (first sentence); a truncated prompt
        # was verified to swing WER wildly on identical audio
        prompt = RASA_TEXTS[lang]["text"].split(".")[0].strip()
        ref = norm_text(prompt)
        wp = os.path.join(resdir, "wav", "rasa-sid%d_eval.wav" % sid)
        if r["rows"] and os.path.exists(wp):
            audio, sr = sf.read(wp, dtype="float32")
            dur = len(audio) / sr
            segs, _ = get_asr().transcribe(
                resample16k(audio, sr), language=RASA_WHISPER.get(lang, "hi"),
                beam_size=1, initial_prompt=prompt)
            hyp = norm_text(" ".join(s.text for s in segs).strip())
            try:
                check_wer = round(float(wer(ref, hyp)), 4)
            except ValueError:
                check_wer = None
        else:
            dur, check_wer = 0.0, None
        card = {
            "voice": vid, "lang": lang, "sid": sid,
            "tuned": {k: w[k] for k in ("noise_scale", "noise_scale_w", "length_scale")
                      if k in w},
            "wer": w.get("wer"), "wer_recheck": check_wer,
            "proxy_mos": w.get("proxy_mos"), "chars_per_sec": w.get("chars_per_sec"),
            "clip_frac": w.get("clip_frac"), "duration_s": round(dur, 2),
            "note": r.get("note", ""),
            "sweep": [{k: row.get(k) for k in
                       ("length_scale", "wer", "proxy_mos",
                        "chars_per_sec", "clip_frac")} for row in r["rows"]],
            "asr": r.get("asr", ""),
        }
        json.dump(card, open(os.path.join(SC, "rasa-%s.json" % vid), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        rows.append(card)
        print("sid %2d %-24s length=%s wer=%s recheck=%s %s" % (
            sid, vid, w.get("length_scale"), w.get("wer"), check_wer,
            ("[" + card["note"] + "]" if card["note"] else "")), flush=True)

    tuned = [c for c in rows if c["wer"] is not None]
    tuned.sort(key=lambda c: (c["wer"], c["voice"]))
    md = ["# Rasa scoreboard (tuned pace per sid, large-v3 WER, proxy MOS)",
          "",
          "> WER is whisper-large-v3 on first-sentence evals — relative within "
          "a sid only; brx/doi/san decoded with the Hindi model. Sids flagged "
          "judge-fail keep length_scale 1.0 (see audit/asr_judge.json). "
          "proxy_mos is a signal heuristic, not a listening test.",
          "",
          "| sid | voice | length | WER | recheck | mos* | cps |",
          "|---|---|---|---|---|---|---|"]
    for c in tuned:
        md.append("| %d | %s | %s | %s | %s | %s | %s |" % (
            c["sid"], c["voice"], c["tuned"].get("length_scale"), c["wer"],
            c["wer_recheck"], c["proxy_mos"], c["chars_per_sec"]))
    md += ["", "Judge-fail sids (default pace, needs ears): " +
           ", ".join("%d %s" % (c["sid"], c["voice"]) for c in rows if c["wer"] is None)]
    open(os.path.join(BASE, "SCOREBOARD_RASA.md"), "w", encoding="utf-8").write(
        "\n".join(md) + "\n")
    print("wrote scorecards/rasa-*.json + SCOREBOARD_RASA.md")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("syspin", "rasa"), default="syspin")
    args = ap.parse_args()
    if args.engine == "rasa":
        return score_rasa()
    import soundfile as sf
    from jiwer import wer
    resdir = os.path.join(BASE, "tune_results")
    voices = sorted(f[:-5] for f in os.listdir(resdir)
                    if f.endswith(".json") and f != "tuning_summary.json")
    rows = []
    for v in voices:
        r = json.load(open(os.path.join(resdir, v + ".json"), encoding="utf-8"))
        lang, w = r["lang"], r["winner"]
        wp = os.path.join(resdir, "wav", v + "_passage.wav")
        audio, sr = sf.read(wp, dtype="float32")
        dur = len(audio) / sr
        # independent re-transcription of the saved winner audio
        segs, _ = get_asr().transcribe(
            resample16k(audio, sr), language=WHISPER_LANG[lang], beam_size=1,
            initial_prompt=EVAL_TEXTS[lang][0])
        hyp = norm_text(" ".join(s.text for s in segs).strip())
        ref = norm_text(" ".join(EVAL_TEXTS[lang]))
        try:
            check_wer = round(float(wer(ref, hyp)), 4)
        except ValueError:
            check_wer = None
        card = {
            "voice": v, "lang": lang,
            "tuned": {k: w[k] for k in ("noise_scale", "noise_scale_w", "length_scale")
                      if k in w},
            "wer": w["wer"], "wer_recheck": check_wer,
            "proxy_mos": w["proxy_mos"], "chars_per_sec": w["chars_per_sec"],
            "clip_frac": w["clip_frac"], "duration_s": round(dur, 2),
            "sweep": [{k: row.get(k) for k in
                       ("length_scale", "apocope", "wer", "proxy_mos",
                        "chars_per_sec", "clip_frac")} for row in r["rows"]],
            "asr": r.get("asr", ""),
        }
        json.dump(card, open(os.path.join(SC, v + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        rows.append(card)
        print("%s length=%s wer=%s recheck=%s mos=%s cps=%s dur=%.1fs" % (
            v, w["length_scale"], w["wer"], check_wer, w["proxy_mos"],
            w["chars_per_sec"], dur), flush=True)

    rows.sort(key=lambda c: (c["wer"] if c["wer"] is not None else 9.0,
                             c["voice"]))
    md = ["# SYSPIN scoreboard (tuned configs, large-v3 WER, proxy MOS)",
          "",
          "> WER is whisper-large-v3 on short eval passages — relative within a "
          "voice only; bho/hne/mai/mag decoded with the Hindi model. "
          "proxy_mos is a signal heuristic, not a listening test.",
          "",
          "| voice | length | WER | recheck | mos* | cps | dur_s |",
          "|---|---|---|---|---|---|---|"]
    for c in rows:
        md.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            c["voice"], c["tuned"].get("length_scale"), c["wer"],
            c["wer_recheck"], c["proxy_mos"], c["chars_per_sec"], c["duration_s"]))
    md += ["",
           "Worst WER voices (tune first / review frontend): " +
           ", ".join("%s (%s)" % (c["voice"], c["wer"]) for c in rows[-3:])]
    open(os.path.join(BASE, "SCOREBOARD.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("wrote scorecards/*.json + SCOREBOARD.md")


if __name__ == "__main__":
    main()
