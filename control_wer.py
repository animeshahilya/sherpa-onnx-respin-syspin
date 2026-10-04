#!/usr/bin/env python3
"""Validate the ASR judge per language (item 3/5 prerequisite).

Transcribes each COMMITTED sample mp3 (upstream's own audio) with the mapped
whisper language and scores WER/CER vs the passage text. A judge that can't
score upstream's own clean audio within reason cannot tune configs:
langs failing here are proxy_mos+pace-sanity only, loudly flagged.
Writes audit/asr_judge.json + prints table.
"""

import io
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tune_inference import (EVAL_TEXTS, RASA_WHISPER, WHISPER_LANG, get_asr,  # noqa: E402
                            norm_text)

BASE = os.path.dirname(os.path.abspath(__file__))

SYS_LANGS = ["hi", "en", "bn", "te", "kn", "mr", "gu", "bho", "hne", "mai", "mag"]
RASA_LANG_OF_SID = {0: "asm", 1: "asm", 2: "bn", 3: "bn", 4: "brx", 5: "brx",
                    6: "doi", 7: "doi", 8: "kn", 9: "kn", 10: "mai", 11: "mal",
                    12: "mr", 13: "mr", 14: "ne", 15: "pan", 16: "pan",
                    17: "san", 18: "tam", 19: "te"}
RASA_MP3 = {0: "vits-rasa-asm-female", 1: "vits-rasa-asm-male", 2: "vits-rasa-bn-female-alt",
            3: "vits-rasa-bn-male-alt", 4: "vits-rasa-brx-female", 5: "vits-rasa-brx-male",
            6: "vits-rasa-doi-female", 7: "vits-rasa-doi-male", 8: "vits-rasa-kn-female-alt",
            9: "vits-rasa-kn-male-alt", 10: "vits-rasa-mai-male-alt", 11: "vits-rasa-mal-female",
            12: "vits-rasa-mr-female-alt", 13: "vits-rasa-mr-male-alt", 14: "vits-rasa-ne-female",
            15: "vits-rasa-pan-female", 16: "vits-rasa-pan-male", 17: "vits-rasa-san-male",
            18: "vits-rasa-tam-female", 19: "vits-rasa-te-female-alt"}


def to_wav(mp3, wav):
    subprocess.run(["ffmpeg", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", wav],
                   capture_output=True, check=True)


def dominant_block(s):
    """Dominant Unicode script block of alphabetic chars, e.g. 'MALAYALAM'."""
    import unicodedata
    from collections import Counter
    c = Counter()
    for ch in s:
        if ch.isalpha():
            try:
                c[unicodedata.name(ch).split()[0]] += 1
            except ValueError:
                pass
    return c.most_common(1)[0][0] if c else None


def main():
    from jiwer import wer, cer
    from generate_rasa_samples import RASA_TEXTS
    from build_dashboard import VOICES_DATA
    syspass = {v["id"]: v["text"] for v in VOICES_DATA}
    out = {}
    jobs = []
    for lang in SYS_LANGS:
        for gender in ("female", "male"):
            vid = f"vits-syspin-{lang}-{gender}"
            mp3 = os.path.join(BASE, "samples", vid + ".mp3")
            if os.path.exists(mp3):
                jobs.append((f"syspin/{vid}", mp3, syspass[lang],
                             WHISPER_LANG.get(lang, "hi")))
    for sid in range(20):
        mp3 = os.path.join(BASE, "samples", RASA_MP3[sid] + ".mp3")
        lang = RASA_LANG_OF_SID[sid]
        if os.path.exists(mp3):
            jobs.append((f"rasa-sid{sid}/{lang}", mp3, RASA_TEXTS[lang]["text"],
                         RASA_WHISPER.get(lang, "hi")))
    for key, mp3, ref_text, wlang in jobs:
        wav = os.path.join(BASE, "audit", "_ctrl.wav")
        to_wav(mp3, wav)
        import soundfile as sf
        w, sr = sf.read(wav, dtype="float32")
        segs, _ = get_asr().transcribe(w, language=wlang, beam_size=1)
        hyp = norm_text(" ".join(s.text for s in segs).strip())
        ref = norm_text(ref_text)
        try:
            wv, cv = round(float(wer(ref, hyp)), 4), round(float(cer(ref, hyp)), 4)
        except ValueError:
            wv, cv = 1.0, 1.0
        # Script confusion (verified: whisper-ml decoded Malayalam audio as
        # Gurmukhi) invalidates the judge regardless of scores.
        script_ok = dominant_block(hyp) == dominant_block(ref)
        ok = script_ok and ((wv is not None and wv < 0.6) or (cv is not None and cv < 0.4))
        out[key] = {"whisper_lang": wlang, "wer": wv, "cer": cv,
                    "judge_ok": bool(ok), "dur_s": round(len(w) / sr, 1)}
        if not script_ok:
            out[key]["note"] = "script confusion: hyp=%s ref=%s" % (
                dominant_block(hyp), dominant_block(ref))
        print("%-22s [%s] WER=%s CER=%s %s" % (
            key, wlang, wv, cv, "OK" if ok else "JUDGE-FAIL"), flush=True)
    json.dump(out, open(os.path.join(BASE, "audit", "asr_judge.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote audit/asr_judge.json")


if __name__ == "__main__":
    main()
