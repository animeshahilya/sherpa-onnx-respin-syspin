#!/usr/bin/env python3
"""Catalog, per-voice configs and samples for Supertonic 3 in the espeak-ng app.

Every Supertonic style (5 female, 5 male) speaks every one of its 31
languages, so like the AI4Bharat Rasa speakers each language x style is its
own voice: key "<code>-supertonic_<female|male>_<n>-medium". All of them share
one model download (the vector estimator, content-addressed like Rasa's
file); the small per-voice config names the language and style and lists the
companion files (duration predictor, text encoder, vocoder, tokenizer table,
tts.json, the style) the app fetches into the same shared store.

Writes (after build_supertonic.py):
  release_assets_supertonic/<key>.onnx.json   per-voice configs (release supertonic-v1)
  <app>/assets/piper/supertonic_voices.json   the app's catalog of them (voices.json format)
  release_assets_samples/<key>.mp3            samples (release samples-v1), 8 steps, CPU, INT8 files

  python build_supertonic_catalog.py <espeak-ng/android> [--no-samples]
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
OUT = BASE / "release_assets_supertonic"
SAMPLES = BASE / "release_assets_samples"
TAG = "supertonic-v1"
RELEASES = "https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/"
STYLES = [(f"{g}{i}", f"{'female' if g == 'F' else 'male'}_{i}") for g in "FM" for i in range(1, 6)]
# family -> (code, native name, English name, country)
LANGS = {
    "en": ("en_US", "English", "English", "United States"), "ko": ("ko_KR", "한국어", "Korean", "South Korea"),
    "ja": ("ja_JP", "日本語", "Japanese", "Japan"), "ar": ("ar_SA", "العربية", "Arabic", "Saudi Arabia"),
    "bg": ("bg_BG", "Български", "Bulgarian", "Bulgaria"), "cs": ("cs_CZ", "Čeština", "Czech", "Czechia"),
    "da": ("da_DK", "Dansk", "Danish", "Denmark"), "de": ("de_DE", "Deutsch", "German", "Germany"),
    "el": ("el_GR", "Ελληνικά", "Greek", "Greece"), "es": ("es_ES", "Español", "Spanish", "Spain"),
    "et": ("et_EE", "Eesti", "Estonian", "Estonia"), "fi": ("fi_FI", "Suomi", "Finnish", "Finland"),
    "fr": ("fr_FR", "Français", "French", "France"), "hi": ("hi_IN", "हिन्दी", "Hindi", "India"),
    "hr": ("hr_HR", "Hrvatski", "Croatian", "Croatia"), "hu": ("hu_HU", "Magyar", "Hungarian", "Hungary"),
    "id": ("id_ID", "Bahasa Indonesia", "Indonesian", "Indonesia"), "it": ("it_IT", "Italiano", "Italian", "Italy"),
    "lt": ("lt_LT", "Lietuvių", "Lithuanian", "Lithuania"), "lv": ("lv_LV", "Latviešu", "Latvian", "Latvia"),
    "nl": ("nl_NL", "Nederlands", "Dutch", "Netherlands"), "pl": ("pl_PL", "Polski", "Polish", "Poland"),
    "pt": ("pt_BR", "Português", "Portuguese", "Brazil"), "ro": ("ro_RO", "Română", "Romanian", "Romania"),
    "ru": ("ru_RU", "Русский", "Russian", "Russia"), "sk": ("sk_SK", "Slovenčina", "Slovak", "Slovakia"),
    "sl": ("sl_SI", "Slovenščina", "Slovenian", "Slovenia"), "sv": ("sv_SE", "Svenska", "Swedish", "Sweden"),
    "tr": ("tr_TR", "Türkçe", "Turkish", "Türkiye"), "uk": ("uk_UA", "Українська", "Ukrainian", "Ukraine"),
    "vi": ("vi_VN", "Tiếng Việt", "Vietnamese", "Vietnam"),
}
# Shared files: name in the voice folder -> release file
COMPANIONS = {"duration_predictor.onnx": "supertonic3-duration_predictor.onnx",
              "text_encoder.onnx": "supertonic3-text_encoder.onnx",
              "vocoder.onnx": "supertonic3-vocoder.onnx",
              "unicode_indexer.json": "supertonic3-unicode_indexer.json",
              "tts.json": "supertonic3-tts.json"}
MODEL = "supertonic3-vector_estimator.onnx"

sys.stdout.reconfigure(encoding="utf-8")
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)


def info(name):
    data = (OUT / name).read_bytes()
    return {"size_bytes": len(data), "md5_digest": hashlib.md5(data).hexdigest()}


def files_entry(name):
    return {f"{TAG}/{name}": info(name)}


def build(app):
    catalog = {}
    shared = {k: {"path": f"{TAG}/{v}", **info(v)} for k, v in COMPANIONS.items()}
    for family, (code, native, english, country) in LANGS.items():
        lang = {"code": code, "family": family, "region": code.split("_")[1], "name_native": native,
                "name_english": english, "country_english": country}
        for style, name in STYLES:
            key = f"{code}-supertonic_{name}-medium"
            cfg = {
                "dataset": "supertonic3",
                "audio": {"sample_rate": 44100, "quality": "medium"},
                "phoneme_type": "supertonic",
                "language": lang,
                "inference": {"noise_scale": 1.0, "length_scale": 1.0, "noise_w": 1.0},
                "num_speakers": 1,
                "phoneme_id_map": {"_": [0]},
                "hop_length": 3072,
                "supertonic": {"lang": family, "style": style, "steps": 8,
                               "files": {**shared, "style.json": {"path": f"{TAG}/supertonic3-style-{style}.json",
                                                                  **info(f"supertonic3-style-{style}.json")}}},
            }
            cfg_name = f"{key}.onnx.json"
            (OUT / cfg_name).write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
            catalog[key] = {
                "key": key, "name": f"supertonic_{name}", "language": lang, "quality": "medium",
                "num_speakers": 1, "base_url": RELEASES, "source": "Supertone Supertonic 3",
                "license": "OpenRAIL-M",
                "files": {**files_entry(MODEL), **files_entry(cfg_name)},
            }
    dst = Path(app) / "assets" / "piper" / "supertonic_voices.json"
    dst.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(catalog)} voices -> {dst}")
    return catalog


def sample_text(app, family, native):
    import re
    res = {"id": "in"}.get(family, family)  # Android's legacy code for Indonesian
    dirs = ("values",) if family == "en" else (f"values-{res}", f"values-b+{res}")
    for d in dirs:
        f = Path(app) / "res" / d / "strings.xml"
        if f.is_file():
            m = re.search(r'name="sample_text"[^>]*>([^<]*)<', f.read_text(encoding="utf-8"))
            if m:
                return m.group(1).replace("\\'", "'").replace("%s", native or family)
    raise SystemExit(f"no sample_text for {family}")


def samples(app, catalog):
    import soundfile as sf
    import check_supertonic as c
    src = BASE / "supertonic_src"
    h = c.helper_module(src)
    tts = c.pipeline(h, OUT, {"cfg": "supertonic3-tts.json", "idx": "supertonic3-unicode_indexer.json",
                              "dp": "supertonic3-duration_predictor.onnx", "te": "supertonic3-text_encoder.onnx",
                              "ve": MODEL, "voc": "supertonic3-vocoder.onnx"})
    SAMPLES.mkdir(exist_ok=True)
    styles = {s: h.load_voice_style([str(OUT / f"supertonic3-style-{s}.json")]) for s, _ in STYLES}
    for i, (key, v) in enumerate(catalog.items(), 1):
        out = SAMPLES / f"{key}.mp3"
        if out.is_file():
            continue
        fam = v["language"]["family"]
        text = sample_text(app, fam, v["language"]["name_native"])
        style = next(s for s, n in STYLES if f"supertonic_{n}-" in key)
        np.random.seed(7)
        wav, dur = tts(text, fam, styles[style], 8, 1.0)
        y = wav[0, :int(tts.sample_rate * dur[0])]
        y = y / max(1e-6, np.abs(y).max()) * 0.9
        sf.write(out, y.astype(np.float32), tts.sample_rate, format="MP3")
        print(f"[{i}/{len(catalog)}] {out.name}: {text[:40]}", flush=True)


if __name__ == "__main__":
    cat = build(sys.argv[1])
    if "--no-samples" not in sys.argv:
        samples(sys.argv[1], cat)
