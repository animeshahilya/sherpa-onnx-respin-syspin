#!/usr/bin/env python3
"""Recorded samples for the espeak-ng app's extra voices (release samples-v1).

The app plays rhasspy's piper-samples for Piper's own voices; voices in its
bundled extra_voices.json (SYSPIN, Rasa, Compact versions, community voices)
had none. This renders each one with its own model the way the app reads it:
the app's sample sentence in the voice's language, letters for text voices,
eSpeak NG IPA (the app's own build) for the rest, the hiss treatment for the
voices the app gives it. One <key>.mp3 per catalog key.

  python build_samples.py <espeak-ng checkout> [key ...]
"""
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import scipy.signal as sg
import soundfile as sf

BASE = Path(__file__).parent
OUT = BASE / "release_assets_samples"
CACHE = BASE / "sample_models"
LOCAL = {"syspin-v2": BASE / "release_assets_syspin_v2", "rasa-v2": BASE / "release_assets_rasa_v2",
         "piper-v2": BASE / "release_assets_piper", "compact-v1": BASE / "release_assets_compact"}
# Mirrors PiperVoiceConfig.HISSY and PiperAudio (TrebleCut, quiet start).
HISSY = {"gu_IN-hetal-medium", "gu_IN-hetal-compact", "en_IN-priya-medium", "en_IN-priya-compact",
         "en_US-ljspeech-high", "en_US-ljspeech-compact"}


def sample_text(app, family, native):
    """The app's R.string.sample_text for the language, as SpeechSynthesis.getSampleText fills it."""
    for d in (f"values-{family}", f"values-b+{family}", "values"):
        f = app / "res" / d / "strings.xml"
        if f.is_file():
            m = re.search(r'name="sample_text"[^>]*>([^<]*)<', f.read_text(encoding="utf-8"))
            if m:
                return m.group(1).replace("\\'", "'").replace("%s", native or family)


def fetch(url, md5):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / (md5 + "-" + url.rsplit("/", 1)[1])
    if not path.is_file():
        data = urllib.request.urlopen(url, timeout=600).read()
        assert hashlib.md5(data).hexdigest() == md5, url
        path.write_bytes(data)
    return path


def local_or_fetch(base, rel, info):
    tag, _, name = rel.rpartition("/")
    if tag in LOCAL:
        for d in (LOCAL[tag], LOCAL[tag] / "configs"):
            if (d / name).is_file():
                return d / name
    return fetch(base + rel, info["md5_digest"])


def ipa(espeak, voice, text):
    exe, data = espeak
    tmp = OUT / "_in.txt"
    tmp.write_text(text, encoding="utf-8")
    r = subprocess.run([str(exe), f"--path={data}", "-q", "--ipa", "-v", voice, "-f", str(tmp)],
                       capture_output=True)
    return " ".join(r.stdout.decode("utf-8").split())


def ids(cfg, text, is_text):
    m = cfg["phoneme_id_map"]
    out = list(m["^"]) + (list(m["_"]) if not is_text else [])
    if is_text:
        chars = []
        for ch in unicodedata.normalize("NFC", text).lower():
            chars += [ch] if ch in m else list(unicodedata.normalize("NFD", ch))
    else:
        chars = list(unicodedata.normalize("NFD", text))
    for ch in chars:
        if ch in m:
            out += m[ch] + m["_"]
    return out + m["$"]


def dehiss(y, sr):
    a = 10 ** (-8 / 40); w = 2 * np.pi * min(6000, 0.45 * sr) / sr; c = np.cos(w)
    s = 2 * np.sqrt(a) * np.sin(w) / (2 * np.sqrt(0.5))
    b = [a * ((a + 1) + (a - 1) * c + s), -2 * a * ((a - 1) + (a + 1) * c), a * ((a + 1) + (a - 1) * c - s)]
    d = [(a + 1) - (a - 1) * c + s, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - s]
    y = sg.lfilter(b, d, y)
    f = sr // 100
    p = np.array([np.mean(y[i * f:(i + 1) * f] ** 2) for i in range((len(y) + f - 1) // f)])
    speech = p[p >= p.max() * 1e-3].mean()
    start = int(np.argmax(p >= speech * 10 ** (-18 / 10)))
    y = y[max(0, start * f - int(0.03 * sr)):].copy()
    y[:sr // 100] *= np.linspace(0, 1, sr // 100)
    return y


def main():
    app = Path(sys.argv[1]) / "android"
    only = set(sys.argv[2:])
    espeak_build = next((Path(sys.argv[1]) / "android" / ".cxx").glob("*/*/espeak-data"))
    espeak = (espeak_build / "src" / "Debug" / "espeak-ng.exe", espeak_build)
    catalog = json.loads((app / "assets" / "piper" / "extra_voices.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    ort.set_default_logger_severity(3)
    for key, v in catalog.items():
        if only and key not in only:
            continue
        model_rel = next(p for p in v["files"] if p.endswith(".onnx"))
        cfg_rel = next(p for p in v["files"] if p.endswith(".json"))
        model = local_or_fetch(v["base_url"], model_rel, v["files"][model_rel])
        cfg = json.loads(local_or_fetch(v["base_url"], cfg_rel, v["files"][cfg_rel]).read_text(encoding="utf-8"))
        lang = v["language"]
        text = sample_text(app, lang["family"], lang.get("name_native"))
        is_text = cfg.get("phoneme_type") == "text"
        x = ids(cfg, text if is_text else ipa(espeak, cfg["espeak"]["voice"], text), is_text)
        sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        inf = cfg.get("inference", {})
        feed = {"input": np.array([x], np.int64), "input_lengths": np.array([len(x)], np.int64),
                "scales": np.array([inf.get("noise_scale", 0.667), inf.get("length_scale", 1.0),
                                    inf.get("noise_w", 0.8)], np.float32)}
        if "sid" in [i.name for i in sess.get_inputs()]:
            feed["sid"] = np.array([cfg.get("default_speaker_id", 0)], np.int64)
        y = sess.run(None, feed)[0].squeeze().astype(np.float64)
        sr = cfg["audio"]["sample_rate"]
        if key in HISSY:
            y = dehiss(y, sr)
        y = y / max(1e-6, float(np.abs(y).max())) * 0.9
        sf.write(OUT / f"{key}.mp3", y.astype(np.float32), sr, format="MP3")
        print(key, f"{len(y) / sr:.1f}s", text, flush=True)


if __name__ == "__main__":
    main()
