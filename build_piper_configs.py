#!/usr/bin/env python3
"""Piper-style configs for all 42 voices, for apps with a Piper "text" frontend
(phoneme_type "text": characters in, no phonemizer), such as the espeak-ng
Android fork's natural voices.

Writes release_assets_piper/<key>.onnx.json (uploaded to the piper-v1 release)
and release_assets_piper/catalog.json, entries for that app's extra_voices.json.

Each config maps Piper's pad/start/end ("_", "^", "$") to the model's blank,
so Piper's phonemes_to_ids() yields the add_blank interspersal the models were
trained with (plus one extra blank at each end). Rasa voices are single-speaker
views of the shared multi-speaker file: num_speakers 1, default_speaker_id = sid.

Languages eSpeak NG has no voice for (Bhojpuri, Chhattisgarhi, Magahi,
Maithili, Sanskrit, Bodo, Dogri) are listed under Hindi, named "<Name> <Language>".

  python build_piper_configs.py
"""

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "release_assets_piper"
RELEASES = "https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/"
CONFIG_TAG = "piper-v1"
SYSPIN_TAG = "v1.1.0-fp16"
RASA_MODEL = BASE / "release_assets_rasa" / "vits-rasa-13-piper-model.onnx"

# voices.json langCode -> (family, region, native name, English name, country)
LANGS = {
    "hi": ("hi", "IN", "हिन्दी", "Hindi", "India"),
    "bn": ("bn", "IN", "বাংলা", "Bengali", "India"),
    "te": ("te", "IN", "తెలుగు", "Telugu", "India"),
    "kn": ("kn", "IN", "ಕನ್ನಡ", "Kannada", "India"),
    "mr": ("mr", "IN", "मराठी", "Marathi", "India"),
    "gu": ("gu", "IN", "ગુજરાતી", "Gujarati", "India"),
    "en": ("en", "IN", "English", "English", "India"),
    "asm": ("as", "IN", "অসমীয়া", "Assamese", "India"),
    "mal": ("ml", "IN", "മലയാളം", "Malayalam", "India"),
    "pan": ("pa", "IN", "ਪੰਜਾਬੀ", "Punjabi", "India"),
    "ne": ("ne", "NP", "नेपाली", "Nepali", "Nepal"),
    "tam": ("ta", "IN", "தமிழ்", "Tamil", "India"),
}
# No eSpeak NG language: read Devanagari, so they join Hindi.
UNDER_HINDI = {"bho": "bhojpuri", "hne": "chhattisgarhi", "mag": "magahi", "mai": "maithili",
               "san": "sanskrit", "brx": "bodo", "doi": "dogri"}


def read_tokens(path):
    ids = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sym, _, num = line.rpartition(" ")
        ids[sym if sym else " "] = int(num)
    return ids


def file_entry(path):
    data = path.read_bytes()
    return {"size_bytes": len(data), "md5_digest": hashlib.md5(data).hexdigest()}


def main():
    import onnx
    OUT.mkdir(exist_ok=True)
    catalog = {}
    for group in json.loads((BASE / "voices.json").read_text(encoding="utf-8")):
        lc = group["langCode"]
        for v in group["voices"]:
            rasa = v.get("engine") == "rasa"
            slug = v["displayName"].lower()
            if lc in UNDER_HINDI:
                slug += "_" + UNDER_HINDI[lc]
                family, region, native, english, country = LANGS["hi"]
            else:
                family, region, native, english, country = LANGS[lc]
            code = f"{family}_{region}"
            key = f"{code}-{slug}-medium"

            if rasa:
                tokens = read_tokens(BASE / "vits-rasa-13" / "tokens.txt")
                blank, model = 0, RASA_MODEL
                model_path = f"{CONFIG_TAG}/{RASA_MODEL.name}"
            else:
                tokens = read_tokens(BASE / v["id"] / "tokens.txt")
                model = BASE / "release_assets_fp16" / f"{v['id']}-model.onnx"
                meta = {p.key: p.value for p in onnx.load(str(model), load_external_data=False).metadata_props}
                blank = int(meta["blank_id"])
                model_path = f"{SYSPIN_TAG}/{model.name}"
            id_map = {s: [i] for s, i in tokens.items() if not s.startswith("<")}
            for special in ("_", "^", "$"):
                id_map[special] = [blank]

            rec = v["recommended"]
            config = {
                "dataset": slug,
                "audio": {"sample_rate": v["sampleRate"], "quality": "medium"},
                "phoneme_type": "text",
                "language": {"code": code, "family": family, "region": region,
                             "name_native": native, "name_english": english,
                             "country_english": country},
                "inference": {"noise_scale": rec["noise_scale"], "length_scale": rec["length_scale"],
                              "noise_w": rec["noise_scale_w"]},
                "num_speakers": 1,
                "default_speaker_id": v.get("sid", 0) if rasa else 0,
                "hop_length": 256,
                "phoneme_id_map": id_map,
            }
            cfg_file = OUT / f"{key}.onnx.json"
            cfg_file.write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

            catalog[key] = {
                "key": key, "name": slug,
                "language": {"code": code, "family": family, "region": region,
                             "name_native": native, "name_english": english, "country_english": country},
                "quality": "medium", "num_speakers": 1,
                "base_url": RELEASES,
                "source": "AI4Bharat Rasa" if rasa else "SYSPIN (IISc SPIRE Lab)",
                "license": "CC-BY-4.0" if rasa else "MIT",
                "files": {model_path: file_entry(model),
                          f"{CONFIG_TAG}/{cfg_file.name}": file_entry(cfg_file)},
            }
    (OUT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"{len(catalog)} configs -> {OUT}")


if __name__ == "__main__":
    main()
