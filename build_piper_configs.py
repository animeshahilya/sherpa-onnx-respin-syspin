#!/usr/bin/env python3
"""Piper-style configs for all 42 voices, for apps with a Piper "text" frontend
(phoneme_type "text": characters in, no phonemizer), such as the espeak-ng
Android fork's natural voices.

Writes release_assets_piper/<key>.onnx.json (uploaded to the piper-v2 release)
and release_assets_piper/catalog.json, entries for that app's extra_voices.json.

Each config maps Piper's pad/start/end ("_", "^", "$") to the model's blank,
so Piper's phonemes_to_ids() yields the add_blank interspersal the models were
trained with (plus one extra blank at each end). Rasa voices are single-speaker
views of the shared multi-speaker file: num_speakers 1, default_speaker_id = sid.

Languages eSpeak NG has no voice for (Bhojpuri, Chhattisgarhi, Magahi,
Maithili, Sanskrit, Bodo, Dogri) keep their own language; the app gives each
its own TTS voice, with eSpeak's Hindi rules as the stand-in.

  python build_piper_configs.py
"""

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "release_assets_piper"
RELEASES = "https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/"
# Configs for SYSPIN + Rasa, Standard and Compact. piper-v1 keeps the older
# ones for apps that pinned their checksums.
CONFIG_TAG = "piper-v2"
# SYSPIN that reads noise_scale/noise_w from `scales` (build_syspin_scales.py),
# Standard and Compact; v1.1.0-fp16/compact-v1 keep the older exports.
SYSPIN_TAG = "syspin-v2"
SYSPIN_DIR = BASE / "release_assets_syspin_v2"
# The value those exports had frozen: the default sound stays the same.
SYSPIN_NOISE_W = 1.0
# Rasa with a speaking style per speaker (build_rasa_styles.py), own release:
# piper-v1/compact-v1 keep the all-ALEXA files for apps that pinned them.
RASA_TAG = "rasa-v2"
RASA_DIR = BASE / "release_assets_rasa_v2"
RASA_MODEL = RASA_DIR / "vits-rasa-13-piper-model.onnx"
# Compact tier (build_compact.py): INT8 decoders, models + configs in their own release.
COMPACT_TAG = "compact-v1"
COMPACT_DIR = BASE / "release_assets_compact"
COMPACT_CONFIGS = COMPACT_DIR / "configs"
# Every kept voice with INT8 weights (build_small.py), under its own file
# name: Standard and Compact alike. syspin-v2, rasa-v2 and compact-v1 keep
# the float files for apps that pinned them.
INT8_TAG = "int8-v1"
INT8_DIR = BASE / "release_assets_int8"
APP_ASSETS = BASE.parent / "espeak-ng" / "android" / "assets" / "piper"
PIPER_SRC = BASE / "piper_src"
# Piper "high" voices given a Compact version: licence from each MODEL_CARD.
# Not here: en_US-lessac (Blizzard 2013 licence, no redistribution),
# en_US-ryan (CC BY-NC-SA), es_MX-claude and en_US-libritts (older exports
# with unnamed graph nodes: no decoder to find, so nothing to quantize).
PIPER_COMPACT = {
    "de_DE-thorsten-high": "CC0", "en_GB-cori-high": "public domain",
"en_US-ljspeech-high": "public domain",
    "es_AR-daniela-high": "CC-BY-SA-4.0", "it_IT-serena-high": "CC-BY-4.0",
    "kk_KZ-issai-high": "CC-BY-4.0", "pl_PL-bass-high": "Apache-2.0",
    "uk_UA-mykyta-high": "Apache-2.0", "uk_UA-oleksa-high": "Apache-2.0",
    "uk_UA-tetiana-high": "Apache-2.0",
}

# The kept rhasspy voices' licences, from each MODEL_CARD (2026-10-08).
# Unclear ones are re-hosted like the rest (user's decision 2026-10-08).
SEE_CARD = "see the voice's MODEL_CARD in rhasspy/piper-voices"
IITM = "IIT Madras Indic TTS licence"
LICENCES = {
    "ar_JO-kareem-medium": SEE_CARD, "bg_BG-dimitar-medium": "CC0", "ca_ES-upc_ona-medium": "CC-BY-SA-3.0-ES",
    "cs_CZ-kasandra-medium": "CC-BY-4.0", "cs_CZ-jirka-medium": "CC0", "cy_GB-gwryw_gogleddol-medium": SEE_CARD,
    "cy_GB-bu_tts-medium": "CC-BY", "da_DK-talesyntese-medium": "CC0", "de_DE-thorsten_emotional-medium": "CC0",
    "el_GR-joy-medium": "CC-BY-NC-4.0", "el_GR-rapunzelina-medium": "CC0",
    "en_US-hfc_female-medium": "CC-BY-NC-SA-4.0", "en_US-amy-medium": SEE_CARD, "es_MX-claude-high": "Apache-2.0",
    "es_ES-sharvard-medium": "CC-BY-3.0", "et_EE-news-medium": "CC-BY", "eu_ES-maider-medium": "CC-BY-4.0",
    "eu_ES-antton-medium": "CC-BY-4.0", "fa_IR-ganji_adabi-medium": "CC0", "fa_IR-ganji-medium": "CC0",
    "fi_FI-harri-medium": "CC0", "fr_FR-tom-medium": "AGPL-3.0", "fr_FR-siwis-medium": "CC-BY-4.0",
    "hi_IN-rohan-medium": IITM, "hi_IN-priyamvada-medium": "CC-BY-NC-SA-4.0", "hu_HU-imre-medium": "CC0",
    "hu_HU-anna-medium": "CC0", "hy_AM-gor-medium": "GPL-2.0", "id_ID-news_tts-medium": SEE_CARD,
    "is_IS-salka-medium": SEE_CARD, "is_IS-ugla-medium": SEE_CARD, "it_IT-serena-medium": "CC-BY-4.0",
    "it_IT-paola-medium": SEE_CARD, "ka_GE-natia-medium": SEE_CARD, "ko_KR-kss-medium": "CC-BY-NC-SA-4.0",
    "lb_LU-marylux-medium": "CC-BY-NC-SA-4.0", "lv_LV-aivars-medium": "CC0", "ml_IN-arjun-medium": SEE_CARD,
    "ml_IN-meera-medium": SEE_CARD, "ne_NP-chitwan-medium": "CC0", "nl_BE-nathalie-medium": "CC0",
    "nl_NL-pim-medium": "CC0", "no_NO-talesyntese-medium": "CC0", "pl_PL-mc_speech-medium": "CC0",
    "pl_PL-gosia-medium": "CC0", "pt_BR-faber-medium": "CC0", "pt_BR-cadu-medium": "CC0", "ro_RO-mihai-medium": "CC0",
    "ru_RU-irina-medium": SEE_CARD, "ru_RU-ruslan-medium": "CC-BY-NC-SA-4.0", "sk_SK-lili-medium": "CC0",
    "sl_SI-artur-medium": "CC-BY-4.0", "sq_AL-edon-medium": "CC0", "sr_RS-serbski_institut-medium": "CC-BY-NC-SA-4.0",
    "sv_SE-nst-medium": "CC0", "sv_SE-lisa-medium": SEE_CARD, "sw_CD-lanfrica-medium": SEE_CARD,
    "te_IN-padmavathi-medium": "CC-BY-4.0", "te_IN-maya-medium": IITM, "tr_TR-dfki-medium": "CC-BY-NC-SA-4.0",
    "ur_PK-aegis_female-medium": "MIT", "ur_PK-fasih-medium": "MIT", "vi_VN-vais1000-medium": "CC-BY-4.0",
    "zh_CN-huayan-medium": SEE_CARD, "de_DE-thorsten-high": "CC0", "kk_KZ-issai-high": "CC-BY-4.0",
    "uk_UA-oleksa-high": "Apache-2.0", "uk_UA-mykyta-high": "Apache-2.0",
}

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
    "bho": ("bho", "IN", "भोजपुरी", "Bhojpuri", "India"),
    "hne": ("hne", "IN", "छत्तीसगढ़ी", "Chhattisgarhi", "India"),
    "mag": ("mag", "IN", "मगही", "Magahi", "India"),
    "mai": ("mai", "IN", "मैथिली", "Maithili", "India"),
    "san": ("sa", "IN", "संस्कृतम्", "Sanskrit", "India"),
    "brx": ("brx", "IN", "बड़ो", "Bodo", "India"),
    "doi": ("doi", "IN", "डोगरी", "Dogri", "India"),
}


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
    COMPACT_CONFIGS.mkdir(parents=True, exist_ok=True)
    catalog = {}
    for group in json.loads((BASE / "voices.json").read_text(encoding="utf-8")):
        lc = group["langCode"]
        for v in group["voices"]:
            rasa = v.get("engine") == "rasa"
            slug = v["displayName"].lower()
            family, region, native, english, country = LANGS[lc]
            code = f"{family}_{region}"
            key = f"{code}-{slug}-medium"

            if rasa:
                tokens = read_tokens(BASE / "vits-rasa-13" / "tokens.txt")
                blank, model = 0, RASA_MODEL
                model_path = f"{RASA_TAG}/{RASA_MODEL.name}"
            else:
                tokens = read_tokens(BASE / v["id"] / "tokens.txt")
                model = SYSPIN_DIR / f"{v['id']}-model.onnx"
                meta = {p.key: p.value for p in onnx.load(str(model), load_external_data=False).metadata_props}
                blank = int(meta["blank_id"])
                model_path = f"{SYSPIN_TAG}/{model.name}"
            if (INT8_DIR / model.name).is_file():  # the same voice, INT8 weights
                model = INT8_DIR / model.name
                model_path = f"{INT8_TAG}/{model.name}"
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
                              "noise_w": rec["noise_scale_w"] if rasa else SYSPIN_NOISE_W},
                "num_speakers": 1,
                "default_speaker_id": v.get("sid", 0) if rasa else 0,
                "hop_length": 256,
                "phoneme_id_map": id_map,
            }
            cfg_file = OUT / f"{key}.onnx.json"
            cfg_file.write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

            entry = {
                "key": key, "name": slug,
                "language": {"code": code, "family": family, "region": region,
                             "name_native": native, "name_english": english, "country_english": country},
                "quality": "medium", "num_speakers": 1,
                # Enhanced-class compute at Standard size: the app labels it by device.
                "heavy": True,
                "base_url": RELEASES,
                "source": "AI4Bharat Rasa" if rasa else "SYSPIN (IISc SPIRE Lab)",
                "license": "CC-BY-4.0" if rasa else "MIT",
                "files": {model_path: file_entry(model),
                          f"{CONFIG_TAG}/{cfg_file.name}": file_entry(cfg_file)},
            }
            catalog[key] = entry

            # Its Compact version: same config but the quality, INT8-decoder model.
            compact_model = INT8_DIR / ("vits-rasa-13-compact.onnx" if rasa else f"{v['id']}-compact.onnx")
            if compact_model.is_file():
                ckey = key.replace("-medium", "-compact")
                ccfg = OUT / f"{ckey}.onnx.json"
                config["audio"]["quality"] = "compact"
                ccfg.write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
                catalog[ckey] = dict(entry, key=ckey, quality="compact", files={
                    f"{INT8_TAG}/{compact_model.name}": file_entry(compact_model),
                    f"{CONFIG_TAG}/{ccfg.name}": file_entry(ccfg)})
    add_piper_compact(catalog)
    add_piper_int8(catalog)
    write_npu_list(catalog)
    (OUT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"{len(catalog)} configs -> {OUT}")


def add_piper_compact(catalog):
    """Compact versions of Piper "high" voices: rhasspy's config with the quality
    changed, and the INT8-decoder model, both in COMPACT_TAG."""
    import urllib.request
    rhasspy = json.loads(urllib.request.urlopen(
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json", timeout=60).read())
    for high, licence in PIPER_COMPACT.items():
        model = COMPACT_DIR / f"{high[:-len('-high')]}-compact.onnx"
        tag = INT8_TAG if (INT8_DIR / model.name).is_file() else COMPACT_TAG
        model = INT8_DIR / model.name if tag == INT8_TAG else model
        if not model.is_file():
            print(f"  skip {high}: no {model.name}")
            continue
        src = rhasspy[high]
        key = high[:-len("-high")] + "-compact"
        config = json.loads((PIPER_SRC / f"{high}.onnx.json").read_text(encoding="utf-8"))
        config["audio"]["quality"] = "compact"
        cfg = COMPACT_CONFIGS / f"{key}.onnx.json"
        cfg.write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        catalog[key] = {
            "key": key, "name": src["name"], "language": src["language"],
            "quality": "compact", "num_speakers": src["num_speakers"],
            "base_url": RELEASES,
            "source": "rhasspy/piper-voices (Compact: animeshahilya)",
            "license": licence,
            "files": {f"{tag}/{model.name}": file_entry(model),
                      f"{COMPACT_TAG}/{cfg.name}": file_entry(cfg)},
        }


def add_piper_int8(catalog):
    """The kept Piper and community voices themselves, INT8 weights
    (build_small.py) under their own keys: same voice, a quarter of the
    space, same speed, so the app offers only these. Model and the voice's
    own config (unchanged) in INT8_TAG."""
    import urllib.request
    import build_small
    rhasspy = json.loads(urllib.request.urlopen(
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json", timeout=60).read())
    extras = json.loads((APP_ASSETS / "extra_voices.json").read_text(encoding="utf-8"))
    for key in build_small.piper_sources(APP_ASSETS):
        model, cfg = INT8_DIR / f"{key}.onnx", INT8_DIR / f"{key}.onnx.json"
        if not model.is_file():
            print(f"  skip {key}: no {model.name}")
            continue
        src = rhasspy.get(key) or extras[key]
        community = key not in rhasspy
        catalog[key] = {
            "key": key, "name": src["name"], "language": src["language"],
            "quality": src["quality"], "num_speakers": src["num_speakers"],
            "base_url": RELEASES,
            "source": (src["source"].split(" (INT8")[0] if community else "rhasspy/piper-voices")
                      + " (INT8: animeshahilya)",
            "license": src["license"] if community else LICENCES[key],
            "files": {f"{INT8_TAG}/{model.name}": file_entry(model),
                      f"{INT8_TAG}/{cfg.name}": file_entry(cfg)},
        }


NPU_TAG = "npu-v1"
NPU_DIR = BASE / "release_assets_npu"


def write_npu_list(catalog):
    """npu_decoders.json for the espeak-ng app's Snapdragon build: voice key ->
    its INT8 NPU decoder (build_npu.py). Standard keys only: the decoder reads
    the Standard encoder's tensors (a Compact encoder hands over different ones)."""
    voices = {}
    for key, entry in catalog.items():
        if entry["quality"] != "medium":
            continue
        model = next(p for p in entry["files"] if p.endswith(".onnx"))
        name = model.split("/")[-1]
        npu = NPU_DIR / ("vits-rasa-13-npu.onnx" if name.startswith("vits-rasa-13")
                         else name.replace("-model.onnx", "-npu.onnx"))
        if npu.is_file():
            voices[key] = dict(file_entry(npu), path=f"{NPU_TAG}/{npu.name}")
    for high in PIPER_COMPACT:
        npu = NPU_DIR / f"{high[:-len('-high')]}-npu.onnx"
        if npu.is_file():
            voices[high] = dict(file_entry(npu), path=f"{NPU_TAG}/{npu.name}")
    out = {"base_url": RELEASES, "voices": dict(sorted(voices.items()))}
    (OUT / "npu_decoders.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(voices)} voices with an NPU decoder")


if __name__ == "__main__":
    main()
