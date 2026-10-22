"""Phonon 31m (Gradium, Apache-2.0) as eSpeak NG Advanced natural voices: release phonon-v1 + the app's catalog.

    python build_phonon_catalog.py <phonon-31m-en-qnn bundle dir> <app android dir>

From Gradium's QNN bundle (xn-ptts v0.4.0 release asset phonon-31m-en-qnn.zip):
- release_assets_phonon/: each voice as phonon-31m-en-voice-<slug>.bin, its
  config <key>.onnx.json, and the shared metadata, tokenizer and BOS data
  (the app downloads those once, like Supertonic's companions);
- release_assets_phonon_ctx/<soc>/phonon-31m-en-<chip>.bin: the compiled
  model per chip, which goes into the npu-ctx-v1-<chip> releases and the
  app's npu_ctx.json (build_npu_ctx_catalog.py reads them);
- <app>/assets/piper/phonon_voices.json, the catalog.

Every voice of the bundle is offered (the user's call) except "default", a
copy of one of them. Voices described as British are en_GB, the rest en_US.
"""
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

TAG = "phonon-v1"
RELEASES = "https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/"
PREFIX = "phonon-31m-en"
CHIPS = {"SM8650": "57", "SM8750": "69", "SM8850": "87"}  # bundle soc_models -> npu-ctx soc ids
LANG = {
    "en_US": {"code": "en_US", "family": "en", "region": "US", "name_native": "English",
              "name_english": "English", "country_english": "United States"},
    "en_GB": {"code": "en_GB", "family": "en", "region": "GB", "name_native": "English",
              "name_english": "English", "country_english": "Great Britain"},
}


def info(path):
    data = Path(path).read_bytes()
    return {"size_bytes": len(data), "md5_digest": hashlib.md5(data).hexdigest()}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main():
    bundle, app = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path("release_assets_phonon")
    out.mkdir(exist_ok=True)
    meta = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    shared = {}
    for name, src in (("metadata.json", "metadata.json"), ("tokenizer.json", "tokenizer.json"),
                      ("bos.bin", "bos.bin")):
        dst = out / f"{PREFIX}-{name}"
        shutil.copyfile(bundle / src, dst)
        shared[name] = {"path": f"{TAG}/{dst.name}", **info(dst)}
    shutil.copyfile(Path(__file__).parent / "phonon" / "LICENSE-phonon.txt", out / f"{PREFIX}-LICENSE.txt") \
        if (Path(__file__).parent / "phonon" / "LICENSE-phonon.txt").exists() else None

    catalog = {}
    for v in meta["voices"]:
        if v["name"] == "default":
            continue
        s = slug(v["name"])
        code = "en_GB" if "British" in v.get("description", "") else "en_US"
        key = f"{code}-phonon_{s}-medium"
        voice_file = out / f"{PREFIX}-voice-{s}.bin"
        shutil.copyfile(bundle / v["file"], voice_file)
        cfg = {
            "dataset": "phonon-31m-en",
            "audio": {"sample_rate": int(v.get("sample_rate", 24000)), "quality": "medium"},
            "phoneme_type": "phonon",
            "language": LANG[code],
            "inference": {"noise_scale": 1.0, "length_scale": 1.0, "noise_w": 1.0},
            "num_speakers": 1,
            "phoneme_id_map": {"_": [0]},
            "phonon": {"voice": v["name"], "files": shared},
        }
        cfg_file = out / f"{key}.onnx.json"
        cfg_file.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
        catalog[key] = {
            "key": key, "name": f"phonon_{s}", "language": LANG[code], "quality": "medium",
            "num_speakers": 1, "base_url": RELEASES, "source": "Gradium Phonon 31M",
            "license": "Apache-2.0",
            "files": {f"{TAG}/{voice_file.name}": info(voice_file), f"{TAG}/{cfg_file.name}": info(cfg_file)},
            "extra_files": {n: {"size_bytes": f["size_bytes"], "md5_digest": f["md5_digest"]}
                            for n, f in shared.items()},
        }

    ctx = Path("release_assets_phonon_ctx")
    for entry in meta["runtime"]["context_binaries"].values():
        for chip in entry["soc_models"]:
            if chip in CHIPS:
                d = ctx / CHIPS[chip]
                d.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(bundle / entry["file"], d / f"{PREFIX}-{chip.lower()}.bin")

    dst = app / "assets" / "piper" / "phonon_voices.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=1)
    print(f"{len(catalog)} voices -> {dst}; release files in {out}, contexts in {ctx}")


if __name__ == "__main__":
    main()
