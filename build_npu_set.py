#!/usr/bin/env python3
"""Hosted NPU files for every voice the espeak-ng app downloads from our releases.

For each distinct model in the app's catalog (assets/piper/extra_voices.json,
paths under int8-v1/), build_npu_enc.py makes the text encoder (256 ids), the
CPU part between, the flow (512 frames; SYSPIN 2048, its letters run ~6.8
frames per id) and, for voices without an INT8 NPU decoder (npu-v1,
npu_decoders.json), a fixed 80-frame decoder. Writes OUT/<stem>-*.onnx and
OUT/npu_files.json (voice key -> files with size and MD5, base_url, tag), the
list the app bundles. Resumable: a model whose files exist is skipped.

  python build_npu_set.py APP_ASSETS_PIPER_DIR OUT [--tag npu-v2]
"""
import argparse
import hashlib
import json
from pathlib import Path

import build_npu_enc

REPO = Path(__file__).resolve().parent
SOURCE = REPO / "release_assets_int8"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("assets", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--tag", default="npu-v2")
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    catalog = json.load(open(args.assets / "extra_voices.json", encoding="utf-8"))
    decoders = json.load(open(args.assets / "npu_decoders.json", encoding="utf-8"))["voices"]
    base = "https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/"
    voices, built = {}, {}
    for key, v in sorted(catalog.items()):
        if "sherpa-onnx-respin-syspin" not in v.get("base_url", ""):
            continue  # not ours to re-host
        models = [f for f in v.get("files", {}) if f.endswith(".onnx") and f.startswith("int8-v1/")]
        if len(models) != 1:
            continue
        stem = Path(models[0]).stem
        src = SOURCE / (stem + ".onnx")
        if not src.is_file():
            print("missing", src)
            continue
        frames = 2048 if stem.startswith("vits-syspin") else 512
        need_dec = key not in decoders
        tagkey = (stem, need_dec)
        if tagkey not in built:
            prefix = args.out / stem
            files = {"enc": Path(f"{prefix}-enc256.onnx"), "mid": Path(f"{prefix}-mid.onnx"),
                     "flow": Path(f"{prefix}-flow{frames}.onnx")}
            if need_dec:
                files["dec"] = Path(f"{prefix}-dec80.onnx")
            only_dec = Path(f"{prefix}-deconly")
            if only_dec.exists():
                files = {"dec": files["dec"]} if need_dec else {}
            elif not all(p.is_file() for p in files.values()):
                print("==", stem, "(decoder)" if need_dec else "", flush=True)
                try:
                    made = build_npu_enc.build(src, str(prefix), 256, frames, decoder=need_dec)
                    if "enc" not in made:  # an older export: its decoder window only
                        files = {"dec": made["dec"]}
                        only_dec.touch()
                except Exception as e:  # the voice keeps its CPU encoder (and decoder) on Snapdragon
                    print("   SKIPPED", stem, type(e).__name__, e, flush=True)
                    for p in files.values():
                        p.unlink(missing_ok=True)
                    built[tagkey] = None
                    continue
            if not files:
                built[tagkey] = None
                continue
            built[tagkey] = {k: {"path": f"{args.tag}/{p.name}", "size_bytes": p.stat().st_size,
                                 "md5_digest": md5(p)} for k, p in files.items()}
            built[tagkey]["ids"] = 256
            built[tagkey]["frames"] = frames
        if built[tagkey] is not None:
            voices[key] = built[tagkey]
    json.dump({"base_url": base, "voices": voices}, open(args.out / "npu_files.json", "w", encoding="utf-8"),
              indent=1, sort_keys=True)
    total = sum(p.stat().st_size for p in args.out.glob("*.onnx"))
    print(len(voices), "voices,", len(built), "file sets,", round(total / 1e6), "MB")


if __name__ == "__main__":
    main()
