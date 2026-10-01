#!/usr/bin/env python3
"""Reproduce the stock-compatible Rasa sherpa-onnx model from public sources.

Pipeline: download MatiasLin export (ungated) -> ONNX surgery (freeze emotion,
pack scales, rename inputs) -> weight-only FP16 -> ORT verify.

  python build_rasa_stock.py --all        # full pipeline
  python build_rasa_stock.py --download   # fetch model.onnx + tokens.txt only
  python build_rasa_stock.py --surgery    # stock-compatible FP32
  python build_rasa_stock.py --fp16       # 59.5MB release asset
  python build_rasa_stock.py --verify     # ORT smoke test (12 priority sids)

Outputs: vits-rasa-13/model.onnx (+tokens.txt, committed), release asset
release_assets_rasa/vits-rasa-13-model.onnx (uploaded to GitHub Release, not git).
"""

import argparse
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).parent
SRC_DIR = BASE / "rasa_src"
OUT_DIR = BASE / "vits-rasa-13"
ASSETS = BASE / "release_assets_rasa"
UPSTREAM = "MatiasLin/sherpa-onnx-vits-rasa-13"
EMOTION_NEUTRAL = 0
PRIORITY_SIDS = {18: "Tamil", 11: "Malayalam", 15: "Punjabi-F", 16: "Punjabi-M",
                 0: "Assamese-F", 1: "Assamese-M", 14: "Nepali", 17: "Sanskrit",
                 4: "Bodo-F", 5: "Bodo-M", 6: "Dogri-F", 7: "Dogri-M"}


def cmd_download():
    from huggingface_hub import hf_hub_download
    import os
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("model.onnx", "tokens.txt"):
        p = hf_hub_download(UPSTREAM, f, local_dir=str(SRC_DIR))
        print(f"{f}: {os.path.getsize(p) / 1024 / 1024:.1f} MB")


def cmd_surgery():
    """Freeze emotion_id and fix metadata; keep sherpa's native multi-speaker
    VITS signature (x, x_length, noise_scale, length_scale, noise_scale_w, sid).

    The comment is not "coqui"/"piper", so sherpa feeds this model through
    RunVits(): five positional inputs + sid. Renaming/packing inputs into the
    coqui `scales` layout makes sherpa segfault (5 tensors vs 4 names).
    """
    import onnx
    from onnx import numpy_helper
    import numpy as np

    src = SRC_DIR / "model.onnx"
    assert src.exists(), "run --download first"
    m = onnx.load(str(src))
    g = m.graph

    emo = numpy_helper.from_array(np.array([EMOTION_NEUTRAL], dtype=np.int64), name="emotion_frozen_idx")
    g.initializer.append(emo)
    for n in g.node:
        for k, s in enumerate(n.input):
            if s == "emotion_id":
                n.input[k] = "emotion_frozen_idx"
    keep = [i for i in g.input if i.name != "emotion_id"]
    del g.input[:]
    g.input.extend(keep)

    # sherpa passes the three scales as shape [1], the export declares them rank-0.
    for i in g.input:
        if i.name in ("noise_scale", "length_scale", "noise_scale_w"):
            i.type.tensor_type.shape.Clear()
            i.type.tensor_type.shape.dim.add().dim_value = 1

    # use_eos_bos=0: sherpa otherwise wraps every sentence in extra 0s and turns
    # a trailing "." into a [0, 0, 0] stub that renders as noise. With it off,
    # the token stream is exactly HF's add_blank output (blank = 0).
    meta = {"emotion_frozen": str(EMOTION_NEUTRAL), "use_eos_bos": "0", "blank_id": "0"}
    props = [p for p in m.metadata_props if p.key not in meta and p.key != "num_emotions"]
    del m.metadata_props[:]
    m.metadata_props.extend(props)
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / "model.onnx"
    onnx.save(m, str(dst))
    shutil.copy2(str(SRC_DIR / "tokens.txt"), str(OUT_DIR / "tokens.txt"))
    onnx.checker.check_model(str(dst))
    names = [i.name for i in onnx.load(str(dst)).graph.input]
    assert names == ["x", "x_length", "noise_scale", "length_scale", "noise_scale_w", "sid"], names
    print(f"stock model: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB), checker OK")


def cmd_fp16():
    from build_fp16_60mb import convert_weight_fp16
    src = OUT_DIR / "model.onnx"
    assert src.exists(), "run --surgery first"
    ASSETS.mkdir(parents=True, exist_ok=True)
    dst = ASSETS / "vits-rasa-13-model.onnx"
    convert_weight_fp16(src, dst)
    shutil.copy2(str(OUT_DIR / "tokens.txt"), str(ASSETS / "vits-rasa-13-tokens.txt"))
    print(f"release asset: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")


def cmd_verify():
    """Smoke test through sherpa-onnx itself, the runtime the app uses.

    Raw-ORT checks missed both a sherpa input-layout segfault and the add_blank
    interspersal, so verify with the real frontend. Each text ends in "." to
    also catch the trailing-stub regression (stub adds ~0.5 s of noise).
    """
    import numpy as np
    import sherpa_onnx
    fp16 = ASSETS / "vits-rasa-13-model.onnx"
    model = str(fp16 if fp16.exists() else OUT_DIR / "model.onnx")
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=model, tokens=str(OUT_DIR / "tokens.txt")),
        num_threads=4)))
    texts = {18: "வணக்கம்", 11: "നമസ്കാരം",
             15: "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", 16: "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ",
             0: "নমস্কাৰ", 1: "নমস্কাৰ",
             14: "नमस्कार", 17: "नमस्कार",
             4: "नमस्कार", 5: "नमस्कार",
             6: "नमस्कार", 7: "नमस्कार"}
    bad = 0
    for sid, text in texts.items():
        wav = np.array(tts.generate(text, sid=sid).samples)
        stub = len(tts.generate(text + ".", sid=sid).samples) - len(wav)
        ok = bool(np.isfinite(wav).all()) and wav.size > 1000 and float(abs(wav).max()) > 1e-4 and stub < 0.2 * tts.sample_rate
        bad += not ok
        print(f"  sid={sid:2d} {PRIORITY_SIDS[sid]:10s} -> {'OK' if ok else 'FAIL'} (trailing '.' adds {stub / tts.sample_rate:.2f}s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--surgery", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    steps = []
    if a.all:
        steps = [cmd_download, cmd_surgery, cmd_fp16, cmd_verify]
    else:
        if a.download:
            steps.append(cmd_download)
        if a.surgery:
            steps.append(cmd_surgery)
        if a.fp16:
            steps.append(cmd_fp16)
        if a.verify:
            steps.append(cmd_verify)
    if not steps:
        ap.print_help()
        sys.exit(2)
    for s in steps:
        s()
