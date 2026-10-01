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
    import onnx
    from onnx import helper, numpy_helper
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

    g.input.append(helper.make_tensor_value_info("scales", onnx.TensorProto.FLOAT, [3]))
    for k, scalar in enumerate(["noise_scale", "length_scale", "noise_scale_w"]):
        idx = numpy_helper.from_array(np.array(k, dtype=np.int64), name=f"scales_idx_{k}")
        g.initializer.append(idx)
        pack = helper.make_node("Gather", ["scales", f"scales_idx_{k}"], [scalar + "__packed"], name=f"PackScales_{scalar}")
        g.node.insert(0, pack)
        for n in g.node:
            if n.name == f"PackScales_{scalar}":
                continue
            for j, s in enumerate(n.input):
                if s == scalar:
                    n.input[j] = scalar + "__packed"
    keep = [i for i in g.input if i.name not in ("noise_scale", "length_scale", "noise_scale_w")]
    del g.input[:]
    g.input.extend(keep)

    ren = {"x": "input", "x_length": "input_lengths"}
    for i in g.input:
        if i.name in ren:
            i.name = ren[i.name]
    for n in g.node:
        for j, s in enumerate(n.input):
            if s in ren:
                n.input[j] = ren[s]

    if "emotion_frozen" not in {p.key for p in m.metadata_props}:
        p = m.metadata_props.add()
        p.key, p.value = "emotion_frozen", str(EMOTION_NEUTRAL)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / "model.onnx"
    onnx.save(m, str(dst))
    shutil.copy2(str(SRC_DIR / "tokens.txt"), str(OUT_DIR / "tokens.txt"))
    onnx.checker.check_model(str(dst))
    assert {i.name for i in onnx.load(str(dst)).graph.input} == {"input", "input_lengths", "scales", "sid"}
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
    import numpy as np
    import onnxruntime as ort
    fp16 = ASSETS / "vits-rasa-13-model.onnx"
    model = str(fp16 if fp16.exists() else OUT_DIR / "model.onnx")
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    vocab = {}
    for line in (OUT_DIR / "tokens.txt").read_text(encoding="utf-8").splitlines():
        if " " in line:
            s, i = line.rsplit(" ", 1)
            try:
                vocab[s] = int(i)
            except ValueError:
                pass
    texts = {18: "\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd", 11: "\u0d28\u0d2e\u0d38\u0d4d\u0d15\u0d3e\u0d30\u0d02",
             15: "\u0a38\u0a24\u0a3f \u0a38\u0a4d\u0a30\u0a40 \u0a05\u0a15\u0a3e\u0a32", 16: "\u0a38\u0a24\u0a3f \u0a38\u0a4d\u0a30\u0a40 \u0a05\u0a15\u0a3e\u0a32",
             0: "\u09a8\u09ae\u09b8\u09cd\u0995\u09be\u09f0", 1: "\u09a8\u09ae\u09b8\u09cd\u0995\u09be\u09f0",
             14: "\u0928\u092e\u0938\u094d\u0915\u093e\u0930", 17: "\u0928\u092e\u0938\u094d\u0915\u093e\u0930",
             4: "\u0928\u092e\u0938\u094d\u0915\u093e\u0930", 5: "\u0928\u092e\u0938\u094d\u0915\u093e\u0930",
             6: "\u0928\u092e\u0938\u094d\u0915\u093e\u0930", 7: "\u0928\u092e\u0938\u094d\u0915\u093e\u0930"}
    bad = 0
    for sid, text in texts.items():
        ids = [vocab[c] for c in text if c in vocab]
        feed = {"input": np.array([ids], dtype=np.int64),
                "input_lengths": np.array([len(ids)], dtype=np.int64),
                "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
                "sid": np.array([sid], dtype=np.int64)}
        wav = sess.run(None, feed)[0]
        ok = bool(np.isfinite(wav).all()) and wav.size > 1000 and float(abs(wav).max()) > 1e-4
        bad += not ok
        print(f"  sid={sid:2d} {PRIORITY_SIDS[sid]:10s} -> {'OK' if ok else 'FAIL'}")
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
