#!/usr/bin/env python3
"""Supertonic 3 for the espeak-ng app's Snapdragon build (release supertonic-v1).

Supertone/supertonic-3 (OpenRAIL-M, pinned below) is four ONNX models: a
duration predictor and text encoder (run on the CPU) and the flow-matching
vector estimator and vocoder (run on the Hexagon NPU through QNN). This
writes release_assets_supertonic/ with:

  - every model with its Conv / MatMul / Gemm weights stored INT8, one scale
    per output channel (build_small.py's per-row best clip), widened back to
    float by a DequantizeLinear: compute stays float, a quarter of the size;
  - the vector estimator made NPU-ready, without fixing its lengths (the app
    pins latent_length / text_length per size bucket when it loads it, and
    ONNX Runtime folds the shape arithmetic before QNN sees the graph):
      * Softplus (QNN has none) -> Log(1 + Exp(x)), the same function;
      * each 4-D batched MatMul (attention, [2,2,L,128] x [2,2,128,50]) ->
        Reshape to 3-D, MatMul, Reshape back, with the shapes taken from the
        tensors: the 4-D form crashes the HTP (DSP restart / error 1003);
  - tts.json, unicode_indexer.json, the ten voice styles and the licence.

Checked here: each model against the original (CPU), and whole sentences
with the same random draw through the reference pipeline (log-mel distance,
as in build_small.py's measurements).

  python build_supertonic.py               build + check
  python build_supertonic.py --check-only
"""

import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference, TensorProto

BASE = Path(__file__).parent
SRC = BASE / "supertonic_src"
OUT = BASE / "release_assets_supertonic"
REPO = "https://huggingface.co/Supertone/supertonic-3/resolve/3cadd1ee6394adea1bd021217a0e650ede09a323/"
MODELS = ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]
STYLES = [f"{g}{i}" for g in "FM" for i in range(1, 6)]
MIN_SIZE = 1024

sys.stdout.reconfigure(encoding="utf-8")
if os.name == "nt":  # long job: idle priority (children inherit)
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)


def fetch():
    SRC.mkdir(exist_ok=True)
    files = [f"onnx/{m}.onnx" for m in MODELS] + ["onnx/tts.json", "onnx/unicode_indexer.json", "LICENSE",
                                                   "README.md"] + [f"voice_styles/{s}.json" for s in STYLES]
    for f in files:
        dst = SRC / f
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(REPO + f, timeout=300) as r, open(str(dst) + ".part", "wb") as o:
                shutil.copyfileobj(r, o)
            os.replace(str(dst) + ".part", dst)
    return files


def softplus(g):
    one = "supertonic_softplus_one"
    g.initializer.append(numpy_helper.from_array(np.array(1.0, np.float32), one))
    out, k = [], 0
    for n in g.node:
        if n.op_type == "Softplus":
            x, y = n.input[0], n.output[0]
            out += [helper.make_node("Exp", [x], [y + "_exp"], name=n.name + "_exp"),
                    helper.make_node("Add", [y + "_exp", one], [y + "_add"], name=n.name + "_add"),
                    helper.make_node("Log", [y + "_add"], [y], name=n.name + "_log")]
            k += 1
        else:
            out.append(n)
    del g.node[:]
    g.node.extend(out)
    return k


def matmul_3d(m):
    """Batched 4-D MatMuls between activations -> Reshape/3-D MatMul/Reshape,
    shapes from Shape() so no length is baked in."""
    inferred = shape_inference.infer_shapes(m)
    rank = {v.name: len(v.type.tensor_type.shape.dim) for v in
            list(inferred.graph.value_info) + list(inferred.graph.input)}
    g = m.graph
    inits = {t.name for t in g.initializer}
    c = lambda name, v: (g.initializer.append(numpy_helper.from_array(np.array(v, np.int64), name)), name)[1]
    neg1, i0, i1, i2, i3, i4 = (c("mm3d_" + n, v) for n, v in
                                (("m1", [-1]), ("0", [0]), ("1", [1]), ("2", [2]), ("3", [3]), ("4", [4])))
    out, k = [], 0
    for n in g.node:
        a, b = (list(n.input) + ["", ""])[:2]
        if n.op_type == "MatMul" and rank.get(a) == 4 and rank.get(b) == 4 and a not in inits and b not in inits:
            p = f"/mm3d_{k}/"
            N = lambda op, ins, outs, **kw: helper.make_node(op, ins, outs, name=p + outs[0].split("/")[-1], **kw)
            out += [N("Shape", [a], [p + "sa"]), N("Shape", [b], [p + "sb"]),
                    N("Slice", [p + "sa", i2, i4], [p + "ta"]), N("Slice", [p + "sb", i2, i4], [p + "tb"]),
                    N("Slice", [p + "sa", i0, i2], [p + "ha"]),
                    N("Concat", [neg1, p + "ta"], [p + "qa"], axis=0), N("Concat", [neg1, p + "tb"], [p + "qb"], axis=0),
                    N("Reshape", [a, p + "qa"], [p + "a3"]), N("Reshape", [b, p + "qb"], [p + "b3"]),
                    N("MatMul", [p + "a3", p + "b3"], [p + "c3"]), N("Shape", [p + "c3"], [p + "sc"]),
                    N("Slice", [p + "sc", i1, i3], [p + "tc"]), N("Concat", [p + "ha", p + "tc"], [p + "qc"], axis=0),
                    helper.make_node("Reshape", [p + "c3", p + "qc"], [n.output[0]], name=p + "out")]
            k += 1
        else:
            out.append(n)
    del g.node[:]
    g.node.extend(out)
    return k


def weight_axes(m):
    """initializer -> output-channel axis, for Conv/ConvTranspose/MatMul/Gemm weights."""
    inits = {t.name for t in m.graph.initializer}
    out = {}
    for n in m.graph.node:
        if n.op_type in ("Conv", "ConvTranspose") and len(n.input) > 1 and n.input[1] in inits:
            g1 = next((a.i for a in n.attribute if a.name == "group"), 1) == 1
            out.setdefault(n.input[1], 1 if n.op_type == "ConvTranspose" and g1 else 0)
        elif n.op_type == "MatMul" and n.input[1] in inits:
            out.setdefault(n.input[1], 1)  # [K, N]: one scale per output column
        elif n.op_type == "Gemm" and len(n.input) > 1 and n.input[1] in inits:
            tb = next((a.i for a in n.attribute if a.name == "transB"), 0)
            out.setdefault(n.input[1], 0 if tb else 1)
    return out


def best_clip(rows, amax):
    best, err = amax.copy(), np.full(amax.shape, np.inf)
    for r in np.linspace(1.0, 0.5, 26):
        c = amax * r
        s = np.where(c > 0, c / 127.0, 1.0)[:, None]
        e = ((np.clip(np.rint(rows / s), -127, 127) * s - rows) ** 2).sum(axis=1)
        better = e < err
        best[better], err[better] = c[better], e[better]
    return best


def int8_weights(m):
    axes = weight_axes(m)
    inits, dq = [], []
    for t in m.graph.initializer:
        a = numpy_helper.to_array(t)
        axis = axes.get(t.name)
        if axis is None or a.dtype != np.float32 or a.size < MIN_SIZE or a.ndim < 2:
            inits.append(t)
            continue
        rows = np.moveaxis(a, axis, 0).reshape(a.shape[axis], -1)
        clip = best_clip(rows, np.abs(rows).max(axis=1))
        scale = np.where(clip > 0, clip / 127.0, 1.0).astype(np.float32)
        shape = [1] * a.ndim
        shape[axis] = -1
        q = np.clip(np.rint(a / scale.reshape(shape)), -127, 127).astype(np.int8)
        inits += [numpy_helper.from_array(q, t.name + "_q"), numpy_helper.from_array(scale, t.name + "_s"),
                  numpy_helper.from_array(np.zeros(scale.shape, np.int8), t.name + "_z")]
        dq.append(helper.make_node("DequantizeLinear", [t.name + "_q", t.name + "_s", t.name + "_z"], [t.name],
                                   axis=axis, name=t.name + "_dq"))
    del m.graph.initializer[:]
    m.graph.initializer.extend(inits)
    nodes = dq + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    return len(dq)


def batch_one(m):
    """batch_size -> 1 (the app synthesizes one text at a time); lengths stay free."""
    for v in list(m.graph.input) + list(m.graph.output):
        for d in v.type.tensor_type.shape.dim:
            if d.dim_param == "batch_size":
                d.ClearField("dim_param")
                d.dim_value = 1


def build():
    fetch()
    OUT.mkdir(exist_ok=True)
    for name in MODELS:
        m = onnx.load(str(SRC / "onnx" / f"{name}.onnx"))
        batch_one(m)
        notes = []
        if name == "vector_estimator":
            notes.append(f"softplus {softplus(m.graph)}")
            notes.append(f"4-D matmuls {matmul_3d(m)}")
        notes.append(f"int8 weights {int8_weights(m)}")
        del m.graph.value_info[:]
        onnx.checker.check_model(m)
        dst = OUT / f"supertonic3-{name}.onnx"
        onnx.save(m, str(dst))
        print(f"{dst.name}: {(SRC / 'onnx' / f'{name}.onnx').stat().st_size / 1e6:.1f} -> "
              f"{dst.stat().st_size / 1e6:.1f} MB ({', '.join(notes)})", flush=True)
    shutil.copyfile(SRC / "onnx" / "tts.json", OUT / "supertonic3-tts.json")
    shutil.copyfile(SRC / "onnx" / "unicode_indexer.json", OUT / "supertonic3-unicode_indexer.json")
    for s in STYLES:
        shutil.copyfile(SRC / "voice_styles" / f"{s}.json", OUT / f"supertonic3-style-{s}.json")
    shutil.copyfile(SRC / "LICENSE", OUT / "supertonic3-LICENSE.txt")


if __name__ == "__main__":
    if "--check-only" not in sys.argv:
        build()
    import check_supertonic
    check_supertonic.main(SRC, OUT)
