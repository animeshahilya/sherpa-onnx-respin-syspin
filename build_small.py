#!/usr/bin/env python3
"""Compact v2: every natural voice's weights in INT8, for low and mid-range phones.

Each Conv / ConvTranspose weight and the symbol embedding is stored as INT8,
one scale per output channel, with the clip that minimises squared error,
and widened back to float by a DequantizeLinear in front of its consumer.
Compute stays float, so only weight rounding changes the sound: no
activation clipping and no noise floor (the pause hiss full INT8 had).
A voice takes a quarter of the space (Piper medium 63 -> 17 MB, Piper high
114 -> 29 MB) and less memory, at the same speed. Applied to a Compact
file (INT8-decoder QDQ, build_compact.py) it shrinks the float rest of the
model and keeps Compact's speed (SYSPIN 44 -> 29 MB).

Measured on the PC with the noise scales at 0 (voice-eval/size/compare.py),
log-mel distance to the original: Priyamvada 1.03 dB, Amy 1.0, Thorsten high
0.91, Rasa 1.09, Rahul 0.94; the shipped Compact is 1.43 (user: "sounds ok").
INT4 (block-wise, opset 21) was 4.3 dB for barely any size gain: rejected.

  python build_small.py IN.onnx OUT.onnx      one model
  python build_small.py --all ESPEAK_NG/android/assets/piper
      writes release_assets_compact_v2/: <lang>-<name>-compact.onnx for each
      kept rhasspy or community voice (original from Hugging Face, cached in
      piper_src/), and the existing SYSPIN, Rasa and Piper-high Compact files
      shrunk.
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

BASE = Path(__file__).parent
OUT = BASE / "release_assets_int8"
PIPER_SRC = BASE / "piper_src"
RHASSPY = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
MIN_SIZE = 1024  # scalars and shape constants stay as they are


def _weights(model):
    """initializer -> axis of its output channels, for Conv / ConvTranspose
    weights and embeddings, looking through an fp16 weight's widening Cast."""
    inits = {t.name for t in model.graph.initializer}
    via = {n.output[0]: n.input[0] for n in model.graph.node
           if n.op_type == "Cast" and n.input[0] in inits}
    # fp16 weights an fp16 decoder computes with directly stay as they are
    half = {t.name for t in model.graph.initializer if t.data_type == TensorProto.FLOAT16}
    out = {}
    for n in model.graph.node:
        if n.op_type in ("Conv", "ConvTranspose") and len(n.input) > 1:
            name, axis = n.input[1], 0
            if n.op_type == "ConvTranspose" and next((a.i for a in n.attribute if a.name == "group"), 1) == 1:
                axis = 1  # [in, out, k]
        elif n.op_type == "Gather":
            name, axis = n.input[0], 0  # one scale per symbol
        else:
            continue
        if name in half:
            continue
        name = via.get(name, name)
        if name in inits:
            out.setdefault(name, axis)
    return out


def _best_clip(rows, amax):
    """Per row, the clip (100% down to 50% of the peak) with the least squared error."""
    best, err = amax.copy(), np.full(amax.shape, np.inf)
    for r in np.linspace(1.0, 0.5, 26):
        c = amax * r
        s = np.where(c > 0, c / 127.0, 1.0)[:, None]
        e = ((np.clip(np.rint(rows / s), -127, 127) * s - rows) ** 2).sum(axis=1)
        better = e < err
        best[better], err[better] = c[better], e[better]
    return best


DECODER_PREFIXES = ("/dec/", "/waveform_decoder/", "/decoder/")
HALF_OPS = {"Conv", "ConvTranspose", "LeakyRelu", "Add", "Div", "Tanh"}


def _fold_upcasts(model):
    """fp16-stored weights (SYSPIN, Rasa) back to float32 initializers."""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    keep = []
    for n in g.node:
        if (n.op_type == "Cast" and n.input[0] in inits
                and inits[n.input[0]].data_type == TensorProto.FLOAT16):
            a = numpy_helper.to_array(inits.pop(n.input[0])).astype(np.float32)
            inits[n.output[0]] = numpy_helper.from_array(a, n.output[0])
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    del g.initializer[:]
    g.initializer.extend(inits.values())


def _half_decoder(model):
    """The HiFi-GAN decoder computes in fp16, its weights stored fp16: ARM
    phones have native fp16 arithmetic (Pixel 8, SYSPIN decoder 2.0x -> 2.8x
    real time; 62-78 dB SNR against float32, i.e. the same sound). conv_pre
    stays float, so the decoder still reads the float latent: the app's
    PiperSplit finds the same cut and the Snapdragon NPU decoders still fit.
    One Cast where float meets fp16, each named into the decoder."""
    g = model.graph
    prefix = lambda name: next(p for p in DECODER_PREFIXES if name.startswith(p))
    half = [n for n in g.node if n.name.startswith(DECODER_PREFIXES) and n.op_type in HALF_OPS
            and "conv_pre" not in n.name]
    inside = {id(n) for n in half}
    made = {o for n in half for o in n.output}
    inits = {t.name: t for t in g.initializer}
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)
    casts, entry, halved = [], {}, set()
    for n in half:
        for k, i in enumerate(n.input):
            if not i or i in made:
                continue
            if i in inits and all(id(u) in inside for u in users[i]):
                halved.add(i)
                continue
            if i not in entry:
                entry[i] = i + "_fp16"
                casts.append(helper.make_node("Cast", [i], [entry[i]], to=TensorProto.FLOAT16,
                                              name=prefix(n.name) + "fp16_in/" + i))
            n.input[k] = entry[i]
    for t in g.initializer:
        if t.name in halved and t.data_type == TensorProto.FLOAT:
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float16), t.name))
    outputs = {o.name for o in g.output}
    for n in half:
        for k, o in enumerate(n.output):
            outside = [u for u in users.get(o, []) if id(u) not in inside]
            if o in outputs or outside:
                for u in users.get(o, []):
                    if id(u) in inside:
                        u.input[:] = [o + "_fp16" if x == o else x for x in u.input]
                n.output[k] = o + "_fp16"
                casts.append(helper.make_node("Cast", [o + "_fp16"], [o], to=TensorProto.FLOAT,
                                              name=prefix(n.name) + "fp32_out/" + o))
    del g.value_info[:]
    # Each Cast right after what it reads (graph order must stay topological).
    after = {}
    for c in casts:
        after.setdefault(c.input[0], []).append(c)
    nodes = [c for i in list(after) if i not in made and not any(i in m.output for m in g.node)
             for c in after.pop(i)]
    for n in g.node:
        nodes.append(n)
        for o in n.output:
            nodes.extend(after.pop(o, []))
    del g.node[:]
    g.node.extend(nodes)


def quantize(src, dst, half_decoder=False):
    model = onnx.load(str(src))
    if half_decoder:
        _fold_upcasts(model)
        _half_decoder(model)
    targets = _weights(model)
    inits, widen = [], {}
    for t in model.graph.initializer:
        a = numpy_helper.to_array(t)
        axis = targets.get(t.name)
        if axis is None or a.dtype not in (np.float32, np.float16) or a.size < MIN_SIZE:
            inits.append(t)
            continue
        a = a.astype(np.float32)
        rows = np.moveaxis(a, axis, 0).reshape(a.shape[axis], -1)
        clip = _best_clip(rows, np.abs(rows).max(axis=1))
        scale = np.where(clip > 0, clip / 127.0, 1.0).astype(np.float32)
        shape = [1] * a.ndim
        shape[axis] = -1
        q = np.clip(np.rint(a / scale.reshape(shape)), -127, 127).astype(np.int8)
        inits += [numpy_helper.from_array(q, t.name + "_q"),
                  numpy_helper.from_array(scale, t.name + "_s"),
                  numpy_helper.from_array(np.zeros(scale.shape, np.int8), t.name + "_z")]
        widen[t.name] = helper.make_node("DequantizeLinear", [t.name + "_q", t.name + "_s", t.name + "_z"],
                                         [t.name], axis=axis, name=t.name + "_dq")
    # An fp16 weight's Cast would now be float -> float: the DequantizeLinear
    # writes its output instead, so each weight's one node reads only
    # initializers (the app's PiperSplit keeps such a node with the half
    # that uses the weight; through a Cast it passed every decoder weight
    # from the encoder).
    nodes = []
    for n in model.graph.node:
        if n.op_type == "Cast" and n.input[0] in widen:
            dq = widen.pop(n.input[0])
            dq.output[0] = n.output[0]
            widen[n.output[0]] = dq
            continue
        nodes.append(n)
    del model.graph.initializer[:]
    model.graph.initializer.extend(inits)
    del model.graph.node[:]
    model.graph.node.extend(list(widen.values()) + nodes)
    onnx.save(model, str(dst))


def _fetch(url, local):
    if not local.is_file():
        tmp = local.with_suffix(local.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while block := r.read(1 << 20):
                f.write(block)
        tmp.replace(local)
    return local


def compact_name(key):
    return key.rsplit("-", 1)[0] + "-compact"


def piper_sources(app_assets):
    """The kept voices that are not SYSPIN/Rasa: key -> (model URL, config
    URL), from rhasspy or the app's community extras on Hugging Face (None
    once cached in piper_src/: the app's list then points at our release)."""
    kept = {k for keys in json.loads((app_assets / "kept_voices.json").read_text(encoding="utf-8")).values()
            for k in keys}
    extras = json.loads((app_assets / "extra_voices.json").read_text(encoding="utf-8"))
    rhasspy = json.loads(urllib.request.urlopen(RHASSPY + "voices.json", timeout=60).read())
    out = {}
    for key in sorted(kept):
        if extras.get(key, {}).get("heavy"):
            continue  # SYSPIN/Rasa: shrunk from their own releases below
        if key in rhasspy:
            base, files = RHASSPY, rhasspy[key]["files"]
        elif key in extras and extras[key]["base_url"].startswith("https://huggingface.co/"):
            base, files = extras[key]["base_url"], extras[key]["files"]
        else:
            out[key] = None
            continue
        model = next(p for p in files if p.endswith(".onnx"))
        config = next(p for p in files if p.endswith(".onnx.json"))
        out[key] = (base + model, base + config)
    return out


def has_fast_compact(key):
    """Piper "high" voices with an INT8-decoder Compact (build_compact.py)."""
    return (BASE / "release_assets_compact" / f"{compact_name(key)}.onnx").is_file() and key.endswith("-high")


def build_all(app_assets):
    """release_assets_int8/ (release int8-v1): every kept voice's model with
    INT8 weights under its own file name - Piper and community voices as
    <key>.onnx (+ their config, unchanged), SYSPIN/Rasa Standard as their
    release names - and the INT8-decoder Compact files shrunk the same way."""
    import shutil
    OUT.mkdir(exist_ok=True)
    PIPER_SRC.mkdir(exist_ok=True)
    jobs = []
    for key, urls in piper_sources(Path(app_assets)).items():
        src, cfg = PIPER_SRC / f"{key}.onnx", PIPER_SRC / f"{key}.onnx.json"
        if urls:
            _fetch(urls[0], src)
            _fetch(urls[1], cfg)
        shutil.copyfile(cfg, OUT / cfg.name)
        # A heavy voice's Standard also gets the fp16 decoder; its Compact
        # (INT8 decoder) is the faster choice.
        jobs.append((src, OUT / src.name, has_fast_compact(key)))
        if has_fast_compact(key):
            c = BASE / "release_assets_compact" / f"{compact_name(key)}.onnx"
            jobs.append((c, OUT / c.name, False))
    for d in ("release_assets_syspin_v2", "release_assets_rasa_v2"):
        for src in sorted((BASE / d).glob("*.onnx")):
            jobs.append((src, OUT / src.name, not src.name.endswith("-compact.onnx")))
    for i, (src, dst, half) in enumerate(jobs, 1):
        if not dst.is_file():
            quantize(src, dst, half_decoder=half)
        print(f"[{i}/{len(jobs)}] {dst.name}: {src.stat().st_size / 1e6:.1f} -> {dst.stat().st_size / 1e6:.1f} MB",
              flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--all":
        build_all(sys.argv[2])
    else:
        quantize(sys.argv[1], sys.argv[2])
