#!/usr/bin/env python3
"""NPU encoder files: a voice's text encoder and flow as fixed-shape graphs
for Qualcomm's HTP, plus the part between them for the CPU.

The espeak-ng app's Snapdragon build ran the whole encoder on the CPU while
the decoder ran on the NPU; on a Galaxy S25 Ultra that encoder was 60-80% of
the time to first audio (Priyamvada: 27-41 ms of a 49 ms phrase). Its text
encoder and flow run on the NPU (text encoder 12 -> 1.3 ms, flow 19.6 ->
0.9 ms, SNR 57 / 64 dB against the CPU); the duration predictor between them
cannot (data-dependent NonZero in its spline flows), so it stays a CPU graph
together with the length regulator and the noise.

Each model (Piper, Coqui SYSPIN, transformers Rasa) is cut by data flow:
  enc   input, input_lengths            -> the text encoder's outputs
        fixed to ENC_IDS ids (padded; input_lengths masks the rest)
  mid   enc outputs, scales (+ sid)     -> the flow's inputs + durations (Ceil)
        (+ input, input_lengths unpadded where it reads them: Rasa's masks)
        dynamic, CPU
  flow  the flow's inputs               -> the decoder's inputs
        fixed to FLOW_FRAMES frames (padded with the frame mask)
Weights stay as the hosted INT8 model has them (DequantizeLinear in front),
so the files are small; the app loads them with session.disable_quant_qdq.
The chain is checked against the whole model with the noise off (padding
included): the latent and the durations must match.

The flow hands out the decoder's inputs by their float names (the Compact
models' quantize pair stays with the decoder, as the app's split keeps it).
With --decoder, also the decoder with its latent fixed to 80 frames, for the
voices with no INT8 NPU decoder (build_npu.py): the app no longer cuts
decoders on the phone.

  python build_npu_enc.py MODEL.onnx OUT_PREFIX [--ids 256] [--frames 512] [--decoder]
writes OUT_PREFIX-enc<ids>.onnx, -mid.onnx, -flow<frames>.onnx (and -dec80.onnx)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.utils
import onnxruntime as ort

sys.setrecursionlimit(100000)

ENC = ("/enc_p/", "/text_encoder/")
FLOW = ("/flow/",)
DEC = ("/dec/", "/waveform_decoder/", "/decoder/")


class Graph:
    def __init__(self, m):
        self.g = m.graph
        self.inits = {i.name for i in self.g.initializer}
        self.prod = {o: n for n in self.g.node for o in n.output}
        self.cons = {}
        for n in self.g.node:
            for i in n.input:
                self.cons.setdefault(i, []).append(n)
        self._const = {}

    def const(self, t):
        """A weight, a Constant, or computed from those alone (INT8 weights' DequantizeLinear)."""
        if t in self._const:
            return self._const[t]
        self._const[t] = False
        p = self.prod.get(t)
        r = t in self.inits or (p is not None and (p.op_type == "Constant" or (
            len(p.input) > 0 and all(not i or self.const(i) for i in p.input))))
        self._const[t] = r
        return r

    def block(self, prefixes):
        s = {id(n) for n in self.g.node if n.name.startswith(prefixes)}
        changed = True
        while changed:
            changed = False
            for n in self.g.node:
                if id(n) in s or n.name.startswith("/"):
                    continue
                ins = [i for i in n.input if i and not self.const(i)]
                if ins and all(i in self.prod and id(self.prod[i]) in s for i in ins):
                    s.add(id(n))
                    changed = True
        return s

    def edges(self, prefixes):
        s = self.block(prefixes)
        outs_graph = {o.name for o in self.g.output}
        ins, outs = [], []
        for n in self.g.node:
            if id(n) not in s:
                continue
            for i in n.input:
                if i and not self.const(i) and (i not in self.prod or id(self.prod[i]) not in s) and i not in ins:
                    ins.append(i)
            for o in n.output:
                if not self.const(o) and o not in outs and (
                        o in outs_graph or any(id(c) not in s for c in self.cons.get(o, []))):
                    outs.append(o)
        return ins, outs

    def ceil(self):
        c = [n.output[0] for n in self.g.node if n.op_type == "Ceil"]
        assert len(c) == 1, c
        return c[0]


def fix_inputs(path, shapes):
    m = onnx.load(str(path))
    for i in m.graph.input:
        if i.name in shapes:
            dims = i.type.tensor_type.shape.dim
            del dims[:]
            for v in shapes[i.name]:
                dims.add().dim_value = v
    for o in m.graph.output:
        o.type.tensor_type.ClearField("shape")
    onnx.save(m, str(path))


def session(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def standard(g, t):
    """The float tensor behind a QuantizeLinear/DequantizeLinear pair (the Compact
    models quantize the latent and Rasa's speaker vector): the app's split hands
    the decoder these float names (PiperSplit keeps the pair with the decoder)."""
    p = g.prod.get(t)
    if p is not None and p.op_type == "DequantizeLinear":
        q = g.prod.get(p.input[0])
        if q is not None and q.op_type == "QuantizeLinear":
            return q.input[0]
    return t


def build(model, prefix, ids_bucket, frames_bucket, decoder=False, window=80):
    m = onnx.load(str(model))
    g = Graph(m)
    _, enc_out = g.edges(ENC)
    flow_in, _ = g.edges(FLOW)
    dec_in, _ = g.edges(DEC)
    graph_inputs = [i.name for i in m.graph.input]
    enc_in = [i for i in ("input", "input_lengths") if i in graph_inputs]
    cond = [i for i in ("scales", "sid") if i in graph_inputs]
    durations = g.ceil()
    if not enc_out or not flow_in:
        # Older exports name only their weights (plain Conv_123 nodes elsewhere):
        # no encoder cut, but the decoder (named /dec/) still gets its window.
        if not decoder or not dec_in:
            raise ValueError("no encoder or flow block found")
        return decoder_only(m, g, prefix, dec_in, window)
    # The decoder's inputs by their float names: the latent (made by the flow)
    # and, for Rasa, the speaker vector (made before it, handed on by mid).
    boundary = [standard(g, t) for t in dec_in]
    latent = [b for t, b in zip(dec_in, boundary) if t not in flow_in]
    assert len(latent) == 1, (dec_in, flow_in)
    latent = latent[0]
    speaker = [b for b in boundary if b != latent]
    mid_out = flow_in + [s for s in speaker if s not in flow_in] + [durations]
    print("  enc out", enc_out, "\n  flow in", flow_in, "\n  boundary", boundary, "\n  durations", durations)

    enc_p = Path(f"{prefix}-enc{ids_bucket}.onnx")
    mid_p = Path(f"{prefix}-mid.onnx")
    flow_p = Path(f"{prefix}-flow{frames_bucket}.onnx")
    dec_p = Path(f"{prefix}-dec{window}.onnx")
    # Extractor, not extract_model: its checker rejects these exports' shapeless
    # value_info. It needs a type for every cut tensor: all of them are float.
    typed = onnx.shape_inference.infer_shapes(m)
    known = {v.name for v in list(typed.graph.value_info) + list(typed.graph.input) + list(typed.graph.output)}
    for t in enc_out + flow_in + dec_in + boundary + [durations]:
        if t not in known:
            typed.graph.value_info.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
            known.add(t)
    ex = onnx.utils.Extractor(typed)
    onnx.save(ex.extract_model(enc_in, enc_out), str(enc_p))
    onnx.save(ex.extract_model(enc_in + enc_out + cond, mid_out), str(mid_p))
    onnx.save(ex.extract_model(flow_in, [latent]), str(flow_p))
    audio_out = m.graph.output[0].name
    if decoder:
        onnx.save(ex.extract_model(boundary, [audio_out]), str(dec_p))

    # Shapes and values of the boundary tensors, from a real run with the noise off.
    probe = onnx.load(str(model))
    for t in dict.fromkeys(enc_out + flow_in + boundary + [durations]):
        probe.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
    probe_p = Path(f"{prefix}-probe.onnx")
    onnx.save(probe, str(probe_p))
    whole = session(probe_p)
    names = [o.name for o in whole.get_outputs()]
    rng = np.random.default_rng(1)
    n = 61
    ids = rng.integers(1, 40, size=(1, n)).astype(np.int64)
    feed = {"input": ids, "input_lengths": np.array([n], np.int64),
            "scales": np.array([0.0, 1.0, 0.0], np.float32)}
    if "sid" in graph_inputs:
        feed["sid"] = np.array([0], np.int64)
    r = dict(zip(names, whole.run(None, feed)))
    probe_p.unlink()

    fix_inputs(enc_p, {"input": [1, ids_bucket], "input_lengths": [1]})
    probe_frames = max(r[t].shape[-1] for t in flow_in)  # Rasa's speaker vector has 1
    fix_inputs(flow_p, {t: list(r[t].shape[:-1]) + [frames_bucket] if r[t].shape[-1] == probe_frames
                        else list(r[t].shape) for t in flow_in})
    if decoder:
        fix_inputs(dec_p, {b: list(r[b].shape[:-1]) + [window] if b == latent else list(r[b].shape)
                           for b in boundary})

    # The chain, padded, against the whole model.
    enc, mid, flow = session(enc_p), session(mid_p), session(flow_p)
    padded = np.zeros((1, ids_bucket), np.int64)
    padded[0, :n] = ids[0]
    e = enc.run(None, {"input": padded, "input_lengths": np.array([n], np.int64)})
    e = [x[..., :n] if x.shape[-1] == ids_bucket else
         (x[:, :n] if x.ndim > 1 and x.shape[1] == ids_bucket else x) for x in e]
    mfeed = dict(zip(enc_out, e))
    mfeed.update({"input": ids, "input_lengths": feed["input_lengths"], "scales": feed["scales"]})
    if "sid" in cond:
        mfeed["sid"] = feed["sid"]
    mfeed = {k: v for k, v in mfeed.items() if k in {i.name for i in mid.get_inputs()}}
    mo = dict(zip(mid_out, mid.run(None, mfeed)))
    frames = max(mo[t].shape[-1] for t in flow_in)
    ffeed = {}
    for t in flow_in:
        x = mo[t]
        if x.shape[-1] == frames:
            pad = np.zeros(x.shape[:-1] + (frames_bucket,), x.dtype)
            pad[..., :frames] = x
            x = pad
        ffeed[t] = x
    lat = flow.run(None, ffeed)[0][..., :frames]
    ref = r[latent]
    err = float(np.abs(lat - ref).max())
    derr = float(np.abs(mo[durations] - r[durations]).max())
    scale = float(np.abs(ref).max())
    serr = max((float(np.abs(mo[s] - r[s]).max()) for s in speaker), default=0.0)
    print(f"  ids {n} frames {frames}: latent max err {err:.2e} (of {scale:.2f}), durations err {derr}, "
          f"speaker err {serr:.1e}")
    assert derr == 0 and serr < 1e-5 and err < 1e-3 * max(1.0, scale), "chain does not match the model"
    if decoder:
        # The fixed window, fed the model's own latent, against the whole model's audio.
        dec = session(dec_p)
        w = min(window, frames)
        dfeed = {b: r[b] for b in speaker}
        win = np.zeros(ref.shape[:-1] + (window,), np.float32)
        win[..., :w] = ref[..., :w]
        dfeed[latent] = win
        audio = dec.run(None, dfeed)[0].reshape(-1)
        full = r[audio_out].reshape(-1)
        hop = audio.size // window
        keep = (w - 12) * hop  # the window's right edge lacks context; the app crops it
        aerr = float(np.abs(audio[:keep] - full[:keep]).max())
        print(f"  decoder window: audio max err {aerr:.2e} over {keep} samples")
        assert aerr < 1e-3, f"decoder window does not match the model ({aerr:.2e})"
    for p in (enc_p, mid_p, flow_p) + ((dec_p,) if decoder else ()):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")
    return {"enc": enc_p, "mid": mid_p, "flow": flow_p, "dec": dec_p if decoder else None,
            "frames": frames_bucket, "ids": ids_bucket}


def decoder_only(m, g, prefix, dec_in, window):
    """The fixed-window decoder alone, checked against the whole model."""
    boundary = [standard(g, t) for t in dec_in]
    dec_p = Path(f"{prefix}-dec{window}.onnx")
    typed = onnx.shape_inference.infer_shapes(m)
    known = {v.name for v in list(typed.graph.value_info) + list(typed.graph.input) + list(typed.graph.output)}
    for t in boundary:
        if t not in known:
            typed.graph.value_info.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    audio_out = m.graph.output[0].name
    onnx.save(onnx.utils.Extractor(typed).extract_model(boundary, [audio_out]), str(dec_p))
    probe = onnx.ModelProto()
    probe.CopyFrom(m)
    for t in boundary:
        probe.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
    probe_p = Path(f"{prefix}-probe.onnx")
    onnx.save(probe, str(probe_p))
    whole = session(probe_p)
    names = [o.name for o in whole.get_outputs()]
    feed = {"input": np.random.default_rng(1).integers(1, 40, size=(1, 61)).astype(np.int64),
            "input_lengths": np.array([61], np.int64), "scales": np.array([0.0, 1.0, 0.0], np.float32)}
    if "sid" in [i.name for i in m.graph.input]:
        feed["sid"] = np.array([0], np.int64)
    r = dict(zip(names, whole.run(None, feed)))
    probe_p.unlink()
    latent = max(boundary, key=lambda b: r[b].shape[-1])
    fix_inputs(dec_p, {b: list(r[b].shape[:-1]) + [window] if b == latent else list(r[b].shape) for b in boundary})
    dec = session(dec_p)
    ref = r[latent]
    w = min(window, ref.shape[-1])
    dfeed = {b: r[b] for b in boundary if b != latent}
    win = np.zeros(ref.shape[:-1] + (window,), np.float32)
    win[..., :w] = ref[..., :w]
    dfeed[latent] = win
    audio = dec.run(None, dfeed)[0].reshape(-1)
    full = r[audio_out].reshape(-1)
    keep = (w - 12) * (audio.size // window)
    aerr = float(np.abs(audio[:keep] - full[:keep]).max())
    print(f"  decoder only (no encoder cut): window audio max err {aerr:.2e}; {dec_p.stat().st_size / 1e6:.1f} MB")
    assert aerr < 1e-3, f"decoder window does not match the model ({aerr:.2e})"
    return {"dec": dec_p}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("model", type=Path)
    a.add_argument("prefix")
    a.add_argument("--ids", type=int, default=256)
    a.add_argument("--frames", type=int, default=512)
    a.add_argument("--decoder", action="store_true", help="also a fixed 80-frame decoder (fp16 on the NPU)")
    args = a.parse_args()
    build(args.model, args.prefix, args.ids, args.frames, args.decoder)


if __name__ == "__main__":
    main()
