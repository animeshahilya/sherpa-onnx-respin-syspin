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

  python build_npu_enc.py MODEL.onnx OUT_PREFIX [--ids 256] [--frames 512]
writes OUT_PREFIX-enc<ids>.onnx, OUT_PREFIX-mid.onnx, OUT_PREFIX-flow<frames>.onnx
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


def build(model, prefix, ids_bucket, frames_bucket):
    m = onnx.load(str(model))
    g = Graph(m)
    _, enc_out = g.edges(ENC)
    flow_in, _ = g.edges(FLOW)
    dec_in, _ = g.edges(DEC)
    graph_inputs = [i.name for i in m.graph.input]
    enc_in = [i for i in ("input", "input_lengths") if i in graph_inputs]
    cond = [i for i in ("scales", "sid") if i in graph_inputs]
    durations = g.ceil()
    # The decoder's inputs the flow does not make (Rasa's speaker vector) come from mid.
    passthrough = [t for t in dec_in if t in flow_in]
    flow_out = [t for t in dec_in if t not in flow_in]
    print("  enc out", enc_out, "\n  flow in", flow_in, "\n  dec in", dec_in, "\n  durations", durations)

    enc_p = Path(f"{prefix}-enc{ids_bucket}.onnx")
    mid_p = Path(f"{prefix}-mid.onnx")
    flow_p = Path(f"{prefix}-flow{frames_bucket}.onnx")
    # Extractor, not extract_model: its checker rejects these exports' shapeless
    # value_info. It needs a type for every cut tensor: all of them are float.
    typed = onnx.shape_inference.infer_shapes(m)
    known = {v.name for v in list(typed.graph.value_info) + list(typed.graph.input) + list(typed.graph.output)}
    for t in enc_out + flow_in + dec_in + [durations]:
        if t not in known:
            typed.graph.value_info.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    ex = onnx.utils.Extractor(typed)
    onnx.save(ex.extract_model(enc_in, enc_out), str(enc_p))
    onnx.save(ex.extract_model(enc_in + enc_out + cond, flow_in + [durations]), str(mid_p))
    onnx.save(ex.extract_model(flow_in, flow_out), str(flow_p))

    # Shapes of the boundary tensors, from a real run, to fix the NPU graphs' inputs.
    probe = onnx.load(str(model))
    for t in enc_out + flow_in + dec_in + [durations]:
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
        feed["sid"] = np.array([3], np.int64)
    r = dict(zip(names, whole.run(None, feed)))
    probe_p.unlink()

    fix_inputs(enc_p, {"input": [1, ids_bucket], "input_lengths": [1]})
    frame_shapes = {}
    probe_frames = max(r[t].shape[-1] for t in flow_in)  # Rasa's speaker vector has 1
    for t in flow_in:
        s = list(r[t].shape)
        frame_shapes[t] = s[:-1] + [frames_bucket] if s[-1] == probe_frames else s
    fix_inputs(flow_p, frame_shapes)

    # The chain, padded, against the whole model.
    enc, mid, flow = session(enc_p), session(mid_p), session(flow_p)
    padded = np.zeros((1, ids_bucket), np.int64)
    padded[0, :n] = ids[0]
    e = enc.run(None, {"input": padded, "input_lengths": np.array([n], np.int64)})
    e = [x[..., :n] if x.shape[-1] == ids_bucket else
         (x[:, :n] if x.ndim > 1 and x.shape[1] == ids_bucket else x) for x in e]
    mfeed = dict(zip(enc_out, e))
    mfeed.update({"input": ids, "input_lengths": feed["input_lengths"]})
    mfeed = {k: v for k, v in mfeed.items() if k in {i.name for i in mid.get_inputs()}}
    mfeed["scales"] = feed["scales"]
    if "sid" in cond:
        mfeed["sid"] = feed["sid"]
    mo = dict(zip(flow_in + [durations], mid.run(None, mfeed)))
    frames = max(mo[t].shape[-1] for t in flow_in)
    ffeed = {}
    for t in flow_in:
        x = mo[t]
        if x.shape[-1] == frames:
            pad = np.zeros(x.shape[:-1] + (frames_bucket,), x.dtype)
            pad[..., :frames] = x
            x = pad
        ffeed[t] = x
    latent = flow.run(None, ffeed)[0][..., :frames]
    ref = r[flow_out[0]]
    err = float(np.abs(latent - ref).max())
    derr = float(np.abs(mo[durations] - r[durations]).max())
    scale = float(np.abs(ref).max())
    print(f"  ids {n} frames {frames}: latent max err {err:.2e} (of {scale:.2f}), durations err {derr}")
    assert derr == 0 and err < 1e-3 * max(1.0, scale), "chain does not match the model"
    for p in (enc_p, mid_p, flow_p):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")
    return passthrough


def main():
    a = argparse.ArgumentParser()
    a.add_argument("model", type=Path)
    a.add_argument("prefix")
    a.add_argument("--ids", type=int, default=256)
    a.add_argument("--frames", type=int, default=512)
    args = a.parse_args()
    build(args.model, args.prefix, args.ids, args.frames)


if __name__ == "__main__":
    main()
