#!/usr/bin/env python3
"""NPU decoders: a voice's HiFi-GAN decoder fully in INT8, for Qualcomm's HTP.

Every decoder op is quantized (Conv, ConvTranspose, LeakyRelu, Add, Div, Tanh,
Mul), so the whole graph runs on the NPU; Compact's float islands (activations,
last stage) make QNN hand those back to the CPU and gain nothing. Inputs and
output are fixed to WINDOW latent frames, the window the espeak-ng app decodes
in (PiperModel.NPU_FRAMES), with the same tensor names as the Standard model's
decoder, so the app feeds it straight from the Standard encoder.

Galaxy S25 Ultra (HTP v79), Kavya: 24.5x real time vs 12.4x for the FP16 path,
same accuracy (speech SNR 12.4 vs 11.9 dB, pause noise -48.2 vs -48.9 dBFS).
Not for CPUs: there its pause noise is -38 dBFS (Compact: -56).

  python build_npu.py MODEL.onnx CONFIG.onnx.json OUT.onnx
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx

import build_compact as bc

WINDOW = 80
FULL_INT8 = ["Conv", "ConvTranspose", "LeakyRelu", "Add", "Div", "Tanh", "Mul"]


def build(model: Path, config, out: Path, sid: int):
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    with tempfile.TemporaryDirectory() as tmp:
        fp32, q = Path(tmp) / "fp32.onnx", Path(tmp) / "q.onnx"
        bc.to_fp32(model, fp32)
        m = onnx.load(str(fp32))
        names = [i.name for i in m.graph.input]
        non_decoder = [n.name for n in m.graph.node if not n.name.startswith(bc.DECODER_PREFIXES)]

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter([bc.feed(s, sid, names) for s in bc.sentences(config, 24)])

            def get_next(self):
                return next(self.it, None)

        quantize_static(str(fp32), str(q), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        op_types_to_quantize=FULL_INT8, nodes_to_exclude=non_decoder)
        inputs = bc.decoder_of(q, out)
        ref_inputs = bc.decoder_of(fp32, Path(tmp) / "ref.onnx")
        assert sorted(inputs) == sorted(ref_inputs), (inputs, ref_inputs)
        # Fixed shapes: the latent window, and conditioning as the model has it.
        probe = onnx.shape_inference.infer_shapes(onnx.load(str(fp32)))
        shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
                  for v in list(probe.graph.value_info) + list(probe.graph.output)}
        d = onnx.load(str(out))
        latent = None
        for i in d.graph.input:
            dims = shapes.get(i.name, [])
            fixed = [1, dims[1], 1] if len(dims) == 3 and dims[2] == 1 else None
            if fixed is None:  # the latent: channels x frames
                latent = i.name
                fixed = [1, dims[1] if len(dims) == 3 and dims[1] else 192, WINDOW]
            i.type.tensor_type.shape.Clear()
            for v in fixed:
                i.type.tensor_type.shape.dim.add().dim_value = v
        o = d.graph.output[0].type.tensor_type.shape
        o.Clear()
        for v in (1, 1, WINDOW * config.get("hop_length", 256)):
            o.dim.add().dim_value = v
        del d.graph.value_info[:]
        onnx.save(d, str(out))
        return latent, Path(tmp) / "ref.onnx", fp32, sorted(inputs)


def verify(out: Path, ref_dec: Path, fp32: Path, inputs, config, sid):
    """INT8 window vs the float decoder on the same real latent window (CPU)."""
    import onnxruntime as ort
    probe = onnx.load(str(fp32))
    for n in inputs:
        probe.graph.output.append(onnx.helper.make_empty_tensor_value_info(n))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "p.onnx"
        onnx.save(probe, str(p))
        P = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        names = [i.name for i in P.get_inputs()]
        seq = max(bc.sentences(config, 8, seed=7), key=len)
        res = dict(zip([o.name for o in P.get_outputs()], P.run(None, bc.feed(seq, sid, names))))
    feed = {}
    for n in inputs:
        t = res[n]
        feed[n] = t[..., :WINDOW] if t.shape[-1] >= WINDOW and t.shape[-1] != 1 else t
    lat = [n for n in inputs if feed[n].shape[-1] == WINDOW][0]
    if feed[lat].shape[-1] < WINDOW:
        raise SystemExit("calibration sentence shorter than a window")
    R = ort.InferenceSession(str(ref_dec), providers=["CPUExecutionProvider"])
    N = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    r = R.run(None, feed)[0].squeeze()
    c = N.run(None, feed)[0].squeeze()
    snr = 10 * np.log10((r ** 2).sum() / ((r - c) ** 2).sum())
    print(f"{out.name}: {out.stat().st_size / 1e6:.1f} MB, inputs {[(n, list(feed[n].shape)) for n in inputs]},"
          f" CPU check SNR {snr:.1f} dB (the HTP computes it differently; S25: ~12 dB, as its FP16 path)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    config = json.loads(a.config.read_text(encoding="utf-8"))
    sid = config.get("default_speaker_id", 0)
    latent, ref_dec, fp32, inputs = None, None, None, None
    with tempfile.TemporaryDirectory():
        latent, ref_dec, fp32, inputs = build(a.model, config, a.out, sid)
    # build() cleaned its temp dir: rebuild the reference for the check.
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "fp32.onnx"
        bc.to_fp32(a.model, fp32)
        ref = Path(tmp) / "ref.onnx"
        bc.decoder_of(fp32, ref)
        verify(a.out, ref, fp32, inputs, config, sid)


if __name__ == "__main__":
    main()
