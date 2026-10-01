#!/usr/bin/env python3
"""Compact tier: a VITS voice with only its HiFi-GAN decoder quantized to INT8.

The decoder is ~95% of the work and runs as int8 QLinearConv on the CPU;
the text encoder, duration predictor and flow stay float, so word timing
and prosody are untouched (whole-model dynamic INT8 shifted durations).
Calibration runs real phoneme/letter sequences from the voice's own id map
through the model. Other float weights are then stored as FP16 like the
Standard files (build_fp16_60mb.convert_weight_fp16).

  python build_compact.py MODEL.onnx CONFIG.onnx.json OUT.onnx [--sid N]

Verify prints the decoder's spectral distance and SNR against the input
model on the same latents, and CPU time for both.
"""

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

DECODER_PREFIXES = ("/dec/", "/waveform_decoder/", "/decoder/")


def to_fp32(src: Path, dst: Path):
    """Folds FP16 weight upcasts (WeightUpcast Cast nodes) back into FP32 initializers."""
    m = onnx.load(str(src))
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    keep = []
    for n in g.node:
        if (n.op_type == "Cast" and len(n.input) == 1 and n.input[0] in inits
                and inits[n.input[0]].data_type == onnx.TensorProto.FLOAT16):
            a = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
            g.initializer.append(numpy_helper.from_array(a, n.output[0]))
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    used = {i for n in g.node for i in n.input}
    live = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(live)
    onnx.save(m, str(dst))


def last_stage(m):
    """The decoder's last upsampling stage, its residual blocks and the output conv:
    kept float. They run at the full sample rate, where INT8's noise floor shows
    (SYSPIN Kavya: pause noise -50.6 -> -55.2 dBFS, speech SNR 19.6 -> 21.2 dB,
    for 2.0x -> 1.7x the float decoder's CPU speed)."""
    import re
    # Piper/Coqui name the upsamplers "ups.N", transformers (Rasa) "upsampler.N".
    ups = {int(x) for n in m.graph.node for x in re.findall(r"/(?:ups|upsampler)\.(\d+)/", n.name)}
    res = {int(x) for n in m.graph.node for x in re.findall(r"/resblocks\.(\d+)/", n.name)}
    if not ups or not res:
        return []
    last = max(ups)
    per_stage = (max(res) + 1) // (last + 1)
    keep = [f"/ups.{last}/", f"/upsampler.{last}/", "conv_post"] + [f"/resblocks.{i}/" for i in range(last * per_stage, max(res) + 1)]
    return [n.name for n in m.graph.node
            if n.name.startswith(DECODER_PREFIXES) and any(k in n.name for k in keep)]


def sentences(config, count, seed=0):
    """Id sequences shaped like speech: words of letters from the voice's map, blank-padded."""
    ids = config["phoneme_id_map"]
    pad = ids["_"][0]
    letters = [v[0] for k, v in ids.items() if len(k) == 1 and k.isalpha()]
    space = ids.get(" ", [pad])[0]
    rnd = random.Random(seed)
    out = []
    for _ in range(count):
        seq = [ids["^"][0], pad]
        for w in range(rnd.randint(4, 12)):
            for _ in range(rnd.randint(2, 7)):
                seq += [rnd.choice(letters), pad]
            seq += [space, pad]
        out.append(seq + [ids["$"][0]])
    return out


def feed(seq, sid, names):
    f = {"input": np.array([seq], np.int64), "input_lengths": np.array([len(seq)], np.int64),
         "scales": np.array([0.667, 1.0, 0.8], np.float32)}
    if "sid" in names:
        f["sid"] = np.array([sid], np.int64)
    return f


def quantize(fp32: Path, out: Path, config, sid):
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    m = onnx.load(str(fp32))
    names = [i.name for i in m.graph.input]
    exclude = [n.name for n in m.graph.node if not n.name.startswith(DECODER_PREFIXES)]
    if len(exclude) == len(m.graph.node):
        # Older exports with unnamed nodes (es_MX-claude, en_US-libritts high):
        # nothing to quantize, and the "Compact" file would just be FP16.
        raise SystemExit(f"{fp32.name}: no named decoder nodes; not convertible")
    exclude += last_stage(m)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([feed(s, sid, names) for s in sentences(config, 24)])

        def get_next(self):
            return next(self.it, None)

    quantize_static(str(fp32), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv", "ConvTranspose"], nodes_to_exclude=exclude)


def decoder_of(model: Path, out: Path):
    """The decoder alone: from the latent (and speaker conditioning) to the audio.
    Inputs are found behind the Quantize/Dequantize/Cast nodes in front of the
    decoder, so the float model and the INT8 one cut at the same tensors."""
    m = onnx.load(str(model))
    producers = {o: n for n in m.graph.node for o in n.output}
    weights = {t.name for t in m.graph.initializer}
    passthrough = ("Cast", "DequantizeLinear", "QuantizeLinear")

    def source(t):
        p = producers.get(t)
        while p is not None and p.op_type in passthrough and not p.name.startswith(DECODER_PREFIXES):
            if all(i in weights or not i for i in p.input):
                return None  # a weight's own node
            t = p.input[0]
            p = producers.get(t)
        return t if p is not None and not p.name.startswith(DECODER_PREFIXES) else None

    inputs = []
    for n in m.graph.node:
        if n.name.startswith(DECODER_PREFIXES):
            for i in n.input:
                t = source(i)
                if t is not None and t not in inputs:
                    inputs.append(t)
    onnx.utils.extract_model(str(model), str(out), inputs, [m.graph.output[0].name],
                             check_model=False)
    return inputs


def verify(reference: Path, compact: Path, config, sid):
    import onnxruntime as ort
    with tempfile.TemporaryDirectory() as tmp:
        ref_dec, cmp_dec = Path(tmp) / "ref_dec.onnx", Path(tmp) / "cmp_dec.onnx"
        fp32 = Path(tmp) / "ref32.onnx"
        to_fp32(reference, fp32)
        ins = decoder_of(fp32, ref_dec)
        cmp_ins = decoder_of(compact, cmp_dec)
        assert sorted(cmp_ins) == sorted(ins), (cmp_ins, ins)
        # Real latents: the reference model's tensors at the decoder's inputs.
        probe = onnx.load(str(fp32))
        for name in ins:
            probe.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
        probe_path = Path(tmp) / "probe.onnx"
        onnx.save(probe, str(probe_path))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        P = ort.InferenceSession(str(probe_path), so, providers=["CPUExecutionProvider"])
        R = ort.InferenceSession(str(ref_dec), so, providers=["CPUExecutionProvider"])
        C = ort.InferenceSession(str(cmp_dec), so, providers=["CPUExecutionProvider"])
        names = [i.name for i in P.get_inputs()]
        lsd, snr, tr, tc, audio = [], [], 0.0, 0.0, 0.0
        rate = config["audio"]["sample_rate"]
        for seq in sentences(config, 6, seed=99):
            outs = P.run(None, feed(seq, sid, names))
            d = dict(zip([o.name for o in P.get_outputs()], outs))
            f = {n: d[n] for n in ins}
            t0 = time.perf_counter(); r = R.run(None, f)[0].squeeze(); tr += time.perf_counter() - t0
            t0 = time.perf_counter(); c = C.run(None, f)[0].squeeze(); tc += time.perf_counter() - t0
            audio += r.size / rate
            snr.append(10 * np.log10((r ** 2).sum() / ((r - c) ** 2).sum()))
            spec = lambda x: np.abs(np.array([np.fft.rfft(np.hanning(1024) * x[i:i + 1024])
                                              for i in range(0, len(x) - 1024, 256)])) + 1e-7
            S, N = spec(r), spec(c)
            lsd.append(np.mean(np.sqrt(np.mean((20 * np.log10(S) - 20 * np.log10(N)) ** 2, axis=1))))
        print(f"decoder: SNR {np.mean(snr):.1f} dB, log-spectral distance {np.mean(lsd):.2f} dB | "
              f"CPU x{audio / tr:.1f} -> x{audio / tc:.1f} real time ({tr / tc:.2f}x faster)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--sid", type=int, default=None)
    a = ap.parse_args()
    from build_fp16_60mb import convert_weight_fp16
    config = json.loads(a.config.read_text(encoding="utf-8"))
    sid = a.sid if a.sid is not None else config.get("default_speaker_id", 0)
    with tempfile.TemporaryDirectory() as tmp:
        fp32, q = Path(tmp) / "fp32.onnx", Path(tmp) / "q.onnx"
        to_fp32(a.model, fp32)
        quantize(fp32, q, config, sid)
        convert_weight_fp16(q, a.out)
    onnx.checker.check_model(str(a.out))
    print(f"{a.out.name}: {a.model.stat().st_size / 1e6:.1f} MB -> {a.out.stat().st_size / 1e6:.1f} MB")
    verify(a.model, a.out, config, sid)


if __name__ == "__main__":
    main()
