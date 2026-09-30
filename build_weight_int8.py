#!/usr/bin/env python3
"""Build ~28MB weight-only INT8 variants of RESPIN/SYSPIN sherpa-onnx VITS voices.

Strategy (weight-only INT8 storage, compute stays FP32):
  - The 109MB FP32 models do all math in float32. Naive full-graph FP16
    breaks onnxruntime (Cast type errors); dynamic INT8 quant (~38MB)
    rewrites compute to INT8 and audibly alters durations (rejected).
  - This script converts only large float32 weight tensors (>=1024 elems,
    ndim>=2: Conv/MatMul weights, ~99.8% of params) to symmetric per-channel
    INT8 plus a DequantizeLinear back to FP32 before each consumer. Biases,
    norms and tiny constants stay FP32. All graph math, inputs and outputs
    stay exactly as before (int64/float32); sherpa-onnx metadata is preserved
    byte-for-byte; onnxruntime constant-folds the dequant at load, so there
    is ~zero runtime cost.
  - Result: ~28-30MB per voice at FP16-grade fidelity (mean rel. weight
    error ~1e-3), deterministic file size.

Usage:
  python build_weight_int8.py                    # build all 22 voices
  python build_weight_int8.py --model vits-syspin-hi-female
  python build_weight_int8.py --verify           # ORT smoke test per voice
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

MIN_ELEMS = 1024

MODELS = [
    "vits-syspin-bho-female", "vits-syspin-bho-male",
    "vits-syspin-bn-female", "vits-syspin-bn-male",
    "vits-syspin-en-female", "vits-syspin-en-male",
    "vits-syspin-gu-female", "vits-syspin-gu-male",
    "vits-syspin-hi-female", "vits-syspin-hi-male",
    "vits-syspin-hne-female", "vits-syspin-hne-male",
    "vits-syspin-kn-female", "vits-syspin-kn-male",
    "vits-syspin-mag-female", "vits-syspin-mag-male",
    "vits-syspin-mai-female", "vits-syspin-mai-male",
    "vits-syspin-mr-female", "vits-syspin-mr-male",
    "vits-syspin-te-female", "vits-syspin-te-male",
]


def quantize_per_channel(arr: np.ndarray, axis: int):
    """Symmetric INT8, one scale per channel along axis. Returns (q, scales)."""
    moved = np.moveaxis(arr, axis, 0)  # [C, ...]
    flat = moved.reshape(moved.shape[0], -1)
    amax = np.abs(flat).max(axis=1)
    amax = np.maximum(amax, 1e-8)
    scales = (amax / 127.0).astype(np.float32)
    q = np.clip(np.round(flat / scales[:, None]), -128, 127).astype(np.int8)
    q = np.moveaxis(q.reshape(moved.shape), 0, axis)
    return q, scales


def convert_weight_int8(src: Path, dst: Path) -> tuple:
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    model = onnx.load(str(src))
    orig_meta = [(p.key, p.value) for p in model.metadata_props]

    # consumer map: initializer name -> set of (op_type, input_index, node)
    consumers: dict = {}
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            consumers.setdefault(inp, []).append((node.op_type, i, node))
    for node in model.graph.node:  # subgraphs (If/Loop) capture outer tensors by name
        for attr in node.attribute:
            for g in ([attr.g] if attr.HasField("g") else list(attr.graphs)):
                for sn in g.node:
                    for i, inp in enumerate(sn.input):
                        consumers.setdefault(inp, []).append((sn.op_type, i, sn))

    def pick_axis(name: str, arr: np.ndarray):
        ops = {op for op, _, _ in consumers.get(name, [])}
        compute_ops = ops & {"Conv", "ConvTranspose", "MatMul", "Gemm"}
        if compute_ops == {"Conv"} or compute_ops == {"ConvTranspose"} \
                or compute_ops == {"Conv", "ConvTranspose"}:
            return 0
        if compute_ops == {"MatMul"}:
            return 1  # W[K, N], scale per output channel
        if compute_ops == {"Gemm"}:
            for op, idx, node in consumers.get(name, []):
                if op == "Gemm" and idx == 1:  # B input of Gemm
                    trans = next((a.i for a in node.attribute if a.name == "transB"), 0)
                    return 0 if trans else 1
            return 1
        if not compute_ops:
            return None  # no compute consumer: leave in FP32
        return -1  # shared across op kinds: per-tensor fallback

    n_conv = n_per_tensor = n_skip = 0
    sum_w2 = sum_e2 = 0.0  # global SNR accumulators (mean-rel-err is
    # dominated by near-zero tensors; energy-weighted SNR is the honest gate)
    deq_nodes = []
    orig_inits = list(model.graph.initializer)
    del model.graph.initializer[:]
    for init in orig_inits:
        if init.data_type != TensorProto.FLOAT:
            model.graph.initializer.append(init)
            continue
        arr = numpy_helper.to_array(init)
        if arr.ndim < 2 or arr.size < MIN_ELEMS:
            model.graph.initializer.append(init)
            n_skip += 1
            continue
        axis = pick_axis(init.name, arr)
        if axis is None:
            model.graph.initializer.append(init)
            n_skip += 1
            continue
        if axis == -1:  # shared weight: per-tensor fallback
            amax = max(float(np.abs(arr).max()), 1e-8)
            scale = np.array([amax / 127.0], dtype=np.float32)
            q = np.clip(np.round(arr / scale[0]), -128, 127).astype(np.int8)
            axis_attr = None
            n_per_tensor += 1
        else:
            q, scale = quantize_per_channel(arr.astype(np.float64), axis)
            axis_attr = axis
            n_conv += 1
        # reconstruction error stat
        if axis_attr is None:
            rec = q.astype(np.float64) * float(scale[0])
        else:
            # exact recompute:
            moved = np.moveaxis(q.astype(np.float64), axis_attr, 0)
            moved = moved * scale.reshape(-1, *([1] * (moved.ndim - 1)))
            rec = np.moveaxis(moved, 0, axis_attr)
        denom = np.abs(arr.astype(np.float64)).mean()
        w64 = arr.astype(np.float64)
        sum_w2 += float((w64 ** 2).sum())
        sum_e2 += float(((w64 - rec) ** 2).sum())
        _ = denom  # (kept for readability of the loop body)

        q_init = numpy_helper.from_array(q, name=init.name + "__w8stored")
        s_init = numpy_helper.from_array(scale, name=init.name + "__w8scale")
        model.graph.initializer.append(q_init)
        model.graph.initializer.append(s_init)
        node = helper.make_node(
            "DequantizeLinear",
            inputs=[q_init.name, s_init.name],
            outputs=[init.name],
            axis=axis_attr if axis_attr is not None else 1,
        )
        # per-tensor scale has 1 elem; axis attr harmless
        safe = init.name.replace("/", "_").replace(".", "_").replace(":", "_")
        node.name = "WeightDequant_" + safe[:180]
        deq_nodes.append(node)

    for n in reversed(deq_nodes):
        model.graph.node.insert(0, n)

    assert [(p.key, p.value) for p in model.metadata_props] == orig_meta
    onnx.save(model, str(dst))
    size_mb = dst.stat().st_size / 1024 / 1024
    snr_db = float(10 * np.log10(sum_w2 / max(sum_e2, 1e-300)))
    print(f"  dequant nodes: {len(deq_nodes)} (per-channel {n_conv}, "
          f"per-tensor {n_per_tensor}, kept-fp32 {n_skip}), "
          f"weight SNR {snr_db:.1f} dB, size: {size_mb:.1f} MB")
    return size_mb, snr_db


def verify(path: Path, tokens_path: Path) -> bool:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ids = []
    for line in tokens_path.read_text(encoding="utf-8").strip().split("\n")[:20]:
        try:
            ids.append(int(line.rsplit(" ", 1)[1]))
        except ValueError:
            continue
    if len(ids) < 5:
        print("  VERIFY FAIL: bad tokens")
        return False
    x = np.array([ids[:10]], dtype=np.int64)
    xl = np.array([len(x[0])], dtype=np.int64)
    s = np.array([0.667, 1.0, 0.8], dtype=np.float32)
    for _ in range(2):
        out = sess.run(None, {"input": x, "input_lengths": xl, "scales": s})[0]
        if not (bool(np.isfinite(out).all()) and out.size > 1000
                and float(abs(out).max()) > 1e-4):
            print(f"  VERIFY FAIL: shape={out.shape}")
            return False
    print(f"  verify: shape={out.shape} max={float(abs(out).max()):.6f} -> OK")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ~28MB weight-INT8 SYSPIN voices")
    ap.add_argument("--model", default="all")
    ap.add_argument("--base-dir", type=Path, default=Path("."))
    ap.add_argument("--assets-dir", type=Path, default=Path("release_assets_int8"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    models = MODELS if args.model == "all" else [args.model]
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for m in models:
        src = args.base_dir / m / "model.onnx"
        tok = args.base_dir / m / "tokens.txt"
        if not src.exists():
            print(f"[{m}] MISSING {src}")
            failures.append(m)
            continue
        dst = args.assets_dir / f"{m}-model.onnx"
        dst_tok = args.assets_dir / f"{m}-tokens.txt"
        if args.skip_existing and dst.exists() and dst.stat().st_size > 20_000_000:
            print(f"[{m}] exists ({dst.stat().st_size/1024/1024:.1f} MB), skipping")
        else:
            print(f"[{m}] converting {src.stat().st_size/1024/1024:.1f} MB FP32 -> W8...")
            size, snr = convert_weight_int8(src, dst)
            if not (15 <= size <= 40):
                print(f"[{m}] WARNING: unexpected size {size:.1f} MB")
                failures.append(m + " (size)")
            if snr < 35.0:
                print(f"[{m}] WARNING: low weight SNR {snr:.1f} dB")
                failures.append(m + " (qerr)")
        shutil.copy2(str(tok), str(dst_tok))
        if args.verify and not verify(dst, tok):
            failures.append(m + " (verify)")
    print("failures:", failures if failures else "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
