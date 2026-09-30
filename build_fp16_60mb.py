#!/usr/bin/env python3
"""Build ~60MB FP16-weight variants of RESPIN/SYSPIN sherpa-onnx VITS voices.

Strategy (weight-only FP16 storage, compute stays FP32):
  - Existing FP32 model.onnx (~109MB) has all compute in float32.
  - Naive full FP16 conversion breaks onnxruntime (Cast/Shape type errors,
    duplicate tensor names from subgraph handling).
  - Dynamic INT8 quant shrinks to ~38MB but audibly alters durations.
  - This script converts only large float32 initializers (>=1024 elems,
    i.e. Conv/MatMul weights, ~99.8% of params) to float16 storage and
    inserts a Cast back to float32 before each consumer. All graph math
    stays FP32, inputs/outputs stay int64/float32, sherpa-onnx metadata
    (model_type=vits, comment=coqui, frontend, sample_rate, blank_id...)
    is preserved byte-for-byte, and onnxruntime CPU loads it fine.
  - Result: ~57-59MB per voice ("around 60MB"), deterministic file size.

Usage:
  python build_fp16_60mb.py                      # build all 22 voices
  python build_fp16_60mb.py --model vits-syspin-hi-female
  python build_fp16_60mb.py --verify             # ORT smoke test (finite, non-silent)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

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


def convert_weight_fp16(src: Path, dst: Path) -> float:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(src))
    orig_meta = [(p.key, p.value) for p in model.metadata_props]

    del_count = 0
    cast_nodes = []
    orig_inits = list(model.graph.initializer)
    del model.graph.initializer[:]
    for init in orig_inits:
        if init.data_type == onnx.TensorProto.FLOAT:
            import numpy as np  # noqa: F401  (needed by numpy_helper)
            arr = numpy_helper.to_array(init)
            if arr.size >= MIN_ELEMS:
                arr16 = arr.astype("float16")
                fp16_init = numpy_helper.from_array(arr16, name=init.name + "__fp16stored")
                model.graph.initializer.append(fp16_init)
                node = onnx.helper.make_node(
                    "Cast",
                    inputs=[fp16_init.name],
                    outputs=[init.name],
                    to=onnx.TensorProto.FLOAT,
                )
                safe = init.name.replace("/", "_").replace(".", "_").replace(":", "_")
                node.name = "WeightUpcast_" + safe[:180]
                cast_nodes.append(node)
                del_count += 1
                continue
        model.graph.initializer.append(init)

    for n in reversed(cast_nodes):
        model.graph.node.insert(0, n)

    # metadata must be unchanged
    assert [(p.key, p.value) for p in model.metadata_props] == orig_meta
    onnx.save(model, str(dst))
    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"  upcast nodes: {del_count}, size: {size_mb:.1f} MB")
    return size_mb


def verify(path: Path, tokens_path: Path) -> bool:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ids = []
    for line in tokens_path.read_text(encoding="utf-8").strip().split("\n")[:20]:
        parts = line.rsplit(" ", 1)
        try:
            ids.append(int(parts[1]))
        except ValueError:
            continue
    if len(ids) < 5:
        print(f"  VERIFY FAIL: could not parse tokens {tokens_path}")
        return False
    x = np.array([ids[:10]], dtype=np.int64)
    xl = np.array([len(x[0])], dtype=np.int64)
    s = np.array([0.667, 1.0, 0.8], dtype=np.float32)
    out = sess.run(None, {"input": x, "input_lengths": xl, "scales": s})[0]
    ok = bool(np.isfinite(out).all()) and out.size > 1000 and float(abs(out).max()) > 1e-4
    print(f"  verify: shape={out.shape} max={float(abs(out).max()):.6f} -> {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ~60MB FP16-weight SYSPIN voices")
    ap.add_argument("--model", default="all", help="'all' or a single model id")
    ap.add_argument("--base-dir", type=Path, default=Path("."))
    ap.add_argument("--assets-dir", type=Path, default=Path("release_assets_fp16"))
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
            print(f"[{m}] MISSING {src}, skipping")
            failures.append(m)
            continue
        dst = args.assets_dir / f"{m}-model.onnx"
        dst_tok = args.assets_dir / f"{m}-tokens.txt"
        if args.skip_existing and dst.exists() and dst.stat().st_size > 50_000_000:
            print(f"[{m}] exists ({dst.stat().st_size/1024/1024:.1f} MB), skipping")
        else:
            print(f"[{m}] converting {src.stat().st_size/1024/1024:.1f} MB FP32 -> FP16 weights...")
            size = convert_weight_fp16(src, dst)
            if not (50 <= size <= 70):
                print(f"[{m}] WARNING: unexpected size {size:.1f} MB (want ~60)")
        shutil.copy2(str(tok), str(dst_tok))
        if args.verify:
            if not verify(dst, tok):
                failures.append(m + " (verify)")
    print("failures:", failures if failures else "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
