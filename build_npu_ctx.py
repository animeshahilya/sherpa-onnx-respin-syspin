"""Compile every hosted NPU graph the app uses into QNN contexts, per Snapdragon target.

The app (eSpeak NG Advanced, Snapdragon build) used to compile these on the
phone with QNN's on-device compiler (libQnnHtpPrepare.so, 35 MB of the APK).
Compiled here instead, a phone downloads its own chip's contexts and the APK
drops the compiler. Proven on a Galaxy S25 Ultra (2026-10-19 spike): same
accuracy, faster loads, no first-use compile.

    python build_npu_ctx.py <npu_files.json> <npu_decoders.json> <out dir> [soc ...]

Sources come from release_assets_npu2 (npu-v2: enc/flow/dec) and
release_assets_npu (npu-v1: INT8 decoders), found by their path's file name.
Options mirror the app's NpuFiles.npuSession exactly:
  fp16 parts: enable_htp_fp16_precision, session.disable_quant_qdq
  INT8 decoders: offload_graph_io_quantization (QDQ kept, no fp16)
both with htp_performance_mode burst and no CPU fallback.

Each output pair is named as the phone names what it compiles itself
(model.1.30.0.<part>.<source size>.ctx.onnx + ..._qnn.bin; INT8:
model.1.30.0.npuq.<size>.onnx + ..._qnn.bin), so the app's cache logic finds
it unchanged. The wrapper's ep_compatibility_info gets the target's HTP arch
(the PC writes 0, which the phone rejects). Outputs go to <out>/<soc>/<md5>/
and a manifest <out>/npu_ctx.json maps soc -> source md5 -> files.

Needs onnxruntime-qnn 2.6.0 (QNN 2.50.40, matching the app's qnn-runtime
2.50.0). Runs at idle priority; already-built outputs are skipped.
"""
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time

SOCS = {"57": "75", "69": "79", "87": "81"}  # soc_model -> HTP arch: 8 Gen 3, 8 Elite, 8 Elite Gen 5
ORT_VERSION = "1.30.0"  # the app's ONNX Runtime: part of the names it looks for
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = [os.path.join(HERE, "release_assets_npu2"), os.path.join(HERE, "release_assets_npu")]
PART_STEM = {"enc": "npuenc", "flow": "npuflow", "dec": "npudec"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def graphs(npu_files, npu_decoders):
    """Distinct source graphs: md5 -> (kind, file name, size)."""
    out = {}
    for entry in json.load(open(npu_files))["voices"].values():
        for part in ("enc", "flow", "dec"):
            if part in entry:
                f = entry[part]
                out[f["md5_digest"]] = (part, os.path.basename(f["path"]), f["size_bytes"])
    for f in json.load(open(npu_decoders))["voices"].values():
        out[f["md5_digest"]] = ("int8", os.path.basename(f["path"]), f["size_bytes"])
    return out


def source(name):
    for d in SOURCES:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(name)


def names(kind, size):
    stem = "npuq" if kind == "int8" else PART_STEM[kind] + "." + str(size) + ".ctx"
    base = f"model.{ORT_VERSION}." + ("npuq." + str(size) if kind == "int8" else stem)
    return base + ".onnx", base + "_qnn.bin"


COMPILE = r'''
import os, sys, onnx
import onnxruntime as ort, onnxruntime_qnn as q
src, out, soc, arch, kind = sys.argv[1:6]
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
dev = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
so = ort.SessionOptions()
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
so.add_session_config_entry("ep.context_enable", "1")
so.add_session_config_entry("ep.context_file_path", out)
so.add_session_config_entry("ep.context_embed_mode", "0")
opts = {"backend_path": q.get_qnn_htp_path(), "htp_performance_mode": "burst", "soc_model": soc}
if kind == "int8":
    opts["offload_graph_io_quantization"] = "1"
else:
    opts["enable_htp_fp16_precision"] = "1"
    so.add_session_config_entry("session.disable_quant_qdq", "1")
so.add_provider_for_devices(dev, opts)
ort.InferenceSession(src, sess_options=so)
m = onnx.load(out, load_external_data=False)
tag = None
for p in m.metadata_props:
    if p.key.startswith("ep_compatibility_info"):
        f = p.value.split(":")
        f[4], f[5] = arch, soc
        p.value = tag = ":".join(f)
onnx.save(m, out)
print(tag)
'''


def main():
    if os.name == "nt":  # idle priority, children inherit
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
    npu_files, npu_decoders, out = sys.argv[1:4]
    socs = sys.argv[4:] or list(SOCS)
    python = os.environ.get("QNN_PYTHON", sys.executable)
    todo = graphs(npu_files, npu_decoders)
    manifest_path = os.path.join(out, "npu_ctx.json")
    manifest = json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {}
    n = 0
    for soc in socs:
        for digest, (kind, name, size) in sorted(todo.items(), key=lambda kv: kv[1][1]):
            n += 1
            ctx_name, bin_name = names(kind, size)
            d = os.path.join(out, soc, digest)
            ctx, binf = os.path.join(d, ctx_name), os.path.join(d, bin_name)
            if os.path.isfile(ctx) and os.path.isfile(binf) and digest in manifest.get(soc, {}):
                continue
            os.makedirs(d, exist_ok=True)
            t = time.time()
            r = subprocess.run([python, "-I", "-c", COMPILE, source(name), ctx, soc, SOCS[soc], kind],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.isfile(binf):
                print(f"FAILED soc {soc} {name}: {r.stderr.strip().splitlines()[-1:]}", flush=True)
                continue
            manifest.setdefault(soc, {})[digest] = {
                "kind": kind, "source": name, "source_size": size, "tag": r.stdout.strip(),
                "files": [{"name": f, "size_bytes": os.path.getsize(os.path.join(d, f)),
                           "md5_digest": md5(os.path.join(d, f))} for f in (ctx_name, bin_name)],
            }
            json.dump(manifest, open(manifest_path, "w"), indent=1)
            print(f"[{n}/{len(todo) * len(socs)}] soc {soc} {name} {time.time() - t:.0f} s", flush=True)
    total = sum(f["size_bytes"] for s in manifest.values() for e in s.values() for f in e["files"])
    print(f"done: {sum(len(s) for s in manifest.values())} contexts, {total / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
