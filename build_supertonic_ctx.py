"""Supertonic's NPU graphs compiled on the PC, per Snapdragon target, weights shared.

    python build_supertonic_ctx.py <vector_estimator.onnx> <vocoder.onnx> <out dir> [soc ...]

Like build_npu_ctx.py (see there), for Supertonic 3: the flow (vector
estimator) and the vocoder each in three size buckets (Supertonic.BUCKETS),
compiled in one process with ep.share_ep_contexts + enable_htp_weight_sharing,
so each stage's buckets share ONE context binary holding the weights once
(184 MB instead of 561 MB; loads in ~1.2 s). QNN needs every non-weight
tensor renamed per bucket for that; the wrappers then get their graph I/O
names back (the app passes the original names - ONNX Runtime matches them by
position) and point at the shared binary under the app's name:

    <md5>.1.30.0.st-<flow|voc><latent>.ctx.onnx   (one per bucket)
    <md5>.1.30.0.st-<flow|voc>-shared_qnn.bin

where <md5> is the vector estimator's (the voice's source model, which keys
the app's shared store). Options mirror Supertonic.npuSession: fp16 HTP,
INT8 weights widened (session.disable_quant_qdq), basic optimization, no CPU
fallback. Proven on a Galaxy S25 Ultra in the 2026-10-19 spike. Output:
<out>/<soc>/supertonic/ and entries in <out>/npu_ctx.json under "supertonic".
"""
import ctypes
import hashlib
import json
import os
import subprocess
import sys

SOCS = {"57": "75", "69": "79", "87": "81"}
ORT_VERSION = "1.30.0"
BUCKETS = [(32, 48), (96, 144), (192, 288)]  # Supertonic.BUCKETS: latent, text lengths

COMPILE = r'''
import os, sys, onnx
import onnxruntime as ort, onnxruntime_qnn as q
model, what, out, soc = sys.argv[1:5]
BUCKETS = %s
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
dev = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
def renamed(src, suffix):
    m = onnx.load(src)
    weights = {i.name for i in m.graph.initializer}
    r = lambda n: n if (not n or n in weights) else n + suffix
    for v in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        v.name = r(v.name)
    for node in m.graph.node:
        node.input[:] = [r(x) for x in node.input]
        node.output[:] = [r(x) for x in node.output]
        node.name = node.name + suffix
    path = os.path.join(out, "src" + suffix + ".onnx")
    onnx.save(m, path, save_as_external_data=True, location="src" + suffix + ".data")
    return path
for i, (lat, txt) in enumerate(BUCKETS):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_session_config_entry("session.disable_quant_qdq", "1")
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_embed_mode", "0")
    so.add_session_config_entry("ep.context_file_path", os.path.join(out, "st-%%s%%d.ctx.onnx" %% (what, lat)))
    so.add_session_config_entry("ep.share_ep_contexts", "1")
    if i == len(BUCKETS) - 1:
        so.add_session_config_entry("ep.stop_share_ep_contexts", "1")
    for k, v in (("batch_size", 1), ("latent_length", lat), ("text_length", txt)):
        so.add_free_dimension_override_by_name(k, v)
    so.add_provider_for_devices(dev, {"backend_path": q.get_qnn_htp_path(), "enable_htp_fp16_precision": "1",
                                      "htp_performance_mode": "burst", "enable_htp_weight_sharing": "1",
                                      "soc_model": soc})
    ort.InferenceSession(renamed(model, "_b%%d" %% lat), sess_options=so)
''' % (BUCKETS,)

FINISH = r'''
import os, sys, onnx
out, what, md5, ver, arch, soc = sys.argv[1:7]
lats = [int(x) for x in sys.argv[7].split(",")]
shared = "%s.%s.st-%s-shared_qnn.bin" % (md5, ver, what)
bins = [f for f in os.listdir(out) if f.endswith("_qnn.bin") and f.startswith("st-")]
assert len(bins) == 1, bins
os.replace(os.path.join(out, bins[0]), os.path.join(out, shared))
for lat in lats:
    m = onnx.load(os.path.join(out, "st-%s%d.ctx.onnx" % (what, lat)), load_external_data=False)
    suffix = "_b%d" % lat
    strip = lambda n: n[:-len(suffix)] if n.endswith(suffix) else n
    for v in list(m.graph.input) + list(m.graph.output):
        v.name = strip(v.name)
    for node in m.graph.node:
        node.input[:] = [strip(x) for x in node.input]
        node.output[:] = [strip(x) for x in node.output]
        for a in node.attribute:
            if a.name == "ep_cache_context":
                a.s = shared.encode()
    for p in m.metadata_props:
        if p.key.startswith("ep_compatibility_info"):
            f = p.value.split(":")
            f[4], f[5] = arch, soc
            p.value = ":".join(f)
    onnx.save(m, os.path.join(out, "%s.%s.st-%s%d.ctx.onnx" % (md5, ver, what, lat)))
    os.remove(os.path.join(out, "st-%s%d.ctx.onnx" % (what, lat)))
for f in os.listdir(out):
    if f.startswith("src_b"):
        os.remove(os.path.join(out, f))
'''


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
    flow, voc, out = sys.argv[1:4]
    socs = sys.argv[4:] or list(SOCS)
    python = os.environ.get("QNN_PYTHON", sys.executable)
    key = md5(flow)
    manifest_path = os.path.join(out, "npu_ctx.json")
    manifest = json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {}
    lats = ",".join(str(b[0]) for b in BUCKETS)
    for soc in socs:
        d = os.path.join(out, soc, "supertonic")
        files = []
        for what, model in (("flow", flow), ("voc", voc)):
            work = os.path.join(d, what)
            os.makedirs(work, exist_ok=True)
            for cmd in ([python, "-I", "-c", COMPILE, os.path.abspath(model), what, os.path.abspath(work), soc],
                        [python, "-I", "-c", FINISH, os.path.abspath(work), what, key, ORT_VERSION, SOCS[soc],
                         soc, lats]):
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise SystemExit(f"soc {soc} {what}: {r.stderr.strip()[-800:]}")
            for f in sorted(os.listdir(work)):
                p = os.path.join(work, f)
                files.append({"name": f, "size_bytes": os.path.getsize(p), "md5_digest": md5(p), "dir": what})
            print(f"soc {soc} {what} done", flush=True)
        manifest.setdefault("supertonic", {})[soc] = {"source_md5": key, "files": files}
        json.dump(manifest, open(manifest_path, "w"), indent=1)


if __name__ == "__main__":
    main()
