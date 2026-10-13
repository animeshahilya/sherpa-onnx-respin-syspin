"""Checks build_supertonic.py's output against the original Supertonic 3.

1. Each rewritten model on random inputs at several lengths (CPU): the NPU
   rewrites (Softplus, 3-D MatMul) alone must match exactly, so the vector
   estimator is also built without INT8 weights and compared.
2. Whole sentences through Supertone's reference pipeline (their py/helper.py,
   pinned), original vs release files, the same random draw, 8 steps:
   log-mel distance aligned by DTW (as build_small.py's measurements; the
   shipped Compact was 1.43 dB, "sounds ok"). WAVs go to supertonic_listen/.
"""

import json
import shutil
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

REF = "https://raw.githubusercontent.com/supertone-inc/supertonic/1e9799e964ea4c0dad7cde993b65c3c813a7b373/py/helper.py"
CORPUS = Path(__file__).parent.parent / "voice-eval" / "corpus"
LANGS = ["en", "hi", "de", "es", "fr", "ru", "ar", "ja"]
LISTEN = Path(__file__).parent / "supertonic_listen"
ort.set_default_logger_severity(3)


def sess(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def feeds(s, L, T, rng):
    out = {}
    for i in s.get_inputs():
        shape = [1 if d == "batch_size" else (L if "latent" in str(d) else T) if isinstance(d, str) else d
                 for d in i.shape]
        if i.type == "tensor(int64)":
            out[i.name] = rng.integers(1, 100, size=shape).astype(np.int64)
        elif "mask" in i.name:
            out[i.name] = np.ones(shape, np.float32)
        elif i.name == "total_step":
            out[i.name] = np.full(shape, 8, np.float32)
        elif i.name == "current_step":
            out[i.name] = np.full(shape, 3, np.float32)
        else:
            out[i.name] = rng.standard_normal(shape).astype(np.float32)
    return out


def per_model(src, out):
    import build_supertonic as b
    tmp = Path(tempfile.mkdtemp())
    m = onnx.load(str(src / "onnx" / "vector_estimator.onnx"))
    b.batch_one(m)
    b.softplus(m.graph)
    b.matmul_3d(m)
    onnx.save(m, str(tmp / "ve_npu_float.onnx"))
    pairs = [("vector_estimator (NPU rewrites only)", src / "onnx" / "vector_estimator.onnx", tmp / "ve_npu_float.onnx")]
    pairs += [(f"{n} (release)", src / "onnx" / f"{n}.onnx", out / f"supertonic3-{n}.onnx") for n in b.MODELS]
    rng = np.random.default_rng(0)
    for label, a, c in pairs:
        sa, sc = sess(a), sess(c)
        for L, T in ((40, 50), (180, 230)):
            f = feeds(sa, L, T, rng)
            ra, rc = sa.run(None, f)[0], sc.run(None, f)[0]
            rel = float(np.sqrt(np.mean((ra - rc) ** 2) / max(np.mean(ra ** 2), 1e-20)))
            print(f"  {label:40s} L={L:3d} T={T:3d} max|diff| {np.abs(ra - rc).max():.3g} rel rms {rel:.4f}")
    shutil.rmtree(tmp, ignore_errors=True)


def helper_module(src):
    path = src / "ref_helper.py"
    if not path.is_file():
        path.write_bytes(urllib.request.urlopen(REF, timeout=60).read())
    sys.path.insert(0, str(src))
    import ref_helper
    return ref_helper


def pipeline(h, onnx_dir, names):
    cfgs = json.loads((Path(onnx_dir) / names["cfg"]).read_text(encoding="utf-8"))
    tp = h.UnicodeProcessor(str(Path(onnx_dir) / names["idx"]))
    s = {k: sess(Path(onnx_dir) / names[k]) for k in ("dp", "te", "ve", "voc")}
    return h.TextToSpeech(cfgs, tp, s["dp"], s["te"], s["ve"], s["voc"])


def logmel_distance(a, b, sr):
    import librosa
    lm = lambda x: 10 * np.log10(np.maximum(librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=2048, hop_length=512, n_mels=80), 1e-8))
    ma, mb = lm(a), lm(b)
    if ma.shape == mb.shape:
        return float(np.mean(np.abs(ma - mb)))
    _, path = librosa.sequence.dtw(X=ma, Y=mb, metric="euclidean")
    return float(np.mean([np.mean(np.abs(ma[:, i] - mb[:, j])) for i, j in path]))


def save(path, x, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def sentences(lang, n=4):
    p = CORPUS / f"{lang}.json"
    if not p.is_file():
        return []
    c = json.loads(p.read_text(encoding="utf-8"))
    return c["sentences"][:n]


def whole(src, out):
    h = helper_module(src)
    orig = pipeline(h, src / "onnx", {"cfg": "tts.json", "idx": "unicode_indexer.json", "dp": "duration_predictor.onnx",
                                      "te": "text_encoder.onnx", "ve": "vector_estimator.onnx", "voc": "vocoder.onnx"})
    rel = pipeline(h, out, {"cfg": "supertonic3-tts.json", "idx": "supertonic3-unicode_indexer.json",
                            "dp": "supertonic3-duration_predictor.onnx", "te": "supertonic3-text_encoder.onnx",
                            "ve": "supertonic3-vector_estimator.onnx", "voc": "supertonic3-vocoder.onnx"})
    styles = {s: h.load_voice_style([str(src / "voice_styles" / f"{s}.json")]) for s in ("F1", "M1")}
    sr = orig.sample_rate
    total = []
    for lang in LANGS:
        ds = []
        for i, text in enumerate(sentences(lang)):
            for sname, st in styles.items():
                np.random.seed(1000 + i)
                a, da = orig(text, lang, st, 8, 1.0)
                np.random.seed(1000 + i)
                b, db = rel(text, lang, st, 8, 1.0)
                a = a[0, :int(sr * da[0])]
                b = b[0, :int(sr * db[0])]
                ds.append(logmel_distance(a, b, sr))
                if i < 2:
                    save(LISTEN / lang / f"{sname}-{i}-original.wav", a, sr)
                    save(LISTEN / lang / f"{sname}-{i}-int8.wav", b, sr)
        if ds:
            total += ds
            print(f"  {lang}: log-mel distance {np.mean(ds):.2f} dB (max {np.max(ds):.2f}, {len(ds)} renders)", flush=True)
    print(f"  all: {np.mean(total):.2f} dB over {len(total)} renders; WAVs in {LISTEN}")


def main(src, out):
    print("per model:")
    per_model(src, out)
    print("whole sentences, 8 steps, same random draw:")
    whole(src, out)
