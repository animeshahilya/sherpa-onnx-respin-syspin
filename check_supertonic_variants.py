"""Which INT8 parts matter: log-mel distance to the original for mixes of
original/INT8 models, plus the original against itself with another random
draw (the model's own take-to-take variation)."""
import os, shutil, sys
from pathlib import Path
import numpy as np
import check_supertonic as c
if os.name == "nt":
    import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
SRC, OUT = Path("supertonic_src"), Path("release_assets_supertonic")
h = c.helper_module(SRC)
orig = {"dp": SRC / "onnx/duration_predictor.onnx", "te": SRC / "onnx/text_encoder.onnx",
        "ve": SRC / "onnx/vector_estimator.onnx", "voc": SRC / "onnx/vocoder.onnx"}
int8 = {k: OUT / f"supertonic3-{n}.onnx" for k, n in
        (("dp", "duration_predictor"), ("te", "text_encoder"), ("ve", "vector_estimator"), ("voc", "vocoder"))}
def tts(parts):
    import json
    cfgs = json.loads((SRC / "onnx/tts.json").read_text(encoding="utf-8"))
    tp = h.UnicodeProcessor(str(SRC / "onnx/unicode_indexer.json"))
    s = {k: c.sess(v) for k, v in parts.items()}
    return h.TextToSpeech(cfgs, tp, s["dp"], s["te"], s["ve"], s["voc"])
variants = {
    "all INT8": int8,
    "INT8, text encoder float": {**int8, "te": orig["te"]},
    "INT8, text encoder + vocoder float": {**int8, "te": orig["te"], "voc": orig["voc"]},
    "only vector estimator INT8": {**orig, "ve": int8["ve"]},
}
ref = tts(orig)
st = h.load_voice_style([str(SRC / "voice_styles/F1.json")])
sr = ref.sample_rate
texts = [(l, t) for l in ("en", "hi", "es") for t in c.sentences(l, 3)]
def render(t, text, lang, seed):
    np.random.seed(seed); w, d = t(text, lang, st, 8, 1.0); return w[0, :int(sr * d[0])]
base = [render(ref, t, l, 1000 + i) for i, (l, t) in enumerate(texts)]
self_d = [c.logmel_distance(base[i], render(ref, t, l, 5000 + i), sr) for i, (l, t) in enumerate(texts)]
print(f"original vs original, other random draw: {np.mean(self_d):.2f} dB", flush=True)
for name, parts in variants.items():
    t = tts(parts)
    d = [c.logmel_distance(base[i], render(t, tx, l, 1000 + i), sr) for i, (l, tx) in enumerate(texts)]
    mb = sum(Path(p).stat().st_size for p in parts.values()) / 1e6
    print(f"{name:38s} {np.mean(d):.2f} dB   ({mb:.0f} MB)", flush=True)
