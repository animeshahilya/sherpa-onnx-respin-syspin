#!/usr/bin/env python3
"""Export ai4bharat/vits_rasa_13 (multi-speaker VITS, 20 sids) to sherpa-onnx ONNX.

Needs:  pip install torch transformers safetensors onnx onnxruntime soundfile
        hf auth login   (gated repo: accept conditions on the HF model page first)

Usage:
  python export_rasa_to_onnx.py --probe                 # no export; dumps config/tokenizer/signature
  python export_rasa_to_onnx.py --export                # FP32 ONNX + tokens.txt
  python export_rasa_to_onnx.py --export --fp16         # + weight-only FP16 (~80MB)
  python export_rasa_to_onnx.py --verify                # ORT smoke test on Tamil/Punjabi/Assamese sids

Design (size-first):
  - ONE model file for all speakers (n_speakers=20, sid selects voice).
    Per-language marginal download cost = 0.
  - Emotion/style is BAKED to a single neutral id (default 0/ALEXA, see --emotion);
    sherpa-onnx VITS has no emotion input, only sid.
  - Weight-only FP16 storage with Cast-back to FP32 (same recipe as
    build_fp16_60mb.py): ~153MB FP32 -> ~80-84MB. No INT8 (prosody shift).
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = "ai4bharat/vits_rasa_13"
OUT_DIR = Path(__file__).parent / "vits-rasa-13"
MODEL_ID = "vits-rasa-13"

# sid plan: NEW languages first (repo already covers hi/bn/te/kn/mr/gu/bho/hne/mai/mag/en)
PRIORITY_SIDS = {
    18: ("TAM_F", "Tamil", "female"),
    11: ("MAL_F", "Malayalam", "female"),
    15: ("PAN_F", "Punjabi", "female"),
    16: ("PAN_M", "Punjabi", "male"),
    0: ("ASM_F", "Assamese", "female"),
    1: ("ASM_M", "Assamese", "male"),
    14: ("NEP_F", "Nepali", "female"),
    17: ("SAN_M", "Sanskrit", "male"),
    4: ("BRX_F", "Bodo", "female"),
    5: ("BRX_M", "Bodo", "male"),
    6: ("DOI_F", "Dogri", "female"),
    7: ("DOI_M", "Dogri", "male"),
}


def load_hf():
    from transformers import AutoModel, AutoTokenizer
    print(f"[{MODEL_ID}] loading {REPO} (trust_remote_code=True)...")
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model.eval()
    return model, tok


def cmd_probe():
    model, tok = load_hf()
    cfg = model.config
    print("=== config ===")
    for k in ("sampling_rate", "sample_rate", "hidden_size", "num_speakers",
              "num_emotions", "num_styles", "vocab_size", "pad_token_id",
              "bos_token_id", "eos_token_id", "model_type", "architectures"):
        if hasattr(cfg, k):
            print(f"  {k} = {getattr(cfg, k)!r}")
    print("  full keys:", sorted(cfg.to_dict().keys())[:40])
    print("=== tokenizer ===")
    print("  class:", type(tok).__name__)
    vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
    print(f"  vocab size: {len(vocab)}")
    items = sorted(vocab.items(), key=lambda kv: kv[1])[:15]
    for s, i in items:
        print(f"    {i}: {s!r}")
    print("=== forward signature ===")
    import inspect
    print("  " + str(inspect.signature(model.forward)))
    print("=== modules (top) ===")
    for name, _ in list(model.named_children()):
        print("  " + name)
    n = sum(p.numel() for p in model.parameters())
    print(f"=== params: {n:,} ({n * 4 / 1024 / 1024:.1f} MB FP32) ===")


def build_wrapper(model, emotion_id: int):
    """Freeze emotion; expose sherpa-onnx VITS multi-speaker forward.

    ASSUMPTION (verified in --probe): model.forward(input_ids, speaker_id,
    emotion_id) -> object with .waveform. If probe shows a different
    signature, adjust here before exporting.
    """
    import torch
    import torch.nn as nn

    class Wrapper(nn.Module):
        def __init__(self, m, emo):
            super().__init__()
            self.m = m
            self.emo = emo

        def forward(self, input_ids, input_lengths, scales, sid):
            # scales: [noise_scale, length_scale, noise_scale_w] (sherpa convention)
            out = self.m(input_ids, speaker_id=sid, emotion_id=self.emo)
            wav = out.waveform if hasattr(out, "waveform") else out[0]
            return wav

    return Wrapper(model, emotion_id).eval()


def cmd_export(args):
    import torch
    model, tok = load_hf()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # tokens.txt from HF tokenizer vocab (sherpa C++ parses `sym id` per line)
    vocab = tok.get_vocab()
    tokens_path = OUT_DIR / "tokens.txt"
    with open(tokens_path, "w", encoding="utf-8") as f:
        for sym, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
            if sym in ("\r", "\n", "\t"):
                continue
            f.write(f"{sym} {idx}\n")
    print(f"[{MODEL_ID}] tokens.txt: {len(vocab)} entries")

    # sample tokenization for export shapes
    sample = " ".join(["\u0ba4\u0bae\u0bbf\u0bb4\u0bcd"])  # Tamil sanity string
    ids = tok(sample, return_tensors="pt")["input_ids"]
    print(f"[{MODEL_ID}] sample ids shape: {tuple(ids.shape)}")

    wrapper = build_wrapper(model, args.emotion)
    x = ids.to(torch.long)
    xl = torch.tensor([x.shape[1]], dtype=torch.long)
    scales = torch.tensor([0.667, 1.0, 0.8], dtype=torch.float32)
    sid = torch.tensor([18], dtype=torch.long)  # Tamil female for tracing

    onnx_path = OUT_DIR / "model.onnx"
    print(f"[{MODEL_ID}] exporting ONNX -> {onnx_path} ...")
    torch.onnx.export(
        wrapper, (x, xl, scales, sid), str(onnx_path),
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["wav"],
        dynamic_axes={"input": {0: "batch", 1: "text_len"},
                      "input_lengths": {0: "batch"},
                      "sid": {0: "batch"}, "wav": {0: "batch", 1: "samples"}},
        opset_version=17, do_constant_folding=True,
    )

    # sherpa-onnx metadata (multi-speaker VITS)
    import onnx
    from transformers import AutoModel as _AM  # noqa (keeps import locality)
    sr = int(getattr(model.config, "sampling_rate",
                     getattr(model.config, "sample_rate", 22050)))
    meta = {
        "model_type": "vits",
        "comment": "ai4bharat-rasa",
        "frontend": "characters",  # CONFIRM in --probe; change to phonemes if BPE/phonemizer
        "language": "multilingual-indic",
        "sample_rate": sr,
        "n_speakers": 20,
        "add_blank": 0,
        "use_eos_bos": 0,
    }
    m = onnx.load(str(onnx_path))
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, str(v)
    onnx.save(m, str(onnx_path))
    print(f"[{MODEL_ID}] FP32 size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")

    if args.fp16:
        from build_fp16_60mb import convert_weight_fp16
        fp16_path = OUT_DIR / "model.fp16.onnx"
        convert_weight_fp16(onnx_path, fp16_path)
        print(f"[{MODEL_ID}] FP16 size: {fp16_path.stat().st_size / 1024 / 1024:.1f} MB")


def cmd_verify(args):
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer
    model_file = OUT_DIR / ("model.fp16.onnx" if (OUT_DIR / "model.fp16.onnx").exists() else "model.onnx")
    assert model_file.exists(), f"missing {model_file}; run --export first"
    sess = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    print("inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    texts = {18: "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0bae\u0bca\u0bb4\u0bbf",
             16: "\u0a38\u0a24\u0a3f \u0a38\u0a4d\u0a30\u0a40 \u0a05\u0a15\u0a3e\u0a32",
             0: "\u0985\u09b8\u09ae\u09c0\u09af\u09bc\u09be \u09ad\u09be\u09b7\u09be"}
    ok_all = True
    for sid, text in texts.items():
        ids = tok(text, return_tensors="pt")["input_ids"].numpy().astype(np.int64)
        xl = np.array([ids.shape[1]], dtype=np.int64)
        scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
        feed = {"input": ids, "input_lengths": xl, "scales": scales}
        if any(i.name == "sid" for i in sess.get_inputs()):
            feed["sid"] = np.array([sid], dtype=np.int64)
        wav = sess.run(None, feed)[0]
        ok = bool(np.isfinite(wav).all()) and wav.size > 1000 and float(abs(wav).max()) > 1e-4
        ok_all &= ok
        print(f"  sid={sid} shape={wav.shape} max={float(abs(wav).max()):.5f} -> {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok_all else 1)


def main():
    ap = argparse.ArgumentParser(description="Export vits_rasa_13 to sherpa-onnx ONNX")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="also build weight-only FP16")
    ap.add_argument("--emotion", type=int, default=0, help="baked style id (default 0/ALEXA per model card example)")
    args = ap.parse_args()
    if args.probe:
        return cmd_probe()
    if args.export:
        return cmd_export(args)
    if args.verify:
        return cmd_verify(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
