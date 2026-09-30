# Sherpa-ONNX RESPIN / SYSPIN Indian Language Voices

Full-precision (FP32) authentic offline TTS models for [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [SherpaVoices](https://github.com/animeshahilya/SherpaVoices), derived from the **RESPIN** and **SYSPIN** initiatives by **IISc Bengaluru (SPIRE Lab)**.

## Overview

The SPIRE Lab at the Indian Institute of Science (IISc Bengaluru) created the **RESPIN** (*REcognizing SPeech in INdian languages*) and **SYSPIN** (*SYnthesizing SPeech in INdian languages*) projects to support speech technology for diverse Indian languages and dialects.

These models provide **authentic, full-precision (FP32) real voices** without INT8 quantization degradation. The full-precision weights preserve the exact timbre, prosody, and natural acoustic fidelity of each speaker.

Pre-converted ONNX models are hosted under [GitHub Releases (v1.0.0)](https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/tag/v1.0.0). Each model is injected with `sherpa-onnx` metadata tags (`model_type=vits`, `comment=coqui`, `frontend=characters`, `sample_rate=22050`).

## Lightweight ~60MB FP16-weight voices (v1.1.0-fp16)

All 22 upstream SYSPIN VITS voices are already covered above — there are no
additional VITS checkpoints upstream (remaining SYSPIN/RESPIN repos are
GlowTTS/Tacotron/ASR, not sherpa-onnx VITS compatible). To make the voices
cheaper to download and cache on-device, [Release v1.1.0-fp16](https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/tag/v1.1.0-fp16)
ships **weight-only FP16 variants at ~55MB each ("around 60MB")** with identical
voice coverage:

- Same 22 model identifiers, same `tokens.txt`, same sherpa-onnx metadata
  (`model_type=vits`, `comment=coqui`, `frontend=characters`, `sample_rate=22050`).
- Only large weight initializers (Conv/MatMul, ≥1024 elems) are stored as FP16
  with a Cast back to FP32, so **all graph compute stays FP32** and the model
  loads on plain onnxruntime CPU. Naive full-FP16 conversion was rejected
  (breaks `Cast` type inference), and dynamic INT8 (∼38MB) was rejected
  (audibly alters durations).
- Built locally with `python build_fp16_60mb.py --verify` (every voice
  smoke-tested in onnxruntime: finite, non-silent output).

```bash
# Download a lightweight voice instead of the 109MB FP32 one
curl -LO https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v1.1.0-fp16/vits-syspin-hi-female-model.onnx
curl -LO https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v1.1.0-fp16/vits-syspin-hi-female-tokens.txt

sherpa-onnx-offline-tts \
  --vits-model=./vits-syspin-hi-female-model.onnx \
  --vits-tokens=./vits-syspin-hi-female-tokens.txt \
  --vits-data-dir="" \
  --output-filename=./output.wav \
  "नमस्ते आप कैसे हैं"
```

## Models in this Repository

| Language | Code | Gender | Model Identifier | Precision | Model Size |
|---|---|---|---|---|---|
| **Hindi** | `hi` | Female | `vits-syspin-hi-female` | FP32 (Real Voice) | ~109 MB |
| **Hindi** | `hi` | Male | `vits-syspin-hi-male` | FP32 (Real Voice) | ~109 MB |
| **Bengali** | `bn` | Female | `vits-syspin-bn-female` | FP32 (Real Voice) | ~109 MB |
| **Bengali** | `bn` | Male | `vits-syspin-bn-male` | FP32 (Real Voice) | ~109 MB |
| **Telugu** | `te` | Female | `vits-syspin-te-female` | FP32 (Real Voice) | ~109 MB |
| **Telugu** | `te` | Male | `vits-syspin-te-male` | FP32 (Real Voice) | ~109 MB |
| **Kannada** | `kn` | Female | `vits-syspin-kn-female` | FP32 (Real Voice) | ~109 MB |
| **Kannada** | `kn` | Male | `vits-syspin-kn-male` | FP32 (Real Voice) | ~109 MB |
| **Marathi** | `mr` | Female | `vits-syspin-mr-female` | FP32 (Real Voice) | ~109 MB |
| **Marathi** | `mr` | Male | `vits-syspin-mr-male` | FP32 (Real Voice) | ~109 MB |
| **Gujarati** | `gu` | Female | `vits-syspin-gu-female` | FP32 (Real Voice) | ~109 MB |
| **Gujarati** | `gu` | Male | `vits-syspin-gu-male` | FP32 (Real Voice) | ~109 MB |
| **Bhojpuri** | `bho` | Female | `vits-syspin-bho-female` | FP32 (Real Voice) | ~109 MB |
| **Bhojpuri** | `bho` | Male | `vits-syspin-bho-male` | FP32 (Real Voice) | ~109 MB |
| **Chhattisgarhi** | `hne` | Female | `vits-syspin-hne-female` | FP32 (Real Voice) | ~109 MB |
| **Chhattisgarhi** | `hne` | Male | `vits-syspin-hne-male` | FP32 (Real Voice) | ~109 MB |
| **Maithili** | `mai` | Female | `vits-syspin-mai-female` | FP32 (Real Voice) | ~109 MB |
| **Maithili** | `mai` | Male | `vits-syspin-mai-male` | FP32 (Real Voice) | ~109 MB |
| **Magahi** | `mag` | Female | `vits-syspin-mag-female` | FP32 (Real Voice) | ~109 MB |
| **Magahi** | `mag` | Male | `vits-syspin-mag-male` | FP32 (Real Voice) | ~109 MB |
| **English (India)** | `en` | Female | `vits-syspin-en-female` | FP32 (Real Voice) | ~109 MB |
| **English (India)** | `en` | Male | `vits-syspin-en-male` | FP32 (Real Voice) | ~109 MB |

## Using with SherpaVoices Android App

In [SherpaVoices](https://github.com/animeshahilya/SherpaVoices), these voices are defined with:
- `source`: `"github_release"`
- `repo`: `"animeshahilya/sherpa-onnx-respin-syspin"`
- `pathPrefix`: `<model-identifier>`
- `rawTokensFile`: `true`

The app downloads the full-precision `model.onnx` and `tokens.txt` directly from GitHub Releases CDN and caches them locally for fully offline synthesis on-device.

## Using with sherpa-onnx CLI

```bash
# Download a voice model and tokens directly from GitHub Releases
curl -LO https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v1.0.0/vits-syspin-hi-female-model.onnx
curl -LO https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v1.0.0/vits-syspin-hi-female-tokens.txt

# Synthesize with sherpa-onnx-offline-tts
sherpa-onnx-offline-tts \
  --vits-model=./vits-syspin-hi-female-model.onnx \
  --vits-tokens=./vits-syspin-hi-female-tokens.txt \
  --vits-data-dir="" \
  --output-filename=./output.wav \
  "नमस्ते आप कैसे हैं"
```

## Ultra-light ~28MB weight-INT8 voices (v1.2.0-int8)

[Release v1.2.0-int8](https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/tag/v1.2.0-int8)
ships all 22 voices at **~28.5MB each** with no compute change:

- Only large weight tensors (Conv/MatMul, ≥1024 elems, ndim≥2) are stored as
  symmetric per-channel INT8 with a `DequantizeLinear` back to FP32, so **all
  graph math stays FP32** (unlike dynamic INT8 quant, which rewrites compute
  to INT8 and audibly alters durations — rejected at ~38MB).
- Same identifiers, tokens and sherpa-onnx metadata as v1.0.0/v1.1.0-fp16.
- Measured global weight SNR is **40–41dB per voice** (transparent rounding);
  every voice is ORT smoke-tested (finite, non-silent output, 2 runs).
- Built with `python build_weight_int8.py --verify`. Prefer this release when
  download size / on-device cache matters; prefer v1.0.0 FP32 when you want
  bit-closest-to-training weights.

## Credits & Licensing

- **Original Models & Checkpoints**: Developed by [SPIRE Lab, IISc Bengaluru](https://spire.ee.iisc.ac.in/) under the RESPIN and SYSPIN initiatives funded by Pratiksha Trust. Original checkpoints are available on Hugging Face at [huggingface.co/SYSPIN](https://huggingface.co/SYSPIN).
- **Inference Runtime**: Powered by [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0).

