# Sherpa-ONNX RESPIN / SYSPIN Indian Language Voices

Full-precision (FP32) authentic offline TTS models for [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [SherpaVoices](https://github.com/animeshahilya/SherpaVoices), derived from the **RESPIN** and **SYSPIN** initiatives by **IISc Bengaluru (SPIRE Lab)**.

## Overview

The SPIRE Lab at the Indian Institute of Science (IISc Bengaluru) created the **RESPIN** (*REcognizing SPeech in INdian languages*) and **SYSPIN** (*SYnthesizing SPeech in INdian languages*) projects to support speech technology for diverse Indian languages and dialects.

These models provide **authentic, full-precision (FP32) real voices** without INT8 quantization degradation. The full-precision weights preserve the exact timbre, prosody, and natural acoustic fidelity of each speaker.

Pre-converted ONNX models are hosted under [GitHub Releases (v1.0.0)](https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/tag/v1.0.0). Each model is injected with `sherpa-onnx` metadata tags (`model_type=vits`, `comment=coqui`, `frontend=characters`, `sample_rate=22050`).

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

## Credits & Licensing

- **Original Models & Checkpoints**: Developed by [SPIRE Lab, IISc Bengaluru](https://spire.ee.iisc.ac.in/) under the RESPIN and SYSPIN initiatives funded by Pratiksha Trust. Original checkpoints are available on Hugging Face at [huggingface.co/SYSPIN](https://huggingface.co/SYSPIN).
- **Inference Runtime**: Powered by [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0).

