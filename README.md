# Sherpa-ONNX RESPIN / SYSPIN Indian Language Voices

Pre-converted and INT8-quantized offline TTS models for [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [SherpaVoices](https://github.com/animeshahilya/SherpaVoices), derived from the **RESPIN** and **SYSPIN** initiatives by **IISc Bengaluru (SPIRE Lab)**.

## Overview

The SPIRE Lab at the Indian Institute of Science (IISc Bengaluru) created the **RESPIN** (*REcognizing SPeech in INdian languages*) and **SYSPIN** (*SYnthesizing SPeech in INdian languages*) projects to support speech technology for diverse Indian languages and dialects.

This repository hosts pre-converted, ready-to-run ONNX models (`model.onnx` + `tokens.txt`) that can be used directly on Android, Linux, macOS, and Windows with **sherpa-onnx** without needing to run Coqui TTS or PyTorch conversion pipelines locally.

## Models in this Repository

Each voice has been INT8-quantized to ~36MB (well within mobile memory constraints) and injected with `sherpa-onnx` metadata tags (`model_type=vits`, `comment=coqui`, `frontend=characters`, `sample_rate=22050`).

| Language | Code | Gender | Model Identifier | Model Size |
|---|---|---|---|---|
| **Hindi** | `hi` | Female | `vits-syspin-hi-female` | ~36 MB |
| **Hindi** | `hi` | Male | `vits-syspin-hi-male` | ~36 MB |
| **Bengali** | `bn` | Female | `vits-syspin-bn-female` | ~36 MB |
| **Telugu** | `te` | Female | `vits-syspin-te-female` | ~36 MB |
| **Kannada** | `kn` | Female | `vits-syspin-kn-female` | ~36 MB |
| **Marathi** | `mr` | Female | `vits-syspin-mr-female` | ~36 MB |
| **Bhojpuri** | `bho` | Female | `vits-syspin-bho-female` | ~36 MB |
| **Chhattisgarhi** | `hne` | Female | `vits-syspin-hne-female` | ~36 MB |
| **Maithili** | `mai` | Female | `vits-syspin-mai-female` | ~36 MB |
| **Magahi** | `mag` | Female | `vits-syspin-mag-female` | ~36 MB |

## Using with SherpaVoices Android App

In [SherpaVoices](https://github.com/animeshahilya/SherpaVoices), these voices are defined as community voices with:
- `source`: `"github"`
- `repo`: `"animeshahilya/sherpa-onnx-respin-syspin"`
- `pathPrefix`: `<model-identifier>`
- `rawTokensFile`: `true`

The app downloads `model.onnx` and `tokens.txt` directly via GitHub raw URLs and caches them locally for fully offline synthesis on-device.

## Using with sherpa-onnx CLI

```bash
# Clone or download the voice folder
git clone https://github.com/animeshahilya/sherpa-onnx-respin-syspin.git

# Synthesize with sherpa-onnx-offline-tts
sherpa-onnx-offline-tts \
  --vits-model=./sherpa-onnx-respin-syspin/vits-syspin-hi-female/model.onnx \
  --vits-tokens=./sherpa-onnx-respin-syspin/vits-syspin-hi-female/tokens.txt \
  --vits-data-dir="" \
  --output-filename=./output.wav \
  "नमस्ते आप कैसे हैं"
```

## Credits & Licensing

- **Original Models & Checkpoints**: Developed by [SPIRE Lab, IISc Bengaluru](https://spire.ee.iisc.ac.in/) under the RESPIN and SYSPIN initiatives funded by Pratiksha Trust. Original checkpoints are available on Hugging Face at [huggingface.co/SYSPIN](https://huggingface.co/SYSPIN).
- **Inference Runtime**: Powered by [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0).
