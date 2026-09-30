# Sherpa-ONNX RESPIN / SYSPIN Indian Language Voices

Lightweight (~55MB) authentic FP16-weight offline TTS models for [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [SherpaVoices](https://github.com/animeshahilya/SherpaVoices), derived from the **RESPIN** and **SYSPIN** initiatives by **IISc Bengaluru (SPIRE Lab)**.

## 🌐 Live Interactive Voice Samples & Evaluation Dashboard

Listen to long-form, phonetically rich speech samples (~15–25s each) for all 22 voices across 11 Indian languages directly in your browser:

👉 **[Live Voice Samples & Audio Evaluation Dashboard](https://animeshahilya.github.io/sherpa-onnx-respin-syspin/)**

- **Audio Playback**: Full-length samples for all 22 voices (Male & Female) testing prosody, commas, and conjunct articulation.
- **Phonetically Balanced Texts**: Tested in native scripts (Devanagari, Bengali, Telugu, Kannada, Gujarati, Latin) with Romanized transliterations and English translations.
- **Direct Checkpoint Links**: Instant downloads for weight-only FP16 (~55 MB) checkpoints and vocabularies.

## Overview

The SPIRE Lab at the Indian Institute of Science (IISc Bengaluru) created the **RESPIN** (*REcognizing SPeech in INdian languages*) and **SYSPIN** (*SYnthesizing SPeech in INdian languages*) projects to support speech technology for diverse Indian languages and dialects.

These models standardize on **lightweight weight-only FP16 (~55 MB)** storage:
- **Graph compute stays Float32**: Only large Conv/MatMul initializers are cast to FP16 in storage, preserving full Float32 acoustic compute and natural phoneme durations on CPU.
- **Zero INT8 degradation**: Avoids robotic artifacts, duration shifting, and quantization noise.
- Pre-converted ONNX models are hosted under [GitHub Releases (v1.1.0-fp16)](https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/tag/v1.1.0-fp16).
- Each model is injected with `sherpa-onnx` metadata tags (`model_type=vits`, `comment=coqui`, `frontend=characters`, `sample_rate=22050`).

## Models in this Repository

| Language | Code | Gender | Model Identifier | Precision | Model Size |
|---|---|---|---|---|---|
| **Hindi** | `hi` | Female | `vits-syspin-hi-female` | Weight-only FP16 | ~55 MB |
| **Hindi** | `hi` | Male | `vits-syspin-hi-male` | Weight-only FP16 | ~55 MB |
| **Bengali** | `bn` | Female | `vits-syspin-bn-female` | Weight-only FP16 | ~55 MB |
| **Bengali** | `bn` | Male | `vits-syspin-bn-male` | Weight-only FP16 | ~55 MB |
| **Telugu** | `te` | Female | `vits-syspin-te-female` | Weight-only FP16 | ~55 MB |
| **Telugu** | `te` | Male | `vits-syspin-te-male` | Weight-only FP16 | ~55 MB |
| **Kannada** | `kn` | Female | `vits-syspin-kn-female` | Weight-only FP16 | ~55 MB |
| **Kannada** | `kn` | Male | `vits-syspin-kn-male` | Weight-only FP16 | ~55 MB |
| **Marathi** | `mr` | Female | `vits-syspin-mr-female` | Weight-only FP16 | ~55 MB |
| **Marathi** | `mr` | Male | `vits-syspin-mr-male` | Weight-only FP16 | ~55 MB |
| **Gujarati** | `gu` | Female | `vits-syspin-gu-female` | Weight-only FP16 | ~55 MB |
| **Gujarati** | `gu` | Male | `vits-syspin-gu-male` | Weight-only FP16 | ~55 MB |
| **Bhojpuri** | `bho` | Female | `vits-syspin-bho-female` | Weight-only FP16 | ~55 MB |
| **Bhojpuri** | `bho` | Male | `vits-syspin-bho-male` | Weight-only FP16 | ~55 MB |
| **Chhattisgarhi** | `hne` | Female | `vits-syspin-hne-female` | Weight-only FP16 | ~55 MB |
| **Chhattisgarhi** | `hne` | Male | `vits-syspin-hne-male` | Weight-only FP16 | ~55 MB |
| **Maithili** | `mai` | Female | `vits-syspin-mai-female` | Weight-only FP16 | ~55 MB |
| **Maithili** | `mai` | Male | `vits-syspin-mai-male` | Weight-only FP16 | ~55 MB |
| **Magahi** | `mag` | Female | `vits-syspin-mag-female` | Weight-only FP16 | ~55 MB |
| **Magahi** | `mag` | Male | `vits-syspin-mag-male` | Weight-only FP16 | ~55 MB |
| **English (India)** | `en` | Female | `vits-syspin-en-female` | Weight-only FP16 | ~55 MB |
| **English (India)** | `en` | Male | `vits-syspin-en-male` | Weight-only FP16 | ~55 MB |

## Using with SherpaVoices Android App

In [SherpaVoices](https://github.com/animeshahilya/SherpaVoices), these voices are defined with:
- `source`: `"github_release"`
- `repo`: `"animeshahilya/sherpa-onnx-respin-syspin"`
- `tag`: `"v1.1.0-fp16"`
- `pathPrefix`: `<model-identifier>`
- `rawTokensFile`: `true`

The app downloads `model.onnx` (~55MB) and `tokens.txt` directly from GitHub Releases CDN and caches them locally for fully offline synthesis on-device.

## Using with sherpa-onnx CLI

```bash
# Download a voice model and tokens directly from GitHub Releases (v1.1.0-fp16)
curl -LO https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v1.1.0-fp16/vits-syspin-hi-female-model.onnx
curl -LO https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v1.1.0-fp16/vits-syspin-hi-female-tokens.txt

# Synthesize with sherpa-onnx-offline-tts
sherpa-onnx-offline-tts \
  --vits-model=./vits-syspin-hi-female-model.onnx \
  --vits-tokens=./vits-syspin-hi-female-tokens.txt \
  --vits-data-dir="" \
  --output-filename=./output.wav \
  "नमस्ते आप कैसे हैं"
```

## Hindi text frontend (hindi_frontend.py)

Character VITS voices must guess the unwritten inherent schwa. This frontend
makes it explicit before tokenization (verified against all 116 hi-female
tokens — zero dropped characters on the test set):

- **Schwa deletion → explicit halant**: final schwa deletes iff the preceding
  vowel is full (`राम`→`राम्`, `देव`→`देव्`, `पास`→`पास्`; `कमल`, `मतलब`,
  `औरत`, `घर`, `वह`/`यह` keep it); medial schwa deletes only in categorical
  VCəCV with a light preceding rhyme, no CCC cluster, no glide
  (`कमरा`→`कम्रा`, `करना`→`कर्ना`, `जनता`→`जन्ता`; `लड़का`, `सहायता`,
  `नमस्ते`, `रुपये`, `दरवाज़ा` keep it). Retroflex-adjacent schwas are never
  touched.
- **Numerals** (ASCII + Devanagari) → Hindi words, Indian system
  (`125`→`एक सौ पच्चीस`, `2026`→`दो हज़ार छब्बीस`).
- **Punctuation**: `।`/`॥`→`.`, quotes/dashes mapped to the supported set,
  unsupported characters dropped (reported, not silent).

```bash
python hindi_frontend.py --text "राम कमरे में है। मेरे पास 125 रुपये हैं।" \
  --tokens vits-syspin-hi-female/tokens.txt --ids
```

Design rule: when in doubt the frontend does nothing — frequent words are
usually already right in the model, and a wrong halant is worse than none.
Rare/OOV words (the ELAICHI low-frequency-bigram cases) are where it helps.

## Credits & Licensing

- **Original Models & Checkpoints**: Developed by [SPIRE Lab, IISc Bengaluru](https://spire.ee.iisc.ac.in/) under the RESPIN and SYSPIN initiatives funded by Pratiksha Trust. Original checkpoints are available on Hugging Face at [huggingface.co/SYSPIN](https://huggingface.co/SYSPIN).
- **Inference Runtime**: Powered by [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0).
