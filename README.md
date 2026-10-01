# Sherpa-ONNX RESPIN / SYSPIN + Rasa Indian Language Voices

42 lightweight offline VITS voices — 22 SYSPIN (`v1.1.0-fp16`, 22050 Hz, ~55 MB each) + 20 Rasa (`v2.0.0-rasa-fp16`, 24000 Hz, one shared 59.5 MB file) — for [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [SherpaVoices](https://github.com/animeshahilya/SherpaVoices).

> 👉 **[Live demo: listen to all named voices](https://animeshahilya.github.io/sherpa-onnx-respin-syspin/)** — samples, transliterations, evaluation rubrics, and one-click model downloads.

## Named voices (no more “Voice 1 / Voice 2”)

| Language | Female voice | Male voice | Code |
|---|---|---|---|
| Hindi (हिन्दी) | **Kavya** | **Vihaan** | `hi` |
| Bengali (বাংলা) | **Riya** | **Sourav** | `bn` |
| Telugu (తెలుగు) | **Sireesha** | **Aditya** | `te` |
| Kannada (ಕನ್ನಡ) | **Ananya** | **Vikram** | `kn` |
| Marathi (मराठी) | **Sneha** | **Omkar** | `mr` |
| Gujarati (ગુજરાતી) | **Hetal** | **Jay** | `gu` |
| Bhojpuri (भोजपुरी) | **Kajal** | **Ranjit** | `bho` |
| Chhattisgarhi (छत्तीसगढ़ी) | **Mamta** | **Bhupesh** | `hne` |
| Maithili (मैथिली) | **Janaki** | **Mithilesh** | `mai` |
| Magahi (मगही) | **Poonam** | **Rakesh** | `mag` |
| English, India | **Priya** | **Rahul** | `en` |

Machine IDs stay stable (`vits-syspin-<lang>-<female|male>`, e.g. `vits-syspin-hi-female`). Display names live in [`voices.json`](./voices.json) with per-voice sample/model/tokens URLs and recommended inference config. Use `fullName` (“Kavya — Hindi Female”) in any UI. Rasa voices carry `engine: "rasa"` + `sid` (0–19) and run at 24000 Hz — see the Rasa section below.

## Rasa engine — 20 more voices, one 59.5 MB file (`v2.0.0-rasa-fp16`)

[AI4Bharat VITS](https://huggingface.co/ai4bharat/vits_rasa_13) (Rasa conversational data), converted via an ungated sherpa-onnx export + repo-side surgery: `emotion_id` frozen to neutral, inputs normalized to stock `{input, input_lengths, scales, sid}`, weight-only FP16. One shared `vits-rasa-13-model.onnx` serves all 20 speakers; per-language download cost is zero.

| sid | Voice | Language | Gender | New / Alternate |
|---|---|---|---|---|
| 18 | **Kaveri** | Tamil (தமிழ்) | Female | NEW |
| 11 | **Aparna** | Malayalam (മലയാളം) | Female | NEW |
| 15 | **Simran** | Punjabi (ਪੰਜਾਬੀ) | Female | NEW |
| 16 | **Harpreet** | Punjabi (ਪੰਜਾਬੀ) | Male | NEW |
| 0 | **Bornali** | Assamese (অসমীয়া) | Female | NEW |
| 1 | **Rituraj** | Assamese (অসমীয়া) | Male | NEW |
| 14 | **Prerana** | Nepali (नेपाली) | Female | NEW |
| 17 | **Vedant** | Sanskrit (संस्कृतम्) | Male | NEW |
| 4 | **Mainao** | Bodo (बड़ो) | Female | NEW |
| 5 | **Sansuma** | Bodo (बड़ो) | Male | NEW |
| 6 | **Sheetal** | Dogri (डोगरी) | Female | NEW |
| 7 | **Vijay** | Dogri (डोगरी) | Male | NEW |
| 2 | **Tithi** | Bengali | Female | Alternate |
| 3 | **Anirban** | Bengali | Male | Alternate |
| 8 | **Spoorthi** | Kannada | Female | Alternate |
| 9 | **Chetan** | Kannada | Male | Alternate |
| 10 | **Shravan** | Maithili | Male | Alternate |
| 12 | **Mrunal** | Marathi | Female | Alternate |
| 13 | **Tejas** | Marathi | Male | Alternate |
| 19 | **Harini** | Telugu | Female | Alternate |

```bash
curl -LO https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v2.0.0-rasa-fp16/vits-rasa-13-model.onnx
curl -LO https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v2.0.0-rasa-fp16/vits-rasa-13-tokens.txt

sherpa-onnx-offline-tts \
  --vits-model=./vits-rasa-13-model.onnx \
  --vits-tokens=./vits-rasa-13-tokens.txt \
  --sid=18 \
  --output-filename=./out.wav \
  "வணக்கம்"
```

Notes: `sid` selects the voice (Tamil Kaveri = 18); emotion is baked to neutral; strip trailing `.?!` (sherpa splits it into a noise-prone stub); page samples are short greeting previews (full passages coming). QA: 20/20 render, FP16 SNR 41.7 dB vs FP32, `onnx.checker` clean.

## Quickstart

```bash
# 1. Download one voice (~55 MB) + vocab
curl -LO https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v1.1.0-fp16/vits-syspin-hi-female-model.onnx
curl -LO https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v1.1.0-fp16/vits-syspin-hi-female-tokens.txt

# 2a. CLI
sherpa-onnx-offline-tts \
  --vits-model=./vits-syspin-hi-female-model.onnx \
  --vits-tokens=./vits-syspin-hi-female-tokens.txt \
  --output-filename=./out.wav \
  "नमस्ते आप कैसे हैं"

# 2b. Python
pip install sherpa-onnx soundfile
```

```python
import sherpa_onnx, soundfile as sf
cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model="./vits-syspin-hi-female-model.onnx",
            tokens="./vits-syspin-hi-female-tokens.txt",
            noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0),
        provider="cpu"),
)
tts = sherpa_onnx.OfflineTts(cfg)
a = tts.generate("नमस्ते आप कैसे हैं", sid=0, speed=1.0)
sf.write("out.wav", a.samples, tts.sample_rate)
```

**Recommended config (all 22 voices):** `noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0, sid=0, speed=1.0, provider=cpu, num_threads=2–4`, `sample_rate=22050`. Single-speaker files — always `sid=0`.

## SherpaVoices (Android, fully offline)

```json
{ "source": "github_release", "repo": "animeshahilya/sherpa-onnx-respin-syspin",
  "tag": "v1.1.0-fp16", "pathPrefix": "vits-syspin-hi-female",
  "rawTokensFile": true, "displayName": "Kavya — Hindi Female",
  "speaker": "Kavya", "language": "Hindi", "gender": "female" }
```

`rawTokensFile: true` is required (Coqui vocab IDs are non-contiguous — normal, not a bug).

## Why FP16 weights (and what NOT to do)

- **Current:** large Conv/MatMul initializers (≥1024 elems, ~99.8% of params) stored as FP16 + Cast to FP32 at load. Graph math stays FP32 → **~109 MB → ~55 MB** with natural durations. See `build_fp16_60mb.py`.
- **Don’t use dynamic INT8** for these voices — it shrinks to ~38 MB but audibly shifts durations / adds robotic artifacts.
- **Don’t do full-FP16 compute** — breaks onnxruntime CPU (Cast/Shape errors).
- Samples are 64 kbps mono MP3 with `preload="none"`; consider Opus 24 kbps for ~40% further saving (see Optimizations below).

## Config audit (all 22 voices pass)

- ONNX metadata (set in `export_respin_syspin_to_onnx.py`): `model_type=vits, comment=coqui, frontend=characters, language=<Name>, sample_rate=22050` + `add_blank/blank_id/n_speakers/use_eos_bos/bos_id/eos_id/pad_id`. Correct for sherpa-onnx character VITS.
- `tokens.txt`: every file has `<PAD> 0`, `<BLNK>` last, a space entry, and script coverage (halant + danda for Devanagari langs, language script chars, a–Z + punctuation for `en`). ID gaps (e.g. hi-female 116 lines, max id 121) are **expected** — pruned Coqui vocab, not corruption.
- Verified: Hindi/Marathi/Bhojpuri/Chhattisgarhi/Maithili/Magahi all contain Devanagari core + halant; Bengali/Telugu/Kannada/Gujarati contain their scripts; English contains a–Z.

If a voice sounds wrong, check in order: (1) `tokens.txt` paired with the right `model.onnx`, (2) `frontend=characters` (not espeak), (3) `sid=0`, (4) text in native script (Hindi schwa helper: `hindi_frontend.py`).

## Files

| File | Purpose |
|---|---|
| `index.html` | GitHub Pages demo (generated — don’t hand-edit; edit `build_dashboard.py`) |
| `build_dashboard.py` | Source of truth for page text + speaker names → regenerates `index.html` |
| `voices.json` | Machine-readable catalog: names, gender, lang, sample/model/tokens URLs, recommended config |
| `generate_samples.py` | Renders `samples/*.mp3` from release FP16 models (`noise_scale=0.667`, 64 kbps mono) |
| `build_fp16_60mb.py` | FP32 → weight-only FP16 converter (+ `--verify` ORT smoke test) |
| `export_respin_syspin_to_onnx.py` | HF Coqui checkpoint → sherpa-onnx ONNX + `tokens.txt` + metadata |
| `hindi_frontend.py` | Optional Hindi schwa/numeral/punctuation normalizer |
| `build_rasa_stock.py` | Rasa pipeline: download ungated export → stock-compat surgery → FP16 → verify |
| `export_rasa_to_onnx.py` | From-scratch Rasa exporter (needs gated HF access; normally not needed) |
| `samples/` | 22 × `.mp3` + `all_22_voices_showcase.mp3` |

Regenerate page: `python build_dashboard.py`.

## Optimizations worth doing next

1. **Samples → Opus 24k** (`ffmpeg -c:a libopus -b:a 24k`): same intelligibility, ~40% smaller than 64k MP3.
2. **Page:** audio `preload="none"` everywhere except active tab (done); add `voices.json` fetch instead of inlining if page exceeds ~100 KB; compile Tailwind instead of Play CDN for production.
3. **Inference:** keep `length_scale=1.0` default; expose 0.9–1.1 slider per voice; trim leading/trailing silence at synthesis time.
4. **Tokens:** keep `rawTokensFile=true`; never renumber IDs.

## Credits & license

- Voices/checkpoints: [SPIRE Lab, IISc Bengaluru](https://spire.ee.iisc.ac.in/) (RESPIN/SYSPIN, Pratiksha Trust), via [HuggingFace SYSPIN](https://huggingface.co/SYSPIN).
- Runtime: [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0). Conversion scripts here follow the repo’s existing license; original model weights follow their upstream terms — check each HF repo before commercial use.
