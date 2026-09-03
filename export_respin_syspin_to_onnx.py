#!/usr/bin/env python3
"""Export IISc SPIRE Lab RESPIN / SYSPIN Coqui VITS models to sherpa-onnx ONNX format.

RESPIN (REcognizing SPeech in INdian languages) and SYSPIN (SYnthesizing SPeech
in INdian languages) are large-scale speech corpora for 9 Indian languages developed
by the SPIRE Lab at the Indian Institute of Science (IISc Bengaluru).

Under the `SYSPIN` organization on Hugging Face (huggingface.co/SYSPIN), SPIRE Lab
released Coqui VITS checkpoints (`best_model.pth` + `config.json`) and TorchScript
models for:
  - Bengali (bn) [Male & Female]
  - Bhojpuri (bho / bh) [Male & Female]
  - Chhattisgarhi (hne) [Male & Female]
  - Hindi (hi) [Male & Female]
  - Kannada (kn) [Male & Female]
  - Magahi (mag) [Male & Female]
  - Maithili (mai) [Male & Female]
  - Marathi (mr) [Male & Female]
  - Telugu (te) [Male & Female]
  - English (en) [Male & Female]

This script:
  1. Downloads checkpoint and configuration files from HuggingFace.
  2. Exports the model to ONNX using Coqui TTS or PyTorch.
  3. Injects sherpa-onnx required metadata properties (`model_type=vits`, `comment=coqui`,
     `frontend=characters`, language, sample_rate, blank_id, etc.).
  4. Generates the corresponding `tokens.txt` character mapping.
  5. Optionally uploads the exported assets to a Hugging Face repository.

Requirements:
  pip install torch onnx onnxruntime huggingface_hub soundfile
  pip install TTS   # Coqui TTS
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Supported RESPIN / SYSPIN models on HuggingFace
RESPIN_SYSPIN_MODELS = {
    # Bengali
    "vits-syspin-bn-female": {
        "repo": "SYSPIN/vits_Bengali_Female",
        "alt_repo": "SYSPIN/tts_vits_coquiai_BengaliFemale",
        "lang": "bn",
        "lang_iso3": "ben",
        "lang_name": "Bengali",
        "gender": "Female",
        "sample_rate": 22050,
    },
    "vits-syspin-bn-male": {
        "repo": "SYSPIN/vits_Bengali_Male",
        "alt_repo": "SYSPIN/tts_vits_coquiai_BengaliMale",
        "lang": "bn",
        "lang_iso3": "ben",
        "lang_name": "Bengali",
        "gender": "Male",
        "sample_rate": 22050,
    },
    # Bhojpuri
    "vits-syspin-bho-female": {
        "repo": "SYSPIN/vits_Bhojpuri_Female",
        "alt_repo": "SYSPIN/tts_vits_coquiai_BhojpuriFemale",
        "lang": "bho",
        "lang_iso3": "bho",
        "lang_name": "Bhojpuri",
        "gender": "Female",
        "sample_rate": 22050,
    },
    "vits-syspin-bho-male": {
        "repo": "SYSPIN/vits_Bhojpuri_Male",
        "alt_repo": "SYSPIN/tts_vits_coquiai_BhojpuriMale",
        "lang": "bho",
        "lang_iso3": "bho",
        "lang_name": "Bhojpuri",
        "gender": "Male",
        "sample_rate": 22050,
    },
    # Chhattisgarhi
    "vits-syspin-hne-female": {
        "repo": "SYSPIN/vits_Chhattisgarhi_Female",
        "alt_repo": "SYSPIN/tts_vits_coquiai_ChhattisgarhiFemale",
        "lang": "hne",
        "lang_iso3": "hne",
        "lang_name": "Chhattisgarhi",
        "gender": "Female",
        "sample_rate": 22050,
    },
    "vits-syspin-hne-male": {
        "repo": "SYSPIN/vits_Chhattisgarhi_Male",
        "alt_repo": "SYSPIN/tts_vits_coquiai_ChhattisgarhiMale",
        "lang": "hne",
        "lang_iso3": "hne",
        "lang_name": "Chhattisgarhi",
        "gender": "Male",
        "sample_rate": 22050,
    },
    # Hindi
    "vits-syspin-hi-female": {
        "repo": "SYSPIN/vits_Hindi_Female",
        "alt_repo": "SYSPIN/tts_vits_coquiai_HindiFemale",
        "lang": "hi",
        "lang_iso3": "hin",
        "lang_name": "Hindi",
        "gender": "Female",
        "sample_rate": 22050,
    },
    "vits-syspin-hi-male": {
        "repo": "SYSPIN/vits_Hindi_Male",
        "alt_repo": "SYSPIN/tts_vits_coquiai_HindiMale",
        "lang": "hi",
        "lang_iso3": "hin",
        "lang_name": "Hindi",
        "gender": "Male",
        "sample_rate": 22050,
    },
    # Kannada
    "vits-syspin-kn-female": {
        "repo": "SYSPIN/vits_Kannada_Female",
        "alt_repo": "SYSPIN/tts_vits_coquiai_KannadaFemale",
        "lang": "kn",
        "lang_iso3": "kan",
        "lang_name": "Kannada",
        "gender": "Female",
        "sample_rate": 22050,
    },
    "vits-syspin-kn-male": {
        "repo": "SYSPIN/vits_Kannada_Male",
        "alt_repo": "SYSPIN/tts_vits_coquiai_KannadaMale",
        "lang": "kn",
        "lang_iso3": "kan",
        "lang_name": "Kannada",
        "gender": "Male",
        "sample_rate": 22050,
    },
    # Magahi
    "vits-syspin-mag-female": {
        "repo": "SYSPIN/vits_Magahi_Female",
        "alt_repo": "SYSPIN/tts_vits_coquiai_MagahiFemale",
        "lang": "mag",
        "lang_iso3": "mag",
        "lang_name": "Magahi",
        "gender": "Female",
        "sample_rate": 22050,
    },
    "vits-syspin-mag-male": {
        "repo": "SYSPIN/vits_Magahi_Male",
        "alt_repo": "SYSPIN/tts_vits_coquiai_MagahiMale",
        "lang": "mag",
        "lang_iso3": "mag",
        "lang_name": "Magahi",
        "gender": "Male",
        "sample_rate": 22050,
    },
    # Maithili
    "vits-syspin-mai-female": {
        "repo": "SYSPIN/vits_Maithili_Female",
        "alt_repo": "SYSPIN/tts_vits_coquiai_MaithiliFemale",
        "lang": "mai",
        "lang_iso3": "mai",
        "lang_name": "Maithili",
        "gender": "Female",
        "sample_rate": 22050,
    },
    "vits-syspin-mai-male": {
        "repo": "SYSPIN/vits_Maithili_Male",
        "alt_repo": "SYSPIN/tts_vits_coquiai_MaithiliMale",
        "lang": "mai",
        "lang_iso3": "mai",
        "lang_name": "Maithili",
        "gender": "Male",
        "sample_rate": 22050,
    },
    # Marathi
    "vits-syspin-mr-female": {
        "repo": "SYSPIN/vits_Marathi_Female",
        "alt_repo": "SYSPIN/tts_vits_coquiai_MarathiFemale",
        "lang": "mr",
        "lang_iso3": "mar",
        "lang_name": "Marathi",
        "gender": "Female",
        "sample_rate": 22050,
    },
    "vits-syspin-mr-male": {
        "repo": "SYSPIN/vits_Marathi_Male",
        "alt_repo": "SYSPIN/tts_vits_coquiai_MarathiMale",
        "lang": "mr",
        "lang_iso3": "mar",
        "lang_name": "Marathi",
        "gender": "Male",
        "sample_rate": 22050,
    },
    # Telugu
    "vits-syspin-te-female": {
        "repo": "SYSPIN/vits_Telugu_Female",
        "alt_repo": "SYSPIN/tts_vits_coquiai_TeluguFemale",
        "lang": "te",
        "lang_iso3": "tel",
        "lang_name": "Telugu",
        "gender": "Female",
        "sample_rate": 22050,
    },
    "vits-syspin-te-male": {
        "repo": "SYSPIN/vits_Telugu_Male",
        "alt_repo": "SYSPIN/tts_vits_coquiai_TeluguMale",
        "lang": "te",
        "lang_iso3": "tel",
        "lang_name": "Telugu",
        "gender": "Male",
        "sample_rate": 22050,
    },
    # English
    "vits-syspin-en-female": {
        "repo": "SYSPIN/vits_English_Female",
        "alt_repo": "SYSPIN/tts_vits_coquiai_EnglishFemale",
        "lang": "en",
        "lang_iso3": "eng",
        "lang_name": "English",
        "gender": "Female",
        "sample_rate": 22050,
    },
    "vits-syspin-en-male": {
        "repo": "SYSPIN/vits_English_Male",
        "alt_repo": "SYSPIN/tts_vits_coquiai_EnglishMale",
        "lang": "en",
        "lang_iso3": "eng",
        "lang_name": "English",
        "gender": "Male",
        "sample_rate": 22050,
    },
    # Gujarati
    "vits-syspin-gu-female": {
        "repo": "SYSPIN/vits_Gujarati_Female",
        "alt_repo": "SYSPIN/tts_vits_coquiai_GujaratiFemale",
        "lang": "gu",
        "lang_iso3": "guj",
        "lang_name": "Gujarati",
        "gender": "Female",
        "sample_rate": 22050,
    },
    "vits-syspin-gu-male": {
        "repo": "SYSPIN/vits_Gujarati_Male",
        "alt_repo": "SYSPIN/tts_vits_coquiai_GujaratiMale",
        "lang": "gu",
        "lang_iso3": "guj",
        "lang_name": "Gujarati",
        "gender": "Male",
        "sample_rate": 22050,
    },
}


def add_metadata_to_onnx(filename: str, metadata: Dict[str, Any]) -> None:
    """Injects key-value metadata properties into an ONNX model file."""
    import onnx

    model = onnx.load(filename)
    # Clear existing metadata props with the same keys if any
    existing_keys = {prop.key for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing_keys:
            for p in model.metadata_props:
                if p.key == key:
                    p.value = str(value)
        else:
            meta = model.metadata_props.add()
            meta.key = key
            meta.value = str(value)

    onnx.save(model, filename)


def generate_tokens_file(tokenizer, output_path: str) -> None:
    """Generates sherpa-onnx tokens.txt from a Coqui TTS character tokenizer."""
    char_to_id = tokenizer.characters._char_to_id
    with open(output_path, "w", encoding="utf-8") as f:
        for char, idx in sorted(char_to_id.items(), key=lambda x: x[1]):
            f.write(f"{char} {idx}\n")


def export_single_model(
    model_key: str,
    output_dir: Path,
    hf_token: Optional[str] = None,
    quantize: bool = False,
    release_assets_dir: Optional[Path] = None,
) -> Path:
    """Exports one RESPIN / SYSPIN model to sherpa-onnx ONNX format."""
    info = RESPIN_SYSPIN_MODELS[model_key]
    out_path = output_dir / model_key
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[{model_key}] Processing {info['lang_name']} ({info['gender']}) from {info['repo']}...")

    if release_assets_dir:
        target_model = release_assets_dir / f"{model_key}-model.onnx"
        target_tokens = release_assets_dir / f"{model_key}-tokens.txt"
        if target_model.exists() and target_model.stat().st_size > 100_000_000 and target_tokens.exists():
            print(f"[{model_key}] Already exported to {target_model.name} ({target_model.stat().st_size/1024/1024:.1f} MB), skipping.")
            return out_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("Please install huggingface_hub: pip install huggingface_hub")

    # Download config.json and checkpoint
    config_file = hf_hub_download(
        repo_id=info["repo"],
        filename="config.json",
        local_dir=out_path,
        token=hf_token,
    )
    checkpoint_file = hf_hub_download(
        repo_id=info["repo"],
        filename="best_model.pth",
        local_dir=out_path,
        token=hf_token,
    )

    onnx_file = out_path / "model.onnx"
    tokens_file = out_path / "tokens.txt"

    # Export using Coqui TTS Vits
    try:
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits
    except ImportError:
        raise RuntimeError(
            "Please install Coqui TTS and onnx: pip install TTS onnx onnxruntime\n"
            "Note: On Python 3.11+, use: pip install coqui-tts onnx"
        )

    config = VitsConfig()
    config.load_json(config_file)

    print(f"[{model_key}] Initializing VITS model from config...")
    vits = Vits.init_from_config(config)
    print(f"[{model_key}] Loading weights from {checkpoint_file}...")
    vits.load_checkpoint(config, checkpoint_file)

    print(f"[{model_key}] Exporting ONNX to {onnx_file}...")
    vits.export_onnx(output_path=str(onnx_file), verbose=False)

    # Invert/write tokens.txt
    print(f"[{model_key}] Generating {tokens_file}...")
    generate_tokens_file(vits.tokenizer, str(tokens_file))

    # Add sherpa-onnx metadata
    print(f"[{model_key}] Adding sherpa-onnx metadata props to {onnx_file.name}...")
    metadata = {
        "model_type": "vits",
        "comment": "coqui",
        "language": info["lang_name"],
        "frontend": "characters",
        "add_blank": int(vits.config.add_blank),
        "blank_id": vits.tokenizer.characters.blank_id,
        "n_speakers": getattr(vits.config.model_args, "num_speakers", 1) or 1,
        "use_eos_bos": int(vits.tokenizer.use_eos_bos),
        "bos_id": vits.tokenizer.characters.bos_id,
        "eos_id": vits.tokenizer.characters.eos_id,
        "pad_id": vits.tokenizer.characters.pad_id,
        "sample_rate": int(getattr(vits.ap, "sample_rate", info["sample_rate"])),
    }
    add_metadata_to_onnx(str(onnx_file), metadata)

    # Optional dynamic INT8 quantization (default: False, full precision real voice)
    if quantize:
        print(f"[{model_key}] Quantizing ONNX model to INT8...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        temp_quant = out_path / "model.quant.onnx"
        quantize_dynamic(str(onnx_file), str(temp_quant), weight_type=QuantType.QUInt8)
        add_metadata_to_onnx(str(temp_quant), metadata)
        os.replace(str(temp_quant), str(onnx_file))

    # Stage flat assets for GitHub Releases if requested
    if release_assets_dir:
        import shutil
        release_assets_dir.mkdir(parents=True, exist_ok=True)
        target_model = release_assets_dir / f"{model_key}-model.onnx"
        target_tokens = release_assets_dir / f"{model_key}-tokens.txt"
        print(f"[{model_key}] Staging release asset: {target_model.name}...")
        shutil.copy2(str(onnx_file), str(target_model))
        shutil.copy2(str(tokens_file), str(target_tokens))

    # Clean up large training checkpoints and cache files
    try:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        if os.path.exists(config_file):
            os.remove(config_file)
        cache_dir = out_path / ".cache"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(str(cache_dir), ignore_errors=True)
    except Exception as e:
        print(f"[{model_key}] Warning cleaning temp files: {e}")

    mode_label = "quantized (INT8)" if quantize else "full-precision FP32 (real voice)"
    print(f"[{model_key}] Successfully exported {mode_label} to {out_path} ({os.path.getsize(onnx_file)/1024/1024:.1f} MB)!")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Export IISc SPIRE Lab RESPIN/SYSPIN Coqui VITS models to sherpa-onnx ONNX"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(RESPIN_SYSPIN_MODELS.keys()) + ["all"],
        default="all",
        help="Which model to export, or 'all' to export all models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to store exported ONNX models and tokens",
    )
    parser.add_argument(
        "--release-assets-dir",
        type=Path,
        default=None,
        help="Optional directory to stage flat release assets ({model}-model.onnx and {model}-tokens.txt)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Whether to dynamically quantize the model to INT8 (default: False, exports full-precision FP32 real voices)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (optional for public models)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported RESPIN/SYSPIN models and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Supported RESPIN / SYSPIN Models:")
        for k, v in sorted(RESPIN_SYSPIN_MODELS.items()):
            print(f"  - {k:<25}: {v['lang_name']} ({v['gender']}) | {v['repo']}")
        return

    models_to_export = (
        list(RESPIN_SYSPIN_MODELS.keys()) if args.model == "all" else [args.model]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for m in models_to_export:
        try:
            export_single_model(
                m,
                args.output_dir,
                hf_token=args.hf_token,
                quantize=args.quantize,
                release_assets_dir=args.release_assets_dir,
            )
        except Exception as e:
            print(f"Error exporting {m}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
