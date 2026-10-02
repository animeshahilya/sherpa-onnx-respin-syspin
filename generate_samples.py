import os
import subprocess
import time
import sherpa_onnx
import soundfile as sf
from build_dashboard import VOICES_DATA
from tts_synth import synthesize_voice

samples = {v["id"]: v["text"] for v in VOICES_DATA}

base_dir = os.path.join(os.path.dirname(__file__), "release_assets_fp16")
out_dir = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(base_dir, exist_ok=True)

langs = ["hi", "en", "bn", "te", "kn", "mr", "gu", "bho", "hne", "mai", "mag"]

print("Starting generation for all 11 languages (FP16)...")
for lang in langs:
    # Generate Female and Male
    for gender in ["female", "male"]:
        m_name = f"vits-syspin-{lang}-{gender}"
        m_path = os.path.join(base_dir, f"{m_name}-model.onnx")
        t_path = os.path.join(base_dir, f"{m_name}-tokens.txt")
        wav_path = os.path.join(out_dir, f"{m_name}.wav")
        mp3_path = os.path.join(out_dir, f"{m_name}.mp3")

        if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 10000:
            print(f"Skipping {m_name} (already generated)")
            continue

        if not os.path.exists(m_path):
            import urllib.request
            print(f"Downloading FP16 model: {m_name}...")
            url = f"https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v1.1.0-fp16/{m_name}-model.onnx"
            urllib.request.urlretrieve(url, m_path)
            tok_url = f"https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/v1.1.0-fp16/{m_name}-tokens.txt"
            urllib.request.urlretrieve(tok_url, t_path)

        t0 = time.time()
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=m_path, tokens=t_path, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0
                ),
                provider="cpu", num_threads=4
            )
        )
        tts = sherpa_onnx.OfflineTts(config)
        # No-retrain pipeline: frontend + per-sentence chunking + trim + norm
        audio = synthesize_voice(tts, sid=0, text=samples[lang], lang=lang,
                                 length_scale=1.0, sample_rate=tts.sample_rate)
        sf.write(wav_path, audio, tts.sample_rate)
        # Convert to lightweight MP3 (64kbps mono)
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-b:a", "64k", mp3_path], capture_output=True)
        # We can remove the raw wav to keep git repo very light, or keep both.
        # Let's keep wav or delete wav if mp3 is good. Keeping mp3 saves 80% disk/bandwidth!
        os.remove(wav_path)
        print(f"[{m_name}] Done: {os.path.getsize(mp3_path)/1024:.1f} KB in {time.time()-t0:.1f}s")

print("All 22 samples generated successfully!")
