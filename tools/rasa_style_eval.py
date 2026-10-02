#!/usr/bin/env python3
"""Which Rasa speaking style is clearest, per speaker?

The shipped Rasa model freezes one style (ALEXA) for all 20 speakers. This
renders 3 sentences x 2 takes per speaker in each candidate style with the
unfrozen export (it still takes emotion_id), tokenized exactly as the
espeak-ng app tokenizes, and scores each take by Whisper character error rate
against the text. Indian scripts are folded together before scoring, since
Whisper sometimes writes Punjabi or Malayalam in Devanagari.

Self-contained: downloads the model (Hugging Face) and voice configs (this
repo's piper-v1 release). Writes results/rasa_styles.tsv and
results/rasa_styles.md (best style per speaker).

    pip install onnxruntime numpy scipy huggingface_hub faster-whisper
    python tools/rasa_style_eval.py
"""
import json, os, statistics, sys, unicodedata, urllib.request
from pathlib import Path

import numpy as np, onnxruntime as ort, scipy.signal as sg
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
CONFIGS = 'https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/piper-v1/'
SR = 24000
STYLES = {0: 'ALEXA', 3: 'BOOK', 9: 'INDICTTS', 10: 'NEWS', 16: 'WIKI'}
TAKES = 2

KN = ['ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ, ನಾವು ಸಂಜೆ ಉದ್ಯಾನವನಕ್ಕೆ ಹೋಗೋಣ.', 'ನಿಮ್ಮ ಫೋನ್‌ನಲ್ಲಿ ಹೊಸ ಸಂದೇಶ ಬಂದಿದೆ.', 'ದಯವಿಟ್ಟು ಬಾಗಿಲು ಮುಚ್ಚಿ ಮತ್ತು ದೀಪ ಆರಿಸಿ.']
TA = ['இன்று வானிலை மிகவும் நன்றாக உள்ளது, மாலையில் பூங்காவிற்கு செல்லலாம்.', 'உங்கள் தொலைபேசியில் புதிய செய்தி வந்துள்ளது.', 'தயவுசெய்து கதவை மூடி விளக்கை அணைக்கவும்.']
TE = ['ఈ రోజు వాతావరణం చాలా బాగుంది, సాయంత్రం పార్కుకు వెళ్దాం.', 'మీ ఫోన్‌లో కొత్త సందేశం వచ్చింది.', 'దయచేసి తలుపు మూసి లైట్ ఆపండి.']
BN = ['আজ আবহাওয়া খুব ভালো, চলো বিকেলে পার্কে যাই।', 'আপনার ফোনে একটি নতুন বার্তা এসেছে।', 'দয়া করে দরজা বন্ধ করে আলো নিভিয়ে দিন।']
MR = ['आज हवामान खूप छान आहे, आपण संध्याकाळी बागेत जाऊया.', 'तुमच्या फोनवर नवीन संदेश आला आहे.', 'कृपया दार बंद करा आणि दिवा बंद करा.']
PA = ['ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ, ਸ਼ਾਮ ਨੂੰ ਪਾਰਕ ਚੱਲੀਏ।', 'ਤੁਹਾਡੇ ਫ਼ੋਨ ਤੇ ਨਵਾਂ ਸੁਨੇਹਾ ਆਇਆ ਹੈ।', 'ਕਿਰਪਾ ਕਰਕੇ ਦਰਵਾਜ਼ਾ ਬੰਦ ਕਰੋ ਅਤੇ ਬੱਤੀ ਬੁਝਾਓ।']
ML = ['ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്, വൈകുന്നേരം പാർക്കിൽ പോകാം.', 'നിങ്ങളുടെ ഫോണിൽ പുതിയ സന്ദേശം വന്നിട്ടുണ്ട്.', 'ദയവായി വാതിൽ അടച്ച് വിളക്ക് അണയ്ക്കുക.']
NE = ['आज मौसम धेरै राम्रो छ, साँझ पार्कमा जाऔं।', 'तपाईंको फोनमा नयाँ सन्देश आएको छ।', 'कृपया ढोका बन्द गर्नुहोस् र बत्ती निभाउनुहोस्।']
BRX = ['नों माबोरै दं? दिनै मौसमआ मोजां।', 'नोंनि फोनाव गोदान खौरां फैदों।', 'अननानै दरजाखौ बन्द खालाम।']
DOI = ['तुस केह् हाल ओ? अज्ज मौसम खरा ऐ।', 'तुंदे फोन उप्पर नमां सनेहा आया ऐ।', 'किरपा करियै दरोआजा बंद करो।']
MAI = ['अहाँ केहन छी? आइ मौसम नीक अछि।', 'अहाँक फोन पर नव संदेश आएल अछि।', 'कृपया केबाड़ बंद करू।']
SA = ['नमस्कारः, भवान् कथम् अस्ति? अद्य वातावरणं शोभनम्।', 'भवतः दूरवाण्यां नूतनः सन्देशः आगतः।', 'कृपया द्वारं पिदधातु।']

# sid, voice config, Whisper language, sentences. Assamese (0, 1) is left
# out: Whisper does not read it (~100% error in every style), no signal.
CASES = [
    (2, 'bn_IN-tithi-medium', 'bn', BN), (3, 'bn_IN-anirban-medium', 'bn', BN),
    (4, 'brx_IN-mainao-medium', 'hi', BRX), (5, 'brx_IN-sansuma-medium', 'hi', BRX),
    (6, 'doi_IN-sheetal-medium', 'hi', DOI), (7, 'doi_IN-vijay-medium', 'hi', DOI),
    (8, 'kn_IN-spoorthi-medium', 'kn', KN), (9, 'kn_IN-chetan-medium', 'kn', KN),
    (10, 'mai_IN-shravan-medium', 'hi', MAI), (11, 'ml_IN-aparna-medium', 'ml', ML),
    (12, 'mr_IN-mrunal-medium', 'mr', MR), (13, 'mr_IN-tejas-medium', 'mr', MR),
    (14, 'ne_NP-prerana-medium', 'ne', NE), (15, 'pa_IN-simran-medium', 'pa', PA),
    (16, 'pa_IN-harpreet-medium', 'pa', PA), (17, 'sa_IN-vedant-medium', 'sa', SA),
    (18, 'ta_IN-kaveri-medium', 'ta', TA), (19, 'te_IN-harini-medium', 'te', TE),
]


def id_map(conf):
    with urllib.request.urlopen(CONFIGS + conf + '.onnx.json') as r:
        return json.loads(r.read().decode('utf-8'))['phoneme_id_map']


def ids(m, text):
    out = list(m['^'])
    for ch in unicodedata.normalize('NFC', text).lower():
        for t in ([ch] if ch in m else list(unicodedata.normalize('NFD', ch))):
            if t in m:
                out += m[t] + m['_']
    return out + m['$']


def norm(s):
    s = unicodedata.normalize('NFD', s)
    s = ''.join(chr(0x0900 + (ord(c) - 0x0900) % 0x80) if 0x0900 <= ord(c) <= 0x0DFF else c for c in s)
    return ''.join(c for c in s if c.isalnum() or unicodedata.category(c).startswith('M'))


def cer(ref, hyp):
    a, b = norm(ref), norm(hyp)
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)] / max(1, len(a))


def main():
    model = hf_hub_download('MatiasLin/sherpa-onnx-vits-rasa-13', 'model.onnx')
    sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
    assert 'emotion_id' in [i.name for i in sess.get_inputs()], 'need the unfrozen export'
    asr = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')
    rows, best = [], {}
    tsv = open(OUT / 'rasa_styles.tsv', 'w', encoding='utf-8')
    tsv.write('sid\tvoice\tstyle\tmean_cer\tworst_cer\n')
    for sid, conf, lang, texts in CASES:
        m = id_map(conf)
        means = {}
        for st, name in STYLES.items():
            scores = []
            for text in texts:
                for _ in range(TAKES):
                    x = ids(m, text)
                    y = sess.run(None, {
                        'x': np.array([x], np.int64), 'x_length': np.array([len(x)], np.int64),
                        'noise_scale': np.array(0.667, np.float32), 'length_scale': np.array(1.0, np.float32),
                        'noise_scale_w': np.array(0.8, np.float32), 'sid': np.array([sid], np.int64),
                        'emotion_id': np.array([st], np.int64)})[0].squeeze()
                    y = y / max(1e-6, float(abs(y).max())) * 0.9
                    z = sg.resample_poly(np.concatenate([np.zeros(SR // 2), y, np.zeros(SR // 2)]), 2, 3)
                    segs, _ = asr.transcribe(z.astype(np.float32), language=lang, beam_size=5)
                    scores.append(cer(text, ''.join(s.text for s in segs)))
            means[name] = statistics.mean(scores)
            tsv.write('%d\t%s\t%s\t%.4f\t%.4f\n' % (sid, conf, name, means[name], max(scores)))
            tsv.flush()
            print(conf, name, '%.1f%%' % (100 * means[name]), flush=True)
        winner = min(means, key=means.get)
        # Keep ALEXA unless another style is clearly better (2 points of CER).
        if means['ALEXA'] - means[winner] < 0.02:
            winner = 'ALEXA'
        best[sid] = winner
        rows.append('| %d | %s | %s | %s |' % (sid, conf, ' | '.join('%.1f%%' % (100 * means[n]) for n in STYLES.values()), winner))
    with open(OUT / 'rasa_styles.md', 'w', encoding='utf-8') as md:
        md.write('# Rasa speaking style per speaker\n\nMean Whisper character error rate, 3 sentences x %d takes. '
                 'Chosen: lowest, unless ALEXA (current) is within 2 points.\n\n' % TAKES)
        md.write('| sid | voice | %s | chosen |\n|%s\n' % (' | '.join(STYLES.values()), '---|' * (len(STYLES) + 3)))
        md.write('\n'.join(rows) + '\n\nSTYLE_BY_SPEAKER = %s\n' % json.dumps(
            [best.get(i, 'ALEXA') for i in range(20)]))
    print('done:', json.dumps({k: best[k] for k in sorted(best)}))


if __name__ == '__main__':
    sys.exit(main())
