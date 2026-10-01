#!/usr/bin/env python3
"""Generates index.html for sherpa-onnx-respin-syspin GitHub Pages dashboard.

Source of truth for speaker display names is voices.json.
Sample passages (text/transliteration/translation/rubric) live in VOICES_DATA below.
Run:  python build_dashboard.py
"""

import html
import json
import os
import urllib.parse
from generate_rasa_samples import RASA_TEXTS

BASE = os.path.dirname(os.path.abspath(__file__))
RELEASE_TAG = "v1.1.0-fp16"
RASA_TAG = "v2.0.0-rasa-fp16"
REPO = "animeshahilya/sherpa-onnx-respin-syspin"

# Rasa engine voices: (sid, sample mp3, display, lang, native, gender, alt?)
RASA_VOICES = [
    (0, "vits-rasa-asm-female", "Bornali", "Assamese", "অসমীয়া", "female", False),
    (1, "vits-rasa-asm-male", "Rituraj", "Assamese", "অসমীয়া", "male", False),
    (2, "vits-rasa-bn-female-alt", "Tithi", "Bengali", "বাংলা", "female", True),
    (3, "vits-rasa-bn-male-alt", "Anirban", "Bengali", "বাংলা", "male", True),
    (4, "vits-rasa-brx-female", "Mainao", "Bodo", "बड़ो", "female", False),
    (5, "vits-rasa-brx-male", "Sansuma", "Bodo", "बड़ो", "male", False),
    (6, "vits-rasa-doi-female", "Sheetal", "Dogri", "डोगरी", "female", False),
    (7, "vits-rasa-doi-male", "Vijay", "Dogri", "डोगरी", "male", False),
    (8, "vits-rasa-kn-female-alt", "Spoorthi", "Kannada", "ಕನ್ನಡ", "female", True),
    (9, "vits-rasa-kn-male-alt", "Chetan", "Kannada", "ಕನ್ನಡ", "male", True),
    (10, "vits-rasa-mai-male-alt", "Shravan", "Maithili", "मैथिली", "male", True),
    (11, "vits-rasa-mal-female", "Aparna", "Malayalam", "മലയാളം", "female", False),
    (12, "vits-rasa-mr-female-alt", "Mrunal", "Marathi", "मराठी", "female", True),
    (13, "vits-rasa-mr-male-alt", "Tejas", "Marathi", "मराठी", "male", True),
    (14, "vits-rasa-ne-female", "Prerana", "Nepali", "नेपाली", "female", False),
    (15, "vits-rasa-pan-female", "Simran", "Punjabi", "ਪੰਜਾਬੀ", "female", False),
    (16, "vits-rasa-pan-male", "Harpreet", "Punjabi", "ਪੰਜਾਬੀ", "male", False),
    (17, "vits-rasa-san-male", "Vedant", "Sanskrit", "संस्कृतम्", "male", False),
    (18, "vits-rasa-tam-female", "Kaveri", "Tamil", "தமிழ்", "female", False),
    (19, "vits-rasa-te-female-alt", "Harini", "Telugu", "తెలుగు", "female", True),
]

RASA_LANG_MAP = {
    "Assamese": "asm",
    "Bengali": "bn",
    "Bodo": "brx",
    "Dogri": "doi",
    "Kannada": "kn",
    "Maithili": "mai",
    "Malayalam": "mal",
    "Marathi": "mr",
    "Nepali": "ne",
    "Punjabi": "pan",
    "Sanskrit": "san",
    "Tamil": "tam",
    "Telugu": "te",
}


def build_rasa_cards():
    cards = []
    for sid, mp3, name, lang, native, gender, alt in RASA_VOICES:
        lang_key = RASA_LANG_MAP[lang]
        info = RASA_TEXTS[lang_key]
        text = info["text"]
        translit = info["transliteration"]
        translation = info["translation"]
        rubric_html = "".join(f'<li class="flex items-start gap-1.5"><span class="text-emerald-400 font-bold">•</span><span>{html.escape(r)}</span></li>' for r in info["rubric"])

        gc = "text-pink-400" if gender == "female" else "text-cyan-400"
        gbg = "bg-pink-500/10 text-pink-400 border border-pink-500/20" if gender == "female" else "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
        dot_color = "bg-pink-400" if gender == "female" else "bg-cyan-400"
        alt_badge = (' <span class="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">alternate</span>'
                     if alt else
                     ' <span class="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">new</span>')

        card_html = f'''<div class="rasa-card p-4 sm:p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-3.5 shadow-lg hover:border-slate-700/80 transition-colors" data-search="{name} {lang} {native} {gender} sid{sid}">
  <div class="flex items-start justify-between gap-2">
    <div>
      <div class="font-bold text-white text-base flex items-center gap-2">
        {name} <span class="text-xs font-semibold px-2 py-0.5 rounded {gbg}">{gender}</span>
        {alt_badge}
      </div>
      <div class="text-xs text-slate-400 mt-0.5">{lang} ({native}) · <span class="font-mono text-slate-500">sid={sid}</span> · 24 kHz</div>
    </div>
    <span class="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">sid={sid}</span>
  </div>

  <div class="space-y-1.5">
    <div class="text-[11px] text-slate-400 flex items-center justify-between">
      <span class="flex items-center gap-1.5">
        <span class="w-2 h-2 rounded-full {dot_color}"></span>
        <span>Synthesized Voice Sample (~17–27s)</span>
      </span>
      <a href="samples/{mp3}.mp3" download class="text-[11px] text-blue-400 hover:text-blue-300 underline font-medium">Download MP3</a>
    </div>
    <audio controls preload="none" class="w-full h-9 rounded-lg bg-slate-900 border border-slate-800">
      <source src="samples/{mp3}.mp3" type="audio/mpeg">
      No audio playback.
    </audio>
  </div>

  <div class="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800/80 space-y-1.5">
    <div class="text-[10px] uppercase font-mono tracking-wider text-slate-400 flex items-center justify-between">
      <span>Test Passage ({len(text)} chars)</span>
      <button data-copy="{urllib.parse.quote(text)}" onclick="copyText(this)" class="text-blue-400 hover:text-blue-300 font-sans text-xs font-semibold">Copy</button>
    </div>
    <div class="text-sm font-medium text-slate-100 leading-relaxed select-all">{html.escape(text)}</div>
  </div>

  <details class="text-xs text-slate-400 group">
    <summary class="cursor-pointer select-none text-slate-400 hover:text-slate-200 font-medium flex items-center justify-between py-1 text-[11px]">
      <span>Pronunciation, Meaning & Rubric</span>
      <span class="text-slate-500 group-open:rotate-180 transition-transform text-[10px]">▼</span>
    </summary>
    <div class="mt-2 space-y-2 pt-2 border-t border-slate-800/60 text-[11px]">
      <div>
        <div class="uppercase font-mono text-[10px] text-slate-500 mb-0.5">Transliteration:</div>
        <div class="italic text-slate-300 leading-relaxed">{html.escape(translit)}</div>
      </div>
      <div>
        <div class="uppercase font-mono text-[10px] text-slate-500 mb-0.5">English Meaning:</div>
        <div class="text-slate-400 leading-relaxed">{html.escape(translation)}</div>
      </div>
      <div>
        <div class="uppercase font-mono text-[10px] text-slate-500 mb-0.5">What to listen for:</div>
        <ul class="space-y-1 text-slate-400">{rubric_html}</ul>
      </div>
    </div>
  </details>

  <div class="flex items-center gap-2 pt-1 border-t border-slate-900 text-[11px]">
    <a class="text-blue-400 hover:text-blue-300 underline" href="samples/{mp3}.mp3" download>MP3 sample</a>
    <span class="text-slate-700">·</span>
    <a class="text-blue-400 hover:text-blue-300 underline" target="_blank" rel="noopener" href="https://github.com/{REPO}/releases/download/{RASA_TAG}/vits-rasa-13-model.onnx">FP16 ONNX 59.5MB</a>
    <span class="text-slate-700">·</span>
    <a class="text-blue-400 hover:text-blue-300 underline" target="_blank" rel="noopener" href="https://github.com/{REPO}/releases/download/{RASA_TAG}/vits-rasa-13-tokens.txt">tokens.txt</a>
  </div>
</div>'''
        cards.append(card_html)
    return "\n".join(cards)


def build_rasa_table_rows():
    rows = []
    for sid, mp3, name, lang, native, gender, alt in RASA_VOICES:
        g = "Female" if gender == "female" else "Male"
        tag = "alternate" if alt else "new"
        rows.append(
            f'<tr class="border-b border-slate-800/60"><td class="py-2 pr-3 font-semibold text-slate-200">{name} <span class="font-mono font-normal text-slate-500">vits-rasa-13 sid={sid} ({tag})</span></td>'
            f'<td class="py-2 pr-3">{lang} ({native})</td><td class="py-2 pr-3">{g}</td>'
            f'<td class="py-2 pr-3"><a class="text-blue-400 underline" href="samples/{mp3}.mp3">mp3</a></td>'
            f'<td class="py-2 pr-3"><a class="text-blue-400 underline" href="https://github.com/{REPO}/releases/download/{RASA_TAG}/vits-rasa-13-model.onnx">onnx 59.5MB shared</a></td>'
            f'<td class="py-2"><a class="text-blue-400 underline" href="https://github.com/{REPO}/releases/download/{RASA_TAG}/vits-rasa-13-tokens.txt">tokens</a></td></tr>'
        )
    return "\n".join(rows)

# Human-friendly speaker names (replaces bare "Voice 1 / Voice 2").
# Also mirrored in voices.json for app / API consumers.
SPEAKERS = {
    "vits-syspin-hi-female": "Kavya",
    "vits-syspin-hi-male": "Vihaan",
    "vits-syspin-bn-female": "Riya",
    "vits-syspin-bn-male": "Sourav",
    "vits-syspin-te-female": "Sireesha",
    "vits-syspin-te-male": "Aditya",
    "vits-syspin-kn-female": "Ananya",
    "vits-syspin-kn-male": "Vikram",
    "vits-syspin-mr-female": "Sneha",
    "vits-syspin-mr-male": "Omkar",
    "vits-syspin-gu-female": "Hetal",
    "vits-syspin-gu-male": "Jay",
    "vits-syspin-bho-female": "Kajal",
    "vits-syspin-bho-male": "Ranjit",
    "vits-syspin-hne-female": "Mamta",
    "vits-syspin-hne-male": "Bhupesh",
    "vits-syspin-mai-female": "Janaki",
    "vits-syspin-mai-male": "Mithilesh",
    "vits-syspin-mag-female": "Poonam",
    "vits-syspin-mag-male": "Rakesh",
    "vits-syspin-en-female": "Priya",
    "vits-syspin-en-male": "Rahul",
}

VOICES_DATA = [
    {
        "id": "hi",
        "name": "Hindi",
        "nativeName": "हिन्दी",
        "script": "Devanagari",
        "models": ["vits-syspin-hi-female", "vits-syspin-hi-male"],
        "text": "भारत एक विशाल और सुंदर देश है, जहां अनेक संस्कृतियों और भाषाओं का अनूठा संगम देखने को मिलता है. यहां के लोग अपनी परंपराओं, कला और इतिहास पर गर्व करते हैं. सुबह के समय ठंडी हवा और पक्षियों की चहचहाहट मन को शांति और ताजगी से भर देती है.",
        "transliteration": "Bharat ek vishal aur sundar desh hai, jahan anek sanskritiyon aur bhashaon ka anootha sangam dekhne ko milta hai. Yahan ke log apni paramparaon, kala aur itihas par garv karte hain. Subah ke samay thandi hawa aur pakshiyon ki chahchahahat man ko shanti aur tazgi se bhar deti hai.",
        "translation": "India is a vast and beautiful country where a unique confluence of diverse cultures and languages can be seen. The people here take immense pride in their traditions, arts, and history. In the morning hours, the cool breeze and the chirping of birds fill the heart with tranquil peace and freshness.",
        "rubric": [
            "Evaluate conjunct clusters (संस्कृतियों, पक्षियों, इतिहास).",
            "Listen for unwritten schwa cadence and natural sentence pauses ('.').",
            "Observe pitch transitions between male and female models."
        ]
    },
    {
        "id": "en",
        "name": "English (India)",
        "nativeName": "Indian English",
        "script": "Latin",
        "models": ["vits-syspin-en-female", "vits-syspin-en-male"],
        "text": "Technology continues to transform the way we communicate, learn, and collaborate across borders. Through dedicated research and open source innovation, high quality speech synthesis is now accessible on everyday devices. Listening to natural prosody and balanced articulation makes digital interactions feel truly seamless.",
        "transliteration": "Technology continues to transform the way we communicate, learn, and collaborate across borders. Through dedicated research and open source innovation, high quality speech synthesis is now accessible on everyday devices. Listening to natural prosody and balanced articulation makes digital interactions feel truly seamless.",
        "translation": "Long-form narrative testing multi-syllabic vocabulary, professional tone, comma breathing pauses, and characteristic Indian English rhotic/alveolar articulation.",
        "rubric": [
            "Check rhythm and timing over polysyllabic words ('collaborate', 'accessible').",
            "Evaluate naturalness of pauses at commas and sentence stops.",
            "Examine acoustic warmth and absence of robotic artifacts."
        ]
    },
    {
        "id": "bn",
        "name": "Bengali",
        "nativeName": "বাংলা",
        "script": "Bengali",
        "models": ["vits-syspin-bn-female", "vits-syspin-bn-male"],
        "text": "আমাদের এই সুন্দর পৃথিবীতে প্রকৃতির অপরূপ রূপ ছড়িয়ে রয়েছে চারিপাশে. সকালের মিষ্টি রোদ আর পাখির কলকাকলি মনকে অনাবিল শান্তিতে ভরিয়ে তোলে. নদী, পাহাড় আর সবুজ মাঠের মেলবন্ধন যেন এক জীবন্ত শিল্পকর্ম.",
        "transliteration": "Amader ei sundor prithibite prokritir oporup rup chhoriye royechhe charipashe. Sokaler mishti rod aar pakhir kolokakoli monke onabil shantite bhoriye tole. Nodi, pahar aar sobuj mather melbandhon jeno ek jibonto shilpokormo.",
        "translation": "In our wonderful world, nature's sublime beauty is spread all around us. The sweet morning sunshine and the melodic twittering of birds fill the soul with pure serenity. The harmony of rivers, hills, and lush green fields resembles a vibrant, living work of art.",
        "rubric": [
            "Verify authentic Bengali rounded vowel coloring (অ/ও-কার).",
            "Test complex conjuncts (পৃথিবীতে, শান্তিতে, শিল্পকর্ম).",
            "Listen for gentle, lyrical sentence cadence."
        ]
    },
    {
        "id": "te",
        "name": "Telugu",
        "nativeName": "తెలుగు",
        "script": "Telugu",
        "models": ["vits-syspin-te-female", "vits-syspin-te-male"],
        "text": "మన భారతదేశం ఎంతో వైవిధ్యభరితమైన సంస్కృతి మరియు చరిత్ర కలిగిన అద్భుతమైన దేశం. ప్రతి ఉదయం సూర్యుని లేలేత కిరణాలు ప్రకృతిని ఎంతో అందంగా తీర్చిదిద్దుతాయి. మానవ జీవనంలో శాంతి, సహనం మరియు పట్టుదల అత్యంత ముఖ్యమైన గుణాలు.",
        "transliteration": "Mana Bharatadesham entho vaividhyabharitamaina sanskruthi mariyu charithra kaligina adbhutamaina desham. Prathi udayam sooryuni leletha kiranaalu prakruthini entho andamgaa theerchididduthaayi. Maanava jeevanamlo shaanthi, sahanam mariyu pattudala atyantha mukhyamaina gunaalu.",
        "translation": "Our India is a wonderful country endowed with an immensely diverse culture and grand history. Every morning, the gentle rays of the sun sculpt nature with breathtaking beauty. In human life, peace, patience, and perseverance are the most vital virtues.",
        "rubric": [
            "Test characteristic Ajanta (vowel-ending) syllable flow.",
            "Check retroflex consonant articulation (ణ, ళ, ట).",
            "Evaluate prosody over complex compound words."
        ]
    },
    {
        "id": "kn",
        "name": "Kannada",
        "nativeName": "ಕನ್ನಡ",
        "script": "Kannada",
        "models": ["vits-syspin-kn-female", "vits-syspin-kn-male"],
        "text": "ನಮ್ಮ ನಾಡು ಅತ್ಯಂತ ಶ್ರೀಮಂತ ಸಾಂಸ್ಕೃತಿಕ ಪರಂಪರೆ ಮತ್ತು ಸುಂದರ ಪ್ರಾಕೃತಿಕ ಸೌಂದರ್ಯವನ್ನು ಹೊಂದಿದೆ. ಬೆಳಗಿನ ತಂಗಾಳಿ ಮತ್ತು ಪಕ್ಷಿಗಳ ಮಧುರ ಧ್ವನಿ ಮನಸ್ಸಿಗೆ ಹೊಸ ಚೈತನ್ಯ ಮತ್ತು ನೆಮ್ಮದಿಯನ್ನು ನೀಡುತ್ತದೆ. ನಾವೆಲ್ಲರೂ ಪರಸ್ಪರ ಪ್ರೀತಿ ಮತ್ತು ಸೌಹಾರ್ದತೆಯಿಂದ ಬಾಳೋಣ.",
        "transliteration": "Namma naadu atyanta shreemanta saamskrutika paramapare mattu sundara praakrutika soundaryavannu hondide. Belagina tangaali mattu pakshigala madhura dhwani manassige hosa chaitanya mattu nemmadiyannu needuttade. Naavellaroo paraspara preeti mattu souhaardateyinda baalona.",
        "translation": "Our land possesses an extraordinarily rich cultural heritage and captivating natural splendor. The gentle morning breeze and sweet songs of birds bring renewed vitality and peaceful solace to the mind. Let us all live together in mutual love and harmony.",
        "rubric": [
            "Evaluate gemination (ದ್ವಿತ್ವ: ನಮ್ಮ, ಮನಸ್ಸಿಗೆ, ನೀಡುತ್ತದೆ).",
            "Check melodic rise-and-fall contour typical of Kannada speech.",
            "Listen for vowel elongation clarity."
        ]
    },
    {
        "id": "mr",
        "name": "Marathi",
        "nativeName": "मराठी",
        "script": "Devanagari",
        "models": ["vits-syspin-mr-female", "vits-syspin-mr-male"],
        "text": "महाराष्ट्र ही संतांची आणि शूरवीरांची पावन भूमी आहे, जिथे संस्कृती आणि निसर्गाचा सुंदर संगम पाहायला मिळतो. सकाळी उगवणारा सूर्य आणि मंद वाहणारा वारा मनाला नवी ऊर्जा देतो. जीवनात सतत प्रयत्नशील राहणे आणि सकारात्मक विचार करणे हीच खरी प्रगतीची गुरुकिल्ली आहे.",
        "transliteration": "Maharashtra hee santanchi aani shoorveeranchi paavan bhoomi aahe, jithe sanskruti aani nisargacha sundar sangam paahayla milto. Sakaali ugavnara soorya aani mand vaahnara vaara manala navi oorja deto. Jeevanat satat prayatnasheel raahne aani sakaratmak vichar karne heech khari pragatichi gurukilli aahe.",
        "translation": "Maharashtra is the sacred land of saints and valorous heroes, where a magnificent harmony of culture and nature is witnessed. The rising morning sun and the gentle breeze instill fresh energy into the soul. Striving continuously and maintaining positive thought is the true key to progress in life.",
        "rubric": [
            "Assess retroflex consonants and typical Marathi nasalizations (संतांची).",
            "Check rhythm in long compound sentences with relative clauses (जिथे, हीच).",
            "Evaluate tone authority in male vs. female speakers."
        ]
    },
    {
        "id": "gu",
        "name": "Gujarati",
        "nativeName": "ગુજરાતી",
        "script": "Gujarati",
        "models": ["vits-syspin-gu-female", "vits-syspin-gu-male"],
        "text": "આપણો દેશ વિવિધ સંસ્કૃતિઓ, તહેવારો અને સુંદર પરંપરાઓનો સંગમ છે. વહેલી સવારનો શીતળ પવન અને પક્ષીઓનો મીઠો કલરવ મનને શાંતિ આપે છે. જીવનમાં મહેનત અને સદ્ભાવના હંમેશાં આગળ વધવાની સાચી દિશા બતાવે છે.",
        "transliteration": "Aapno desh vividh sanskrutio, tahevaro ane sundar paramparaono sangam chhe. Vaheli savaarno sheetal pavan ane pakshiono meetho kalrav manne shaanti aape chhe. Jeevanma mahenat ane sadbhavna hamesha aagad vadhavani saachi disha bataave chhe.",
        "translation": "Our country is a grand confluence of diverse cultures, vibrant festivals, and cherished traditions. The cool breeze of early dawn and the sweet chirping of birds bring serene calmness to the mind. Hard work and goodwill in life always illuminate the true path forward.",
        "rubric": [
            "Test characteristic Gujarati vowel distinctions (મુક્ત સ્વર).",
            "Check conjunct rendering (સંસ્કૃતિઓ, સદ્ભાવના).",
            "Observe warm, gentle conversational inflection."
        ]
    },
    {
        "id": "bho",
        "name": "Bhojpuri",
        "nativeName": "भोजपुरी",
        "script": "Devanagari",
        "models": ["vits-syspin-bho-female", "vits-syspin-bho-male"],
        "text": "हमार गांव के सुबह बहुत सुहावन होला, जब पूरब से सुरुज देव के किरण फूटेला. खेतन में हरियर फसल लहराए लागेला आ चिड़ियन के चहचहाहट से पूरा माहौल गूंज उठेेला. सब लोग मिलजुल के आपन काम करे निकल पड़ेला.",
        "transliteration": "Hamaar gaanv ke subah bahut suhaawan hola, jab poorab se suruj dev ke kiran phootela. Khetan mein hariyar phasal lahraaye laagela aa chiriyan ke chahchahaahat se poora maahol goonj uthela. Sab log miljool ke aapan kaam kare nikal padela.",
        "translation": "The morning in our village is immensely pleasant when the first rays of the Sun God burst from the east. The lush green crops begin swaying gently in the fields and the chirping of birds makes the entire air resonate. Everyone heads out together in good spirits to begin their daily work.",
        "rubric": [
            "Evaluate authentic regional verbal markers (होला, फूटेला, लागेला, पड़ेला).",
            "Listen for rustic warmth and natural village narrative prosody.",
            "Check nasal vowels (गांव, खेतन)."
        ]
    },
    {
        "id": "hne",
        "name": "Chhattisgarhi",
        "nativeName": "छत्तीसगढ़ी",
        "script": "Devanagari",
        "models": ["vits-syspin-hne-female", "vits-syspin-hne-male"],
        "text": "हमार छत्तीसगढ़ महतारी के पावन भुइयां म चारों कोति हरियर रुख-राई अउ सुग्घर पहाड़ हे. बिहनिया के बेरा म जब चिरई-चिरगुन मन चहचहाथे, त मन म भारी आनंद भर जाथे. सब झन मया-पिरीत के संग मिलजुल के जिनगी बितोथे.",
        "transliteration": "Hamaar Chhattisgadh mahtaari ke paavan bhuiyaan ma chaaro koti hariyar rukh-raai au sugghar pahaad he. Bihaniya ke bera ma jab chirai-chirgun man chahchahaathe, ta man ma bhaari aanand bhar jaathe. Sab jhan maya-pirit ke sang miljool ke jingi bitothe.",
        "translation": "On the sacred soil of our Mother Chhattisgarh, lush green trees and scenic mountains stretch in all directions. In the quiet dawn, when the little birds start chirping, the heart fills with deep joy. Everyone leads their lives harmoniously together with mutual love and affection.",
        "rubric": [
            "Evaluate dialectal vocabulary (रुख-राई, सुग्घर, बिहनिया, मया-पिरीत).",
            "Check postpositions (म, हे) and verb endings (चहचहाथे, जाथे, बितोथे).",
            "Observe gentle, musical eastern-central cadence."
        ]
    },
    {
        "id": "mai",
        "name": "Maithili",
        "nativeName": "मैथिली",
        "script": "Devanagari",
        "models": ["vits-syspin-mai-female", "vits-syspin-mai-male"],
        "text": "हमर मिथिलाक पावन धरती पर संस्कृति, विद्या आ कलाक अनुपम संगम देखय लेल भेटैत अछि. प्रात:कालक शीतल बयार आ मन्दिरक शंखनाद सं सम्पूर्ण वातावरण पवित्र भऽ जाइत अछि. अपन भाषा आ मातृभूमिक सेवा करब हमर परम कर्तव्य अछि.",
        "transliteration": "Hamar Mithilaak paavan dhartee par sanskriti, vidya aa kalaak anupam sangam dekhay lel bhetait achhi. Praatahkaalak sheetal bayaar aa mandirak shankhnaad san sampoorna vaataavaran pavitra bha jaait achhi. Apan bhaasha aa maatribhoomik seva karab hamar param kartavya achhi.",
        "translation": "Upon the sacred soil of Mithila, a peerless confluence of culture, learning, and fine arts is witnessed. The cool dawn breeze and the sounding of temple conches make the entire atmosphere deeply sanctified. Serving our mother tongue and homeland is our highest duty.",
        "rubric": [
            "Check genitive nominal affix -क (मिथिलाक, कलाक, प्रात:कालक).",
            "Evaluate verbal copulas (भेटैत अछि, भऽ जाइत अछि).",
            "Listen for classic, scholarly Mithila diction."
        ]
    },
    {
        "id": "mag",
        "name": "Magahi",
        "nativeName": "मगही",
        "script": "Devanagari",
        "models": ["vits-syspin-mag-female", "vits-syspin-mag-male"],
        "text": "मगध के ई ऐतिहासिक धरती पर ज्ञान आ तपस्या के बड़ा महत्व रहल हई. सबेरे-सबेरे जब सुरुज भगवान निकले हथ, त गांवे-घर के लोग आपन-आपन काम में लग जा हथ. आपस में प्रेम आ सद्भाव से रहे में ही जिनगी के असली सुख मिलो हई.",
        "transliteration": "Magadh ke ee aitihasik dhartee par gyaan aa tapasya ke bada mahatva rahal hayi. Sabere-sabere jab suruj bhagwaan nikle hath, ta gaanve-ghar ke log aapan-aapan kaam mein lag ja hath. Aapas mein prem aa sadbhaav se rahe mein hee jingi ke asli sukh milo hayi.",
        "translation": "On this historic soil of Magadha, knowledge and dedication have always held immense importance. In the early morning as the Sun God ascends, people across every household set out to their chores. Living together in peace and goodwill is the true joy of life.",
        "rubric": [
            "Assess Magahi honorific verbal agreement (निकले हथ, लग जा हथ).",
            "Listen for regional markers (हई, मिलो हई).",
            "Evaluate lively colloquial flow."
        ]
    }
]

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SYSPIN / RESPIN Indian Voices — 22 Offline TTS Voices for sherpa-onnx</title>
  <meta name="description" content="42 lightweight offline VITS voices (22 SYSPIN + 20 Rasa, 19 language entries) for sherpa-onnx. Named voices, FP16, listen and download.">
  <meta property="og:title" content="SYSPIN / RESPIN Indian Voices — Listen & Download">
  <meta property="og:description" content="Hindi, Bengali, Telugu, Kannada, Marathi, Gujarati, Bhojpuri, Chhattisgarhi, Maithili, Magahi + Indian English. Named voices, 22050 Hz, ~55 MB FP16 each.">
  <meta property="og:type" content="website">
  <meta name="theme-color" content="#020617">
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎙️</text></svg>">
  <link rel="preconnect" href="https://cdn.tailwindcss.com">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @keyframes pulse-slow { 0%,100% {opacity:1} 50% {opacity:.4} }
    .pulse-indicator { animation: pulse-slow 2s cubic-bezier(.4,0,.6,1) infinite; }
    :focus-visible { outline: 2px solid #60a5fa; outline-offset: 2px; }
    audio::-webkit-media-controls-panel { background: #020617; }
  </style>
</head>
"""

BODY_TOP = """<body class="bg-slate-950 text-slate-100 antialiased p-4 md:p-8 min-h-screen">
  <a href="#voices" class="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:bg-blue-600 focus:text-white focus:px-3 focus:py-1 focus:rounded-lg focus:text-xs">Skip to voices</a>
  <div class="max-w-6xl mx-auto space-y-6">

    <header class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div class="space-y-2">
          <div class="flex flex-wrap items-center gap-2">
            <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">42 named voices</span>
            <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20">19 language entries &bull; 2 engines</span>
            <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-purple-500/10 text-purple-400 border border-purple-500/20">sherpa-onnx VITS &bull; 22050 Hz</span>
          </div>
          <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight text-white">SYSPIN / RESPIN Indian Voices</h1>
          <p class="text-sm text-slate-400 max-w-3xl">Lightweight offline TTS from IISc Bengaluru SPIRE Lab, converted to <span class="text-slate-200 font-medium">weight-only FP16 (~55 MB each)</span> for <a class="text-blue-400 underline" href="https://github.com/k2-fsa/sherpa-onnx">sherpa-onnx</a> and <a class="text-blue-400 underline" href="https://github.com/animeshahilya/SherpaVoices">SherpaVoices</a>. Each voice now has a proper name — e.g. <span class="text-slate-200">Kavya (Hindi Female)</span>, <span class="text-slate-200">Vihaan (Hindi Male)</span> — instead of Voice 1 / Voice 2.</p>
          <p class="text-xs text-slate-500">Source repo: <a href="https://github.com/REPO" class="font-mono text-blue-400 underline">REPO</a> &middot; Release <span class="font-mono">TAG</span> &middot; <a href="#about" class="underline">About this project</a> &middot; <a href="#all-voices" class="underline">All-voices table</a></p>
        </div>
        <div class="flex flex-wrap gap-2 text-xs">
          <div class="px-3 py-2 rounded-xl bg-slate-800/80 border border-slate-700"><div class="text-slate-400">Precision</div><div class="font-bold text-slate-200">FP16 weights, FP32 compute</div></div>
          <div class="px-3 py-2 rounded-xl bg-slate-800/80 border border-slate-700"><div class="text-slate-400">Runtime</div><div class="font-bold text-slate-200">Fully offline on CPU</div></div>
        </div>
      </div>
      <div class="mt-4 flex flex-col sm:flex-row gap-2">
        <label for="voiceSearch" class="sr-only">Search voices or languages</label>
        <input id="voiceSearch" type="search" placeholder="Search: e.g. hindi, Kavya, telugu, female…" autocomplete="off" class="w-full px-4 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-sm placeholder:text-slate-600 focus:border-blue-500">
      </div>
    </header>

    <section aria-label="Master showcase" class="bg-gradient-to-r from-blue-950/60 via-slate-900 to-indigo-950/60 border border-blue-800/40 rounded-2xl p-5 shadow-xl space-y-3">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div class="space-y-1">
          <div class="flex items-center gap-2">
            <span class="px-2 py-0.5 rounded text-[11px] font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30">🎙️ Master showcase — all 22 voices</span>
            <span class="text-xs text-slate-400">~3m 48s &middot; one continuous stream</span>
          </div>
          <p class="text-xs text-slate-300">Each speaker introduces themselves in their native script for sherpa-onnx offline synthesis. One file, quick comparison.</p>
        </div>
        <a href="samples/all_22_voices_showcase.mp3" download class="px-3.5 py-1.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white flex items-center gap-1.5 whitespace-nowrap">Download master MP3 (3.5 MB)</a>
      </div>
      <audio controls preload="none" class="w-full h-10 rounded-xl bg-slate-950 border border-slate-800"><source src="samples/all_22_voices_showcase.mp3" type="audio/mpeg">Your browser does not support audio playback.</audio>
    </section>

    <nav aria-label="Languages"><div id="langTabs" role="tablist" aria-label="Choose language" class="flex items-center gap-2 overflow-x-auto pb-2 border-b border-slate-800"></div></nav>
    <main id="voices"><div id="activeVoiceContainer" class="space-y-6" role="tabpanel" aria-live="polite"></div></main>

    <section id="rasa" class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <h2 class="text-lg font-bold text-white">Rasa engine — 20 more voices, one 59.5 MB file</h2>
          <p class="text-xs text-slate-400 mt-1">AI4Bharat VITS (multi-speaker, <span class="font-mono">sid</span> 0–19) with stock sherpa-onnx inputs, emotion frozen to neutral, weight-only FP16. Tamil, Malayalam, Punjabi, Assamese, Nepali, Sanskrit, Bodo, Dogri — plus alternate Bengali, Kannada, Maithili, Marathi, Telugu voices. 24 kHz. Full-length conversational test passages (~17–27s each) with native scripts, Romanized transliterations, English meanings, and phonetic rubrics (sentence-synthesized with natural breathing pauses). Release <span class="font-mono">v2.0.0-rasa-fp16</span>.</p>
        </div>
        <div class="flex gap-2 text-xs font-mono whitespace-nowrap">
          <a class="px-3 py-2 rounded-lg bg-blue-600/10 border border-blue-500/30 text-blue-300" target="_blank" rel="noopener" href="https://github.com/REPO/releases/download/RASATAG/vits-rasa-13-model.onnx">model.onnx 59.5MB</a>
          <a class="px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-300" target="_blank" rel="noopener" href="https://github.com/REPO/releases/download/RASATAG/vits-rasa-13-tokens.txt">tokens.txt</a>
        </div>
      </div>
      <div id="rasaGrid" class="grid grid-cols-1 md:grid-cols-2 gap-3">
__RASA_CARDS__
      </div>
      <p class="text-[11px] text-slate-500">Tip: search above filters these cards too. CLI needs <code class="font-mono">--sid=&lt;id&gt;</code> (e.g. Tamil Kaveri is <code class="font-mono">sid=18</code>). Full sid table in <code class="font-mono">voices.json</code> and the README.</p>
    </section>
""".replace("REPO", REPO).replace("TAG", RELEASE_TAG).replace("RASATAG", RASA_TAG)

BODY_MID = """
    <section id="about" class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
      <h2 class="text-lg font-bold text-white">About this project</h2>
      <div class="grid md:grid-cols-2 gap-4 text-sm text-slate-300 leading-relaxed">
        <div class="space-y-2">
          <p><strong class="text-white">What it is.</strong> RESPIN (REcognizing SPeech in INdian languages) and SYSPIN (SYnthesizing SPeech in INdian languages) are open speech corpora by <a class="text-blue-400 underline" href="https://spire.ee.iisc.ac.in/">SPIRE Lab, IISc Bengaluru</a>. This repo converts their Coqui VITS checkpoints to sherpa-onnx ONNX + <code class="font-mono text-xs">tokens.txt</code>.</p>
          <p><strong class="text-white">Why FP16 weights.</strong> Only large Conv/MatMul initializers (&ge;1024 elems) are stored as FP16 with a Cast back to FP32, so graph compute stays FP32 on CPU. ~109 MB &rarr; ~55 MB with no INT8 duration shift or robotic artifacts.</p>
        </div>
        <div class="space-y-2">
          <p><strong class="text-white">How to use.</strong> Download <code class="font-mono text-xs">*-model.onnx</code> + <code class="font-mono text-xs">*-tokens.txt</code> from <a class="text-blue-400 underline" href="https://github.com/REPO/releases/tag/TAG">Release TAG</a>. CLI: <code class="font-mono text-xs">sherpa-onnx-offline-tts --vits-model=… --vits-tokens=…</code>. Python: <code class="font-mono text-xs">pip install sherpa-onnx soundfile</code>. Android: SherpaVoices with <code class="font-mono text-xs">source=github_release, repo=REPO, tag=TAG, rawTokensFile=true</code>.</p>
          <p><strong class="text-white">Recommended config.</strong> <code class="font-mono text-xs">noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0, sid=0, speed=1.0, provider=cpu</code>. Single-speaker per file, so keep <code class="font-mono text-xs">sid=0</code>. See <code class="font-mono text-xs">voices.json</code> for per-voice names, links, and defaults.</p>
        </div>
      </div>
    </section>

    <section class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
      <h2 class="text-lg font-bold text-white">Synthesize locally</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
        <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div class="font-bold text-slate-200 font-sans text-sm">CLI</div>
          <pre class="overflow-x-auto p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-300 leading-relaxed"># 1. Download (FP16 ~55 MB each)
curl -LO https://github.com/REPO/releases/download/TAG/vits-syspin-hi-female-model.onnx
curl -LO https://github.com/REPO/releases/download/TAG/vits-syspin-hi-female-tokens.txt

# 2. Synthesize
sherpa-onnx-offline-tts --vits-model=./vits-syspin-hi-female-model.onnx --vits-tokens=./vits-syspin-hi-female-tokens.txt --output-filename=./out.wav "नमस्ते आप कैसे हैं"</pre>
        </div>
        <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div class="font-bold text-slate-200 font-sans text-sm">Python</div>
          <pre class="overflow-x-auto p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-300 leading-relaxed">import sherpa_onnx, soundfile as sf
cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
  vits=sherpa_onnx.OfflineTtsVitsModelConfig(model="./vits-syspin-hi-female-model.onnx",
    tokens="./vits-syspin-hi-female-tokens.txt", noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0),
  provider="cpu"))
tts = sherpa_onnx.OfflineTts(cfg)
a = tts.generate("your text here", sid=0, speed=1.0)
sf.write("out.wav", a.samples, tts.sample_rate)</pre>
        </div>
      </div>
    </section>

    <section id="all-voices" class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
      <h2 class="text-lg font-bold text-white">All voices (no-JS index + SEO)</h2>
      <p class="text-xs text-slate-400">Every named voice, sample, model, and vocabulary. Works without JavaScript; the tabs above are a faster filter for the same files.</p>
      <div class="overflow-x-auto"><table class="w-full text-xs">
        <thead><tr class="text-left text-slate-400 border-b border-slate-800"><th class="py-2 pr-3">Voice</th><th class="py-2 pr-3">Language</th><th class="py-2 pr-3">Gender</th><th class="py-2 pr-3">Sample</th><th class="py-2 pr-3">Model</th><th class="py-2">Tokens</th></tr></thead>
        <tbody>
""".replace("REPO", REPO).replace("TAG", RELEASE_TAG)

BODY_END = """
        </tbody>
      </table></div>
      <noscript><p class="text-xs text-amber-300">JavaScript is off — use the table above; all MP3 / ONNX / tokens links work directly.</p></noscript>
    </section>

    <footer class="text-xs text-slate-500 flex flex-col md:flex-row justify-between gap-2 pb-6">
      <span>Voices: SPIRE Lab, IISc Bengaluru (RESPIN/SYSPIN, via HuggingFace SYSPIN). Runtime: k2-fsa/sherpa-onnx (Apache-2.0). Converter: this repo (see export + FP16 scripts).</span>
      <span><a class="underline" href="https://github.com/REPO">Repo</a> &middot; <a class="underline" href="https://github.com/REPO/releases/tag/TAG">Release TAG</a> &middot; <a class="underline" href="voices.json">voices.json</a></span>
    </footer>
  </div>
""".replace("REPO", REPO).replace("TAG", RELEASE_TAG)


def build_table_rows():
    rows = []
    for v in VOICES_DATA:
        for m in v["models"]:
            sp = SPEAKERS.get(m, m)
            female = "female" in m
            g = "Female" if female else "Male"
            rows.append(
                f'<tr class="border-b border-slate-800/60"><td class="py-2 pr-3 font-semibold text-slate-200">{sp} <span class="font-mono font-normal text-slate-500">{m}</span></td>'
                f'<td class="py-2 pr-3">{v["name"]} ({v["nativeName"]})</td><td class="py-2 pr-3">{g}</td>'
                f'<td class="py-2 pr-3"><a class="text-blue-400 underline" href="samples/{m}.mp3">mp3</a></td>'
                f'<td class="py-2 pr-3"><a class="text-blue-400 underline" href="https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{m}-model.onnx">onnx ~55MB</a></td>'
                f'<td class="py-2"><a class="text-blue-400 underline" href="https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{m}-tokens.txt">tokens</a></td></tr>'
            )
    return "\n".join(rows) + "\n" + build_rasa_table_rows()


def build_html():
    data_json = json.dumps(VOICES_DATA, ensure_ascii=False)
    spk_json = json.dumps(SPEAKERS, ensure_ascii=False)
    top = BODY_TOP.replace("__RASA_CARDS__", build_rasa_cards())
    js = """<script>
    const VOICES_DATA = __DATA__;
    const SPEAKERS = __SPK__;
    const RELEASE = { repo: "__REPO__", tag: "__TAG__" };
    let currentLangIdx = 0;
    const q = new URLSearchParams(location.search).get("lang");
    if (q) { const i = VOICES_DATA.findIndex(v => v.id === q); if (i >= 0) currentLangIdx = i; }

    function modelUrl(m, ext) { return `https://github.com/${RELEASE.repo}/releases/download/${RELEASE.tag}/${m}-${ext}`; }
    function esc(s) { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

    function renderTabs(filter="") {
      const box = document.getElementById("langTabs"); box.innerHTML = "";
      VOICES_DATA.forEach((v, idx) => {
        const hay = (v.name + " " + v.nativeName + " " + v.id + " " + v.models.map(m => SPEAKERS[m]||m).join(" ")).toLowerCase();
        if (filter && !hay.includes(filter.toLowerCase())) return;
        const b = document.createElement("button");
        b.type = "button"; b.setAttribute("role","tab");
        b.setAttribute("aria-selected", idx === currentLangIdx ? "true" : "false");
        b.className = "px-4 py-2 rounded-xl text-xs font-semibold whitespace-nowrap transition-all " + (idx === currentLangIdx ? "bg-blue-600 text-white shadow-lg ring-2 ring-blue-400/50" : "bg-slate-900 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-800");
        b.textContent = `${v.name} (${v.nativeName})`;
        b.onclick = () => { currentLangIdx = idx; history.replaceState(null,"","?lang="+v.id); renderTabs(document.getElementById("voiceSearch").value); renderActive(); };
        box.appendChild(b);
      });
      if (!box.children.length) box.innerHTML = '<span class="text-xs text-slate-500 px-2 py-2">No match — clear search.</span>';
    }

    function copyText(btn) {
      const t = decodeURIComponent(btn.getAttribute("data-copy") || "");
      navigator.clipboard.writeText(t).then(() => { const o = btn.textContent; btn.textContent = "Copied!"; setTimeout(() => btn.textContent = o, 1200); });
    }
    window.copyText = copyText;

    function renderActive() {
      const v = VOICES_DATA[currentLangIdx];
      const c = document.getElementById("activeVoiceContainer");
      const cards = v.models.map(m => {
        const sp = SPEAKERS[m] || m, female = m.includes("female");
        return `<div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
          <div class="flex items-start justify-between gap-2">
            <div><div class="text-base font-bold text-white">${esc(sp)} <span class="text-xs font-medium ${female ? "text-pink-400" : "text-cyan-400"}">${female ? "Female" : "Male"}</span></div>
            <div class="font-mono text-[11px] text-slate-500">${m}</div></div>
            <span class="text-[11px] px-2 py-0.5 rounded font-semibold ${female ? "bg-pink-500/10 text-pink-400 border border-pink-500/20" : "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"}">${esc(v.name)}</span>
          </div>
          <audio controls preload="none" class="w-full h-9"><source src="samples/${m}.mp3" type="audio/mpeg">No audio.</audio>
          <div class="flex gap-2 text-[11px]"><a href="samples/${m}.mp3" download class="text-blue-400 underline">Sample MP3</a><span class="text-slate-600">·</span><a class="text-blue-400 underline" target="_blank" rel="noopener" href="${modelUrl(m,"model.onnx")}">FP16 ONNX ~55MB</a><span class="text-slate-600">·</span><a class="text-blue-400 underline" target="_blank" rel="noopener" href="${modelUrl(m,"tokens.txt")}">tokens.txt</a></div>
        </div>`;
      }).join("");
      c.innerHTML = `<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-2 space-y-4">
          <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-5">
            <div class="flex items-center justify-between gap-2"><span class="px-3 py-1 rounded-full text-xs font-bold bg-blue-500/10 text-blue-400 border border-blue-500/20">Test passage (${v.text.length} chars)</span><span class="text-xs text-slate-400">~18–25s per voice</span></div>
            <div class="p-5 rounded-xl bg-slate-950 border border-slate-800"><div class="text-[11px] uppercase font-mono text-slate-400 mb-2 flex justify-between"><span>Native (${esc(v.name)} · ${esc(v.script)})</span><button data-copy="${encodeURIComponent(v.text)}" onclick="copyText(this)" class="text-blue-400">Copy</button></div><div class="text-lg leading-relaxed">${esc(v.text)}</div></div>
            <div class="p-5 rounded-xl bg-slate-950 border border-slate-800"><div class="text-[11px] uppercase font-mono text-slate-400 mb-2">Transliteration</div><div class="text-sm italic text-slate-300">${esc(v.transliteration)}</div></div>
            <div class="p-5 rounded-xl bg-slate-950 border border-slate-800"><div class="text-[11px] uppercase font-mono text-slate-400 mb-2">English meaning</div><div class="text-xs text-slate-400">${esc(v.translation)}</div></div>
          </div>
          <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6"><h4 class="text-sm font-bold mb-3">What to listen for</h4><ul class="space-y-2 text-xs text-slate-300">${v.rubric.map(r => `<li>• ${esc(r)}</li>`).join("")}</ul></div>
        </div>
        <div class="space-y-4"><div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4"><h4 class="text-sm font-bold">Voices — ${esc(v.name)}</h4>${cards}</div></div>
      </div>`;
    }

    document.getElementById("voiceSearch").addEventListener("input", e => { renderTabs(e.target.value); filterRasa(e.target.value); });
    function filterRasa(f) {
      f = (f || "").toLowerCase();
      document.querySelectorAll(".rasa-card").forEach(c => {
        c.style.display = (!f || (c.getAttribute("data-search") || "").toLowerCase().includes(f)) ? "" : "none";
      });
    }
    document.addEventListener("play", e => document.querySelectorAll("audio").forEach(a => { if (a !== e.target) a.pause(); }), true);
    document.addEventListener("keydown", e => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      if (document.activeElement && document.activeElement.getAttribute("role") === "tab") {
        currentLangIdx = (currentLangIdx + (e.key === "ArrowRight" ? 1 : -1) + VOICES_DATA.length) % VOICES_DATA.length;
        renderTabs(document.getElementById("voiceSearch").value); renderActive();
      }
    });
    renderTabs(); renderActive();
  </script>""".replace("__DATA__", data_json).replace("__SPK__", spk_json).replace("__REPO__", REPO).replace("__TAG__", RELEASE_TAG)
    return HTML_HEAD + top + BODY_MID + build_table_rows() + BODY_END + js + "\n</body>\n</html>\n"


if __name__ == "__main__":
    out = os.path.join(BASE, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_html())
    print(f"Generated {out}")
