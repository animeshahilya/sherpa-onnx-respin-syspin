#!/usr/bin/env python3
"""Hindi text frontend for sherpa-onnx SYSPIN VITS voices (character frontend).

Pipeline: normalize -> numerals-to-words -> schwa-deletion (explicit halant
insertion) -> punctuation mapping -> token-id filtering against tokens.txt.

Schwa deletion (the part that changes pronunciation):
  Devanagari does not write the inherent schwa, so a character model must
  guess where it is pronounced. We make it explicit by inserting halant
  (U+094D) wherever the schwa is deleted in speech, e.g. "राम" -> "राम्",
  "कमरा" -> "कम्रा". Rules (moraic-weight based, standard Hindi):
  - Final schwa: delete iff the preceding vowel is a FULL vowel.
    "राम"->"राम्" (aa full), "देव"->"देव्", but "कमल" (prev = schwa),
    "मतलब", "औरत", "घर", "वह"/"यह", monosyllables ("न") keep it.
  - Medial schwa (never the first akshara): delete in V C_schwa C V,
    right-to-left, iff (a) the preceding rhyme is light (short vowel, no
    coda incl. anusvara/visarga), and (b) deleting does not build a
    3-consonant cluster (i.e. the following consonant is not itself followed
    by an explicit halant). "कमरा"->"कम्रा", "करना"->"कर्ना",
    "मतलब"->"मत्लब", "दरवाज़ा"->"दर्वाज़ा", but "लड़का" (heavy coda),
    "सहायता" (long vowel), "नमस्ते" (would make mst cluster) keep it.
  Words with explicit halants, conjuncts, vowel signs are untouched except
  through these rules. Latin-script words pass through (model has some ASCII
  coverage); unsupported characters are dropped with a warning count.

Numerals: ASCII + Devanagari digits -> Hindi words (Indian system:
सौ/हज़ार/लाख/करोड़). Punctuation "।"/"॥" -> "."; quotes/dashes mapped to the
model's supported set (see tokens.txt); everything else is dropped.

Usage:
  python hindi_frontend.py --text "राम कमरे में है। मेरे पास 125 रुपये हैं।"
  python hindi_frontend.py --tokens vits-syspin-hi-female/tokens.txt --ids  # print id seq
"""

import argparse
import re
import sys
import unicodedata

HALANT = "\u094d"
NUKTA = "\u093c"
ANUSVARA = "\u0902"
VISARGA = "\u0903"
CHANDRA = "\u0901"

VOWEL_SIGNS = set("\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0945\u0946\u0947\u0948\u0949\u094a\u094b\u094c\u093e\u093f\u0955\u0956\u0957")
LONG_SIGNS = set("\u093e\u0940\u0942\u0944\u0947\u0948\u0949\u094a\u094b\u094c")  # aa ii uu vocR e ai o au (+candra)
SHORT_A_SIGNS = set()  # inherent schwa has no sign
NASALS = set([ANUSVARA, CHANDRA, "\u0901"])
LONG_INDEP = set("\u0906\u0908\u090a\u090f\u0910\u0911\u0912\u0913\u0914")

DEV_CONS_START, DEV_CONS_END = 0x0915, 0x0939
RETROFLEX = set("\u091f\u0920\u0921\u0922\u0923\u0922\u093c\u0922\u093c\u0937")  # ट ठ ड ढ ण ड़ ढ़ ष (dupes harmless)


def is_consonant(ch: str) -> bool:
    return len(ch) == 1 and (DEV_CONS_START <= ord(ch) <= DEV_CONS_END
                             or ch in ("\u0958\u0959\u095a\u095b\u095c\u095d\u095e\u095f"))


def split_aksharas(word: str):
    """Split a Devanagari word into aksharas: [consonant(+nukta)(+halant+consonant...)]
    each possibly followed by vowel sign / anusvara / visarga."""
    aks, cur = [], ""
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        if is_consonant(ch):
            if cur:
                aks.append(cur)
            cur = ch
            i += 1
            # absorb nukta, halant+consonant clusters, then signs
            while i < n:
                if word[i] == NUKTA:
                    cur += word[i]
                    i += 1
                elif word[i] == HALANT and i + 1 < n and is_consonant(word[i + 1]):
                    cur += word[i] + word[i + 1]
                    i += 2
                elif word[i] in VOWEL_SIGNS or word[i] in NASALS or word[i] in (VISARGA, "\u093d"):
                    cur += word[i]
                    i += 1
                else:
                    break
        else:
            if cur:
                aks.append(cur)
                cur = ""
            aks.append(ch)  # independent vowel, digit, punct...
            i += 1
    if cur:
        aks.append(cur)
    return aks


def ak_has_explicit_halant(ak: str) -> bool:
    return HALANT in ak


def ak_vowel(ak: str):
    """Return 'full' (explicit vowel sign other than inherent), 'long', 'schwa', or 'none'."""
    for ch in ak:
        if ch in VOWEL_SIGNS:
            return "long" if ch in LONG_SIGNS else "full"
    # independent vowels
    if ak and "\u0904" <= ak[0] <= "\u0914":
        ch = ak[0]
        if ch in LONG_INDEP:
            return "long"
        return "full"
    if ak and is_consonant(ak[0]):
        return "schwa"
    return "none"


def ak_rhyme_heavy(ak: str) -> bool:
    """Moraic weight of the akshara rhyme: heavy if long/full-non-schwa vowel
    after a cluster, long vowel, diphthong, or any nasal/visarga coda... simplified:
    heavy if it carries a LONG vowel sign, an independent long vowel, a nasal
    (anusvara/chandrabindu), visarga, or ends in an explicit consonant cluster."""
    for ch in ak:
        if ch in LONG_SIGNS or ch in NASALS or ch == VISARGA:
            return True
    if ak and ak[0] in LONG_INDEP:
        return True
    # cluster coda, e.g. "स्त", "र्त"
    cons_count = sum(1 for ch in ak if is_consonant(ch))
    if cons_count >= 2:
        return True
    return False


def ak_starts_consonant(ak: str) -> bool:
    return bool(ak) and is_consonant(ak[0])


def ak_ends_bare_consonant(ak: str) -> bool:
    """Ends with a consonant carrying the inherent schwa (no vowel sign, no halant)."""
    if not ak or not is_consonant(ak[0]):
        return False
    if ak_has_explicit_halant(ak):
        # ends bare only if text after last halant is consonant(s) w/o sign
        tail = ak.split(HALANT)[-1]
        tail = tail.lstrip(NUKTA)
        return bool(tail) and all(is_consonant(c) for c in tail)
    # no halant: bare iff no vowel sign / nasal / visarga present
    return not any(c in VOWEL_SIGNS or c in NASALS or c == VISARGA for c in ak[1:])


def delete_schwa_word(word: str) -> str:
    aks = split_aksharas(word)
    if not aks:
        return word
    # classify aksharas that are consonant-initial
    is_cak = [ak_starts_consonant(a) for a in aks]
    if not any(is_cak):
        return word
    delete = [False] * len(aks)

    # --- final schwa: delete iff preceding vowel is FULL (non-schwa) ---
    last = len(aks) - 1
    if is_cak[last] and ak_ends_bare_consonant(aks[last]):
        # find preceding vowel: vowel sign on same akshara? none (bare). Look left.
        prev_vowel = None
        for j in range(last - 1, -1, -1):
            v = ak_vowel(aks[j])
            if v in ("full", "long"):
                prev_vowel = v
                break
            if v == "schwa":
                prev_vowel = "schwa"
                break
        # monosyllabic bare-consonant word -> keep (e.g. "न")
        n_cak = sum(is_cak)
        if n_cak > 1 and prev_vowel in ("full", "long"):
            delete[last] = True

    # --- medial schwas, right to left, never the first akshara ---
    first_cak = next(i for i, c in enumerate(is_cak) if c)
    for i in range(len(aks) - 2, -1, -1):
        if not is_cak[i] or i == first_cak:
            continue
        if not ak_ends_bare_consonant(aks[i]):
            continue
        nxt = aks[i + 1]
        if not ak_starts_consonant(nxt):
            continue
        # categorical context only: following akshara must have a full vowel
        if ak_vowel(nxt) not in ("full", "long"):
            continue
        # glide guard: schwa is kept before य/व (रुपये, दवा, कवि)
        if nxt[0] in ("\u092f", "\u0935"):
            continue
        # retroflex guard: no deletion next to ट ठ ड ढ ण ड़ ढ़ ष
        # (लड़का, सड़क, मटका, पटना, संतरा keep their schwa)
        c1 = aks[i][0]
        if c1 in RETROFLEX or nxt[0] in RETROFLEX:
            continue
        # (b) avoid CCC: next consonant followed by explicit halant cluster
        if ak_has_explicit_halant(nxt):
            continue
        # (a) preceding rhyme must be light
        if i - 1 >= 0 and ak_rhyme_heavy(aks[i - 1]):
            continue
        # also the schwa's own akshara must be light (bare single consonant)
        cons_count = sum(1 for ch in aks[i] if is_consonant(ch))
        if cons_count > 1:
            continue
        delete[i] = True

    out = []
    for a, d in zip(aks, delete):
        out.append(a + HALANT if d else a)
    return "".join(out)


# ---------------- numerals ----------------
DEV_DIGITS = "०१२३४५६७८९"
ONES = ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
        "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस"]
TENS = {30: "तीस", 40: "चालीस", 50: "पचास", 60: "साठ", 70: "सत्तर", 80: "अस्सी", 90: "नब्बे"}
TENS_INFIX = {2: "बीस", 3: "तीस", 4: "चालीस", 5: "पचास", 6: "साठ", 7: "सत्तर", 8: "अस्सी", 9: "नब्बे"}
# 21-29,31-39... Hindi irregulars (standard forms)
TWENTIES = {21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस", 26: "छब्बीस",
            27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस"}
THIRTIES = {31: "इकतीस", 32: "बत्तीस", 33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस", 36: "छत्तीस",
            37: "सैंतीस", 38: "अड़तीस", 39: "उनतालीस"}
FORTIES = {41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस", 44: "चवालीस", 45: "पैंतालीस",
           46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास"}
FIFTIES = {51: "इक्यावन", 52: "बावन", 53: "तिरपन", 54: "चौवन", 55: "पचपन", 56: "छप्पन",
           57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ"}
SIXTIES = {61: "इकसठ", 62: "बासठ", 63: "तिरसठ", 64: "चौंसठ", 65: "पैंसठ", 66: "छियासठ",
           67: "सड़सठ", 68: "अड़सठ", 69: "उनहत्तर"}
SEVENTIES = {71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
             76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उनासी"}
EIGHTIES = {81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी", 85: "पचासी", 86: "छियासी",
            87: "सत्तासी", 88: "अट्ठासी", 89: "नवासी"}
NINETIES = {91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे", 95: "पचानवे", 96: "छियानवे",
            97: "सत्तानवे", 98: "अट्ठानवे", 99: "निन्यानवे"}
UNDER100 = {}
UNDER100.update({i: w for i, w in enumerate(ONES)})
UNDER100.update(TENS)
for d in (TWENTIES, THIRTIES, FORTIES, FIFTIES, SIXTIES, SEVENTIES, EIGHTIES, NINETIES):
    UNDER100.update(d)


def under_100(n: int) -> str:
    return UNDER100[n]


def number_to_hindi(n: int) -> str:
    if n < 0:
        return "ऋण " + number_to_hindi(-n)
    if n < 100:
        return under_100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        out = ("एक " if h == 1 else under_100(h) + " ") + "सौ"
        return out if r == 0 else out + " " + number_to_hindi(r)
    if n < 100000:
        th, r = divmod(n, 1000)
        out = ("एक " if th == 1 else number_to_hindi(th) + " ") + "हज़ार"
        return out if r == 0 else out + " " + number_to_hindi(r)
    if n < 10000000:
        lk, r = divmod(n, 100000)
        out = ("एक " if lk == 1 else number_to_hindi(lk) + " ") + "लाख"
        return out if r == 0 else out + " " + number_to_hindi(r)
    cr, r = divmod(n, 10000000)
    out = ("एक " if cr == 1 else number_to_hindi(cr) + " ") + "करोड़"
    return out if r == 0 else out + " " + number_to_hindi(r)


def expand_numerals(text: str) -> str:
    def dev_to_int(tok: str) -> int:
        return int("".join(str(DEV_DIGITS.index(c)) if c in DEV_DIGITS else c for c in tok))

    def repl(m):
        tok = m.group(0)
        try:
            return number_to_hindi(dev_to_int(tok))
        except (ValueError, KeyError):
            return tok

    return re.sub(r"[0-9\u0966-\u096f]+", repl, text)


# ---------------- punctuation / normalize ----------------
PUNCT_MAP = {
    "\u0965": ".",  # Double danda (॥)
    "\u0964": ".",  # Danda / purna viram (।)
    "!": "!",
    "?": "?",
    ",": ",",
    ".": ".",
    ":": ":",
    ";": ";",
    "-": "-",
    "\u2013": "-",  # En-dash (–)
    "\u2014": "-",  # Em-dash (—)
    "(": "(",
    ")": ")",
    '"': '"',
    "\u201c": '"',  # Left double curly quote (“)
    "\u201d": '"',  # Right double curly quote (”)
    "'": "'",
    "\u2018": "'",  # Left single curly quote (‘)
    "\u2019": "'",  # Right single curly quote (’)
    "{": "",
    "}": "",
}
DROP = set("@#$%^&*+=/\\|<>[]~`")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200d", "").replace("\u200e", "").replace("\u00a0", " ")
    text = expand_numerals(text)
    out = []
    for ch in text:
        if ch in PUNCT_MAP:
            out.append(PUNCT_MAP[ch])
        elif ch in DROP:
            continue
        else:
            out.append(ch)
    text = "".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def frontend(text: str) -> str:
    """Full pipeline -> model-ready text (still graphemes)."""
    text = normalize(text)
    words = []
    for tok in text.split(" "):
        if re.fullmatch(r"[A-Za-z]+", tok or ""):
            words.append(tok)  # Latin passthrough (partial ASCII coverage)
        else:
            words.append(delete_schwa_word(tok))
    return " ".join(w for w in words if w)


def to_ids(text: str, vocab: dict) -> tuple:
    """Map processed text to token ids; returns (ids, dropped_chars)."""
    ids, dropped = [], set()
    for ch in text:
        if ch in vocab:
            ids.append(vocab[ch])
        else:
            dropped.add(ch)
    return ids, dropped


def load_vocab(tokens_path: str) -> dict:
    vocab = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            ch, idx = line.rsplit(" ", 1)
            try:
                vocab[ch] = int(idx)
            except ValueError:
                continue
    return vocab


def main() -> int:
    ap = argparse.ArgumentParser(description="Hindi frontend for SYSPIN VITS voices")
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--ids", action="store_true")
    args = ap.parse_args()

    print("IN :", args.text)
    out = frontend(args.text)
    print("OUT:", out)
    if args.tokens:
        vocab = load_vocab(args.tokens)
        ids, dropped = to_ids(out, vocab)
        print("IDS:", ids)
        if dropped:
            print("DROPPED:", sorted(dropped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
