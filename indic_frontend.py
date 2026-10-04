#!/usr/bin/env python3
"""Text frontends for non-Devanagari SYSPIN VITS voices (bn/te/kn/gu/en).

Why this exists: audit of every vits-syspin-*/tokens.txt showed
  - bn, kn: NO digit support at all (native or ASCII) -> digits are dropped
  - kn, gu: NO danda (U+0964) -> sentence-final dandas are dropped
  - te: only partial native-digit coverage (ASCII digits ARE covered)
So numerals MUST be expanded to words and punctuation normalized, or input
text silently loses content. (See audit_frontend.py.)

Each language provides:
  NATIVE_DIGITS : 10-char string, source-order 0-9 (all five scripts)
  UNDER100      : dict int -> word for 0..99 (full tables for bn/gu with
                  irregular forms; compositional tens+units for te/kn)
  scales        : hundred / thousand / lakh / crore (+ million/billion: en)
  POINT, PERCENT: decimal-point and percent words
  CURRENCY      : symbol -> word (Rs/Tk/$ only; keeps it predictable)
  PUNCT_MAP     : script punctuation -> model-supported chars

Bengali apocope (delete_schwa_bn, EXPERIMENTAL): Bengali's inherent vowel
is [O] and word-final inherent vowels are usually silent in speech
(e.g. jal not *jalo). Rule mirrors hindi_frontend's final-schwa logic but
Bengali-tuned: bare final consonant, >=2 aksharas, preceding akshara has a
full/long vowel; never medial; khanda-ta (t, U+09CE) never touched; glide
guard before y/w (y U+09DF, w U+09F1). Validate with tune_inference.py
--ablate-frontend before trusting it; the model learned implicit apocope
from 50h of raw-text training data, so this only biases ambiguous cases.
"""

import re
import unicodedata

# ---------------------------------------------------------------- Bengali

BN_DIGITS = "০১২৩৪৫৬৭৮৯"
BN_UNDER100 = {
    0: "শূন্য", 1: "এক", 2: "দুই", 3: "তিন", 4: "চার", 5: "পাঁচ",
    6: "ছয়", 7: "সাত", 8: "আট", 9: "নয়", 10: "দশ",
    11: "এগারো", 12: "বারো", 13: "তেরো", 14: "চৌদ্দ", 15: "পনেরো",
    16: "ষোলো", 17: "সতেরো", 18: "আঠারো", 19: "ঊনিশ", 20: "বিশ",
    21: "একুশ", 22: "বাইশ", 23: "তেইশ", 24: "চব্বিশ", 25: "পঁচিশ",
    26: "ছাব্বিশ", 27: "সাতাশ", 28: "আঠাশ", 29: "ঊনত্রিশ", 30: "ত্রিশ",
    31: "একত্রিশ", 32: "বত্রিশ", 33: "তেত্রিশ", 34: "চৌত্রিশ",
    35: "পঁয়ত্রিশ", 36: "ছত্রিশ", 37: "সাঁইত্রিশ", 38: "আটত্রিশ",
    39: "ঊনচল্লিশ", 40: "চল্লিশ", 41: "একচল্লিশ", 42: "বিয়াল্লিশ",
    43: "তেতাল্লিশ", 44: "চুয়াল্লিশ", 45: "পঁয়তাল্লিশ",
    46: "ছেচল্লিশ", 47: "সাতচল্লিশ", 48: "আটচল্লিশ",
    49: "ঊনপঞ্চাশ", 50: "পঞ্চাশ", 51: "একান্ন", 52: "বাহান্ন",
    53: "তিপ্পান্ন", 54: "চুয়ান্ন", 55: "পঞ্চান্ন", 56: "ছাপ্পান্ন",
    57: "সাতান্ন", 58: "আটান্ন", 59: "ঊনষাট", 60: "ষাট",
    61: "একষট্টি", 62: "বাষট্টি", 63: "তেষট্টি", 64: "চৌষট্টি",
    65: "পঁয়ষট্টি", 66: "ছেষট্টি", 67: "সাতষট্টি", 68: "আটষট্টি",
    69: "ঊনসত্তর", 70: "সত্তর", 71: "একাত্তর", 72: "বাহাত্তর",
    73: "তিয়াত্তর", 74: "চুয়াত্তর", 75: "পঁচাত্তর", 76: "ছিয়াত্তর",
    77: "সাতাত্তর", 78: "আটাত্তর", 79: "ঊনআশি", 80: "আশি",
    81: "একাশি", 82: "বিরাশি", 83: "তিরাশি", 84: "চুরাশি",
    85: "পঁচাশি", 86: "ছিয়াশি", 87: "সাতাশি", 88: "আটাশি",
    89: "ঊননব্বই", 90: "নব্বই", 91: "একানব্বই", 92: "বিরানব্বই",
    93: "তিরানব্বই", 94: "চুরানব্বই", 95: "পঁচানব্বই",
    96: "ছিয়ানব্বই", 97: "সাতানব্বই", 98: "আটানব্বই",
    99: "নিরানব্বই",
}
assert len(BN_UNDER100) == 100 and min(BN_UNDER100) == 0 and max(BN_UNDER100) == 99


def bn_number(n: int) -> str:
    if n < 0:
        return "ঋণ " + bn_number(-n)
    if n < 100:
        return BN_UNDER100[n]
    if n < 1000:
        h, r = divmod(n, 100)
        out = "একশত" if h == 1 else BN_UNDER100[h] + " শত"
        return out if r == 0 else out + " " + bn_number(r)
    if n < 100000:
        th, r = divmod(n, 1000)
        out = "এক হাজার" if th == 1 else bn_number(th) + " হাজার"
        return out if r == 0 else out + " " + bn_number(r)
    if n < 10000000:
        lk, r = divmod(n, 100000)
        out = "এক লাখ" if lk == 1 else bn_number(lk) + " লাখ"
        return out if r == 0 else out + " " + bn_number(r)
    cr, r = divmod(n, 10000000)
    out = "এক কোটি" if cr == 1 else bn_number(cr) + " কোটি"
    return out if r == 0 else out + " " + bn_number(r)


# ---------------------------------------------------------------- Telugu

TE_DIGITS = "౦౧౨౩౪౫౬౭౮౯"
TE_UNDER21 = {
    0: "సున్న", 1: "ఒకటి", 2: "రెండు", 3: "మూడు", 4: "నాలుగు",
    5: "ఐదు", 6: "ఆరు", 7: "ఏడు", 8: "ఎనిమిది", 9: "తొమ్మిది",
    10: "పది", 11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు",
    14: "పద్నాలుగు", 15: "పదిహేను", 16: "పదహారు", 17: "పదిహేడు",
    18: "పద్దెనిమిది", 19: "పందొమ్మిది", 20: "ఇరవై",
}
TE_TENS = {20: "ఇరవై", 30: "ముప్పై", 40: "నలభై", 50: "యాభై", 60: "అరవై",
           70: "డెబ్బై", 80: "ఎనభై", 90: "తొంభై"}


def te_under100(n: int) -> str:
    if n <= 20:
        return TE_UNDER21[n]
    t, u = divmod(n, 10)
    return TE_TENS[t * 10] if u == 0 else TE_TENS[t * 10] + " " + TE_UNDER21[u]


def te_number(n: int) -> str:
    if n < 0:
        return "మైనస్ " + te_number(-n)
    if n < 100:
        return te_under100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        out = "వంద" if h == 1 else TE_UNDER21[h] + " వందలు"
        return out if r == 0 else out + " " + te_number(r)
    if n < 100000:
        th, r = divmod(n, 1000)
        out = "వెయ్యి" if th == 1 else te_number(th) + " వేలు"
        return out if r == 0 else out + " " + te_number(r)
    if n < 10000000:
        lk, r = divmod(n, 100000)
        out = "లక్ష" if lk == 1 else te_number(lk) + " లక్షలు"
        return out if r == 0 else out + " " + te_number(r)
    cr, r = divmod(n, 10000000)
    out = "కోటి" if cr == 1 else te_number(cr) + " కోట్లు"
    return out if r == 0 else out + " " + te_number(r)


# ---------------------------------------------------------------- Kannada

KN_DIGITS = "೦೧೨೩೪೫೬೭೮೯"
KN_UNDER21 = {
    0: "ಸೊನ್ನೆ", 1: "ಒಂದು", 2: "ಎರಡು", 3: "ಮೂರು", 4: "ನಾಲ್ಕು",
    5: "ಐದು", 6: "ಆರು", 7: "ಏಳು", 8: "ಎಂಟು", 9: "ಒಂಬತ್ತು",
    10: "ಹತ್ತು", 11: "ಹನ್ನೊಂದು", 12: "ಹನ್ನೆರಡು", 13: "ಹದಿಮೂರು",
    14: "ಹದಿನಾಲ್ಕು", 15: "ಹದಿನೈದು", 16: "ಹದಿನಾರು", 17: "ಹದಿನೇಳು",
    18: "ಹದಿನೆಂಟು", 19: "ಹತ್ತೊಂಬತ್ತು", 20: "ಇಪ್ಪತ್ತು",
}
KN_TENS = {20: "ಇಪ್ಪತ್ತು", 30: "ಮೂವತ್ತು", 40: "ನಲವತ್ತು", 50: "ಐವತ್ತು",
           60: "ಅರವತ್ತು", 70: "ಎಪ್ಪತ್ತು", 80: "ಎಂಬತ್ತು", 90: "ತೊಂಬತ್ತು"}


def kn_under100(n: int) -> str:
    if n <= 20:
        return KN_UNDER21[n]
    t, u = divmod(n, 10)
    return KN_TENS[t * 10] if u == 0 else KN_TENS[t * 10] + " " + KN_UNDER21[u]


def kn_number(n: int) -> str:
    if n < 0:
        return "ಮೈನಸ್ " + kn_number(-n)
    if n < 100:
        return kn_under100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        out = "ನೂರು" if h == 1 else KN_UNDER21[h] + " ನೂರು"
        return out if r == 0 else out + " " + kn_number(r)
    if n < 100000:
        th, r = divmod(n, 1000)
        out = "ಒಂದು ಸಾವಿರ" if th == 1 else kn_number(th) + " ಸಾವಿರ"
        return out if r == 0 else out + " " + kn_number(r)
    if n < 10000000:
        lk, r = divmod(n, 100000)
        out = "ಒಂದು ಲಕ್ಷ" if lk == 1 else kn_number(lk) + " ಲಕ್ಷ"
        return out if r == 0 else out + " " + kn_number(r)
    cr, r = divmod(n, 10000000)
    out = "ಒಂದು ಕೋಟಿ" if cr == 1 else kn_number(cr) + " ಕೋಟಿ"
    return out if r == 0 else out + " " + kn_number(r)


# ---------------------------------------------------------------- Gujarati

GU_DIGITS = "૦૧૨૩૪૫૬૭૮૯"
GU_UNDER100 = {
    0: "શૂન્ય", 1: "એક", 2: "બે", 3: "ત્રણ", 4: "ચાર", 5: "પાંચ",
    6: "છ", 7: "સાત", 8: "આઠ", 9: "નવ", 10: "દસ",
    11: "અગિયાર", 12: "બાર", 13: "તેર", 14: "ચૌદ", 15: "પંદર",
    16: "સોળ", 17: "સત્તર", 18: "અઢાર", 19: "ઓગણીસ", 20: "વીસ",
    21: "એકવીસ", 22: "બાવીસ", 23: "ત્રેવીસ", 24: "ચોવીસ",
    25: "પચ્ચીસ", 26: "છવ્વીસ", 27: "સત્તાવીસ", 28: "અઠ્ઠાવીસ",
    29: "ઓગણત્રીસ", 30: "ત્રીસ", 31: "એકત્રીસ", 32: "બત્રીસ",
    33: "તેત્રીસ", 34: "ચોત્રીસ", 35: "પાંત્રીસ", 36: "છત્રીસ",
    37: "સાડત્રીસ", 38: "આડત્રીસ", 39: "ઓગણચાલીસ", 40: "ચાલીસ",
    41: "એકતાલીસ", 42: "બેતાલીસ", 43: "તેતાલીસ", 44: "ચુંમાલીસ",
    45: "પિસ્તાલીસ", 46: "છેતાલીસ", 47: "સુડતાલીસ", 48: "અડતાલીસ",
    49: "ઓગણપચાસ", 50: "પચાસ", 51: "એકાવન", 52: "બાવન",
    53: "ત્રેપન", 54: "ચોપન", 55: "પંચાવન", 56: "છપ્પન",
    57: "સત્તાવન", 58: "અઠ્ઠાવન", 59: "ઓગણસાઠ", 60: "સાઠ",
    61: "એકસઠ", 62: "બાસઠ", 63: "ત્રેસઠ", 64: "ચોસઠ",
    65: "પાંસઠ", 66: "છાસઠ", 67: "સડસઠ", 68: "અડસઠ",
    69: "ઓગણસિત્તેર", 70: "સિત્તેર", 71: "એકોતેર", 72: "બોતેર",
    73: "તોતેર", 74: "ચુમ્મોતેર", 75: "પંચોતેર", 76: "છોતેર",
    77: "સિત્ત્યોતેર", 78: "ઇઠ્યોતેર", 79: "ઓગણએંસી", 80: "એંસી",
    81: "એક્યાસી", 82: "બ્યાસી", 83: "ત્યાસી", 84: "ચોર્યાસી",
    85: "પંચાસી", 86: "છ્યાસી", 87: "સિત્યાસી", 88: "ઇઠ્યાસી",
    89: "નેવ્યાસી", 90: "નેવું", 91: "એકાણું", 92: "બાણું",
    93: "ત્રાણું", 94: "ચોરાણું", 95: "પંચાણું", 96: "છન્નું",
    97: "સત્તાણું", 98: "અઠ્ઠાણું", 99: "નવ્વાણું",
}
assert len(GU_UNDER100) == 100 and min(GU_UNDER100) == 0 and max(GU_UNDER100) == 99


def gu_number(n: int) -> str:
    if n < 0:
        return "માઇનસ " + gu_number(-n)
    if n < 100:
        return GU_UNDER100[n]
    if n < 1000:
        h, r = divmod(n, 100)
        out = "એકસો" if h == 1 else ("બસો" if h == 2 else GU_UNDER100[h] + " સો")
        return out if r == 0 else out + " " + gu_number(r)
    if n < 100000:
        th, r = divmod(n, 1000)
        out = "એક હજાર" if th == 1 else gu_number(th) + " હજાર"
        return out if r == 0 else out + " " + gu_number(r)
    if n < 10000000:
        lk, r = divmod(n, 100000)
        out = "એક લાખ" if lk == 1 else gu_number(lk) + " લાખ"
        return out if r == 0 else out + " " + gu_number(r)
    cr, r = divmod(n, 10000000)
    out = "એક કરોડ" if cr == 1 else gu_number(cr) + " કરોડ"
    return out if r == 0 else out + " " + gu_number(r)


# ---------------------------------------------------------------- English

EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
           "eight", "nine", "ten", "eleven", "twelve", "thirteen",
           "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
           "nineteen"]
EN_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
           60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def en_under100(n: int) -> str:
    if n < 20:
        return EN_ONES[n]
    t, u = divmod(n, 10)
    return EN_TENS[t * 10] if u == 0 else EN_TENS[t * 10] + "-" + EN_ONES[u]


def en_number(n: int) -> str:
    if n < 0:
        return "minus " + en_number(-n)
    if n < 100:
        return en_under100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        out = EN_ONES[h] + " hundred"
        return out if r == 0 else out + " " + en_number(r)
    for scale, word in ((1000000000, "billion"), (1000000, "million"), (1000, "thousand")):
        if n >= scale:
            q, r = divmod(n, scale)
            out = en_number(q) + " " + word
            return out if r == 0 else out + " " + en_number(r)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- config

LANG = {
    "bn": {"digits": BN_DIGITS, "number": bn_number, "point": "দশমিক",
           "percent": "শতাংশ", "currency": {"₹": "টাকা", "৳": "টাকা", "$": "ডলার"},
           "minus": "ঋণ"},
    "te": {"digits": TE_DIGITS, "number": te_number, "point": "దశాంశం",
           "percent": "శాతం", "currency": {"₹": "రూపాయలు", "$": "డాలర్లు"},
           "minus": None},
    "kn": {"digits": KN_DIGITS, "number": kn_number, "point": "ದಶಾಂಶ",
           "percent": "ಶೇಕಡಾ", "currency": {"₹": "ರೂಪಾಯಿ", "$": "ಡಾಲರ್"},
           "minus": None},
    "gu": {"digits": GU_DIGITS, "number": gu_number, "point": "દશાંશ",
           "percent": "ટકા", "currency": {"₹": "રૂપિયા", "$": "ડોલર"},
           "minus": None},
    "en": {"digits": "0123456789", "number": en_number, "point": "point",
           "percent": "percent", "currency": {"₹": "rupees", "$": "dollars"},
           "minus": "minus"},
}
for _lc, _cfg in LANG.items():
    assert len(_cfg["digits"]) == 10, _lc
del _lc, _cfg

DROP = set("@#$%^&*+=/\\|<>[]~`")
PUNCT_MAP = {
    "।": ".",  # danda (all four Indic scripts use U+0964)
    "॥": ".",  # double danda
    "!": "!", "?": "?", ",": ",", ".": ".", ":": ":", ";": ";",
    "-": "-", "–": "-", "—": "-", "(": "(", ")": ")",
    '"': '"', "“": '"', "”": '"',
    "'": "'", "‘": "'", "’": "'",
    "{": "", "}": "",
}


def _digit_class(cfg) -> str:
    return "[" + re.escape(cfg["digits"]) + "0-9]"


def expand_numerals(text: str, lang: str) -> str:
    """Native + ASCII digit runs -> words. Handles Indian/Western commas,
    decimals (digit-by-digit fraction) and % / currency suffixes."""
    cfg = LANG[lang]
    dc = _digit_class(cfg)

    def parse_int(tok: str) -> int:
        t = tok.replace(",", "")
        return int("".join(str(cfg["digits"].index(c)) if c in cfg["digits"] else c
                           for c in t))

    def digit_words(tok: str) -> str:
        return " ".join(cfg["number"](parse_int(c.replace(",", ""))) for c in tok if c != ",")

    # decimals FIRST (dot only) so the integer pass keeps comma groupings
    def repl_dec(m):
        ip, fp = m.group(1), m.group(2)
        try:
            return cfg["number"](parse_int(ip)) + " " + cfg["point"] + " " + digit_words(fp)
        except (ValueError, KeyError):
            return m.group(0)
    text = re.sub("(" + dc + r"+)\.(" + dc + r"+)", repl_dec, text)

    def repl_int(m):
        tok = m.group(0)
        try:
            return cfg["number"](parse_int(tok))
        except (ValueError, KeyError):
            return tok
    text = re.sub(dc + r"+(?:," + dc + r"+)*", repl_int, text)

    if cfg["minus"]:
        text = re.sub(r"(?<!\d)-(?=" + dc + r"+)", cfg["minus"] + " ", text)

    for sym, word in cfg["currency"].items():
        text = text.replace(sym, " " + word + " ")
    text = text.replace("%", " " + cfg["percent"] + " ")
    return text


def normalize(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200d", "").replace("\u200e", "").replace("\u00a0", " ")
    text = expand_numerals(text, lang)
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


# ------------------------------------------------- Bengali apocope (exp.)

BN_HALANT = "্"  # U+09CD
BN_NUKTA = "়"  # U+09BC
BN_ANUSVARA = "ং"  # U+0982
BN_VISARGA = "ঃ"  # U+0983
BN_CHANDRA = "ঁ"  # U+0981
BN_KHANDA_TA = "ৎ"  # U+09CE: inherently vowelless, never takes schwa
BN_VOWEL_SIGNS = set(" া ি ী ু ূ ৃ ৄ ে ৈ ো ৌ ৢ ৣ".split())
BN_LONG_SIGNS = set("া ী ূ ে ৈ ো ৌ".split())
BN_LONG_INDEP = set("আ ঈ ঊ এ ঐ ও ঔ".split())
BN_GLA = set("য়ৱ")  # য U+09AF? no: য় U+09DF, ৱ U+09F1


def _bn_is_cons(ch: str) -> bool:
    return len(ch) == 1 and ("\u0980" <= ch <= "\u09ff" and
            ("\u0995" <= ch <= "\u09b9" or "\u09dc" <= ch <= "\u09df"
             or "\u09ab" <= ch <= "\u09b0" or ch in "য়ড়ঢ়লশষসহৎ"))


def bn_split_aksharas(word: str):
    aks, cur = [], ""
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if _bn_is_cons(ch) and ch != BN_KHANDA_TA:
            if cur:
                aks.append(cur)
            cur = ch
            i += 1
            while i < n:
                if word[i] == BN_NUKTA:
                    cur += word[i]
                    i += 1
                elif word[i] == BN_HALANT and i + 1 < n and _bn_is_cons(word[i + 1]):
                    cur += word[i] + word[i + 1]
                    i += 2
                elif word[i] in BN_VOWEL_SIGNS or word[i] in (BN_ANUSVARA, BN_VISARGA, BN_CHANDRA):
                    cur += word[i]
                    i += 1
                else:
                    break
        else:
            if cur:
                aks.append(cur)
                cur = ""
            aks.append(ch)
            i += 1
    if cur:
        aks.append(cur)
    return aks


def _bn_vowel(ak: str):
    for ch in ak:
        if ch in BN_VOWEL_SIGNS:
            return "long" if ch in BN_LONG_SIGNS else "full"
    if ak and "অ" <= ak[0] <= "ঔ":
        return "long" if ak[0] in BN_LONG_INDEP else "full"
    if ak and _bn_is_cons(ak[0]):
        return "schwa"  # inherent ô
    return "none"


def _bn_ends_bare(ak: str) -> bool:
    if not ak or not _bn_is_cons(ak[0]) or ak[0] == BN_KHANDA_TA:
        return False
    if BN_HALANT in ak:
        tail = ak.split(BN_HALANT)[-1].lstrip(BN_NUKTA)
        return bool(tail) and all(_bn_is_cons(c) for c in tail)
    return not any(c in BN_VOWEL_SIGNS or c in (BN_ANUSVARA, BN_VISARGA, BN_CHANDRA)
                   for c in ak[1:])


def delete_schwa_bn(word: str) -> str:
    """Word-final inherent-ô deletion only (conservative, experimental)."""
    aks = bn_split_aksharas(word)
    is_cak = [bool(a) and _bn_is_cons(a[0]) and a[0] != BN_KHANDA_TA for a in aks]
    if sum(is_cak) < 2 or not is_cak[-1] or not _bn_ends_bare(aks[-1]):
        return word
    if aks[-1][0] in BN_GLA or any(a and a[0] in BN_GLA for a in [aks[-2]]):
        return word
    prev = None
    for j in range(len(aks) - 2, -1, -1):
        v = _bn_vowel(aks[j])
        if v in ("full", "long"):
            prev = v
            break
        if v == "schwa":
            prev = "schwa"
            break
    if prev in ("full", "long"):
        aks[-1] = aks[-1] + BN_HALANT
    return "".join(aks)


BN_TRAILING_PUNCT = ".,?!:;,—–'\"()"


def frontend(text: str, lang: str = "bn", apocope: bool = True) -> str:
    """Full pipeline -> model-ready text. apocope toggles Bengali final-ô
    deletion (see tune_inference.py --ablate-frontend for validation)."""
    text = normalize(text, lang)
    if lang == "bn" and apocope:
        words = []
        for tok in text.split(" "):
            if re.fullmatch(r"[A-Za-z]+", tok or ""):
                words.append(tok)
            else:
                # detach trailing punct so 'জল.' still gets apocope
                i = len(tok)
                while i > 0 and tok[i - 1] in BN_TRAILING_PUNCT:
                    i -= 1
                core, tail = tok[:i], tok[i:]
                words.append(delete_schwa_bn(core) + tail if core else tok)
        text = " ".join(w for w in words if w)
    return text
