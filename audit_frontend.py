#!/usr/bin/env python3
"""Audit text frontend coverage for SYSPIN voices (item 1).

Part A -- vocab coverage: per voice, does tokens.txt contain native digits
(all 10), ASCII digits, danda, halant/hasanta, nukta, anusvara, visarga,
and basic punctuation?
Part B -- dropped-character test: run realistic eval texts (dashboard
passages + digit/punctuation stress sentences) through old behavior (raw
passthrough for bn/te/kn/gu/en) vs new frontend, report dropped chars.

Writes audit/vocab_coverage.json and audit/dropped_chars.json, prints a
markdown summary table to audit/audit_report.md (console mangles Indic
scripts, so everything substantive goes to files).
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hindi_frontend import load_vocab as hi_load_vocab  # noqa: E402
from indic_frontend import LANG, frontend as indic_frontend  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(BASE, "audit")
os.makedirs(AUDIT, exist_ok=True)

VOICE_LANGS = ["hi", "bn", "te", "kn", "mr", "gu", "bho", "hne", "mai", "mag", "en"]

# script-specific checks: (label, chars)
SCRIPT_CHECKS = {
    "hi": [("native_digits", "०१२३४५६७८९"), ("danda", "।"),
           ("halant", "्"), ("nukta", "़"), ("anusvara", "ं"), ("visarga", "ः")],
    "bn": [("native_digits", "০১২৩৪৫৬৭৮৯"), ("danda", "।"),
           ("hasanta", "্"), ("nukta", "়"), ("anusvara", "ং"),
           ("visarga", "ঃ"), ("khanda_ta", "ৎ")],
    "te": [("native_digits", "౦౧౨౩౪౫౬౭౮౯"), ("danda", "।"),
           ("halant", "్"), ("anusvara", "ం"), ("visarga", "ః")],
    "kn": [("native_digits", "೦೧೨೩೪೫೬೭೮೯"), ("danda", "।"),
           ("halant", "್"), ("anusvara", "ಂ"), ("visarga", "ಃ")],
    "gu": [("native_digits", "૦૧૨૩૪૫૬૭૮૯"), ("danda", "।"),
           ("halant", "્"), ("anusvara", "ં"), ("visarga", "ઃ")],
    "en": [],
    "mr": [("native_digits", "०१२३४५६७८९"), ("danda", "।"), ("halant", "्")],
    "bho": [("native_digits", "०१२३४५६७८९"), ("danda", "।"), ("halant", "्")],
    "hne": [("native_digits", "०१२३४५६७८९"), ("danda", "।"), ("halant", "्")],
    "mai": [("native_digits", "०१२३४५६७८९"), ("danda", "।"), ("halant", "्")],
    "mag": [("native_digits", "०१२३४५६७८९"), ("danda", "।"), ("halant", "्")],
}
ASCII_DIGITS = "0123456789"
PUNCT = "?!,.:'\"()-;:"

# stress sentences: native digits, ascii digits, decimals, %, currency,
# danda, quotes, dashes -- one per non-Devanagari lang + Devanagari control
STRESS = {
    "bn": "২০২৪ সালে দাম ১২৫ টাকা। 12.5% ছাড়! “ভালো”-মন্দ (সব)।",
    "te": "2024లో ధర ౧౨౫ రూపాయలు. 12.5% తగ్గింపు! “మంచి”-చెడు (అన్నీ).",
    "kn": "2024ರಲ್ಲಿ ಬೆಲೆ 125 ರೂಪಾಯಿ. 12.5% ರಿಯಾಯಿತಿ! “ಒಳ್ಳೆಯದು”-ಕೆಟ್ಟದು (ಎಲ್ಲ).",
    "gu": "૨૦૨૪માં ભાવ ૧૨૫ રૂપિયા. 12.5% વળતર! “સારું”-ખરાબ (બધું).",
    "en": "In 2024 the price is $125. 12.5% off! \"Good\"-bad (all).",
    "hi": "२०२४ में दाम १२५ रुपये। 12.5% छूट! \"अच्छा\"-बुरा (सब)।",
}


def load_voice_vocab(lang, gender="female"):
    return hi_load_vocab(os.path.join(BASE, f"vits-syspin-{lang}-{gender}", "tokens.txt"))


def dropped(text, vocab):
    return sorted({ch for ch in text if ch not in vocab and not ch.isspace()},
                  key=ord)


def validate_tokens_file(path):
    """Replicate sherpa-onnx ReadTokens exactly (offline-tts-character-
    frontend.cc): C-locale whitespace split; a bare-number line means the
    SPACE token; <PAD>/<EOS>/<BOS>/<BLNK> skipped. Returns (issues, n_syms).

    Real bugs this caught: mai-male release tokens had a TAB-typo space
    line (id 48) PLUS a second space line (id 81) -> sherpa hard-aborts
    the whole process on load. mr-male has a junk U+200A hair-space token
    (harmless: never matches text, but should be cleaned)."""
    import unicodedata
    issues = []
    seen = {}
    c_ws = set(" \t\n\r\v\f")
    for i, raw in enumerate(open(path, encoding="utf-8"), 1):
        line = raw.rstrip("\n").rstrip("\r")
        # C++ operator>> tokenization: skip ASCII whitespace runs
        toks = []
        cur = ""
        for ch in line:
            if ch in c_ws:
                if cur:
                    toks.append(cur)
                    cur = ""
            else:
                cur += ch
        if cur:
            toks.append(cur)
        if not toks:
            continue
        if len(toks) == 1:
            try:
                sym, idv = " ", int(toks[0])
            except ValueError:
                issues.append("line %d: unparsable %r" % (i, line))
                continue
        else:
            sym, idv = toks[0], toks[1]
            try:
                idv = int(idv)
            except ValueError:
                issues.append("line %d: bad id %r" % (i, line))
                continue
        if sym in ("<PAD>", "<EOS>", "<BOS>", "<BLNK>"):
            continue
        # NOTE: sherpa does NOT NFC-normalize (raw codepoints only); neither
        # do we here. (Python NFC decomposes Devanagari nukta letters like
        # U+095B into 2 chars, which would false-positive every nukta entry.)
        if len(sym) != 1:
            issues.append("line %d: multi-codepoint sym %r" % (i, line))
            continue
        if sym in (" ",):
            pass  # the one legitimate whitespace token
        elif any(unicodedata.category(c) in ("Zs", "Zl", "Zp", "Cc", "Cf") for c in sym):
            issues.append("line %d: whitespace/control sym %s (id %d) -- junk token" %
                          (i, " ".join("U+%04X" % ord(c) for c in sym), idv))
        if sym in seen:
            issues.append("line %d: DUPLICATE sym U+%04X (id %d, first id %d)"
                          % (i, ord(sym), idv, seen[sym]))
        else:
            seen[sym] = idv
    return issues, len(seen)


def old_behavior(text, lang):
    """Pre-change behavior: raw passthrough for bn/te/kn/gu/en."""
    return text


def main():
    try:
        from build_dashboard import VOICES_DATA
        passages = {v["id"]: v["text"] for v in VOICES_DATA}
    except ImportError:
        passages = {}

    cov, drops = {}, {}
    for lang in VOICE_LANGS:
        vocab = load_voice_vocab(lang)
        entry = {"vocab_size": len(vocab)}
        for label, chars in SCRIPT_CHECKS.get(lang, []):
            have = [c for c in chars if c in vocab]
            entry[label] = {"have": len(have), "total": len(chars),
                            "missing": [hex(ord(c)) for c in chars if c not in vocab]}
        entry["ascii_digits"] = {"have": sum(c in vocab for c in ASCII_DIGITS),
                                 "total": 10}
        entry["punct"] = {c: (c in vocab) for c in PUNCT}
        entry["space"] = " " in vocab
        cov[lang] = entry

        texts = []
        if lang in passages:
            texts.append(("dashboard", passages[lang]))
        if lang in STRESS:
            texts.append(("stress", STRESS[lang]))
        lang_drops = {}
        for name, text in texts:
            before = dropped(old_behavior(text, lang), vocab)
            after_text = (indic_frontend(text, lang) if lang in LANG
                          else __import__("hindi_frontend").frontend(text))
            after = dropped(after_text, vocab)
            lang_drops[name] = {
                "before": ["U+%04X" % ord(c) for c in before],
                "after": ["U+%04X" % ord(c) for c in after],
                "after_text": after_text,
            }
        drops[lang] = lang_drops

    json.dump(cov, open(os.path.join(AUDIT, "vocab_coverage.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(drops, open(os.path.join(AUDIT, "dropped_chars.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # Part C: tokens-file validation with sherpa's exact parser semantics.
    tokrep = {}
    for lang in VOICE_LANGS:
        for gender in ("female", "male"):
            vid = f"vits-syspin-{lang}-{gender}"
            for label, path in (
                    ("committed", os.path.join(BASE, vid, "tokens.txt")),
                    ("release", os.path.join(BASE, "release_assets_fp16", vid + "-tokens.txt"))):
                if not os.path.exists(path):
                    tokrep[f"{vid}/{label}"] = {"missing": True}
                    continue
                issues, n = validate_tokens_file(path)
                tokrep[f"{vid}/{label}"] = {"issues": issues, "symbols": n}
    json.dump(tokrep, open(os.path.join(AUDIT, "tokens_validation.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    lines = ["# Frontend audit", "",
             "## A. Vocab coverage (female voice per lang)",
             "",
             "| lang | size | nat_digits | ascii_digits | danda | halant | space |",
             "|---|---|---|---|---|---|---|"]
    for lang in VOICE_LANGS:
        e = cov[lang]
        def cell(k):
            v = e.get(k)
            return ("--" if v is None else "%d/%d" % (v["have"], v["total"]))
        lines.append("| %s | %d | %s | %d/10 | %s | %s | %s |" % (
            lang, e["vocab_size"], cell("native_digits"), e["ascii_digits"]["have"],
            cell("danda"), cell("halant") if "halant" in e else cell("hasanta"),
            "Y" if e["space"] else "N"))
    lines += ["", "## B. Dropped chars: before (raw) vs after (frontend)", ""]
    for lang in VOICE_LANGS:
        for name, d in drops[lang].items():
            lines.append("- %s/%s: before=%d %s | after=%d %s" % (
                lang, name, len(d["before"]), d["before"], len(d["after"]), d["after"]))
    lines += ["", "## C. tokens.txt validation (sherpa-exact parse)", ""]
    bad = {k: v for k, v in tokrep.items() if v.get("issues")}
    if bad:
        for k, v in sorted(bad.items()):
            lines.append("- %s: %s" % (k, "; ".join(v["issues"])))
    else:
        lines.append("all 44 tokens files clean.")
    open(os.path.join(AUDIT, "audit_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # console-safe summary: counts only
    for lang in VOICE_LANGS:
        for name, d in drops[lang].items():
            print("%s/%s dropped: before=%d after=%d" % (lang, name, len(d["before"]), len(d["after"])))
    print("wrote audit/*.json + audit/audit_report.md")


if __name__ == "__main__":
    main()
