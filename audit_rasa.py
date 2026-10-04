#!/usr/bin/env python3
"""Audit Rasa frontend coverage (mirrors audit_frontend.py parts A-C).

- Validates vits-rasa-13/tokens.txt (+ release copy) with the sherpa-exact
  parser from audit_frontend (dup/junk detection).
- Runs all 13 RASA_TEXTS passages through old behavior (raw passthrough,
  as generate_rasa_samples.py does today) vs new frontend_for_lang, and
  reports dropped chars per language.
Writes audit/rasa_audit.json + audit/rasa_report.md.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_frontend import validate_tokens_file  # noqa: E402
from generate_rasa_samples import RASA_TEXTS  # noqa: E402
from tts_synth import frontend_for_lang  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(BASE, "audit")


def load_vocab(path):
    vocab = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.strip():
            ch, idx = line.rsplit(" ", 1)
            vocab[ch] = int(idx)
    return vocab


def main():
    rep = {"tokens": {}, "langs": {}}
    for label, path in (("committed", os.path.join(BASE, "vits-rasa-13", "tokens.txt")),
                        ("release", os.path.join(BASE, "release_assets_rasa",
                                                  "vits-rasa-13-tokens.txt"))):
        if not os.path.exists(path):
            rep["tokens"][label] = {"missing": True}
            continue
        issues, n = validate_tokens_file(path)
        rep["tokens"][label] = {"issues": issues, "symbols": n}

    vocab = load_vocab(os.path.join(BASE, "vits-rasa-13", "tokens.txt"))

    def dropped(text):
        return sorted({ch for ch in text if ch not in vocab and not ch.isspace()},
                      key=ord)

    for lang, data in sorted(RASA_TEXTS.items()):
        text = data["text"]
        before = dropped(text)
        after_text = frontend_for_lang(text, lang)
        after = dropped(after_text)
        rep["langs"][lang] = {
            "before": ["U+%04X" % ord(c) for c in before],
            "after": ["U+%04X" % ord(c) for c in after],
        }
        print("%s dropped: before=%d %s | after=%d %s" % (
            lang, len(before), rep["langs"][lang]["before"],
            len(after), rep["langs"][lang]["after"]))

    json.dump(rep, open(os.path.join(AUDIT, "rasa_audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    lines = ["# Rasa audit", "",
             "## tokens.txt validation"]
    for label, v in rep["tokens"].items():
        lines.append("- %s: %s" % (label, v.get("issues") or "clean (%d symbols)" % v.get("symbols", 0)))
    lines += ["", "## passage dropped chars: raw vs frontend", ""]
    for lang, d in sorted(rep["langs"].items()):
        lines.append("- %s: before=%d %s | after=%d %s" % (
            lang, len(d["before"]), d["before"], len(d["after"]), d["after"]))
    open(os.path.join(AUDIT, "rasa_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote audit/rasa_audit.json + audit/rasa_report.md")


if __name__ == "__main__":
    main()
