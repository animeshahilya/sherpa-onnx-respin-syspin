"""Writes the espeak-ng app's JVM test cases for Supertonic.preprocess from
Supertone's reference (py/helper.py, pinned in check_supertonic.py):
lang <TAB> input <TAB> expected, inputs escaped as JSON strings."""
import json, sys
from pathlib import Path
import check_supertonic as c
h = c.helper_module(Path("supertonic_src"))
tp = h.UnicodeProcessor("release_assets_supertonic/supertonic3-unicode_indexer.json")
cases = [
    ("en", "Hello world"), ("en", "Settings, double tap to open."), ("en", "  spaces   and\ttabs\nnew line  "),
    ("en", "“Quoted” and ‘single’ — dash – en-dash"), ("en", "e.g., this i.e., that @home"),
    ("en", "Emoji \U0001F600 here ❤ ok"), ("en", "Wait ,really ?Yes !Done ;semi :colon"),
    ("en", "He said \"\"hi\"\" ''twice''"), ("en", "path/to#tag [x] {y} a|b"), ("en", "Price: $5.2M (approx)"),
    ("en", "Ends with quote\""), ("en", "Café naïve résumé"), ("en", " nbsp inside em"),
    ("hi", "नमस्ते, आज मौसम बहुत अच्छा है।"), ("hi", "बैटरी 45% बची है"), ("ko", "안녕하세요 반갑습니다"),
    ("ja", "こんにちは、元気ですか"), ("ar", "مرحبا بالعالم"), ("ru", "Привет, мир"), ("de", "Grüße aus München"),
    ("fr", "L'été dernier, «bonjour»"), ("vi", "Xin chào thế giới"), ("el", "Γειά σου κόσμε;"), ("tr", "İstanbul'da ışık"),
    ("es", "¿Qué tal? ¡Muy bien!"), ("en", "♥ love © 2024 \ back"), ("en", "a_b→c←d´e`f"), ("en", "x...")]
out = []
for lang, text in cases:
    exp = tp._preprocess_text(text, lang)
    out.append(f"{lang}\t{json.dumps(text, ensure_ascii=True)}\t{json.dumps(exp, ensure_ascii=True)}")
dst = Path(sys.argv[1])
dst.write_text("\n".join(out) + "\n", encoding="utf-8")
print(len(out), "cases ->", dst)
