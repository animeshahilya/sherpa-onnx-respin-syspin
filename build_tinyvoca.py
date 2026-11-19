#!/usr/bin/env python3
"""tiny-voca (n4ze3m/tiny-voca) for the espeak-ng app's slim ONNX Runtime.

The app's runtime (android/tools/ort-voices/voices_ops.config) has no kernels
for ten operators the export uses (LessOrEqual: not at opset 17's version 16).
Each is rewritten with operators it has, exactly (same results for every
input, not approximations). A boolean "or" is done as numbers: ONNX Runtime
1.30 folds Not(And(Not a, Not b)) back into Or.

  IsNaN(x)    -> Not(Equal(x, x))
  Greater(a,b)-> Less(b, a)
  LessOrEqual(a,b) -> 0.5 < float(a < b) + float(a == b)
  Or(a,b)     -> 0.5 < float(a) + float(b)
  Xor(a,b)    -> Not(Equal(a, b))                       (bool)
  Flatten(axis=1) -> Reshape(x, [0, -1])
  Abs(x)      -> Max(x, Neg x)
  Floor(x)    -> t = float(int64(x)); t - (x < t)       (float32)
  Round(x)    -> half to even, from Floor(x) and the exact x - Floor(x)
  Mod(a, b>0) -> r = a - (a / b) * b; r < 0 ? r + b : r (int64, Python sign)

Then the outputs are checked against the original with the same noise seed,
and every operator, as written and after ONNX Runtime's optimizer, against the
runtime's list at the exact kernel version opset 17 needs (run it with the
app's ONNX Runtime version: 1.30).

  python build_tinyvoca.py <tiny-voca onnx> <out onnx> [<tiny-voca config.json>]
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

COUNT = [0]
VOWEL_LETTERS = "aeiouæɐɑɒɔəɚɛɜɪʊʌᵻ"
# eSpeak symbols the voice lacks -> its nearest phoneme (as TinyVoca.FALLBACK)
FALLBACK = {"r": "ɹ", "x": "k", "ç": "k", "ɬ": "l", "ʲ": "j", "ɑ": "ɑː", "e": "eɪ", "u": "uː"}


def tokenize(ipa, id_map):
    """eSpeak IPA to (tokens, stress) as the app's TinyVoca.ids does: greedy
    longest match, ˈ/ˌ on the next vowel, <sp> between words, a closing '.'."""
    keys = sorted((k for k in id_map if not k.startswith("<") and k not in "_^$"), key=len, reverse=True)
    one = lambda k: id_map[k][0]
    ids, stress, pending = [one("^")], [0], 0
    s, i = ipa.strip(), 0
    while i < len(s):
        c = s[i]
        if c in "ˈˌ":
            pending, i = (1 if c == "ˈ" else 2), i + 1
            continue
        if c.isspace():
            if ids[-1] not in (one("<sp>"), one("^")):
                ids.append(one("<sp>")); stress.append(0)
            i += 1
            continue
        k = next((k for k in keys if s.startswith(k, i)), None)
        used = len(k) if k else 1
        if k is None:
            k = FALLBACK.get(c)
            if k not in id_map:
                i += 1
                continue
        st = 0
        if pending and (k == "n̩" or any(ch in VOWEL_LETTERS for ch in k)):
            st, pending = pending, 0
        ids.append(one(k)); stress.append(st)
        i += used
    if ids[-1] == one("<sp>"):
        ids.pop(); stress.pop()
    if ids[-1] not in [one(p) for p in ",.?!;:"]:
        ids.append(one(".")); stress.append(0)
    ids.append(one("$")); stress.append(0)
    return ids, stress


def name(base):
    COUNT[0] += 1
    return f"{base}/tv{COUNT[0]}"


def const(nodes, value, dtype):
    out = name("c")
    nodes.append(helper.make_node("Constant", [], [out], value=numpy_helper.from_array(np.array(value, dtype=dtype))))
    return out


def node(nodes, op, inputs, base, **attrs):
    out = name(base)
    nodes.append(helper.make_node(op, inputs, [out], name=out, **attrs))
    return out


def floor(nodes, x, base):
    t = node(nodes, "Cast", [node(nodes, "Cast", [x], base, to=TensorProto.INT64)], base, to=TensorProto.FLOAT)
    below = node(nodes, "Cast", [node(nodes, "Less", [x, t], base)], base, to=TensorProto.FLOAT)
    return node(nodes, "Sub", [t, below], base)


def rewrite(n):
    """Nodes replacing n, the last one writing n's output under its name."""
    nodes, base, out = [], n.name or n.output[0], n.output[0]
    i = list(n.input)
    op = n.op_type
    if op == "IsNaN":
        last = ("Not", [node(nodes, "Equal", [i[0], i[0]], base)])
    elif op == "Greater":
        last = ("Less", [i[1], i[0]])
    elif op == "LessOrEqual":
        # At opset 17 its kernel is version 16, which the slim runtime lacks; ORT then
        # expands ONNX's own definition, Or(Less, Equal), and has no Or either.
        f = [node(nodes, "Cast", [node(nodes, o, i[:2], base)], base, to=TensorProto.FLOAT) for o in ("Less", "Equal")]
        last = ("Less", [const(nodes, 0.5, np.float32), node(nodes, "Add", f, base)])
    elif op == "Or":
        # As numbers: ONNX Runtime 1.30 folds Not(And(Not a, Not b)) back into Or
        f = [node(nodes, "Cast", [x], base, to=TensorProto.FLOAT) for x in i[:2]]
        last = ("Less", [const(nodes, 0.5, np.float32), node(nodes, "Add", f, base)])
    elif op == "Xor":
        last = ("Not", [node(nodes, "Equal", [i[0], i[1]], base)])
    elif op == "Flatten":
        assert helper.get_attribute_value(n.attribute[0]) == 1
        last = ("Reshape", [i[0], const(nodes, [0, -1], np.int64)])
    elif op == "Abs":
        last = ("Max", [i[0], node(nodes, "Neg", [i[0]], base)])
    elif op == "Floor":
        t = node(nodes, "Cast", [node(nodes, "Cast", [i[0]], base, to=TensorProto.INT64)], base, to=TensorProto.FLOAT)
        below = node(nodes, "Cast", [node(nodes, "Less", [i[0], t], base)], base, to=TensorProto.FLOAT)
        last = ("Sub", [t, below])
    elif op == "Round":
        # f = floor(x) and x - f are exact in float32 (x + 0.5 is not: 0.49999997 + 0.5 == 1.0)
        x = i[0]
        f = floor(nodes, x, base)
        frac = node(nodes, "Sub", [x, f], base)
        half_c = const(nodes, 0.5, np.float32)
        half = floor(nodes, node(nodes, "Mul", [f, half_c], base), base)
        odd = node(nodes, "Equal", [node(nodes, "Sub", [f, node(nodes, "Mul", [half, const(nodes, 2.0, np.float32)], base)], base),
                                    const(nodes, 1.0, np.float32)], base)
        # up = frac > 0.5 or (frac == 0.5 and f odd); the "or" as numbers, as Or above
        above = node(nodes, "Cast", [node(nodes, "Less", [half_c, frac], base)], base, to=TensorProto.FLOAT)
        tie_odd = node(nodes, "Cast", [node(nodes, "And", [node(nodes, "Equal", [frac, half_c], base), odd], base)],
                       base, to=TensorProto.FLOAT)
        up = node(nodes, "Less", [half_c, node(nodes, "Add", [above, tie_odd], base)], base)
        last = ("Where", [up, node(nodes, "Add", [f, const(nodes, 1.0, np.float32)], base), f])
    elif op == "Mod":
        assert helper.get_attribute_value(n.attribute[0]) == 0  # Python sign; the divisor is a positive constant
        a, b = i
        r = node(nodes, "Sub", [a, node(nodes, "Mul", [node(nodes, "Div", [a, b], base), b], base)], base)
        neg = node(nodes, "Less", [r, const(nodes, 0, np.int64)], base)
        last = ("Where", [neg, node(nodes, "Add", [r, b], base), r])
    else:
        return None
    nodes.append(helper.make_node(last[0], last[1], [out], name=base))
    return nodes


def convert(src, dst):
    m = onnx.load(src)
    consts = {nd.output[0]: numpy_helper.to_array(nd.attribute[0].t) for nd in m.graph.node if nd.op_type == "Constant"}
    out_nodes, changed = [], 0
    for n in m.graph.node:
        if n.op_type == "Mod":
            assert (consts[n.input[1]] > 0).all()
        new = rewrite(n)
        if new is None:
            out_nodes.append(n)
        else:
            out_nodes.extend(new)
            changed += 1
    del m.graph.node[:]
    m.graph.node.extend(out_nodes)
    # ONNX Runtime fuses MatMul + 1-D bias Add + Gelu into BiasGelu (com.microsoft),
    # which the slim runtime lacks; the same bias as [1, 1, C] (same broadcast
    # over [B, T, C]) keeps it as Gelu, which it has.
    users = {}
    for n in m.graph.node:
        for x in n.input:
            users.setdefault(x, []).append(n.op_type)
    reshaped = 0
    for t in m.graph.initializer:
        if t.name.endswith(".expand.bias") and len(t.dims) == 1 and users.get(t.name) == ["Add"]:
            t.dims[:] = [1, 1, t.dims[0]]
            reshaped += 1
    print(f"{reshaped} Gelu biases as [1, 1, C]")
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    left = {n.op_type for n in m.graph.node} & {"IsNaN", "Greater", "LessOrEqual", "Or", "Xor", "Flatten", "Abs", "Floor", "Round", "Mod"}
    print(f"{changed} nodes rewritten, none left: {not left}")
    return m


def seeded(path, tmp):
    m = onnx.load(path)
    for n in m.graph.node:
        if n.op_type == "RandomNormalLike":
            n.attribute.append(helper.make_attribute("seed", 7.0))
    onnx.save(m, tmp)
    return tmp


def check(src, dst):
    import onnxruntime as ort
    a = ort.InferenceSession(seeded(src, dst + ".a.tmp"), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(seeded(dst, dst + ".b.tmp"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    worst = 0.0
    for n in (3, 9, 40, 120, 300, 510):
        for voice in (0, 1):
            for speed in (0.7, 1.0, 2.0, 3.0):
                feed = {"tokens": np.concatenate([[1], rng.integers(5, 76, n), [6, 2]]).astype(np.int64)[None],
                        "voice": np.array([voice], np.int64), "speed": np.array([speed], np.float32),
                        "pitch_shift": np.array([0.0], np.float32), "pitch_range": np.array([1.0], np.float32)}
                feed["stress"] = rng.integers(0, 3, feed["tokens"].shape).astype(np.int64)
                ya, da = a.run(None, feed)
                yb, db = b.run(None, feed)
                assert (da == db).all() and ya.shape == yb.shape, (n, voice, speed)
                worst = max(worst, float(np.abs(ya - yb).max()))
    import os
    os.remove(dst + ".a.tmp")
    os.remove(dst + ".b.tmp")
    print(f"48 runs: durations identical, audio max difference {worst:.2e}")


LANGUAGE = {"code": "en_US", "family": "en", "region": "US", "name_native": "English",
            "name_english": "English", "country_english": "United States"}


def configs(tv_config, out_dir):
    """One app config per voice over the shared model (phoneme_type "tiny_voca").

    The vocabulary goes in as phoneme_id_map (one id each), with Piper's
    _ ^ $ naming its pad, start and end; the voice is default_speaker_id,
    as each Rasa speaker's config names its speaker.
    """
    import json
    import os
    c = json.load(open(tv_config, encoding="utf-8"))
    vocab = c["vocab"]
    ids = {k: [v] for k, v in vocab.items()}
    ids.update({"_": [vocab["<pad>"]], "^": [vocab["<bos>"]], "$": [vocab["<eos>"]]})
    for index, voice in enumerate(c["voices"]):
        key = f"en_US-tiny_voca_{voice.lower()}-medium"
        cfg = {"audio": {"sample_rate": c["sample_rate"], "quality": "medium"},
               "espeak": {"voice": "en-us"}, "language": LANGUAGE, "dataset": f"tiny_voca_{voice.lower()}",
               "phoneme_type": "tiny_voca", "phoneme_id_map": ids, "num_symbols": len(vocab),
               "num_speakers": 1, "speaker_id_map": {}, "default_speaker_id": index,
               "hop_length": c["hop"], "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
               "tiny_voca": {"source": "n4ze3m/tiny-voca", "step": c["step"], "voice": voice}}
        path = os.path.join(out_dir, key + ".onnx.json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("wrote", path)


OPS_CONFIG = "../espeak-ng/android/tools/ort-voices/voices_ops.config"


def check_ops(path, ops_config=OPS_CONFIG):
    import os
    import onnxruntime as ort
    from onnx import defs
    have = set()
    for line in open(ops_config, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            dom, ver, ops = line.strip().split(";")
            have |= {(dom, int(ver), o.strip()) for o in ops.split(",")}
    opt = path + ".opt.tmp"
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.optimized_model_filepath = opt
    o.log_severity_level = 3
    ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
    for p in (path, opt):
        miss = set()
        for n in onnx.load(p).graph.node:
            dom = n.domain or "ai.onnx"
            if dom == "ai.onnx":
                v = defs.get_schema(n.op_type, 17, "").since_version
                if (dom, v, n.op_type) not in have:
                    miss.add(f"{n.op_type}({v})")
            elif dom == "com.microsoft.nchwc":
                continue  # x86-only layouts: the phone's optimizer never makes them
            elif not any(h[0] == dom and h[2] == n.op_type for h in have):
                miss.add(f"{dom}.{n.op_type}")
        assert not miss, f"{p}: no kernel in the slim runtime for {sorted(miss)}"
    os.remove(opt)
    print(f"every operator has a kernel in the slim runtime (ONNX Runtime {ort.__version__}, as written and optimized)")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
    check(sys.argv[1], sys.argv[2])
    check_ops(sys.argv[2])
    if len(sys.argv) > 3:
        import os
        configs(sys.argv[3], os.path.dirname(os.path.abspath(sys.argv[2])))
