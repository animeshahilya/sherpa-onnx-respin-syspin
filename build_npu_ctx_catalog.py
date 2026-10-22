"""The app's npu_ctx.json, and one upload folder per chip, from the compiled NPU contexts.

    python build_npu_ctx_catalog.py <app npu_ctx.json out> [ctx dir] [supertonic ctx dir]

Reads <ctx dir>/npu_ctx.json (build_npu_ctx.py) and <supertonic ctx
dir>/npu_ctx.json (build_supertonic_ctx.py) and writes the app's list
(eSpeak NG Advanced, assets/piper/npu_ctx.json, read by
PiperDownloads.fetchNpuDecoders):

  {"base_url": ..., "targets": {"SM8750": {"release": "npu-ctx-v1-sm8750",
     "graphs": {<source md5>: [{"name", "size_bytes", "md5_digest"}, ...]},
     "supertonic": {"source_md5", "files": [...]}}}}

Release assets are named <source md5>-<name> for the graphs (names repeat
across voices) and <name> for Supertonic (already unique). Hard links of
every file under those names go to upload/<release>/ for `gh release upload`.
"""
import json
import os
import sys

BASE_URL = "https://github.com/animeshahilya/sherpa-onnx-respin-syspin/releases/download/"
SOC_TARGET = {"57": "SM8650", "69": "SM8750", "87": "SM8850"}
RELEASE = "npu-ctx-v1-{}"


def link(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copyfile(src, dst)


def main():
    out = sys.argv[1]
    ctx_dir = sys.argv[2] if len(sys.argv) > 2 else "release_assets_npu_ctx"
    st_dir = sys.argv[3] if len(sys.argv) > 3 else "release_assets_npu_ctx_st"
    graphs = json.load(open(os.path.join(ctx_dir, "npu_ctx.json")))
    st = json.load(open(os.path.join(st_dir, "npu_ctx.json"))).get("supertonic", {})
    targets = {}
    for soc, target in SOC_TARGET.items():
        release = RELEASE.format(target.lower())
        up = os.path.join("upload", release)
        os.makedirs(up, exist_ok=True)
        entry = {"release": release, "graphs": {}}
        for md5, g in sorted(graphs.get(soc, {}).items()):
            entry["graphs"][md5] = [{k: f[k] for k in ("name", "size_bytes", "md5_digest")} for f in g["files"]]
            for f in g["files"]:
                link(os.path.join(ctx_dir, soc, md5, f["name"]), os.path.join(up, md5 + "-" + f["name"]))
        if soc in st:
            entry["supertonic"] = {"source_md5": st[soc]["source_md5"],
                                   "files": [{k: f[k] for k in ("name", "size_bytes", "md5_digest")}
                                             for f in st[soc]["files"]]}
            for f in st[soc]["files"]:
                link(os.path.join(st_dir, soc, "supertonic", f["dir"], f["name"]), os.path.join(up, f["name"]))
        targets[target] = entry
        n = len(os.listdir(up))
        size = sum(os.path.getsize(os.path.join(up, f)) for f in os.listdir(up))
        print(f"{target}: {len(entry['graphs'])} graphs, {n} assets, {size / 2**30:.2f} GB -> {up}")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"base_url": BASE_URL, "targets": targets}, f, separators=(",", ":"), sort_keys=True)
    print(f"wrote {out} ({os.path.getsize(out) // 1024} kB)")


if __name__ == "__main__":
    main()
