#!/usr/bin/env python3

# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

import argparse, io, json, os, sys, glob, shutil, tarfile, zipfile, subprocess, base64
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset")
INDEX_DIR = os.path.join(ROOT, "indices")
IMAGES_OUT = os.path.join(ROOT, "images")
sys.path.insert(0, os.path.join(ROOT, "prepare"))
from sources import SOURCES
from vcr_overlay import apply_overlay


def _download(url, dest, insecure=False):
    if os.path.exists(dest):
        print(f"    already downloaded: {os.path.basename(dest)}"); return
    print(f"    downloading {url}")
    cmd = ["curl", "-L", "--fail", "-o", dest, url]
    if insecure:
        cmd.insert(1, "-k")
    subprocess.check_call(cmd)


def _extract(archive_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = archive_path.lower()
    if n.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive_path) as t: t.extractall(out_dir)
    elif n.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as z: z.extractall(out_dir)
    elif n.endswith(".rar"):
        tool = shutil.which("unar") or shutil.which("unrar")
        if not tool:
            raise RuntimeError("install `unar` or `unrar` to extract .rar archives")
        cmd = [tool, "-o", out_dir, archive_path] if "unar" in tool \
              else [tool, "x", "-o" + out_dir, archive_path]
        subprocess.check_call(cmd)
    else:
        raise RuntimeError(f"unknown archive type: {archive_path}")


def obtain(label, cfg):
    if cfg["mode"] == "hf":
        return
    archives = cfg.get("archive")
    if isinstance(archives, str):
        archives = [archives]
    for arc in (archives or []):
        ap = os.path.join(DATASET_DIR, arc)
        if cfg["mode"] == "auto":
            os.makedirs(DATASET_DIR, exist_ok=True)
            _download(cfg["url"], ap, insecure=cfg.get("insecure", False))
        elif not os.path.exists(ap):
            raise FileNotFoundError(
                f"[{label}] place `{arc}` in dataset/ (see prepare/DATA_PREPARATION.md). Source: {cfg.get('url')}")
        if cfg.get("unpack"):
            root = os.path.join(DATASET_DIR, cfg.get("image_root", label) + "_unpacked")
            if not (os.path.isdir(root) and any(os.scandir(root))):
                _extract(ap, root)


def _decode(v):
    if isinstance(v, Image.Image): return v
    if isinstance(v, (bytes, bytearray)): return Image.open(io.BytesIO(v))
    if isinstance(v, dict) and v.get("bytes"): return Image.open(io.BytesIO(v["bytes"]))
    if isinstance(v, dict) and v.get("path") and os.path.exists(v["path"]): return Image.open(v["path"])
    if isinstance(v, str) and len(v) > 64 and v[:20].isascii() and "/" not in v[:8]:
        try: return Image.open(io.BytesIO(base64.b64decode(v)))
        except Exception: return None
    return None


def _save(im, target, size):
    out = os.path.join(ROOT, target)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im = im.convert("RGB")
    if size and list(im.size) != list(size):
        im = im.resize(tuple(size), Image.LANCZOS)
    im = apply_overlay(im, os.path.basename(target))
    im.save(out, quality=95)


def materialize_hf(entries):
    from datasets import load_dataset
    groups = {}
    for e in entries:
        hf = e["hf"]
        groups.setdefault((hf["repo"], hf.get("subset"), hf["split"]), []).append(e)
    for (repo, subset, split), items in groups.items():
        want = {}
        for e in items:
            want.setdefault(e["hf"]["index"], []).append(e)
        ds = load_dataset(repo, subset, split=split, streaming=True) if subset \
            else load_dataset(repo, split=split, streaming=True)
        imgcols = [k for k, v in ds.features.items()
                   if v.__class__.__name__ == "Image"
                   or (v.__class__.__name__ in ("List", "Sequence")
                       and getattr(v, "feature", None).__class__.__name__ == "Image")] \
            or [k for k in ds.features if "image" in k.lower() or k.lower() in ("img", "decoded_image")]
        for i, ex in enumerate(ds):
            if i not in want:
                continue
            imgs = []
            for c in imgcols:
                v = ex.get(c)
                for cv in (v if isinstance(v, (list, tuple)) else [v]):
                    im = _decode(cv)
                    if im is not None:
                        imgs.append(im)
            for e in want[i]:
                pos = e["hf"].get("img_pos", 0)
                if pos < len(imgs):
                    _save(imgs[pos], e["target"], e["size"])
            want.pop(i, None)
            if not want:
                break


def materialize_pk(entries):
    import pickle, numpy as np
    groups = {}
    for e in entries:
        groups.setdefault(e["pk"]["file"], []).append(e)
    for fname, items in groups.items():
        hits = glob.glob(os.path.join(DATASET_DIR, "**", fname), recursive=True)
        if not hits:
            print(f"    pickle {fname} not found under dataset/ — skipping {len(items)}"); continue
        data = pickle.load(open(hits[0], "rb"))
        for e in items:
            a = np.asarray(data[e["pk"]["index"]][e["pk"]["key"]])
            if a.dtype != "uint8":
                a = a.astype("uint8")
            _save(Image.fromarray(a if a.ndim == 2 else a[..., :3]), e["target"], e["size"])


def materialize_file(label, cfg, entries):
    arc = os.path.join(DATASET_DIR, cfg["archive"] if isinstance(cfg.get("archive"), str)
                       else cfg.get("image_root", label))
    want = {}
    for e in entries:
        want.setdefault(e["archive_path"], []).append(e)
    if arc.endswith((".tar", ".tar.gz", ".tgz")) and os.path.exists(arc):
        remaining = dict(want)
        with tarfile.open(arc, "r|*") as tf:
            for m in tf:
                if m.name in remaining:
                    data = tf.extractfile(m).read()
                    for e in remaining.pop(m.name):
                        _save(Image.open(io.BytesIO(data)), e["target"], e["size"])
                    if not remaining:
                        break
        return
    if arc.endswith(".zip") and os.path.exists(arc):
        with zipfile.ZipFile(arc) as zf:
            names = set(zf.namelist())
            for ap, es in want.items():
                if ap in names:
                    data = zf.read(ap)
                    for e in es:
                        _save(Image.open(io.BytesIO(data)), e["target"], e["size"])
        return
    base = {}
    for p in glob.glob(os.path.join(DATASET_DIR, "**", "*.*"), recursive=True):
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            base.setdefault(os.path.basename(p), p)
    for e in entries:
        src = base.get(os.path.basename(e.get("archive_path", e["source_file"])))
        if src:
            _save(Image.open(src), e["target"], e["size"])


def reconstruct(label, cfg):
    idx_path = os.path.join(INDEX_DIR, f"{label}.json")
    if not os.path.exists(idx_path):
        print(f"  [{label}] no index file — skipping."); return 0
    idx = json.load(open(idx_path))
    entries = [e for e in idx["samples"] if any(k in e for k in ("hf", "pk", "file"))]
    missing = len(idx["samples"]) - len(entries)
    if [e for e in entries if "hf" in e]:
        materialize_hf([e for e in entries if "hf" in e])
    if [e for e in entries if "pk" in e]:
        materialize_pk([e for e in entries if "pk" in e])
    if [e for e in entries if "file" in e]:
        materialize_file(label, cfg, [e for e in entries if "file" in e])
    done = sum(1 for e in idx["samples"] if os.path.exists(os.path.join(ROOT, e["target"])))
    print(f"  [{label}] reconstructed {done}/{idx['n']} images"
          + (f"  ({missing} not yet mapped to a source)" if missing else ""))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for label, cfg in SOURCES.items():
            p = os.path.join(INDEX_DIR, f"{label}.json")
            cov = "-"
            if os.path.exists(p):
                idx = json.load(open(p)); cov = f"{idx.get('covered', '?')}/{idx['n']}"
            print(f"  {cfg['mode']:6s} idx={cov:>9s}  {label:16s} {cfg.get('license', '')}")
        return

    total = 0
    for label in ([args.only] if args.only else list(SOURCES)):
        cfg = SOURCES[label]
        print(f"[{label}] mode={cfg['mode']}")
        try:
            obtain(label, cfg)
            total += reconstruct(label, cfg)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  SKIP: {e}")
    print(f"\nDone. {total} images reconstructed into images/.")


if __name__ == "__main__":
    main()
