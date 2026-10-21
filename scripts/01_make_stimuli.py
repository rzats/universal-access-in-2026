"""Generate the synthetic stimulus set: images + exact ground truth.

Design: genre (signage / packaging / document) x quality (clean / blur / severe)
x framing (none / crop / occlusion) x name-slot condition (word / nonce / lure),
REPS images per cell. Output: data/images/<id>.jpg and data/stimuli.jsonl.
"""
import argparse, itertools, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from textgen import Gen
from render import render, gt_tokens

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--seed", type=int, default=20261021)
ap.add_argument("--out", default="data")
ap.add_argument("--only", type=int, default=0, help="debug: stop after N images")
a = ap.parse_args()

lex = set()
with open(os.path.join(a.out, "assets", "uk_full.txt"), encoding="utf-8") as f:
    for ln in f:
        w = ln.split(" ")[0].strip().lower()
        if w:
            lex.add(w)
os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
GENRES, QUAL, FRAME, COND = ["signage", "packaging", "document"], ["clean", "blur", "severe"], ["none", "crop", "occlusion"], ["word", "nonce", "lure"]
cells = list(itertools.product(GENRES, QUAL, FRAME, COND))
n = 0
with open(os.path.join(a.out, "stimuli.jsonl"), "w", encoding="utf-8") as out:
    for rep in range(a.reps):
        for g, q, fr, c in cells:
            sid = f"{g[:3]}-{q[:2]}-{fr[:2]}-{c[:2]}-{rep}"
            rng = random.Random(f"{a.seed}-{sid}")
            gen = Gen(rng, lex)
            for attempt in range(20):
                kind, L = getattr(gen, g)(c)
                lines = []
                for role, pieces in L:
                    raw = "".join(p for p, _ in pieces)
                    cats = [cat for p, cat in pieces for _ in p]
                    lines.append({"role": role, "raw": raw, "cats": cats})
                spec = {"id": sid, "genre": g, "kind": kind, "quality": q, "framing": fr, "cond": c,
                        "rep": rep, "lines": lines}
                img, jpg = render(spec, rng)
                toks = gt_tokens(lines)
                hidden = sum(t["state"] != "visible" for t in toks)
                # crop/occlusion stimuli must actually hide something; 'none' must hide nothing
                if (fr == "none") == (hidden == 0):
                    break
            spec["attempts"] = attempt + 1
            spec["tokens"] = toks
            img.save(os.path.join(a.out, "images", sid + ".jpg"), quality=jpg)
            for ln in spec["lines"]:  # per-character slot category, one letter each
                ln["cats"] = "".join({"word": "w", "name": "n", "nonce": "x", "lure": "l", "num": "d"}[c] for c in ln["cats"])
            out.write(json.dumps(spec, ensure_ascii=False) + "\n")
            n += 1
            if a.only and n >= a.only:
                sys.exit(0)
print("wrote", n, "stimuli")
