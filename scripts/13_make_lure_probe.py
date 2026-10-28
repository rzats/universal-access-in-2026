"""Near-word probe on real documents.

For every document page used in the benchmark, one target word (a common Ukrainian word of
6-12 letters that occurs once on the page) is overpainted with a near-word: the same word
with one letter replaced by another of the same class, not in the word list. Each document
yields a pair of stimuli with identical framing (full) and quality (clean or blur):
  lprobe-XXX-o  original page        (target = the real word)
  lprobe-XXX-l  edited page          (target = the near-word)
The question is whether models read the near-word as printed or report the real word.
Output: data/probe/stimuli.jsonl and data/probe/images/ (UAIS_SET=probe).
"""
import json, os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from realstim import photograph, gt_tokens, lines_from_pdf
from textnorm import toks

SEED = 20261023
VOW, CONS = "аеиіоуяюєї", "бвгґджзклмнпрстфхцчшщ"
SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"


def lexicon():
    lex = set()
    for ln in open(os.path.join(ROOT, "data", "assets", "uk_full.txt"), encoding="utf-8"):
        w, _, c = ln.strip().partition(" ")
        if w and int(c or 1) >= 2:
            lex.add(w.lower())
    return lex


def words(line):
    """(start, end) char spans of space-separated words with boxes."""
    out, s = [], None
    for i, b in enumerate(line["boxes"] + [None]):
        if b is not None and s is None:
            s = i
        elif b is None and s is not None:
            out.append((s, i)); s = None
    return out


def lure_of(w, rng, lex):
    idx = [i for i in range(1, len(w) - 1) if w[i].lower() in VOW + CONS]
    rng.shuffle(idx)
    for i in idx:
        pool = VOW if w[i].lower() in VOW else CONS
        for c in rng.sample(pool, len(pool)):
            if c == w[i].lower():
                continue
            c2 = c.upper() if w[i].isupper() else c
            cand = w[:i] + c2 + w[i + 1:]
            if cand.lower() not in lex:
                return cand
    return None


def main():
    lex = lexicon()
    stim = [json.loads(l) for l in open(os.path.join(ROOT, "data", "real", "stimuli.jsonl"), encoding="utf-8")]
    docs = [s for s in stim if s["genre"] == "document"]
    os.makedirs(os.path.join(ROOT, "data", "probe", "images"), exist_ok=True)
    out = open(os.path.join(ROOT, "data", "probe", "stimuli.jsonl"), "w", encoding="utf-8")
    n = 0
    for k, s in enumerate(docs):
        rng = random.Random(f"{SEED}-{s['id']}")
        pdf = os.path.join(ROOT, "data", "raw_docs", s["src"].split(":", 1)[1])
        img, lines = lines_from_pdf(pdf, s["attribution"].get("page", 1), dpi=150, max_words=150)
        if img is None:
            continue
        counts = {}
        for l in lines:
            for t in toks(l["raw"]):
                counts[t] = counts.get(t, 0) + 1
        cands = []
        for li, l in enumerate(lines):
            for a, b in words(l):
                w = l["raw"][a:b]
                core = w.strip(".,:;!?()«»\"'’")
                if core != w or not (6 <= len(w) <= 12) or not w.isalpha():
                    continue
                if w.lower() not in lex or counts.get(w.lower(), 0) != 1 or w.isupper():
                    continue
                cands.append((li, a, b, w))
        if not cands:
            continue
        li, a, b, w = rng.choice(cands)
        lw = lure_of(w, rng, lex)
        if lw is None:
            continue
        # overpaint the word box and typeset the near-word to the same width
        box = np.array([p for q in lines[li]["boxes"][a:b] for p in q])
        x0, y0 = box.min(0)
        x1, y1 = box.max(0)
        arr = np.array(img.convert("L"), float)
        patch = arr[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
        ink = int(np.percentile(patch, 3)) if patch.size else 0
        bg = tuple(int(v) for v in np.median(np.array(img)[max(0, int(y0) - 3):int(y1) + 4, max(0, int(x0) - 6):int(x1) + 7].reshape(-1, 3), 0))
        edited = img.copy()
        d = ImageDraw.Draw(edited)
        d.rectangle([x0 - 1, y0, x1 + 1, y1], fill=bg)
        size = int((y1 - y0) * 0.95)
        f = ImageFont.truetype(SERIF, max(6, size))
        while f.getlength(lw) > (x1 - x0) * 1.04 and size > 6:
            size -= 1
            f = ImageFont.truetype(SERIF, size)
        asc, desc = f.getmetrics()
        d.text((x0, y1 - asc - desc * 0.35), lw, font=f, fill=(ink, ink, ink))
        quality = rng.choice(["clean", "blur"])
        # the target must be legible in the final 1024-px photograph: letter height >= 11 px
        if (y1 - y0) * 1024 / (max(img.size) * 1.3) < 11:
            continue
        for tag, im, word in [("o", img, w), ("l", edited, lw)]:
            L = [dict(l) for l in lines]
            L[li] = dict(L[li], raw=L[li]["raw"][:a] + word + L[li]["raw"][b:])
            r2 = random.Random(f"{SEED}-{s['id']}-photo")  # identical degradation for both versions
            ph, jpg, info = photograph(im, L, quality, "none", r2, sheet=True)
            sid = f"lprobe-{k:03d}-{tag}"
            path = os.path.join(ROOT, "data", "probe", "images", sid + ".jpg")
            q_ = jpg
            while True:
                ph.save(path, quality=q_)
                if os.path.getsize(path) <= 240_000 or q_ <= 30:
                    break
                q_ -= 5
            info["jpeg_saved"] = q_
            rec = {"id": sid, "genre": "document", "kind": s["kind"], "quality": quality, "framing": "none",
                   "cond": "probe_" + ("original" if tag == "o" else "nearword"), "rep": 0, "src": s["src"],
                   "attribution": s["attribution"], "framing_info": info,
                   "target": {"printed": word.lower(), "real": w.lower(), "near": lw.lower(), "line": li},
                   "lines": [{"raw": l["raw"], "vis": l["vis"], **({"bbox": l["bbox"]} if "bbox" in l else {})} for l in L], "tokens": gt_tokens(L, lex)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("probe stimuli:", n)


if __name__ == "__main__":
    main()
