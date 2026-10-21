"""Build the real-image benchmark: real Ukrainian packaging photographs, real documents
and real signage photographs, each assigned one (quality, framing) condition and degraded
with the same procedure as the synthetic study.

Sources and ground truth
  packaging  Open Food Facts raw photographs (data/off, scripts/10_harvest_off.py); ground truth
             = Google Cloud Vision OCR stored by OFF (character boxes), plus the contributor's
             transcription, product name and brand as additional reference tokens
  document   real PDFs with a text layer (data/raw_docs); ground truth = PDF text layer with word
             boxes; the page region is "photographed" as a sheet lying on a surface
  signage    Wikimedia Commons photographs (data/raw_signs); ground truth = reference
             transcription (data/raw_signs/transcriptions.jsonl); no character boxes, so
             signage is only degraded in quality (framing = none)
Output: data/real/images/<id>.jpg, data/real/stimuli.jsonl
"""
import itertools, json, os, random, sys
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from realstim import photograph, gt_tokens, lines_from_gcv, lines_from_pdf
from textnorm import toks

SEED = 20261022
OUT = os.path.join(ROOT, "data", "real")
QUAL, FRAME = ["clean", "blur", "severe"], ["none", "crop", "occlusion"]


def lexicon():
    lex = set()
    for ln in open(os.path.join(ROOT, "data", "assets", "uk_full.txt"), encoding="utf-8"):
        w, _, c = ln.strip().partition(" ")
        if w and int(c or 1) >= 2:
            lex.add(w.lower())
    return lex


def cells(n, rng, frames=FRAME):
    """A balanced, shuffled sequence of (quality, framing) cells of length n."""
    base = list(itertools.product(QUAL, frames))
    seq = []
    while len(seq) < n:
        b = base[:]
        rng.shuffle(b)
        seq += b
    return seq[:n]


def tess_agreement(img, lines):
    """Share of text-layer word tokens that Tesseract also reads on the clean page render."""
    import subprocess, tempfile
    T = os.path.join(ROOT, "data", "assets", "tess")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.save(f.name)
        out = subprocess.run([os.path.join(T, "squashfs-root", "AppRun"), f.name, "-", "-l", "ukr", "--psm", "3"],
                             capture_output=True, text=True, timeout=300,
                             env=dict(os.environ, TESSDATA_PREFIX=os.path.join(T, "tessdata"), OMP_THREAD_LIMIT="1")).stdout
    ref = [t for l in lines for t in toks(l["raw"]) if not t.isdigit() and len(t) > 2]
    got = set(toks(out))
    return sum(t in got for t in ref) / max(1, len(ref))


def sources():
    items = []
    # packaging
    p = os.path.join(ROOT, "data", "off", "candidates.jsonl")
    for c in (json.loads(l) for l in open(p, encoding="utf-8")):
        if not c["ok"]:
            continue
        fta = json.load(open(os.path.join(ROOT, "data", "off", c["key"] + ".ocr.json"), encoding="utf-8"))
        if c["panel"] == "front":  # independent check of the OCR: the contributor's product name must be read
            name = [t for t in toks(c.get("product_name_uk") or "") if len(t) > 2]
            got = set(toks(fta.get("text", "")))
            if not name or sum(t in got for t in name) / len(name) < 0.5:
                continue
        lines = lines_from_gcv(fta, c["scale"])
        extra = toks(" ".join(x for x in [c.get("ingredients_text_uk"), c.get("product_name_uk"),
                                         c.get("brands") if isinstance(c.get("brands"), str) else ""] if x))
        items.append({"genre": "packaging", "kind": c["panel"], "src": f"off:{c['key']}",
                      "attribution": {"source": "Open Food Facts", "url": c["url"], "licence": "CC BY-SA 3.0"},
                      "image": os.path.join(ROOT, "data", "off", c["key"] + ".jpg"), "lines": lines,
                      "extra": sorted(set(extra)), "sheet": False})
    # documents
    p = os.path.join(ROOT, "data", "raw_docs", "manifest.jsonl")
    if os.path.exists(p):
        for d in (json.loads(l) for l in open(p, encoding="utf-8")):
            pdf = os.path.join(ROOT, "data", "raw_docs", d["file"])
            best = None
            # cover pages of bulletins carry little text: take the richest of the first pages
            for page in ([2, 3, 4] if d.get("category") == "book" else [1]):
                try:
                    img, lines = lines_from_pdf(pdf, page, dpi=150, max_words=150)
                except Exception as e:
                    continue
                if img is None:
                    continue
                nw = sum(len(toks(l["raw"])) for l in lines)
                if best is None or nw > best[0]:
                    best = (nw, img, lines, page)
            if best is None or best[0] < 25:
                print("skip (too little text)", d["file"]); continue
            nw, img, lines, page = best
            # quality gate: the text layer must match what is drawn (Tesseract on the clean render)
            agree = tess_agreement(img, lines)
            if agree < 0.6:
                print("skip (text layer disagrees with rendering)", d["file"], round(agree, 2)); continue
            items.append({"genre": "document", "kind": d.get("category", "document"), "src": f"doc:{d['file']}",
                          "attribution": {"source": d.get("publisher"), "url": d.get("source_url"), "page": page,
                                          "title": d.get("title"), "licence": d.get("licence_note"),
                                          "tesseract_agreement": round(agree, 3)},
                          "pil": img, "lines": lines, "extra": [], "sheet": True})
    # signage
    p = os.path.join(ROOT, "data", "real", "sign_transcriptions.jsonl")
    if os.path.exists(p):
        man = {json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "raw_signs", "manifest.jsonl"), encoding="utf-8")}
        for t in (json.loads(l) for l in open(p, encoding="utf-8")):
            if not t.get("use", True):
                continue
            m = man.get(t["file"], {})
            lines = [{"raw": ln, "boxes": [None] * len(ln)} for ln in t["lines"] if ln.strip()]
            items.append({"genre": "signage", "kind": m.get("category", "sign"), "src": f"sign:{t['file']}",
                          "attribution": {"source": "Wikimedia Commons", "url": m.get("page_url"),
                                          "author": m.get("author"), "licence": m.get("licence")},
                          "image": os.path.join(ROOT, "data", "raw_signs", t["file"]), "lines": lines,
                          # text the transcriber could not confirm (tiny, damaged, partly hidden) is accepted
                          # as correct when quoted, but does not count towards coverage
                          "extra": sorted(set(toks(" ".join(t.get("uncertain", []))))), "sheet": False, "no_boxes": True})
    return items


def main():
    lex = lexicon()
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    items = sources()
    rng0 = random.Random(SEED)
    by = {}
    for it in items:
        by.setdefault(it["genre"], []).append(it)
    n = 0
    with open(os.path.join(OUT, "stimuli.jsonl"), "w", encoding="utf-8") as out:
        for g in ["signage", "packaging", "document"]:
            its = sorted(by.get(g, []), key=lambda x: x["src"])
            frames = ["none"] if g == "signage" else FRAME
            seq = cells(len(its), random.Random(f"{SEED}-{g}"), frames)
            for k, (it, (q, fr)) in enumerate(zip(its, seq)):
                sid = f"{g[:3]}-{k:03d}"
                rng = random.Random(f"{SEED}-{sid}")
                img = it.get("pil") or Image.open(it["image"]).convert("RGB")
                lines = [dict(l) for l in it["lines"]]
                if it.get("no_boxes"):
                    for l in lines:
                        l["vis"] = [2] * len(l["raw"])
                    s = 1024 / max(img.size)
                    img2, jpg, info = photograph(img.resize((int(img.size[0] * s), int(img.size[1] * s))),
                                                 [{"raw": " ", "boxes": [[(0, 0), (1, 0), (1, 1), (0, 1)]]}], q, "none", rng)
                else:
                    for attempt in range(10):
                        lines = [dict(l) for l in it["lines"]]  # conf travels with the line
                        img2, jpg, info = photograph(img, lines, q, fr, rng, sheet=it["sheet"])
                        hidden = sum(v != 2 for l in lines for v in l["vis"])
                        if fr == "none" or hidden > 0:
                            break
                toks_ = gt_tokens(lines, lex)
                vis_set = {t["tok"] for t in toks_}
                extra = [{"tok": e, "chars": e, "line": -1, "cat": "num" if e.isdigit() else ("word" if e in lex else "name"),
                          "state": "visible", "vis": [2] * len(e), "role": "extra", "extra": True}
                         for e in it["extra"] if e not in vis_set]
                # the CLI drops image attachments above ~256 KB: re-encode until the file is below 240 KB
                path = os.path.join(OUT, "images", sid + ".jpg")
                q_ = jpg
                while True:
                    img2.save(path, quality=q_)
                    if os.path.getsize(path) <= 240_000 or q_ <= 30:
                        break
                    q_ -= 5
                info["jpeg_saved"] = q_
                rec = {"id": sid, "genre": g, "kind": it["kind"], "quality": q, "framing": fr, "cond": "real", "rep": 0,
                       "src": it["src"], "attribution": it["attribution"], "framing_info": info,
                       "lines": [{"raw": l["raw"], "vis": l["vis"]} for l in lines], "tokens": toks_ + extra}
                rec["n_lowconf"] = sum(1 for t in toks_ if t.get("lowconf"))
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print("wrote", n, {g: len(v) for g, v in by.items()})


if __name__ == "__main__":
    main()
