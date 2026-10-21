"""Label every claimed token of every response and attach detector features.

Output (git-ignored, contains model text): runs/claims.jsonl, one record per claimed
token of a (model, prompt, image) response, and runs/responses.jsonl, one record per
response with coverage statistics. Aggregates are computed by 05_analyse.py.

Detector features for a claimed token o of response R (model m, image i):
  reread      o appears among the tokens of m's own transcription of i ('trans' prompt)
  reread_unc  o is marked '(?)' in that transcription
  xmodel      fraction of the other main models whose description or transcription of i
              contains o
  ocr         o appears in the Tesseract reading of i (psm 3 or 11)
  ocr_sim     best normalised edit similarity of o to any Tesseract token
  lex         o is in the Ukrainian word list (log10 frequency otherwise 0)
  hedged      the sentence quoting o carries a hedging cue
"""
import collections, json, math, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from paths import DATA, RUNS, RESULTS  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "src"))
from scoring import claims_desc, claims_trans, label
from textnorm import toks, sim

MAIN = ["haiku", "sonnet", "opus"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def main():
    stim = {s["id"]: s for s in load_jsonl(os.path.join(DATA, "stimuli.jsonl"))}
    freq = {}
    for ln in open(os.path.join(ROOT, "data", "assets", "uk_full.txt"), encoding="utf-8"):
        w, _, c = ln.strip().partition(" ")
        if w:
            freq[w.lower()] = freq.get(w.lower(), 0) + int(c or 1)
    lexicon = {w for w, c in freq.items() if c >= 2}
    # real names the generator draws from also count as "real words" when a near-word is normalised to them
    import textgen
    known = set(lexicon)
    for lst in (textgen.BRAND_WORDS, textgen.SURNAMES, textgen.DRUGS, textgen.STREETS, textgen.CITIES):
        for w in lst:
            known.update(toks(w))
    ocr = {r["id"]: set(toks(r["psm3"])) | set(toks(r["psm11"])) for r in load_jsonl(os.path.join(RUNS, "ocr.jsonl"))}
    resp = {}
    for m in MAIN + ["fable"]:
        for p in ["desc", "trans"]:
            for r in load_jsonl(os.path.join(RUNS, "vlm", f"{m}.{p}.jsonl")):
                resp[(m, p, r["id"])] = r["result"]
    readtoks = {k: set(toks(re.sub(r"\[[^\]]*\]", " ", v))) for k, v in resp.items()}
    unc = {}
    for (m, p, i), v in resp.items():
        if p == "trans":
            unc[(m, i)] = {c["tok"] for c in claims_trans(v) if c["uncertain"]}

    out_c = open(os.path.join(RUNS, "claims.jsonl"), "w", encoding="utf-8")
    out_r = open(os.path.join(RUNS, "responses.jsonl"), "w", encoding="utf-8")
    for (m, p, i), text in sorted(resp.items()):
        s = stim[i]
        gt = s["tokens"]
        cl = claims_desc(text) if p == "desc" else claims_trans(text)
        n_lab = collections.Counter()
        matched = set()
        for c in cl:
            o = c["tok"]
            lab = label(o, gt, known)
            if lab["label"] in ("misread", "fabricated") and any(g.get("lowconf") and g["tok"] == lab["gt"] for g in gt) \
                    and lab["sim"] >= 0.5:
                lab["label"] = "unscored"  # the matched reference word is itself an uncertain OCR reading
            n_lab[lab["label"]] += 1
            if lab["label"] == "correct":
                matched.add(lab["gt"])
            others = [mm for mm in MAIN if mm != m]
            xm = [any(o in readtoks.get((mm, pp, i), set()) for pp in ["desc", "trans"]) for mm in others
                  if any((mm, pp, i) in readtoks for pp in ["desc", "trans"])]
            ocr_t = ocr.get(i)
            rec = {"id": i, "model": m, "prompt": p, "tok": o, "span": c["span"], "hedged": c["hedged"],
                   "uncertain": c["uncertain"], **lab,
                   "genre": s["genre"], "kind": s["kind"], "quality": s["quality"], "framing": s["framing"],
                   "cond": s["cond"], "rep": s["rep"], "is_num": o.isdigit(), "len": len(o),
                   "reread": (o in readtoks[(m, "trans", i)]) if (m, "trans", i) in readtoks else None,
                   "reread_unc": (o in unc[(m, i)]) if (m, i) in unc else None,
                   "xmodel": (sum(xm) / len(xm)) if xm else None,
                   "ocr": (o in ocr_t) if ocr_t is not None else None,
                   "ocr_sim": max([sim(o, t) for t in ocr_t] or [0.0]) if ocr_t is not None else None,
                   "lex": o in lexicon, "lexfreq": math.log10(freq.get(o, 0) + 1)}
            out_c.write(json.dumps(rec, ensure_ascii=False) + "\n")
        vis = [g for g in gt if g["state"] == "visible" and not g.get("extra")]
        out_r.write(json.dumps({"id": i, "model": m, "prompt": p, "genre": s["genre"], "quality": s["quality"],
                                "framing": s["framing"], "cond": s["cond"], "rep": s["rep"], "n_claims": len(cl),
                                **{f"n_{k}": n_lab[k] for k in ["correct", "completion", "misread", "fabricated"]},
                                "n_hedged_spans": len({c["span"] for c in cl if c["hedged"]}),
                                "n_gt_visible": len(vis),
                                "n_gt_visible_read": sum(1 for g in vis if g["tok"] in matched),
                                "says_illegible": bool(re.search(r"нерозбірлив|не видно|не можу прочитати|неможливо прочитати|не вдається прочитати|закрит|обрізан", text, re.I)),
                                "chars": len(text)}, ensure_ascii=False) + "\n")
    out_c.close()
    out_r.close()


if __name__ == "__main__":
    main()
