"""Failure cases from the real-image runs (paper/figures/fig_failures.pdf): a lexicalised
near-word (probe), a Ukrainian-letter loss, a completion of cropped text and a digit error.
Cases are drawn with a fixed seed among descriptions by Sonnet/Opus/Fable where the matched
reference line has a known position, so the crop can be shown next to the claim."""
import json, os, random, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from scoring import quoted_spans

NAMES = {"haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5", "opus": "Opus 5.5", "fable": "Fable 5.1"}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def line_of(stim, tok):
    for g in stim["tokens"]:
        if g["tok"] == tok and g.get("line", -1) >= 0:
            return stim["lines"][g["line"]]
    return None


def crop(path, bbox, pad=70, frac=None, half=190):
    """crop around a line; if frac (relative position of the token in the line) is given,
    centre a window of +-half px on the token."""
    im = Image.open(path)
    if not bbox:
        return im
    x0, y0, x1, y1 = bbox
    if frac is not None:
        cx = x0 + frac * (x1 - x0)
        x0, x1 = cx - half + pad, cx + half - pad
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.size[0], x1 + pad), min(im.size[1], y1 + pad)))


def span_of(results, model, sid, k):
    _, sp = quoted_spans(results[(model, sid)])
    return sp[k][0] if k < len(sp) else ""


UA2RU = {("і", "и"), ("ї", "і"), ("ї", "и"), ("є", "е"), ("ґ", "г")}


def ua_loss(gt, tok):
    """claimed form replaces a Ukrainian-specific letter by a Russian-shared one (same length)."""
    if len(gt) != len(tok) or gt == tok:
        return False
    d = [(a, b) for a, b in zip(gt, tok) if a != b]
    return all(x in UA2RU for x in d)


def same_context(span, line, tok):
    """the quoted span shares at least one other word with the printed line."""
    from textnorm import toks
    a = {t for t in toks(span) if not t.isdigit() and len(t) > 2 and t != tok}
    b = {t for t in toks(line) if not t.isdigit() and len(t) > 2}
    return len(a & b) > 0


def frac_of(text, needle):
    import re
    pat = "".join(re.escape(ch) + (r"[ .,:]*" if ch.isdigit() else "") for ch in needle)
    m = re.search(pat, text.lower())
    return (m.start() + m.end()) / 2 / max(1, len(text)) if m else None


def window(text, needle, w=28):
    """text around the first occurrence of needle (digits may be separated by . , : or spaces)."""
    import re
    pat = "".join(re.escape(ch) + (r"[ .,:]*" if ch.isdigit() else "") for ch in needle)
    m = re.search(pat, text.lower())
    if not m:
        return text[:2 * w]
    a, b = max(0, m.start() - w), min(len(text), m.end() + w)
    return ("…" if a else "") + text[a:b] + ("…" if b < len(text) else "")


def main():
    rng = random.Random(5)
    cases = []
    # 1. near-word probe: the real word reported instead of the printed near-word
    pst = {s["id"]: s for s in load(os.path.join(ROOT, "data", "probe", "stimuli.jsonl"))}
    pc = load(os.path.join(ROOT, "runs", "probe", "claims.jsonl"))
    pres = {(json.loads(l)["model"], json.loads(l)["id"]): json.loads(l)["result"]
            for m in NAMES for l in open(os.path.join(ROOT, "runs", "probe", "vlm", f"{m}.desc.jsonl"), encoding="utf-8")}
    cand = [c for c in pc if c["id"].endswith("-l") and c["model"] in ("sonnet", "opus", "fable")
            and c["tok"] == pst[c["id"]]["target"]["real"]]
    if cand:
        c = rng.choice(sorted(cand, key=lambda x: (x["id"], x["model"])))
        s = pst[c["id"]]
        ln = s["lines"][s["target"]["line"]]
        cases.append({"img": crop(os.path.join(ROOT, "data", "probe", "images", c["id"] + ".jpg"), ln.get("bbox"), frac=frac_of(ln["raw"], s["target"]["printed"])),
                      "title": "Near-word normalised to the real word (edited document)",
                      "printed": s["target"]["printed"], "claim": c["tok"], "model": NAMES[c["model"]],
                      "cause": "language prior overrides a clearly printed one-letter difference"})
    # 2-4 from the real benchmark
    st = {s["id"]: s for s in load(os.path.join(ROOT, "data", "real", "stimuli.jsonl"))}
    C = [c for c in load(os.path.join(ROOT, "runs", "real", "claims.jsonl")) if c["prompt"] == "desc"
         and c["model"] in ("sonnet", "opus", "fable")]
    res = {(json.loads(l)["model"], json.loads(l)["id"]): json.loads(l)["result"]
           for m in NAMES for l in open(os.path.join(ROOT, "runs", "real", "vlm", f"{m}.desc.jsonl"), encoding="utf-8")}
    picks = [
        ("Ukrainian letter replaced by a Russian-like spelling", lambda c: c["label"] == "misread" and ua_loss(c["gt"], c["tok"]) and len(c["gt"]) >= 5,
         "letters specific to Ukrainian (і, ї, є, ґ) are normalised away"),
        ("Completion of text cut off by the frame edge", lambda c: c["label"] == "completion" and c["framing"] == "crop" and len(c["gt"]) >= 6,
         "the hidden part of the word is predictable, so it is filled in"),
        ("Digit misread", lambda c: c["label"] == "misread" and c.get("sub") == "digit" and 3 <= len(c["gt"]) == len(c["tok"])
         and sum(x != y for x, y in zip(c["gt"], c["tok"])) == 1 and c["quality"] != "severe",
         "small or blurred digits carry no language prior to correct them"),
    ]
    for title, f, cause in picks:
        cand = sorted([c for c in C if f(c)], key=lambda x: (x["id"], x["model"], x["tok"]))
        rng.shuffle(cand)
        for c in cand:
            ln = line_of(st[c["id"]], c["gt"])
            if ln is None or (st[c["id"]]["genre"] != "signage" and "bbox" not in ln):
                continue
            if st[c["id"]].get("framing_info", {}).get("occluder") == "finger":  # finger occluder not shown
                continue
            if title.startswith("Digit") and not same_context(span_of(res, c["model"], c["id"], c["span"]), ln["raw"], c["tok"]):
                continue
            cases.append({"img": crop(os.path.join(ROOT, "data", "real", "images", c["id"] + ".jpg"), ln.get("bbox"), frac=frac_of(ln["raw"], c["gt"])),
                          "title": title, "printed": window(ln["raw"], c["gt"]),
                          "claim": window(span_of(res, c["model"], c["id"], c["span"]), c["tok"]),
                          "token": c["tok"], "gt": c["gt"], "model": NAMES[c["model"]], "cause": cause, "id": c["id"]})
            break
    fig, axs = plt.subplots(len(cases), 2, figsize=(7.0, 1.55 * len(cases)), gridspec_kw={"width_ratios": [1.0, 1.0]})
    for (a, b), cs in zip(axs, cases):
        a.imshow(cs["img"]); a.set_axis_off()
        b.set_axis_off()
        txt = (f"{cs['title']}\n\nPrinted: «{cs['printed']}»\n{cs['model']} quoted: «{cs['claim']}»\n"
               f"Cause: {cs['cause']}.")
        import textwrap
        b.text(0, 0.5, "\n".join(textwrap.fill(t, 58) for t in txt.split("\n")), fontsize=7.5, va="center", ha="left")
    fig.tight_layout()
    out = os.path.join(ROOT, "paper", "figures", "fig_failures.pdf")
    fig.savefig(out, dpi=200)
    json.dump([{k: v for k, v in c.items() if k != "img"} for c in cases], open(os.path.join(ROOT, "results", "failure_cases.json"), "w"),
              ensure_ascii=False, indent=1)
    print("cases:", [c["title"] for c in cases])


if __name__ == "__main__":
    main()
