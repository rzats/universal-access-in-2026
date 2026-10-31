"""Annotated real-image examples (paper/figures/fig_annotated.pdf).

For a few real stimuli, the quoted text of one model's description is shown next to the
image, every quoted word coloured by the deterministic label: correct, completion, or
hallucinated. Each hallucination label is additionally checked by the Claude audit rater
(same prompt as scripts/15_audit_raters.py), which separates confirmed hallucinations
from probable reference gaps (text printed but missing from the reference). Words the
combined run-time rule flags (Section 3.6) are marked with a dagger.
Rater outputs are cached in runs/audit/annotated.jsonl.
"""
import importlib.util, json, os, re, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from scoring import quoted_spans
from textnorm import toks

spec = importlib.util.spec_from_file_location("aud", os.path.join(ROOT, "scripts", "15_audit_raters.py"))
aud = importlib.util.module_from_spec(spec); spec.loader.exec_module(aud)

CASES = [("pac-067", "sonnet"), ("doc-022", "opus")]  # chosen for illustration (full frame, blur)
NAME = {"haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5", "opus": "Opus 5.5", "fable": "Fable 5.1"}
COL = {"correct": "#52514e", "completion": "#6250d6", "confirmed": "#e34948", "gap": "#eda100"}
CACHE = os.path.join(ROOT, "runs", "audit", "annotated.jsonl")


def rate(row):
    """Claude rater verdict for one token (cached)."""
    cache = {json.loads(l)["key"]: json.loads(l) for l in open(CACHE, encoding="utf-8")} if os.path.exists(CACHE) else {}
    if row["key"] in cache:
        return cache[row["key"]]
    imgdir = os.path.join(ROOT, "runs", "audit", "img")
    os.makedirs(imgdir, exist_ok=True)
    name = "ann_" + re.sub(r"\W", "_", row["id"]) + ".jpg"
    with open(os.path.join(ROOT, row["image"]), "rb") as a, open(os.path.join(imgdir, name), "wb") as b:
        b.write(a.read())
    p = subprocess.run(["claude", "-p", aud.prompt_of(row) + f"\n\n@{name}", "--model", aud.CLAUDE, "--output-format", "json",
                        "--tools", "", "--setting-sources", "", "--no-session-persistence"], cwd=imgdir,
                       capture_output=True, text=True, timeout=300)
    d = json.loads(p.stdout)
    v, printed, reason = aud.parse(d["result"])
    rec = {"key": row["key"], "verdict": v, "printed": printed, "reason": reason, "cost": d.get("total_cost_usd", 0)}
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def draw_words(ax, lines, x0=0.0, y0=0.98, fs=6.6, lh=0.055, width=1.0):
    """Lay out (word, colour, mark) items line by line with wrapping, in axes coordinates."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    y = y0
    for items in lines:
        x = x0
        for w, c, mark in items:
            t = ax.text(x, y, w + ("†" if mark else ""), color=c, fontsize=fs, va="top", ha="left", transform=ax.transAxes,
                        fontweight="bold" if c in (COL["confirmed"], COL["gap"], COL["completion"]) else "normal")
            bb = t.get_window_extent(renderer=r).transformed(ax.transAxes.inverted())
            if bb.x1 > width and x > x0:  # wrap
                t.remove()
                y -= lh; x = x0 + 0.03
                t = ax.text(x, y, w + ("†" if mark else ""), color=c, fontsize=fs, va="top", ha="left", transform=ax.transAxes,
                            fontweight="bold" if c != COL["correct"] else "normal")
                bb = t.get_window_extent(renderer=r).transformed(ax.transAxes.inverted())
            x = bb.x1 + 0.012
        y -= lh * 1.15
        if y < 0.02:
            break


def main():
    stim = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "real", "stimuli.jsonl"), encoding="utf-8")}
    C = [json.loads(l) for l in open(os.path.join(ROOT, "runs", "real", "claims.jsonl"), encoding="utf-8")]
    fig, axs = plt.subplots(len(CASES), 2, figsize=(7.0, 3.3 * len(CASES)), gridspec_kw={"width_ratios": [1, 1.25]})
    summary, pending = [], []
    for (sid, m), (ai, at) in zip(CASES, axs):
        s = stim[sid]
        R = {json.loads(l)["id"]: json.loads(l)["result"] for l in open(os.path.join(ROOT, "runs", "real", "vlm", f"{m}.desc.jsonl"), encoding="utf-8")}
        _, spans = quoted_spans(R[sid])
        cc = [c for c in C if c["id"] == sid and c["model"] == m and c["prompt"] == "desc"]
        lab = {}
        for c in cc:  # token label per (span, token)
            st = c["label"]
            if c["label"] in ("misread", "fabricated"):
                line = next((s["lines"][g["line"]]["raw"] for g in s["tokens"] if g["tok"] == c["gt"] and g.get("line", -1) >= 0), "")
                row = {"key": f"ann:{m}:{sid}:{c['span']}:{c['tok']}", "id": sid, "image": os.path.join("data", "real", "images", sid + ".jpg"),
                       "tok": c["tok"], "label": c["label"], "line": line, "span": spans[c["span"]][0] if c["span"] < len(spans) else ""}
                v = rate(row)
                st = "confirmed" if v["printed"] == "no" else "gap"
                summary.append({"id": sid, "model": m, "tok": c["tok"], "label": c["label"], "gt": c["gt"], "rater_printed": v["printed"], "reason": v["reason"]})
            flag = (c["reread"] is False) or (c["ocr"] is False and (c["xmodel"] or 0) < 0.5)
            lab[(c["span"], c["tok"])] = (st, flag)
        ai.imshow(Image.open(os.path.join(ROOT, "data", "real", "images", sid + ".jpg"))); ai.set_axis_off()
        ai.set_title(f"{s['genre']} ({s['kind']}), {s['quality']}", fontsize=7)
        at.set_axis_off()
        lines, seen = [], set()
        for k, (sp, _, _) in enumerate(spans):
            key = sp.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            items = [("«", COL["correct"], False)]
            for w in sp.split():
                tt = toks(w)
                st, fl = "correct", False
                for t in tt:
                    if (k, t) in lab:
                        st2, fl2 = lab[(k, t)]
                        if st2 != "correct":
                            st = st2
                        fl = fl or fl2
                items.append((w, COL[st], fl))
            items.append(("»", COL["correct"], False))
            lines.append(items)
        at.set_title(f"Quoted text in the {NAME[m]} description", fontsize=7)
        pending.append((at, lines[:14]))
    handles = [plt.Line2D([], [], color=COL[k], lw=4) for k in ["correct", "completion", "confirmed", "gap"]]
    fig.legend(handles, ["correct", "completion", "hallucination (confirmed by rater)", "labelled hallucination, judged printed (reference gap)"],
               loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2, fontsize=6.5, frameon=False)
    fig.text(0.5, 0.005, "† flagged by the training-free run-time rule", ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.canvas.draw()
    for at, lines in pending:  # lay out words only after the axes have their final size
        draw_words(at, lines)
    fig.savefig(os.path.join(ROOT, "paper", "figures", "fig_annotated.pdf"), dpi=400)
    json.dump(summary, open(os.path.join(ROOT, "results", "annotated_cases.json"), "w"), ensure_ascii=False, indent=1)
    for x in summary:
        print(x)


if __name__ == "__main__":
    main()
