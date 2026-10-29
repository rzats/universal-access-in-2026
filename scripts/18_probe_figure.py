"""Near-word probe figure (paper/figures/fig_probe.pdf).
(a) two probe pairs: crop of the original and the edited real document page around the target;
(b) per model, what the description quoted for the target on edited pages (printed near-word /
real word / other form / not read) and how often the original word was read on unedited pages."""
import json, os, random, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAME = {"haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5", "opus": "Opus 5.5", "fable": "Fable 5.1"}
INK, GRID = "#0b0b0b", "#e4e3df"


def crop(sid, line, word, half=170, pad=45):
    im = Image.open(os.path.join(ROOT, "data", "probe", "images", sid + ".jpg"))
    x0, y0, x1, y1 = line["bbox"]
    i = line["raw"].lower().find(word)
    cx = x0 + (i + len(word) / 2) / max(1, len(line["raw"])) * (x1 - x0) if i >= 0 else (x0 + x1) / 2
    return im.crop((max(0, cx - half), max(0, y0 - pad), min(im.size[0], cx + half), min(im.size[1], y1 + pad)))


def main():
    st = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "probe", "stimuli.jsonl"), encoding="utf-8")}
    P = json.load(open(os.path.join(ROOT, "results", "probe.json")))
    C = [json.loads(l) for l in open(os.path.join(ROOT, "runs", "probe", "claims.jsonl"), encoding="utf-8")]
    norm = {}
    for c in C:
        if c["id"].endswith("-l") and c["tok"] == st[c["id"]]["target"]["real"]:
            norm.setdefault(c["id"], set()).add(c["model"])
    # pairs where at least three models normalised the near-word, clean quality, chosen with a fixed seed
    cand = sorted(k for k, v in norm.items() if len(v) >= 3 and st[k]["quality"] == "clean")
    random.Random(3).shuffle(cand)
    pairs = cand[:2]
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
    for r, lid in enumerate(pairs):
        s = st[lid]; o = st[lid[:-1] + "o"]
        t = s["target"]
        for cidx, (x, word, lab) in enumerate([(o, t["real"], "original"), (s, t["printed"], "edited")]):
            ax = fig.add_subplot(gs[r, cidx])
            ax.imshow(crop(x["id"], x["lines"][t["line"]], word)); ax.set_axis_off()
            ax.set_title(f"{lab}: «{word}»", fontsize=7, color=INK)
    ax = fig.add_subplot(gs[:, 2])
    models = [m for m in NAME if m in P]
    cats = [("exact", "printed near-word", "#2a78d6"), ("normalised", "real word", "#eb6834"),
            ("other", "other form", "#eda100"), ("absent", "not read", "#c9c8c3")]
    x = np.arange(len(models)); bottom = np.zeros(len(models))
    for k, lab, col in cats:
        v = np.array([100 * P[m]["nearword"][k] / P[m]["n_pairs"] for m in models])
        ax.bar(x, v, 0.6, bottom=bottom, color=col, label=lab, edgecolor="white", linewidth=0.8, zorder=2)
        bottom += v
    ax.scatter(x, [100 * P[m]["orig_exact"]["p"] for m in models], marker="D", s=22, color=INK, zorder=3,
               label="original word read\n(unedited page)")
    ax.set_xticks(x); ax.set_xticklabels([NAME[m].replace(" ", "\n") for m in models], fontsize=7)
    ax.set_ylabel("Edited pages (%)", fontsize=8); ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.set_title("What the description quoted", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "paper", "figures", "fig_probe.pdf"), dpi=200)
    json.dump({"pairs": pairs, "targets": [st[p]["target"] for p in pairs]}, open(os.path.join(ROOT, "results", "probe_figure.json"), "w"),
              ensure_ascii=False, indent=1)
    print(pairs, [st[p]["target"] for p in pairs])


if __name__ == "__main__":
    main()
