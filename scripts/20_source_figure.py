"""Study 2 hallucination rate by source category (paper/figures/fig_sources.pdf):
token-level rate of description words per document category / packaging panel, per model
(Wilson 95% intervals), with the number of images per category."""
import collections, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAME = {"haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5", "opus": "Opus 5.5", "fable": "Fable 5.1"}
COL = {"haiku": "#2a78d6", "sonnet": "#eb6834", "opus": "#1baf7a", "fable": "#eda100"}
LAB = {"ingredients": "packaging:\ningredients", "front": "packaging:\nfront", "form": "application\nforms", "medical": "medical\nforms",
       "leaflet": "medicine\nleaflets", "bill": "bills,\ntariffs", "bank": "bank\nforms", "education": "education",
       "notice": "notices", "book": "book\nannotations"}


def wilson(k, n, z=1.96):
    if not n:
        return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), c + h


def main():
    st = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "real", "stimuli.jsonl"), encoding="utf-8")}
    C = [json.loads(l) for l in open(os.path.join(ROOT, "runs", "real", "claims.jsonl"), encoding="utf-8")]
    C = [c for c in C if c["prompt"] == "desc" and c["label"] != "unscored" and c["id"] in st]
    nimg = collections.Counter(s["kind"] for s in st.values())
    cats = [k for k in LAB if k in nimg]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    w = 0.2
    out = {}
    for j, m in enumerate(NAME):
        vals = []
        for k in cats:
            cc = [c for c in C if c["model"] == m and st[c["id"]]["kind"] == k]
            vals.append(wilson(sum(c["label"] in ("misread", "fabricated") for c in cc), len(cc)))
            out[f"{m}.{k}"] = {"k": sum(c["label"] in ("misread", "fabricated") for c in cc), "n": len(cc)}
        v = np.array(vals) * 100
        x = np.arange(len(cats)) + (j - 1.5) * w
        ax.bar(x, v[:, 0], w * 0.92, color=COL[m], label=NAME[m], zorder=2)
        ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor="#52514e", elinewidth=0.6, capsize=1.2, zorder=3)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels([f"{LAB[k]}\n(n={nimg[k]})" for k in cats], fontsize=6.3)
    ax.set_ylabel("Hallucinated words (%)", fontsize=8)
    ax.yaxis.grid(True, color="#e4e3df", lw=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "paper", "figures", "fig_sources.pdf"))
    json.dump({"n_images": nimg, "rates": out}, open(os.path.join(ROOT, "results", "sources.json"), "w"), indent=1)
    print(nimg)


if __name__ == "__main__":
    main()
