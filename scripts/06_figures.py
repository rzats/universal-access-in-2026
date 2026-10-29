"""Figures for the paper (PDF, vector). Reads results/summary.json and runs/claims.jsonl."""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from paths import DATA, RUNS, RESULTS  # noqa: E402

OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(RESULTS, "summary.json"), encoding="utf-8"))
SUF = "" if os.environ.get("UAIS_SET") == "synthetic" else "_real"  # Study 1 / Study 2 figure names
COL = {"haiku": "#2a78d6", "sonnet": "#eb6834", "opus": "#1baf7a", "fable": "#eda100"}
NAME = {"haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5", "opus": "Opus 5.5", "fable": "Fable 5.1"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
models = [m for m in ["haiku", "sonnet", "opus", "fable"] if f"{m}.desc" in S["rates"]]


def bars(ax, groups, getter, ylabel):
    w = 0.8 / len(models)
    x = np.arange(len(groups))
    for k, m in enumerate(models):
        vals = [getter(m, g) for g in groups]
        p = np.array([v["p"] if v and v["p"] is not None else np.nan for v in vals])
        lo = np.array([v["lo"] if v and v["lo"] is not None else np.nan for v in vals])
        hi = np.array([v["hi"] if v and v["hi"] is not None else np.nan for v in vals])
        ax.bar(x + (k - (len(models) - 1) / 2) * w, p * 100, w * 0.92, color=COL[m], label=NAME[m], zorder=2)
        ax.errorbar(x + (k - (len(models) - 1) / 2) * w, p * 100, yerr=[(p - lo) * 100, (hi - p) * 100],
                    fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.set_xticks(x)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ---- Fig 2: image-level any-hallucination by quality, faceted by framing
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
for ax, fr, title in zip(axs, ["none", "crop", "occlusion"], ["Full frame", "Cropped by frame edge", "Finger / glare occlusion"]):
    bars(ax, ["clean", "blur", "severe"], lambda m, q: S["img_cells"][m].get(f"{q}|{fr}"),
         "Responses with ≥1 hallucinated word (%)" if fr == "none" else "")
    ax.set_xticklabels(["clean", "blur", "severe"])
    ax.set_title(title, fontsize=8, color=INK)
    ax.set_ylim(0, 100)
axs[0].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_cells" + SUF + ".pdf"))
plt.close(fig)

# ---- Fig 3: token hallucination rate by ground-truth category
cats = ["word", "name", "num"]
labels = ["common word", "name or rare word", "digits"]
fig, ax = plt.subplots(figsize=(4.6, 2.4))
bars(ax, cats, lambda m, c: S["factors"]["gt_cat"][m].get(c), "Hallucinated words (%)")
ax.set_xticklabels(labels)
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_categories" + SUF + ".pdf"))
plt.close(fig)

# ---- Fig 4: ROC curves of detectors, one panel per main model
C = [json.loads(l) for l in open(os.path.join(RUNS, "claims.jsonl"), encoding="utf-8")]
mm = [m for m in ["haiku", "sonnet", "opus"] if m in S["detection"]]
fig, axs = plt.subplots(1, len(mm), figsize=(7.0, 3.6), sharey=True)
axs = np.atleast_1d(axs)
DET = [("reread", "Re-reading", "#2a78d6"), ("xmodel", "Other models", "#eb6834"),
       ("ocr_sim", "Tesseract OCR", "#1baf7a"), ("lex", "Lexicon", "#eda100"), ("hedged", "Hedging cue", "#6250d6")]
for ax, m in zip(axs, mm):
    cc = [c for c in C if c["model"] == m and c["prompt"] == "desc" and all(c[k] is not None for k in ("reread", "xmodel", "ocr"))]
    y = np.array([c["label"] in ("misread", "fabricated") for c in cc], int)
    for key, lab, col in DET:
        s = np.array([float(c[key]) for c in cc])
        s = s if key == "hedged" else 1 - s
        f, t, _ = roc_curve(y, s)
        ax.plot(f, t, color=col, lw=1.4, label=f"{lab} ({S['detection'][m]['single'][key]['auroc']:.2f})")
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, ls="--")
    ax.set_title(NAME[m], fontsize=8, color=INK)
    ax.set_xlabel("False-flag rate on correct words")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, title="AUROC", title_fontsize=6)
axs[0].set_ylabel("Hallucinated words caught")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_roc" + SUF + ".pdf"))
plt.close(fig)

# ---- Fig 1: example stimuli (one row per genre: clean / degraded / framed)
ST = [json.loads(l) for l in open(os.path.join(DATA, "stimuli.jsonl"), encoding="utf-8")]
def pick(g, q, fr):
    for x in ST:
        if x["genre"] == g and x["quality"] == q and x["framing"] == fr:
            return x
if SUF == "":  # Study 1: one example per genre and condition, as in the first version
    ex = [x for x in ST if x["id"] in ("sig-cl-no-lu-0", "pac-bl-oc-no-1", "doc-se-cr-wo-2", "pac-cl-cr-no-0", "doc-bl-oc-lu-3", "sig-se-no-no-0")]
    ex = ex[:6]
else:
  ex = [pick("packaging", "clean", "none"), pick("packaging", "blur", "crop"), pick("packaging", "severe", "occlusion"),
        pick("document", "clean", "none"), pick("document", "blur", "occlusion"), pick("document", "severe", "crop")]
fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.8)) if SUF == "" else plt.subplots(2, 3, figsize=(7.0, 4.6))
for ax, e in zip(axs.flat, ex):
    ax.set_axis_off()
    if e is None:
        continue
    ax.imshow(Image.open(os.path.join(DATA, "images", e["id"] + ".jpg")))
    ax.set_title(f"{e['genre']}, {e['quality']}, " + {"none": "full frame", "crop": "cropped", "occlusion": "occluded"}[e["framing"]],
                 fontsize=7, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_examples" + ("_synth" if SUF == "" else "") + ".pdf"), dpi=170)
plt.close(fig)
print("figures written to", OUT)
