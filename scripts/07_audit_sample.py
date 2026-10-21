"""Draw a stratified random sample of labelled description tokens for a manual audit of
the automatic labeller. Prints context; the judgments are recorded in results/audit.json."""
import json, os, random, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from paths import DATA, RUNS, RESULTS  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "src"))
from scoring import quoted_spans
rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
C = [json.loads(l) for l in open(os.path.join(RUNS, "claims.jsonl"), encoding="utf-8")]
MODELS = ("haiku", "sonnet", "opus", "fable")
C = [c for c in C if c["prompt"] == "desc"]
stim = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "stimuli.jsonl"), encoding="utf-8")}
resp = {}
for m in MODELS:
    for l in open(os.path.join(RUNS, "vlm", f"{m}.desc.jsonl"), encoding="utf-8"):
        d = json.loads(l); resp[(m, d["id"])] = d["result"]
sample = []
for lab, n in [("misread", 50), ("fabricated", 50), ("completion", 30), ("correct", 50)]:
    xs = [c for c in C if c["label"] == lab]
    sample += rng.sample(xs, min(n, len(xs)))
for k, c in enumerate(sample):
    t, sp = quoted_spans(resp[(c["model"], c["id"])])
    s = sp[c["span"]][0] if c["span"] < len(sp) else "?"
    g = [x for x in stim[c["id"]]["tokens"] if x["tok"] == c["gt"]]
    line = stim[c["id"]]["lines"][g[0]["line"]]["raw"] if g else ""
    vis = "".join(str(v) for v in g[0]["vis"]) if g else ""
    print(f"{k:3d} {c['model'][:2]} {c['label']:10s} tok={c['tok']!r} gt={c['gt']!r}[{c['gt_state']} {vis}] span=«{s[:70]}» line=«{line}»")
