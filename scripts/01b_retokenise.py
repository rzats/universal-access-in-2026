"""Recompute ground-truth tokens of data/stimuli.jsonl from the stored per-character
text, visibility and slot category (after a change to the tokeniser), without
re-rendering the images."""
import json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from render import gt_tokens
CAT = {"w": "word", "n": "name", "x": "nonce", "l": "lure", "d": "num"}
p = os.path.join(ROOT, "data", "stimuli.jsonl")
S = [json.loads(l) for l in open(p, encoding="utf-8")]
for s in S:
    lines = [dict(ln, cats=[CAT[c] for c in ln["cats"]]) for ln in s["lines"]]
    s["tokens"] = gt_tokens(lines)
with open(p, "w", encoding="utf-8") as f:
    for s in S:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print("retokenised", len(S))
