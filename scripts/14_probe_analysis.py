"""Near-word probe: does the description quote the printed word, or normalise it?

For each model and probe pair, the target token is looked up among the tokens quoted in
the description (runs/probe/claims.jsonl): 'exact' = the printed form is quoted,
'normalised' = (near-word page only) the real word is quoted instead, 'other' = a
different form within edit similarity 0.5, 'absent' = not read. Writes results/probe.json.
"""
import collections, json, math, os, sys
from scipy.stats import binomtest
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from textnorm import sim

MODELS = ["haiku", "sonnet", "opus", "fable"]


def wilson(k, n, z=1.96):
    if n == 0:
        return {"k": 0, "n": 0, "p": None, "lo": None, "hi": None}
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "p": round(p, 4), "lo": round(max(0, c - h), 4), "hi": round(min(1, c + h), 4)}


def main():
    stim = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "probe", "stimuli.jsonl"), encoding="utf-8")}
    C = [json.loads(l) for l in open(os.path.join(ROOT, "runs", "probe", "claims.jsonl"), encoding="utf-8")]
    toks = collections.defaultdict(set)
    for c in C:
        if c["prompt"] == "desc":
            toks[(c["model"], c["id"])].add(c["tok"])
    res = {}
    for m in MODELS:
        out = {"original": collections.Counter(), "nearword": collections.Counter()}
        pair = []
        for sid, s in stim.items():
            if (m, sid) not in toks and not any(k[0] == m for k in toks):
                continue
            t = toks.get((m, sid), set())
            tg = s["target"]
            v = "nearword" if s["cond"] == "probe_nearword" else "original"
            if tg["printed"] in t:
                o = "exact"
            elif v == "nearword" and tg["real"] in t:
                o = "normalised"
            elif any(sim(x, tg["printed"]) >= 0.5 and len(x) >= 4 for x in t):
                o = "other"
            else:
                o = "absent"
            out[v][o] += 1
            if v == "nearword":
                twin = stim.get(sid[:-1] + "o")
                if twin:
                    t2 = toks.get((m, twin["id"]), set())
                    pair.append((twin["target"]["printed"] in t2, o == "exact"))
        if not sum(out["nearword"].values()):
            continue
        read = [p for p in pair if p[0] or p[1]]
        both_read = sum(1 for a, b in pair if a)
        # among pairs where the real word was read correctly on the original page
        cond = [b for a, b in pair if a]
        res[m] = {
            "n_pairs": len(pair),
            "original": {k: out["original"][k] for k in ["exact", "other", "absent"]},
            "nearword": {k: out["nearword"][k] for k in ["exact", "normalised", "other", "absent"]},
            "orig_exact": wilson(out["original"]["exact"], sum(out["original"].values())),
            "near_exact": wilson(out["nearword"]["exact"], sum(out["nearword"].values())),
            "near_normalised": wilson(out["nearword"]["normalised"], sum(out["nearword"].values())),
            "near_exact_given_orig_read": wilson(sum(cond), len(cond)),
            "p_mcnemar": float(binomtest(sum(1 for a, b in pair if a and not b), sum(1 for a, b in pair if a != b), 0.5).pvalue)
            if any(a != b for a, b in pair) else 1.0,
        }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "probe.json"), "w"), indent=1)
    for m, r in res.items():
        print(m, r["n_pairs"], r["original"], r["nearword"], r["near_exact_given_orig_read"]["p"])


if __name__ == "__main__":
    main()
