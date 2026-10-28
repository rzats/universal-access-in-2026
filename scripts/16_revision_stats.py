"""Revision analyses (writes results/revision.json):
  1. paired model-vs-model comparisons on shared images, per dataset: exact McNemar on
     image-level 'any hallucinated word' and a paired cluster-bootstrap CI on the
     difference in token hallucination rate; Holm correction across model pairs
  2. condition contrasts (quality, framing, description vs transcription) with bootstrap
     CIs on differences and exact tests, Holm-corrected within each family
  3. run-to-run stability from two repeat runs of the description prompt on one synthetic
     image per design cell, and sample-to-sample disagreement as a detector signal
  4. cost and latency per model, prompt and dataset (same CLI path for all)
  5. synthetic-vs-real comparison: token length, category mix, letter frequencies
"""
import collections, itertools, json, math, os, random, sys
import numpy as np
from scipy.stats import binomtest, fisher_exact
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from scoring import claims_desc, label
from textnorm import toks

MODELS = ["haiku", "sonnet", "opus", "fable"]
HALL = {"misread", "fabricated"}
DS = {"synthetic": ("data", "runs"), "real": (os.path.join("data", "real"), os.path.join("runs", "real"))}
B = 2000


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def wilson(k, n, z=1.96):
    if n == 0:
        return {"k": 0, "n": 0, "p": None, "lo": None, "hi": None}
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "p": round(p, 4), "lo": round(max(0, c - h), 4), "hi": round(min(1, c + h), 4)}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def mcnemar(a, b):
    b10 = sum(x and not y for x, y in zip(a, b)); b01 = sum(y and not x for x, y in zip(a, b))
    return b10, b01, (binomtest(b10, b10 + b01, 0.5).pvalue if b10 + b01 else 1.0)


def per_image(C, m, p="desc"):
    """image -> [n_hall, n_claims] for one model/prompt."""
    d = collections.defaultdict(lambda: [0, 0])
    for c in C:
        if c["model"] == m and c["prompt"] == p:
            d[c["id"]][0] += c["label"] in HALL
            d[c["id"]][1] += 1
    return d


def boot_diff(da, db, ids, rng, paired=True):
    """Bootstrap CI of rate_a - rate_b; paired resamples the same images for both."""
    ids = list(ids)
    vals = []
    for _ in range(B):
        sa = [rng.choice(ids) for _ in ids]
        sb = sa if paired else [rng.choice(ids) for _ in ids]
        ha = sum(da[i][0] for i in sa); na = sum(da[i][1] for i in sa)
        hb = sum(db[i][0] for i in sb); nb = sum(db[i][1] for i in sb)
        if na and nb:
            vals.append(ha / na - hb / nb)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def group_boot(dg1, dg2, rng):
    """Unpaired bootstrap CI of rate(group1) - rate(group2) over disjoint image sets."""
    i1, i2 = list(dg1), list(dg2)
    vals = []
    for _ in range(B):
        s1 = [rng.choice(i1) for _ in i1]; s2 = [rng.choice(i2) for _ in i2]
        n1 = sum(dg1[i][1] for i in s1); n2 = sum(dg2[i][1] for i in s2)
        if n1 and n2:
            vals.append(sum(dg1[i][0] for i in s1) / n1 - sum(dg2[i][0] for i in s2) / n2)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def rate(d, ids=None):
    ids = d.keys() if ids is None else ids
    n = sum(d[i][1] for i in ids)
    return sum(d[i][0] for i in ids) / n if n else None


def model_pairs(C, rng):
    out, ps = [], []
    for a, b in itertools.combinations(MODELS, 2):
        da, db = per_image(C, a), per_image(C, b)
        ids = sorted(set(da) & set(db))
        if not ids:
            continue
        ia = [da[i][0] > 0 for i in ids]; ib = [db[i][0] > 0 for i in ids]
        b10, b01, p = mcnemar(ia, ib)
        out.append({"a": a, "b": b, "n_images": len(ids), "img_a": round(sum(ia) / len(ids), 4), "img_b": round(sum(ib) / len(ids), 4),
                    "a_only": b10, "b_only": b01, "p_mcnemar": float(p),
                    "tok_a": round(rate(da, ids), 4), "tok_b": round(rate(db, ids), 4),
                    "tok_diff": round(rate(da, ids) - rate(db, ids), 4), "tok_diff_ci": boot_diff(da, db, ids, rng)})
        ps.append(p)
    for r, h in zip(out, holm(ps)):
        r["p_holm"] = float(h)
    return out


def conditions(C, R, rng):
    """Contrasts the prose uses, per model (description prompt)."""
    res, ps = [], []
    cond = {r["id"]: r for r in R}
    for m in MODELS:
        d = per_image(C, m)
        if not d:
            continue
        for fac, a, b in [("quality", "severe", "clean"), ("quality", "blur", "clean"), ("framing", "occlusion", "none"),
                          ("framing", "crop", "none"), ("genre", "signage", "document"), ("genre", "packaging", "document")]:
            g1 = {i: d[i] for i in d if cond.get(i, {}).get(fac) == a}
            g2 = {i: d[i] for i in d if cond.get(i, {}).get(fac) == b}
            if len(g1) < 5 or len(g2) < 5:
                continue
            k1 = sum(v[0] > 0 for v in g1.values()); k2 = sum(v[0] > 0 for v in g2.values())
            p = fisher_exact([[k1, len(g1) - k1], [k2, len(g2) - k2]])[1]
            res.append({"model": m, "factor": fac, "a": a, "b": b, "tok_a": round(rate(g1), 4), "tok_b": round(rate(g2), 4),
                        "tok_diff_ci": group_boot(g1, g2, rng), "img_a": round(k1 / len(g1), 4), "img_b": round(k2 / len(g2), 4),
                        "n_a": len(g1), "n_b": len(g2), "p_fisher_img": float(p)})
            ps.append(p)
        # description vs transcription on the same images (paired)
        dt = per_image(C, m, "trans")
        ids = sorted(set(d) & set(dt))
        if ids:
            ia = [d[i][0] > 0 for i in ids]; it = [dt[i][0] > 0 for i in ids]
            b10, b01, p = mcnemar(ia, it)
            res.append({"model": m, "factor": "prompt", "a": "desc", "b": "trans", "tok_a": round(rate(d, ids), 4),
                        "tok_b": round(rate(dt, ids), 4), "tok_diff_ci": boot_diff(d, dt, ids, rng),
                        "img_a": round(sum(ia) / len(ids), 4), "img_b": round(sum(it) / len(ids), 4), "n_a": len(ids), "n_b": len(ids),
                        "a_only": b10, "b_only": b01, "p_mcnemar": float(p)})
            ps.append(p)
    for r, h in zip(res, holm(ps)):
        r["p_holm"] = float(h)
    return res


def cost_latency(runs):
    out = {}
    for m in MODELS:
        for p in ["desc", "trans"]:
            R = load(os.path.join(ROOT, runs, "vlm", f"{m}.{p}.jsonl"))
            if not R:
                continue
            u = [r["usage"] for r in R]
            inp = [x.get("input_tokens", 0) + x.get("cache_creation_input_tokens", 0) + x.get("cache_read_input_tokens", 0) for x in u]
            outp = [x.get("output_tokens", 0) for x in u]
            out[f"{m}.{p}"] = {"n": len(R), "in_tokens": round(float(np.mean(inp))), "out_tokens": round(float(np.mean(outp))),
                               "usd_per_image": round(float(np.mean([r["cost"] for r in R])), 4),
                               "sec_median": round(float(np.median([r["secs"] for r in R])), 1),
                               "sec_p90": round(float(np.percentile([r["secs"] for r in R], 90)), 1)}
    return out


def stability(rng):
    stim = {s["id"]: s for s in load(os.path.join(ROOT, "data", "stimuli.jsonl"))}
    lex = set()
    for ln in open(os.path.join(ROOT, "data", "assets", "uk_full.txt"), encoding="utf-8"):
        w, _, c = ln.strip().partition(" ")
        if w and int(c or 1) >= 2:
            lex.add(w.lower())
    sub = set(open(os.path.join(ROOT, "runs", "repeat_subset.txt")).read().split())
    out = {}
    for m in MODELS:
        runs = {"r1": {r["id"]: r["result"] for r in load(os.path.join(ROOT, "runs", "vlm", f"{m}.desc.jsonl")) if r["id"] in sub}}
        for t in ["r2", "r3"]:
            runs[t] = {r["id"]: r["result"] for r in load(os.path.join(ROOT, "runs", "vlm", f"{m}.desc_{t}.jsonl"))}
        ids = sorted(set.intersection(*[set(v) for v in runs.values()]))
        if len(ids) < 20:
            continue
        lab = {}  # (run, id) -> list of (tok, hall)
        for t, rr in runs.items():
            for i in ids:
                lab[(t, i)] = [(c["tok"], label(c["tok"], stim[i]["tokens"], lex)["label"]) for c in claims_desc(rr[i])]
        anyh = {t: [any(l in HALL for _, l in lab[(t, i)]) for i in ids] for t in runs}
        flips = sum(len({anyh[t][k] for t in runs}) > 1 for k in range(len(ids)))
        trate = {t: sum(l in HALL for i in ids for _, l in lab[(t, i)]) / max(1, sum(len(lab[(t, i)]) for i in ids)) for t in runs}
        # sample-to-sample disagreement detector on run-1 tokens
        y, s = [], []
        for i in ids:
            others = [set(x for x, _ in lab[(t, i)]) for t in ["r2", "r3"]]
            for tok, l in lab[("r1", i)]:
                if l in ("correct", "misread", "fabricated"):
                    y.append(l in HALL); s.append(sum(tok not in o for o in others) / 2)
        out[m] = {"n_images": len(ids), "img_any_by_run": {t: round(sum(v) / len(v), 4) for t, v in anyh.items()},
                  "flip_rate": wilson(flips, len(ids)), "token_rate_by_run": {t: round(v, 4) for t, v in trate.items()},
                  "token_rate_spread": round(max(trate.values()) - min(trate.values()), 4),
                  "token_rate_sd": round(float(np.std(list(trate.values()), ddof=1)), 4),
                  "disagreement_auroc": round(roc_auc_score(y, s), 4) if 0 < sum(y) < len(y) else None,
                  "n_tokens": len(y), "n_hall": int(sum(y))}
    return out


def compare_sets():
    """Synthetic vs real: token length, category mix and Ukrainian-specific letter frequencies."""
    res = {}
    natural = collections.Counter()
    for ln in open(os.path.join(ROOT, "data", "assets", "uk_full.txt"), encoding="utf-8"):
        w, _, c = ln.strip().partition(" ")
        if w:
            for ch in w.lower():
                natural[ch] += int(c or 1)
    def letfreq(counter):
        tot = sum(v for k, v in counter.items() if "а" <= k <= "я" or k in "іїєґ'")
        return {ch: round(counter[ch] / tot, 5) for ch in "їєґі"}
    res["letters_natural"] = letfreq(natural)
    for ds, (data, _) in DS.items():
        S = load(os.path.join(ROOT, data, "stimuli.jsonl"))
        T = [t for s in S for t in s["tokens"] if not t.get("extra")]
        letters = [t for t in T if not t["tok"].isdigit()]
        cat = collections.Counter(("name" if t["cat"] in ("name", "nonce", "lure") else t["cat"]) for t in T)
        r = {"n_images": len(S), "n_tokens": len(T), "tokens_per_image": round(len(T) / len(S), 1),
             "letter_token_len_mean": round(float(np.mean([len(t["tok"]) for t in letters])), 2),
             "letter_token_len_median": float(np.median([len(t["tok"]) for t in letters])),
             "len_hist": {str(k): v for k, v in sorted(collections.Counter(min(len(t["tok"]), 15) for t in letters).items())},
             "category_share": {k: round(v / len(T), 4) for k, v in cat.items()},
             "not_visible_share": round(sum(t["state"] != "visible" for t in T) / len(T), 4),
             "letters_all": letfreq(collections.Counter(ch for t in letters for ch in t["tok"]))}
        if ds == "synthetic":
            r["letters_invented"] = letfreq(collections.Counter(ch for t in letters if t["cat"] in ("nonce",) for ch in t["tok"]))
            r["letters_invented_over_natural"] = {ch: round(r["letters_invented"][ch] / res["letters_natural"][ch], 2) for ch in "їєґі"}
        res[ds] = r
    return res


def main():
    rng = random.Random(11)
    out = {}
    for ds, (data, runs) in DS.items():
        C = [c for c in load(os.path.join(ROOT, runs, "claims.jsonl")) if c["label"] != "unscored"]
        R = load(os.path.join(ROOT, runs, "responses.jsonl"))
        out[ds] = {"model_pairs": model_pairs(C, rng), "conditions": conditions(C, R, rng), "cost": cost_latency(runs)}
    out["probe_cost"] = cost_latency(os.path.join("runs", "probe"))
    out["stability"] = stability(rng)
    out["sets"] = compare_sets()
    json.dump(out, open(os.path.join(ROOT, "results", "revision.json"), "w"), indent=1, ensure_ascii=False)
    for ds in DS:
        for r in out[ds]["model_pairs"]:
            print(ds, r["a"], r["b"], r["img_a"], r["img_b"], round(r["p_holm"], 4), r["tok_diff"], r["tok_diff_ci"])
    print(json.dumps(out["stability"], indent=0)[:1500])
    print(json.dumps(out["sets"]["synthetic"].get("letters_invented_over_natural")), out["sets"]["letters_natural"])


if __name__ == "__main__":
    main()
