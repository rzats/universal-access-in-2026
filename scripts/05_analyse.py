"""All reported numbers: hallucination rates, factor breakdowns, detection performance.

Reads runs/claims.jsonl and runs/responses.jsonl (from 04_score.py); writes the
response-free aggregate results/summary.json and LaTeX-ready tables in results/.
"""
import collections, json, math, os, random

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MAIN = ["haiku", "sonnet", "opus"]
MODELS = MAIN + ["fable"]
HALL = {"misread", "fabricated"}
rng = random.Random(7)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 4), round(max(0, c - h), 4), round(min(1, c + h), 4)]


def prop(k, n):
    w = wilson(k, n)
    return {"k": k, "n": n, "p": w[0], "lo": w[1], "hi": w[2]}


def cluster_boot(ids, fn, B=1000):
    """Percentile CI of fn(sample) resampling images (clusters) with replacement."""
    u = sorted(set(ids))
    by = collections.defaultdict(list)
    for j, i in enumerate(ids):
        by[i].append(j)
    vals = []
    for _ in range(B):
        idx = [j for i in (rng.choice(u) for _ in u) for j in by[i]]
        v = fn(idx)
        if v is not None:
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else [None, None]


def auc(y, s):
    return roc_auc_score(y, s) if 0 < sum(y) < len(y) else None


def main():
    C = [json.loads(l) for l in open(os.path.join(ROOT, "runs", "claims.jsonl"), encoding="utf-8")]
    R = [json.loads(l) for l in open(os.path.join(ROOT, "runs", "responses.jsonl"), encoding="utf-8")]
    S = {}
    for c in C:
        c["hall"] = c["label"] in HALL
        c["unsup"] = c["label"] in HALL | {"completion"}
    stim_ids = sorted({r["id"] for r in R})
    fable_ids = sorted({r["id"] for r in R if r["model"] == "fable"})
    S["n_images"] = len(stim_ids)
    S["n_fable_images"] = len(fable_ids)
    S["n_claims"] = {f"{m}.{p}": sum(1 for c in C if c["model"] == m and c["prompt"] == p) for m in MODELS for p in ["desc", "trans"]}
    S["n_responses"] = {f"{m}.{p}": sum(1 for r in R if r["model"] == m and r["prompt"] == p) for m in MODELS for p in ["desc", "trans"]}

    # ---------- RQ1: how often ----------
    rates = {}
    for m in MODELS:
        for p in ["desc", "trans"]:
            cc = [c for c in C if c["model"] == m and c["prompt"] == p]
            rr = [r for r in R if r["model"] == m and r["prompt"] == p]
            if not cc:
                continue
            lab = collections.Counter(c["label"] for c in cc)
            ids = [c["id"] for c in cc]
            hall = [c["hall"] for c in cc]
            unh = [c for c in cc if c["hall"] and not c["hedged"]]
            img_any = [r["n_misread"] + r["n_fabricated"] > 0 for r in rr]
            img_unh = collections.Counter(c["id"] for c in unh)
            rates[f"{m}.{p}"] = {
                "claims": len(cc), "labels": dict(lab),
                "hall_token": prop(sum(hall), len(cc)),
                "hall_token_cluster_ci": cluster_boot(ids, lambda idx: sum(hall[j] for j in idx) / len(idx), 500),
                "misread_token": prop(lab["misread"], len(cc)),
                "fabricated_token": prop(lab["fabricated"], len(cc)),
                "completion_token": prop(lab["completion"], len(cc)),
                "img_any_hall": prop(sum(img_any), len(rr)),
                "img_any_unhedged_hall": prop(sum(1 for r in rr if img_unh[r["id"]] > 0), len(rr)),
                "coverage": round(sum(r["n_gt_visible_read"] for r in rr) / max(1, sum(r["n_gt_visible"] for r in rr)), 4),
                "says_illegible": prop(sum(r["says_illegible"] for r in rr), len(rr)),
                "hedged_share_of_hall": prop(sum(1 for c in cc if c["hall"] and c["hedged"]), sum(hall)),
                "hedged_share_of_correct": prop(sum(1 for c in cc if c["label"] == "correct" and c["hedged"]), lab["correct"]),
            }
    S["rates"] = rates

    # factor breakdowns for the description prompt (token-level hallucination rate)
    fac = {}
    for f in ["quality", "framing", "genre", "gt_cat", "is_num", "cond"]:
        fac[f] = {}
        for m in MODELS:
            cc = [c for c in C if c["model"] == m and c["prompt"] == "desc"]
            levels = sorted({str(c[f]) for c in cc})
            fac[f][m] = {lv: prop(sum(c["hall"] for c in cc if str(c[f]) == lv), sum(1 for c in cc if str(c[f]) == lv)) for lv in levels}
    S["factors"] = fac
    # image-level any hallucination by quality x framing (desc)
    cells = {}
    for m in MODELS:
        rr = [r for r in R if r["model"] == m and r["prompt"] == "desc"]
        cells[m] = {f"{q}|{fr}": prop(sum(r["n_misread"] + r["n_fabricated"] > 0 for r in rr if r["quality"] == q and r["framing"] == fr),
                                      sum(1 for r in rr if r["quality"] == q and r["framing"] == fr))
                    for q in ["clean", "blur", "severe"] for fr in ["none", "crop", "occlusion"]}
    S["img_cells"] = cells

    # subtypes of misreadings and fabrication, desc
    sub = {}
    for m in MODELS:
        cc = [c for c in C if c["model"] == m and c["prompt"] == "desc" and c["hall"]]
        sub[m] = dict(collections.Counter((c["label"] + ":" + c.get("sub", "none")) for c in cc))
    S["subtypes"] = sub

    # nonce / lure: lexicalisation of invented names (desc + trans)
    lexi = {}
    for m in MODELS:
        for p in ["desc", "trans"]:
            cc = [c for c in C if c["model"] == m and c["prompt"] == p and c["gt_cat"] in ("lure", "nonce", "word", "name")
                  and c["label"] in ("correct", "misread")]
            if not cc:
                continue
            lexi[f"{m}.{p}"] = {cat: {"misread": prop(sum(c["label"] == "misread" for c in cc if c["gt_cat"] == cat),
                                                      sum(1 for c in cc if c["gt_cat"] == cat)),
                                      "lexicalised": prop(sum(c.get("sub") == "lexicalised" for c in cc if c["gt_cat"] == cat),
                                                          sum(1 for c in cc if c["gt_cat"] == cat)),
                                      "ua_letter": prop(sum(c.get("sub") == "russified" for c in cc if c["gt_cat"] == cat),
                                                        sum(1 for c in cc if c["gt_cat"] == cat))}
                                for cat in ["word", "name", "nonce", "lure"]}
    S["lexicalisation"] = lexi

    # completion behaviour: claims whose best GT match is a not-fully-visible token
    comp = {}
    for m in MODELS:
        for p in ["desc", "trans"]:
            cc = [c for c in C if c["model"] == m and c["prompt"] == p and c["framing"] != "none"]
            if not cc:
                continue
            nv = [c for c in cc if c["gt_state"] in ("partial", "hidden") and c["label"] != "correct"]
            comp[f"{m}.{p}"] = {"completion": sum(c["label"] == "completion" for c in nv),
                                "wrong_guess": sum(c["label"] in HALL for c in nv),
                                "hedged_completion": prop(sum(c["hedged"] for c in nv if c["label"] == "completion"),
                                                          sum(c["label"] == "completion" for c in nv))}
    S["completion"] = comp

    # desc vs trans: paired image-level any-hallucination (exact McNemar)
    pair = {}
    for m in MAIN:
        a = {r["id"]: r["n_misread"] + r["n_fabricated"] > 0 for r in R if r["model"] == m and r["prompt"] == "desc"}
        b = {r["id"]: r["n_misread"] + r["n_fabricated"] > 0 for r in R if r["model"] == m and r["prompt"] == "trans"}
        ids = sorted(set(a) & set(b))
        if not ids:
            continue
        b10 = sum(a[i] and not b[i] for i in ids)
        b01 = sum(b[i] and not a[i] for i in ids)
        pv = binomtest(b10, b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
        pair[m] = {"n": len(ids), "desc_only": b10, "trans_only": b01, "both": sum(a[i] and b[i] for i in ids),
                   "p_mcnemar": float(pv)}
    S["desc_vs_trans"] = pair

    # ---------- RQ2: detection ----------
    det = {}
    feats = ["reread", "xmodel", "ocr", "ocr_sim", "lex", "lexfreq", "hedged", "reread_unc", "is_num", "len"]
    risk = {  # higher = more suspicious
        "reread": lambda c: 1 - c["reread"], "xmodel": lambda c: 1 - c["xmodel"], "ocr": lambda c: 1 - c["ocr"],
        "ocr_sim": lambda c: 1 - c["ocr_sim"], "lex": lambda c: 1 - c["lex"], "hedged": lambda c: float(c["hedged"]),
        "reread_unc": lambda c: float(c["reread_unc"]),
    }
    for m in MAIN:
        cc = [c for c in C if c["model"] == m and c["prompt"] == "desc" and all(c[k] is not None for k in ("reread", "xmodel", "ocr"))]
        if len(cc) < 50:
            continue
        y = np.array([c["hall"] for c in cc], int)
        ids = [c["id"] for c in cc]
        d = {"n": len(cc), "pos": int(y.sum()), "single": {}}
        for k, fn in risk.items():
            s = np.array([fn(c) for c in cc], float)
            a = auc(y, s)
            d["single"][k] = {"auroc": round(a, 4) if a else None,
                              "auroc_ci": cluster_boot(ids, lambda idx: auc(y[idx], s[idx]), 300)}
        # combined logistic regression, 5-fold cross-validation grouped by image
        X = np.array([[1 - c["reread"], 1 - c["xmodel"], 1 - c["ocr"], 1 - c["ocr_sim"], 1 - c["lex"],
                       c["lexfreq"], float(c["hedged"]), float(c["reread_unc"]), float(c["is_num"]), min(c["len"], 15) / 15]
                      for c in cc])
        uid = sorted(set(ids))
        rng.shuffle(uid)
        fold = {i: k % 5 for k, i in enumerate(uid)}
        pred = np.zeros(len(cc))
        for k in range(5):
            tr = np.array([fold[i] != k for i in ids])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X[tr], y[tr])
            pred[~tr] = clf.predict_proba(X[~tr])[:, 1]
        a = auc(y, pred)
        d["combined"] = {"auroc": round(a, 4), "auprc": round(average_precision_score(y, pred), 4),
                         "base_rate": round(float(y.mean()), 4),
                         "auroc_ci": cluster_boot(ids, lambda idx: auc(y[idx], pred[idx]), 300)}
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, y)
        d["combined"]["coef"] = dict(zip(["no_reread", "no_xmodel", "no_ocr", "ocr_dissim", "not_lex", "lexfreq",
                                          "hedged", "reread_unc", "is_num", "len"], [round(float(v), 3) for v in clf.coef_[0]]))
        # operating points: recall of hallucinated tokens at fixed false-flag rates on correct tokens
        neg = np.sort(pred[y == 0])[::-1]
        ops = {}
        for fpr in [0.05, 0.10, 0.20]:
            thr = neg[int(fpr * len(neg))]
            ops[str(fpr)] = {"recall": round(float((pred[y == 1] > thr).mean()), 4),
                             "precision": round(float((pred > thr)[y == 1].sum() / max(1, (pred > thr).sum())), 4)}
        d["combined"]["operating_points"] = ops
        # simple deployable rule: flag a quoted token unless both the model's own re-reading and OCR or another model confirm it
        rule = np.array([(not c["reread"]) or (not c["ocr"] and c["xmodel"] < 0.5) for c in cc])
        d["rule_reread_or_unconfirmed"] = {"recall": round(float(rule[y == 1].mean()), 4),
                                           "false_flag": round(float(rule[y == 0].mean()), 4),
                                           "precision": round(float(y[rule].mean()), 4) if rule.any() else None}
        rule2 = np.array([(not c["reread"]) for c in cc])
        d["rule_reread_only"] = {"recall": round(float(rule2[y == 1].mean()), 4), "false_flag": round(float(rule2[y == 0].mean()), 4)}
        # image-level: does the max token risk flag images with at least one hallucination?
        img_y, img_s = collections.defaultdict(int), collections.defaultdict(float)
        for c, p_ in zip(cc, pred):
            img_y[c["id"]] |= c["hall"]
            img_s[c["id"]] = max(img_s[c["id"]], p_)
        iy = [img_y[i] for i in sorted(img_y)]
        isc = [img_s[i] for i in sorted(img_y)]
        d["image_level_auroc"] = round(auc(iy, isc), 4) if auc(iy, isc) else None
        # unsupported (incl. completions) as the target
        yu = np.array([c["unsup"] for c in cc], int)
        d["combined_unsupported_auroc"] = round(auc(yu, pred), 4)
        det[m] = d
    S["detection"] = det

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=1)
    for k, v in rates.items():
        print(k, v["claims"], "hall", v["hall_token"]["p"], "img_any", v["img_any_hall"]["p"], "compl", v["completion_token"]["p"], "cov", v["coverage"])
    for m, d in det.items():
        print(m, {k: v["auroc"] for k, v in d["single"].items()}, "comb", d["combined"]["auroc"], d["combined"]["operating_points"])


if __name__ == "__main__":
    main()
