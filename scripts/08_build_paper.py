"""Assemble the single-file manuscript from the Springer Nature template and the
section sources, define every reported number as a macro computed from
results/summary.json, and compile with tectonic.

Sources:  paper/template_main.tex (official sn-article template, body replaced by %BODY%)
          paper/body_*.tex       (sections, in the order listed below)
Output:   paper/universal-access-in-2026.tex and .pdf
"""
import json, os, re, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from paths import DATA, RUNS, RESULTS  # noqa: E402

P = os.path.join(ROOT, "paper")
ORDER = ["body_title", "body_front", "body_method", "body_results", "body_discussion", "body_back"]
APPENDIX = "body_appendix"
SYN = json.load(open(os.path.join(ROOT, "results", "summary.json"), encoding="utf-8"))
REAL = json.load(open(os.path.join(ROOT, "results", "real", "summary.json"), encoding="utf-8"))
REV = json.load(open(os.path.join(ROOT, "results", "revision.json"), encoding="utf-8"))
AUD = json.load(open(os.path.join(ROOT, "results", "audit_raters.json"), encoding="utf-8")) \
    if os.path.exists(os.path.join(ROOT, "results", "audit_raters.json")) else None
S, A = SYN, None  # macros() reads the current study through these globals


def pc(x, nd=1):
    return f"{100 * x:.{nd}f}"


def ci(d, nd=1):
    return f"{pc(d['p'], nd)}\\% [{pc(d['lo'], nd)}, {pc(d['hi'], nd)}]"


def macros():
    m = {}
    stim = [json.loads(l) for l in open(os.path.join(DATA, "stimuli.jsonl"), encoding="utf-8")]
    toks = [t for s in stim for t in s["tokens"]]
    m["NImages"] = str(len(stim))
    m["NImgPerCell"] = str(len(stim) // 81)
    m["NGTTokens"] = f"{len(toks):,}".replace(",", "\\,")
    m["NGTNotVisible"] = f"{sum(t['state'] != 'visible' for t in toks):,}".replace(",", "\\,")
    m["NFableImages"] = str(S["n_fable_images"])
    m["NAudit"] = str(A["n"]) if A else "??"
    for key, r in S["rates"].items():
        mod, pr = key.split(".")
        tag = mod.capitalize() + pr.capitalize()
        m[f"Claims{tag}"] = f"{r['claims']:,}".replace(",", "\\,")
        m[f"Hall{tag}"] = ci(r["hall_token"])
        m[f"HallP{tag}"] = pc(r["hall_token"]["p"])
        m[f"Img{tag}"] = ci(r["img_any_hall"], 0)
        m[f"ImgP{tag}"] = pc(r["img_any_hall"]["p"], 0)
        m[f"ImgUnh{tag}"] = pc(r["img_any_unhedged_hall"]["p"], 0)
        m[f"Compl{tag}"] = pc(r["completion_token"]["p"])
        m[f"Cov{tag}"] = pc(r["coverage"])
    NM = {"haiku": "Haiku", "sonnet": "Sonnet", "opus": "Opus", "fable": "Fable"}
    for mod in NM:
        F = S["factors"]
        if mod not in F["gt_cat"] or not F["gt_cat"][mod]:
            continue
        T = NM[mod]
        for cat, nm in [("word", "Word"), ("name", "Name"), ("nonce", "Nonce"), ("lure", "Lure"), ("num", "Num")]:
            if cat in F["gt_cat"][mod]:
                m[f"Cat{nm}{T}"] = pc(F["gt_cat"][mod][cat]["p"])
        for q in ["clean", "blur", "severe"]:
            m[f"Qual{q.capitalize()}{T}"] = pc(F["quality"][mod][q]["p"])
        for fr, nm in [("none", "Full"), ("crop", "Crop"), ("occlusion", "Occ")]:
            m[f"Frame{nm}{T}"] = pc(F["framing"][mod][fr]["p"])
        for g in ["signage", "packaging", "document"]:
            m[f"Genre{g.capitalize()}{T}"] = pc(F["genre"][mod][g]["p"])
        r = S["rates"][f"{mod}.desc"]
        m[f"HedgeHall{T}"] = pc(r["hedged_share_of_hall"]["p"], 0)
        m[f"HedgeCorr{T}"] = pc(r["hedged_share_of_correct"]["p"], 0)
        m[f"Illegible{T}"] = pc(r["says_illegible"]["p"], 0)
        m[f"ImgDigit{T}"] = pc(r["img_any_digit_hall"]["p"], 0)
        m[f"ImgCleanFull{T}"] = pc(r["img_any_hall_clean_full"]["p"], 0)
        c = S["completion"].get(f"{mod}.desc")
        if c:
            m[f"NCompl{T}"] = str(c["completion"])
            m[f"ComplHedged{T}"] = pc(c["hedged_completion"]["p"], 0) if c["hedged_completion"]["p"] is not None else "--"
        c = S["completion"].get(f"{mod}.trans")
        if c:
            m[f"NComplTrans{T}"] = str(c["completion"])
        L = S["lexicalisation"].get(f"{mod}.desc")
        if L and L.get("lure", {}).get("lexicalised", {}).get("p") is not None:
            m[f"LureLex{T}"] = pc(L["lure"]["lexicalised"]["p"])
            m[f"LureMis{T}"] = pc(L["lure"]["misread"]["p"])
            m[f"NonceUa{T}"] = pc(L["nonce"]["ua_letter"]["p"])
            m[f"NonceMis{T}"] = pc(L["nonce"]["misread"]["p"])
        sub = S["subtypes"].get(mod, {})
        tot = sum(sub.values())
        if tot:
            for k, nm in [("misread:digit", "Digit"), ("misread:russified", "Ua"), ("misread:lexicalised", "Lex"),
                          ("misread:other", "Other"), ("fabricated:none", "Fab")]:
                m[f"Sub{nm}{T}"] = pc(sub.get(k, 0) / tot, 0)
        if mod in S["desc_vs_trans"]:
            d = S["desc_vs_trans"][mod]
            m[f"McDesc{T}"] = str(d["desc_only"])
            m[f"McTrans{T}"] = str(d["trans_only"])
            m[f"McP{T}"] = f"={d['p_mcnemar']:.3f}" if d["p_mcnemar"] >= 0.001 else "<0.001"
        if mod in S["detection"]:
            d = S["detection"][mod]
            for k, nm in [("reread", "Reread"), ("xmodel", "Xmodel"), ("ocr_sim", "Ocr"), ("lex", "Lex"), ("hedged", "Hedge"),
                          ("reread_unc", "Unc")]:
                m[f"Auc{nm}{T}"] = f"{d['single'][k]['auroc']:.2f}"
            cb = d["combined"]
            m[f"AucComb{T}"] = f"{cb['auroc']:.2f}"
            m[f"AucCombCI{T}"] = f"{cb['auroc']:.3f} [{cb['auroc_ci'][0]:.3f}, {cb['auroc_ci'][1]:.3f}]"
            m[f"Ap{T}"] = f"{cb['auprc']:.2f}"
            m[f"Base{T}"] = pc(cb["base_rate"])
            for f_, nm in [("0.05", "Five"), ("0.1", "Ten"), ("0.2", "Twenty")]:
                m[f"Rec{nm}{T}"] = pc(cb["operating_points"][f_]["recall"], 0)
                m[f"Prec{nm}{T}"] = pc(cb["operating_points"][f_]["precision"], 0)
            m[f"RuleRec{T}"] = pc(d["rule_reread_or_unconfirmed"]["recall"], 0)
            m[f"RuleFf{T}"] = pc(d["rule_reread_or_unconfirmed"]["false_flag"], 1)
            m[f"RuleRereadRec{T}"] = pc(d["rule_reread_only"]["recall"], 0)
            m[f"ImgAuc{T}"] = f"{d['image_level_auroc']:.2f}"
            m[f"AucUnsup{T}"] = f"{d['combined_unsupported_auroc']:.2f}"
            m[f"AucDigits{T}"] = f"{d['combined_digits_auroc']:.2f}"
    if A:
        m["AuditHallPrec"] = pc(A["hall_precision"]["p"], 0)
        m["AuditHallPrecCI"] = ci(A["hall_precision"], 0)
        m["AuditHallLo"] = pc(A["hall_precision"]["lo"], 0)
        m["AuditHallHi"] = pc(A["hall_precision"]["hi"], 0)
        m["AuditCorrPrec"] = pc(A["correct_precision"]["p"], 0)
        m["AuditComplPrec"] = pc(A["completion_precision"]["p"], 0)
        m["AuditNHall"] = str(A["hall_precision"]["n"])
        m["AuditNCorr"] = str(A["correct_precision"]["n"])
        m["AuditNCompl"] = str(A["completion_precision"]["n"])
    def rng_(vals, nd=1, pct=True):
        lo, hi = min(vals), max(vals)
        f = (lambda v: pc(v, nd)) if pct else (lambda v: f"{v:.2f}")
        return f"{f(lo)}--{f(hi)}" if f(lo) != f(hi) else f(lo)
    allm = [k for k in ["haiku", "sonnet", "opus", "fable"] if f"{k}.desc" in S["rates"]]
    main3 = [k for k in ["haiku", "sonnet", "opus"] if k in S["detection"]]
    m["RangeHall"] = rng_([S["rates"][f"{k}.desc"]["hall_token"]["p"] for k in allm])
    m["RangeImg"] = rng_([S["rates"][f"{k}.desc"]["img_any_hall"]["p"] for k in allm], 0)
    m["RangeCleanFull"] = rng_([S["rates"][f"{k}.desc"]["img_any_hall_clean_full"]["p"] for k in main3], 0)
    m["RangeNameHall"] = rng_([S["factors"]["gt_cat"][k]["name"]["p"] for k in main3])
    m["RangeNumHall"] = rng_([S["factors"]["gt_cat"][k]["num"]["p"] for k in main3])
    m["RangeWordHall"] = rng_([S["factors"]["gt_cat"][k]["word"]["p"] for k in main3])
    m["RangeAucComb"] = rng_([S["detection"][k]["combined"]["auroc"] for k in main3], pct=False)
    m["RangeRecTen"] = rng_([S["detection"][k]["combined"]["operating_points"]["0.1"]["recall"] for k in main3], 0)
    m["RangeHedgeHall"] = rng_([S["rates"][f"{k}.desc"]["hedged_share_of_hall"]["p"] for k in main3], 0)
    m["RangeImgUnh"] = rng_([S["rates"][f"{k}.desc"]["img_any_unhedged_hall"]["p"] for k in allm], 0)
    m["RangeCov"] = rng_([S["rates"][f"{k}.desc"]["coverage"] for k in allm])
    m["RangeIllegible"] = rng_([S["rates"][f"{k}.desc"]["says_illegible"]["p"] for k in allm], 0)
    m["RangeRuleRec"] = rng_([S["detection"][k]["rule_reread_or_unconfirmed"]["recall"] for k in main3], 0)
    m["RangeRuleFf"] = rng_([S["detection"][k]["rule_reread_or_unconfirmed"]["false_flag"] for k in main3], 0)
    m["RangeRereadRec"] = rng_([S["detection"][k]["rule_reread_only"]["recall"] for k in main3], 0)
    # stimulus-set composition
    m["NSign"] = str(sum(s_["genre"] == "signage" for s_ in stim))
    m["NPack"] = str(sum(s_["genre"] == "packaging" for s_ in stim))
    m["NDoc"] = str(sum(s_["genre"] == "document" for s_ in stim))
    man = os.path.join(ROOT, "data", "raw_docs", "manifest.jsonl")
    if os.path.exists(man):
        nd = sum(1 for _ in open(man, encoding="utf-8"))
        m["NDocsCollected"] = str(nd)
        m["NDocsRejected"] = str(nd - int(m["NDoc"]))
    # near-word probe
    PR = os.path.join(ROOT, "results", "probe.json")
    if os.path.exists(PR):
        P = json.load(open(PR))
        first = next(iter(P.values()))
        m["NProbePairs"] = str(first["n_pairs"])
        for mod, r in P.items():
            T = mod.capitalize()
            m[f"PrOrig{T}"] = pc(r["orig_exact"]["p"], 0)
            m[f"PrNear{T}"] = pc(r["near_exact"]["p"], 0)
            m[f"PrNorm{T}"] = pc(r["near_normalised"]["p"], 0)
            m[f"PrNormCI{T}"] = f"{pc(r['near_normalised']['lo'], 0)}--{pc(r['near_normalised']['hi'], 0)}"
            m[f"PrCond{T}"] = pc(r["near_exact_given_orig_read"]["p"], 0)
            m[f"PrCondN{T}"] = str(r["near_exact_given_orig_read"]["n"])
            m[f"PrP{T}"] = f"={r['p_mcnemar']:.3f}" if r["p_mcnemar"] >= 0.001 else "<0.001"
        m["RangePrNorm"] = rng_([r["near_normalised"]["p"] for r in P.values()], 0)
        m["RangePrNear"] = rng_([r["near_exact"]["p"] for r in P.values()], 0)
        m["RangePrOrig"] = rng_([r["orig_exact"]["p"] for r in P.values()], 0)
    # spend across all sweeps (synthetic pilot, real benchmark, probe, discarded oversize runs)
    import glob
    tot = 0.0
    for f in glob.glob(os.path.join(ROOT, "runs", "**", "*.jsonl"), recursive=True):
        if "/vlm" in f or "vlm_oversize" in f:
            tot += sum(json.loads(l).get("cost", 0) for l in open(f, encoding="utf-8"))
    m["SpendTotal"] = f"{tot:.0f}"
    return m


NAMES = {"haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5", "opus": "Opus 5.5", "fable": "Fable 5.1"}


def table_rates(S):
    rows = []
    for mod in NAMES:
        for pr, pn in [("desc", "description"), ("trans", "transcription")]:
            r = S["rates"].get(f"{mod}.{pr}")
            if not r:
                continue
            h = r["hall_token"]
            rows.append(f"{NAMES[mod]} & {pn} & {r['claims']:,} & {pc(h['p'], 2)} [{pc(h['lo'], 2)}, {pc(h['hi'], 2)}] & "
                        f"{pc(r['misread_token']['p'], 2)} & {pc(r['fabricated_token']['p'], 2)} & {pc(r['completion_token']['p'], 2)} & "
                        f"{pc(r['img_any_hall']['p'], 1)} [{pc(r['img_any_hall']['lo'], 1)}, {pc(r['img_any_hall']['hi'], 1)}] & "
                        f"{pc(r['coverage'], 1)} \\\\")
    return "\n".join(rows).replace(",", "{,}")


def table_detect(S):
    rows = []
    keys = [("reread", "Re-reading (own transcription)"), ("reread_unc", "Re-reading uncertainty mark (?)"),
            ("xmodel", "Cross-model agreement"), ("ocr_sim", "Tesseract OCR similarity"), ("lex", "Lexicon membership"),
            ("hedged", "Hedging cue in sentence")]
    mods = [m for m in ["haiku", "sonnet", "opus"] if m in S["detection"]]
    for k, nm in keys:
        cells = []
        for mod in mods:
            d = S["detection"][mod]["single"][k]
            cells.append(f"{d['auroc']:.3f} [{d['auroc_ci'][0]:.2f}, {d['auroc_ci'][1]:.2f}]")
        rows.append(nm + " & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    cells = []
    for mod in mods:
        d = S["detection"][mod]["combined"]
        cells.append(f"\\textbf{{{d['auroc']:.3f}}} [{d['auroc_ci'][0]:.2f}, {d['auroc_ci'][1]:.2f}]")
    rows.append("Combined (logistic, grouped 5-fold CV) & " + " & ".join(cells) + " \\\\")
    rows.append("\\quad average precision (base rate) & " + " & ".join(
        f"{S['detection'][m]['combined']['auprc']:.3f} ({pc(S['detection'][m]['combined']['base_rate'], 1)}\\%)" for m in mods) + " \\\\")
    for f_, nm in [("0.05", "5"), ("0.1", "10"), ("0.2", "20")]:
        rows.append(f"\\quad recall at {nm}\\% false flags & " + " & ".join(
            pc(S["detection"][m]["combined"]["operating_points"][f_]["recall"], 1) + "\\%" for m in mods) + " \\\\")
    rows.append("Rule: not re-read, or unconfirmed by OCR and other models & " + " & ".join(
        f"{pc(S['detection'][m]['rule_reread_or_unconfirmed']['recall'], 1)}\\% / {pc(S['detection'][m]['rule_reread_or_unconfirmed']['false_flag'], 1)}\\%" for m in mods) + " \\\\")
    rows.append("Tokens (hallucinated) & " + " & ".join(f"{S['detection'][m]['n']:,} ({S['detection'][m]['pos']})".replace(",", "{,}") for m in mods) + " \\\\")
    return "\n".join(rows)


def table_pairs():
    rows = []
    for ds, nm in [("synthetic", "Study 1 (synthetic)"), ("real", "Study 2 (real)")]:
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{nm}}}}} \\\\")
        for r in REV[ds]["model_pairs"]:
            rows.append(f"{NAMES[r['a']]} vs {NAMES[r['b']]} & {r['n_images']} & {pc(r['img_a'], 1)} / {pc(r['img_b'], 1)} & "
                        f"{r['a_only']}/{r['b_only']} & {p3(r['p_holm'])} & "
                        f"{pc(r['tok_diff'], 2)} [{pc(r['tok_diff_ci'][0], 2)}, {pc(r['tok_diff_ci'][1], 2)}] \\\\")
    return "\n".join(rows)


def table_cost():
    rows = []
    for ds, nm in [("synthetic", "Study 1"), ("real", "Study 2")]:
        for k, v in REV[ds]["cost"].items():
            mod, pr = k.split(".")
            rows.append(f"{nm} & {NAMES[mod]} & {'description' if pr == 'desc' else 'transcription'} & {v['n']} & "
                        f"{v['in_tokens']:,} & {v['out_tokens']:,} & {v['usd_per_image']:.4f} & {v['sec_median']:.1f} ({v['sec_p90']:.1f}) \\\\".replace(",", "{,}"))
    return "\n".join(rows)


def table_stab():
    rows = []
    for mod, v in REV["stability"].items():
        r = v["token_rate_by_run"]
        rows.append(f"{NAMES[mod]} & {v['n_images']} & {pc(v['img_any_by_run']['r1'], 0)} / {pc(v['img_any_by_run']['r2'], 0)} / {pc(v['img_any_by_run']['r3'], 0)} & "
                    f"{pc(v['flip_rate']['p'], 0)} [{pc(v['flip_rate']['lo'], 0)}, {pc(v['flip_rate']['hi'], 0)}] & "
                    f"{pc(r['r1'], 2)} / {pc(r['r2'], 2)} / {pc(r['r3'], 2)} & {pc(v['token_rate_spread'], 2)} & "
                    f"{v['disagreement_auroc']:.2f} \\\\")
    return "\n".join(rows)


def table_breakdown():
    rows = []
    for ds, S_, nm in [("synthetic", SYN, "1"), ("real", REAL, "2")]:
        F = S_["factors"]
        for fac, levels in [("genre", ["signage", "packaging", "document"]), ("quality", ["clean", "blur", "severe"]),
                            ("framing", ["none", "crop", "occlusion"]), ("gt_cat", ["word", "name", "nonce", "lure", "num"])]:
            for lv in levels:
                cells = []
                for mod in ["haiku", "sonnet", "opus", "fable"]:
                    d = F[fac].get(mod, {}).get(lv)
                    cells.append(f"{pc(d['p'], 1)} ({d['n']:,})".replace(",", "{,}") if d and d["n"] else "--")
                if all(c == "--" for c in cells):
                    continue
                lab = {"none": "full frame", "word": "common word", "name": "name / rare word", "nonce": "invented name",
                       "lure": "near-word name", "num": "digits"}.get(lv, lv)
                rows.append(f"{nm} & {fac.replace('gt_cat', 'category')} & {lab} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    return "\n".join(rows[:-1])


TABLES = {"rates": lambda: table_rates(SYN), "realrates": lambda: table_rates(REAL), "detect": lambda: table_detect(SYN),
          "realdetect": lambda: table_detect(REAL), "pairs": table_pairs, "cost": table_cost, "stab": table_stab,
          "breakdown": table_breakdown}


def p3(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def rev_macros():
    """Numbers from results/revision.json and results/audit_raters.json."""
    m = {}
    st = REV.get("stability", {})
    if st:
        m["StabFlipRange"] = rng2([v["flip_rate"]["p"] for v in st.values()], 0)
        m["StabSpreadMax"] = pc(max(v["token_rate_spread"] for v in st.values()), 2)
        m["StabDisAucRange"] = rng2([v["disagreement_auroc"] for v in st.values() if v["disagreement_auroc"]], 2, False)
        m["StabN"] = str(next(iter(st.values()))["n_images"])
        for mod, v in st.items():
            T = mod.capitalize()
            m[f"StabFlip{T}"] = pc(v["flip_rate"]["p"], 0)
            m[f"StabDisAuc{T}"] = f"{v['disagreement_auroc']:.2f}" if v["disagreement_auroc"] else "--"
    sets = REV["sets"]
    for ch, nm in [("ї", "Yi"), ("є", "Ye"), ("ґ", "Ghe")]:
        m[f"Over{nm}"] = f"{sets['synthetic']['letters_invented_over_natural'][ch]:.0f}"
        m[f"Real{nm}Share"] = f"{100 * sets['real']['letters_all'][ch]:.2f}"
        m[f"Nat{nm}Share"] = f"{100 * sets['letters_natural'][ch]:.2f}"
        m[f"Syn{nm}Share"] = f"{100 * sets['synthetic']['letters_all'][ch]:.2f}"
    for ds, pre in [("synthetic", "Syn"), ("real", "Real")]:
        x = sets[ds]
        m[f"{pre}TokLen"] = f"{x['letter_token_len_mean']:.1f}"
        m[f"{pre}TokPerImg"] = f"{x['tokens_per_image']:.0f}"
        m[f"{pre}NameShare"] = pc(x["category_share"].get("name", 0), 0)
        m[f"{pre}NumShare"] = pc(x["category_share"].get("num", 0), 0)
        m[f"{pre}WordShare"] = pc(x["category_share"].get("word", 0), 0)
    # model pairs: count of Holm-significant image-level differences
    for ds, pre in [("synthetic", "Syn"), ("real", "Real")]:
        mp = REV[ds]["model_pairs"]
        m[f"{pre}NPairsSig"] = str(sum(r["p_holm"] < 0.05 for r in mp))
        m[f"{pre}NPairs"] = str(len(mp))
        m[f"{pre}NTokSig"] = str(sum(r["tok_diff_ci"][0] > 0 or r["tok_diff_ci"][1] < 0 for r in mp))
    if AUD:
        for ds, pre in [("synthetic", "Syn"), ("real", "Real")]:
            a = AUD.get(ds)
            if not a:
                continue
            m[f"{pre}AudN"] = str(sum(v["n"] for k, v in a["resolved"].items() if k != "hallucination"))
            m[f"{pre}AudKappa"] = f"{a['kappa']:.2f}" if a.get("kappa") is not None else "--"
            m[f"{pre}AudAgree"] = pc(a["raw_agreement"], 0)
            m[f"{pre}AudNBoth"] = str(a["n_both"])
            for k, nm in [("hallucination", "Hall"), ("correct", "Corr"), ("completion", "Compl"), ("misread", "Mis"), ("fabricated", "Fab")]:
                r = a["resolved"][k]
                m[f"{pre}Aud{nm}"] = pc(r["p"], 0) if r["p"] is not None else "--"
                m[f"{pre}Aud{nm}CI"] = f"{pc(r['lo'], 0)}--{pc(r['hi'], 0)}" if r["p"] is not None else "--"
                for rater, rn in [("claude", "Cl"), ("gemma", "Gm")]:
                    rr = a[rater][k]
                    m[f"{pre}Aud{nm}{rn}"] = pc(rr["p"], 0) if rr["p"] is not None else "--"
        gm = AUD.get("gemma_meta") or {}
        m["OllamaVersion"] = str(gm.get("ollama_version", "--"))
    return m


def rng2(vals, nd=1, pct=True):
    lo, hi = min(vals), max(vals)
    f = (lambda v: pc(v, nd)) if pct else (lambda v: f"{v:.{nd}f}")
    return f"{f(lo)}--{f(hi)}" if f(lo) != f(hi) else f(lo)


def main():
    tpl = open(os.path.join(P, "template_main.tex"), encoding="utf-8").read()
    body = "\n\n".join(open(os.path.join(P, b + ".tex"), encoding="utf-8").read()
                       for b in ORDER if os.path.exists(os.path.join(P, b + ".tex")))
    app = open(os.path.join(P, APPENDIX + ".tex"), encoding="utf-8").read()
    body = body.replace("\\bibliography{refs}", app + "\n\\bibliography{refs}")
    global S, DATA
    S, DATA = SYN, os.path.join(ROOT, "data")
    mac = macros()
    S, DATA = REAL, os.path.join(ROOT, "data", "real")
    real = macros()
    shared = [k for k in real if k.startswith(("NSign", "NPack", "NDoc", "NProbe", "Pr", "RangePr", "SpendTotal"))]
    mac.update({k: real[k] for k in shared})
    mac.update({"Real" + k: v for k, v in real.items() if k not in shared})
    mac.update(rev_macros())
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)(?![A-Za-z])", body)) - {"LaTeX", "TeX", "Large", "Huge", "LARGE"}
    missing = sorted(u for u in used if u not in mac and u not in ("LaTeX", "TeX"))
    if missing:
        print("WARNING undefined number macros:", missing)
    defs = "%% reported numbers, generated from results/summary.json by scripts/08_build_paper.py\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(mac.items()) if k in used)
    for tag, fn in TABLES.items():
        body = body.replace(f"%TABLE:{tag}%", fn())
    out = tpl.replace("%BODY%", body).replace("\\begin{document}", defs + "\n\\begin{document}", 1)
    out = out.replace("\\bibliography{sn-bibliography}", "\\bibliography{refs}")
    if "---" in re.sub(r"%.*", "", out):
        print("WARNING: em-dash (---) present in manuscript")
    tex = os.path.join(P, "universal-access-in-2026.tex")
    open(tex, "w", encoding="utf-8").write(out)
    r = subprocess.run([os.path.expanduser("~/.local/bin/tectonic"), "-k", os.path.basename(tex)], cwd=P,
                       capture_output=True, text=True)
    warn = [l for l in (r.stdout + r.stderr).splitlines() if re.search(r"error|undefined|Citation|multiply", l, re.I)]
    print("\n".join(warn[:40]))
    print("tectonic exit", r.returncode)


if __name__ == "__main__":
    main()
