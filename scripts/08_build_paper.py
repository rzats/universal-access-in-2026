"""Assemble the single-file manuscript from the Springer Nature template and the
section sources, define every reported number as a macro computed from
results/summary.json, and compile with tectonic.

Sources:  paper/template_main.tex (official sn-article template, body replaced by %BODY%)
          paper/body_*.tex       (sections, in the order listed below)
Output:   paper/universal-access-in-2026.tex and .pdf
"""
import json, os, re, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = os.path.join(ROOT, "paper")
ORDER = ["body_title", "body_front", "body_method", "body_results", "body_discussion", "body_back"]
S = json.load(open(os.path.join(ROOT, "results", "summary.json"), encoding="utf-8"))
A = json.load(open(os.path.join(ROOT, "results", "audit.json"), encoding="utf-8")) if os.path.exists(os.path.join(ROOT, "results", "audit.json")) else None


def pc(x, nd=1):
    return f"{100 * x:.{nd}f}"


def ci(d, nd=1):
    return f"{pc(d['p'], nd)}\\% [{pc(d['lo'], nd)}, {pc(d['hi'], nd)}]"


def macros():
    m = {}
    stim = [json.loads(l) for l in open(os.path.join(ROOT, "data", "stimuli.jsonl"), encoding="utf-8")]
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
        if L:
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
            m[f"McP{T}"] = f"{d['p_mcnemar']:.3f}" if d["p_mcnemar"] >= 0.001 else "<0.001"
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
        m["AuditCorrPrec"] = pc(A["correct_precision"]["p"], 0)
        m["AuditComplPrec"] = pc(A["completion_precision"]["p"], 0)
        m["AuditNHall"] = str(A["hall_precision"]["n"])
        m["AuditNCorr"] = str(A["correct_precision"]["n"])
        m["AuditNCompl"] = str(A["completion_precision"]["n"])
    m["SpendTotal"] = f"{S.get('spend_usd', 0):.0f}"
    return m


NAMES = {"haiku": "Haiku 5.5", "sonnet": "Sonnet 5.5", "opus": "Opus 5.5", "fable": "Fable 5.1"}


def table_rates():
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


def table_detect():
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


def main():
    tpl = open(os.path.join(P, "template_main.tex"), encoding="utf-8").read()
    body = "\n\n".join(open(os.path.join(P, b + ".tex"), encoding="utf-8").read()
                       for b in ORDER if os.path.exists(os.path.join(P, b + ".tex")))
    mac = macros()
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", body))
    missing = sorted(u for u in used if u not in mac and u not in ("LaTeX", "TeX"))
    if missing:
        print("WARNING undefined number macros:", missing)
    defs = "%% reported numbers, generated from results/summary.json by scripts/08_build_paper.py\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(mac.items()) if k in used)
    body = body.replace("%TABLE:rates%", table_rates()).replace("%TABLE:detect%", table_detect())
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
