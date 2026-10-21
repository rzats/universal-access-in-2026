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
    return m


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
