"""Automated audit of the deterministic labeller by two independent automated raters.

Sample: stratified random sample of description tokens, per dataset (synthetic, real),
per model and per label. Each rater sees the stimulus image, the printed reference line
the token was matched to, the quoted span it came from, the extracted token and the
label with its definition, and answers whether the label is right.

Raters
  claude  Claude Sonnet 5.5 via the Claude Code CLI (same path as the evaluated runs)
  gemma   gemma3:12b served by Ollama (local), temperature 0, one request at a time

Usage:  python3 scripts/15_audit_raters.py sample synthetic|real   # (re)writes that part of runs/audit/sample.jsonl
        python3 scripts/15_audit_raters.py claude|gemma # resumable rating passes
        python3 scripts/15_audit_raters.py analyse     # results/audit_raters.json
"""
import base64, collections, json, math, os, random, re, subprocess, sys, time
import requests

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from scoring import quoted_spans
from textnorm import tokens

OUT = os.path.join(ROOT, "runs", "audit")
OLLAMA = "http://127.0.0.1:11434"
GEMMA = "gemma3:12b"
CLAUDE = "claude-sonnet-5-5"
LABELS = ["correct", "completion", "misread", "fabricated"]
DEFS = {
    "correct": "the token matches text that is printed AND visible in the image (an explicit fragment of a visible word also counts)",
    "completion": "the token matches a printed word that is partly or wholly cut off by the frame edge or covered (by a finger or glare), so the model completed text it could not see",
    "misread": "the token is a wrong reading of a printed word (wrong letters or digits, a different word form, a Russian-like spelling, a 'corrected' spelling)",
    "fabricated": "the token does not correspond to any printed text in the image; the model invented it",
}
PROMPT = """You are auditing an automatic scorer of image descriptions written for blind users.
A vision-language model described the attached image and quoted text it claimed to read.
The automatic scorer compared one quoted word with the reference text of the image and assigned a label.

Reference line the word was matched to (printed text, ground truth): «{line}»
Quoted span from the description: «{span}»
Quoted word being scored (normalised to lower case): «{tok}»
Label assigned by the scorer: {label}
Definition of this label: {definition}

Look at the image and the reference, then answer two questions.
1. Is the assigned label right for this word?
2. Does the quoted word reproduce text that is printed AND fully visible in the image (ignoring capitalisation and punctuation)? Answer "yes" if so; answer "no" if the word is not printed, is printed differently, or is printed but cut off or covered.
Answer with JSON only:
{{"verdict": "right" or "wrong", "printed": "yes" or "no", "reason": "<at most 15 words>"}}"""


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def sample(which):
    rng = random.Random(f"20261028-{which}")
    rows = []
    for ds, runs, data, per in [("synthetic", "runs", "data", 25), ("real", os.path.join("runs", "real"), os.path.join("data", "real"), 15)]:
        if ds != which:
            continue
        C = [c for c in load(os.path.join(ROOT, runs, "claims.jsonl")) if c["prompt"] == "desc" and c["label"] in LABELS]
        stim = {s["id"]: s for s in load(os.path.join(ROOT, data, "stimuli.jsonl"))}
        resp = {}
        for m in ["haiku", "sonnet", "opus", "fable"]:
            for r in load(os.path.join(ROOT, runs, "vlm", f"{m}.desc.jsonl")):
                resp[(m, r["id"])] = r["result"]
        for m in ["haiku", "sonnet", "opus", "fable"]:
            for lab in LABELS:
                xs = [c for c in C if c["model"] == m and c["label"] == lab]
                for c in rng.sample(xs, min(per, len(xs))):
                    s = stim[c["id"]]
                    _, sp = quoted_spans(resp[(m, c["id"])])
                    span = sp[c["span"]][0] if c["span"] < len(sp) else ""
                    line = ""
                    for g in s["tokens"]:
                        if g["tok"] == c["gt"] and g.get("line", -1) >= 0:
                            line = s["lines"][g["line"]]["raw"]
                            break
                    rows.append({"key": f"{ds}:{m}:{c['id']}:{c['span']}:{c['tok']}", "dataset": ds, "model": m,
                                 "id": c["id"], "image": os.path.join(data, "images", c["id"] + ".jpg"),
                                 "tok": c["tok"], "label": lab, "gt": c["gt"], "line": line, "span": span,
                                 "genre": c["genre"], "quality": c["quality"], "framing": c["framing"]})
    seen, out = set(), [r for r in load(os.path.join(OUT, "sample.jsonl")) if r["dataset"] != which]
    seen = {r["key"] for r in out}
    for r in rows:
        if r["key"] not in seen:
            seen.add(r["key"]); out.append(r)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sample.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("sample:", len(out), collections.Counter((r["dataset"], r["label"]) for r in out))


def prompt_of(r):
    return PROMPT.format(line=r["line"] or "(no reference line: the word matched nothing printed)", span=r["span"],
                         tok=r["tok"], label=r["label"], definition=DEFS[r["label"]])


def parse(text):
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        d = json.loads(m.group(0))
        v = str(d.get("verdict", "")).lower()
        pr = str(d.get("printed", "")).lower()
        return ("right" if v.startswith("right") else "wrong" if v.startswith("wrong") else None), \
            ("yes" if pr.startswith("y") else "no" if pr.startswith("n") else None), d.get("reason", "")
    except Exception:
        return None, None, text[:100]


def rate(rater):
    S = load(os.path.join(OUT, "sample.jsonl"))
    path = os.path.join(OUT, f"rater_{rater}.jsonl")
    done = {r["key"] for r in load(path)}
    imgdir = os.path.join(OUT, "img")
    os.makedirs(imgdir, exist_ok=True)
    meta = {}
    if rater == "gemma":
        meta = {"ollama_version": requests.get(OLLAMA + "/api/version", timeout=10).json().get("version"),
                "model": GEMMA, "options": {"temperature": 0, "seed": 7, "num_predict": 120}}
    with open(path, "a", encoding="utf-8") as f:
        for i, r in enumerate(S):
            if r["key"] in done:
                continue
            p = prompt_of(r)
            t0 = time.time()
            for attempt in range(4):
                try:
                    if rater == "gemma":
                        b64 = base64.b64encode(open(os.path.join(ROOT, r["image"]), "rb").read()).decode()
                        resp = requests.post(OLLAMA + "/api/chat", timeout=600, json={
                            "model": GEMMA, "stream": False, "options": meta["options"],
                            "messages": [{"role": "user", "content": p, "images": [b64]}]}).json()
                        text, cost = resp["message"]["content"], 0.0
                    else:
                        name = f"a{i:04d}.jpg"
                        dst = os.path.join(imgdir, name)
                        if not os.path.exists(dst):
                            with open(os.path.join(ROOT, r["image"]), "rb") as a, open(dst, "wb") as b:
                                b.write(a.read())
                        proc = subprocess.run(["claude", "-p", p + f"\n\n@{name}", "--model", CLAUDE, "--output-format", "json",
                                               "--tools", "", "--setting-sources", "", "--no-session-persistence"],
                                              cwd=imgdir, capture_output=True, text=True, timeout=300)
                        d = json.loads(proc.stdout)
                        text, cost = d["result"], d.get("total_cost_usd", 0.0)
                    break
                except Exception as e:
                    print("retry", r["key"], str(e)[:120], flush=True)
                    time.sleep(15 * (attempt + 1))
            else:
                if rater == "gemma":
                    print("server unreachable; stopping (resumable)", flush=True)
                    return
                continue
            v, printed, reason = parse(text)
            f.write(json.dumps({"key": r["key"], "verdict": v, "printed": printed, "reason": reason, "raw": text[:400], "cost": cost,
                                "secs": round(time.time() - t0, 1), **({"meta": meta} if meta else {})}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(rater, i + 1, flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return {"k": 0, "n": 0, "p": None, "lo": None, "hi": None}
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "p": round(p, 4), "lo": round(max(0, c - h), 4), "hi": round(min(1, c + h), 4)}


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


def analyse():
    S = {r["key"]: r for r in load(os.path.join(OUT, "sample.jsonl"))}
    # drop items whose label changed after sampling (look-alike folding fix applied after the draw)
    from textnorm import _fix_script
    cur = {}
    for ds, runs in [("synthetic", "runs"), ("real", os.path.join("runs", "real"))]:
        for c in load(os.path.join(ROOT, runs, "claims.jsonl")):
            if c["prompt"] == "desc":
                cur[f"{ds}:{c['model']}:{c['id']}:{c['span']}:{c['tok']}"] = c["label"]
    stale = [k for k, r in S.items() if cur.get(":".join(k.split(":")[:4] + [_fix_script(r["tok"])])) != r["label"]]
    S = {k: r for k, r in S.items() if k not in stale}
    R = {name: {r["key"]: r for r in load(os.path.join(OUT, f"rater_{name}.jsonl")) if r.get("printed")}
         for name in ["claude", "gemma"]}
    res = {"raters": {"claude": CLAUDE, "gemma": GEMMA}, "n_sample": len(S), "n_stale_dropped": len(stale),
           "primary": "for misread/fabricated labels: the rater judges the quoted word NOT printed-and-visible; "
                      "for correct labels: judged printed-and-visible; for completions: judged not printed-and-visible",
           "resolution": "an item is confirmed only if every rater that rated it confirms it (conservative); "
                         "items rated by neither rater are excluded"}
    gm = load(os.path.join(OUT, "rater_gemma.jsonl"))
    if gm:
        res["gemma_meta"] = gm[0].get("meta")

    def ok(lab, rr):  # does the rater confirm the label's claim about printed text?
        return (rr["printed"] == "yes") if lab == "correct" else (rr["printed"] == "no")

    for ds in ["synthetic", "real"]:
        out = {}
        keys = [k for k, r in S.items() if r["dataset"] == ds]
        for name in ["claude", "gemma"]:
            got = [k for k in keys if k in R[name]]
            out[name] = {lab: wilson(sum(ok(lab, R[name][k]) for k in got if S[k]["label"] == lab),
                                     sum(1 for k in got if S[k]["label"] == lab)) for lab in LABELS}
            hk = [k for k in got if S[k]["label"] in ("misread", "fabricated")]
            out[name]["hallucination"] = wilson(sum(ok(S[k]["label"], R[name][k]) for k in hk), len(hk))
            out[name]["exact_label"] = {lab: wilson(sum(R[name][k]["verdict"] == "right" for k in got if S[k]["label"] == lab),
                                                    sum(1 for k in got if S[k]["label"] == lab)) for lab in LABELS}
        both = [k for k in keys if k in R["claude"] and k in R["gemma"]]
        out["n_both"] = len(both)
        out["kappa"] = kappa([R["claude"][k]["printed"] == "yes" for k in both], [R["gemma"][k]["printed"] == "yes" for k in both])
        out["raw_agreement"] = round(sum(R["claude"][k]["printed"] == R["gemma"][k]["printed"] for k in both) / max(1, len(both)), 4)
        out["kappa_exact_label"] = kappa([R["claude"][k]["verdict"] == "right" for k in both], [R["gemma"][k]["verdict"] == "right" for k in both])
        rated = [k for k in keys if k in R["claude"] or k in R["gemma"]]
        comb = {k: all(ok(S[k]["label"], R[n][k]) for n in ["claude", "gemma"] if k in R[n]) for k in rated}
        out["resolved"] = {lab: wilson(sum(comb[k] for k in rated if S[k]["label"] == lab),
                                       sum(1 for k in rated if S[k]["label"] == lab)) for lab in LABELS}
        hk = [k for k in rated if S[k]["label"] in ("misread", "fabricated")]
        out["resolved"]["hallucination"] = wilson(sum(comb[k] for k in hk), len(hk))
        res[ds] = out
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "audit_raters.json"), "w"), indent=1)
    print(json.dumps({ds: {k: v for k, v in res[ds].items() if k in ("kappa", "raw_agreement", "n_both")} for ds in ["synthetic", "real"]}))
    for ds in ["synthetic", "real"]:
        print(ds, {lab: res[ds]["resolved"][lab]["p"] for lab in LABELS + ["hallucination"]})


if __name__ == "__main__":
    {"sample": lambda: sample(sys.argv[2]), "claude": lambda: rate("claude"), "gemma": lambda: rate("gemma"), "analyse": analyse}[sys.argv[1]]()
