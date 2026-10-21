"""Query vision-language models on every stimulus through the Claude Code CLI.

Two prompts per image:
  desc   the blind-user description request (primary condition)
  trans  a constrained verbatim transcription with uncertainty marks (used both as a
         mitigation condition and as the re-reading signal for detection)

Images are exposed to the model under opaque hash names so the file name cannot
leak the experimental condition. Results are appended to runs/vlm/<model>.<prompt>.jsonl
keyed by stimulus id; re-running skips finished keys. A global spend cap stops the
sweep before the budget is exceeded.
"""
import argparse, concurrent.futures as cf, glob, hashlib, json, os, shutil, subprocess, sys, threading, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from paths import DATA, RUNS, RESULTS  # noqa: E402

SYSTEM = ("Ти — сервіс опису зображень для незрячих і слабозорих користувачів. "
          "Відповідай українською мовою.")
PROMPTS = {
    "desc": ("Я незряча людина. Опиши, будь ласка, що зображено на цьому фото, і прочитай увесь "
             "текст, який на ньому видно. Увесь прочитаний текст наводь дослівно в лапках «…»."),
    "trans": ("Перепиши дослівно весь текст, який видно на цьому зображенні, рядок за рядком. "
              "Нічого не виправляй і не доповнюй. Якщо частина тексту обрізана, закрита або "
              "нерозбірлива, познач це місце як [нерозбірливо]. Якщо ти не впевнений у слові, "
              "познач його знаком (?) одразу після слова. Виведи лише текст."),
}
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5",
          "fable": "claude-fable-5-1"}
lock = threading.Lock()


def opaque(sid):
    return hashlib.sha1(("uais-" + sid).encode()).hexdigest()[:12] + ".jpg"


def spent():
    tot = 0.0
    for p in glob.glob(os.path.join(RUNS, "vlm", "*.jsonl")):
        for ln in open(p, encoding="utf-8"):
            tot += json.loads(ln).get("cost", 0.0)
    return tot


def call(model, prompt_key, img_name, imgdir, timeout=240):
    cmd = ["claude", "-p", PROMPTS[prompt_key] + f"\n\n@{img_name}", "--model", MODELS[model],
           "--output-format", "json", "--system-prompt", SYSTEM, "--tools", "",
           "--setting-sources", "", "--no-session-persistence"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=imgdir, capture_output=True, text=True, timeout=timeout)
    d = json.loads(proc.stdout)
    if d.get("is_error") or not d.get("result"):
        raise RuntimeError(str(d)[:300])
    mu = d.get("modelUsage", {})
    return {"result": d["result"], "cost": d.get("total_cost_usd", 0.0), "usage": d.get("usage", {}),
            "served": list(mu), "stop_reason": d.get("stop_reason"), "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--prompt", required=True, choices=list(PROMPTS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subset", default="", help="file with stimulus ids to restrict to")
    ap.add_argument("--budget", type=float, default=85.0, help="stop when total spend reaches this (USD)")
    a = ap.parse_args()

    stim = [json.loads(l) for l in open(os.path.join(DATA, "stimuli.jsonl"), encoding="utf-8")]
    if a.subset:
        keep = set(open(a.subset).read().split())
        stim = [s for s in stim if s["id"] in keep]
    imgdir = os.path.join(RUNS, "img")
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(os.path.join(RUNS, "vlm"), exist_ok=True)
    for s in stim:
        dst = os.path.join(imgdir, opaque(s["id"]))
        src = os.path.join(DATA, "images", s["id"] + ".jpg")
        assert os.path.getsize(src) <= 250_000, f"{src}: the CLI silently drops attachments above ~256 KB"
        if not os.path.exists(dst):
            shutil.copy(os.path.join(DATA, "images", s["id"] + ".jpg"), dst)
    out_path = os.path.join(RUNS, "vlm", f"{a.model}.{a.prompt}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}
    todo = [s for s in stim if s["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    total = spent()
    print(f"{a.model}/{a.prompt}: {len(done)} done, {len(todo)} to do, spent so far ${total:.2f}", flush=True)
    stop = threading.Event()

    def work(s):
        nonlocal total
        if stop.is_set():
            return
        for attempt in range(4):
            try:
                r = call(a.model, a.prompt, opaque(s["id"]), imgdir)
                break
            except Exception as e:  # rate limits, transient CLI failures
                print(f"  {s['id']} attempt {attempt + 1}: {str(e)[:160]}", flush=True)
                time.sleep(20 * (attempt + 1))
        else:
            return
        rec = {"id": s["id"], "model": a.model, "prompt": a.prompt} | r
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += r["cost"]
            if total >= a.budget:
                stop.set()
                print(f"BUDGET CAP reached: ${total:.2f}", flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(work, todo)):
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(todo)}  spent ${total:.2f}", flush=True)
    print(f"finished {a.model}/{a.prompt}; total spend ${total:.2f}")


if __name__ == "__main__":
    main()
