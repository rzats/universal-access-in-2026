"""Run Tesseract 5 (tessdata_best 'ukr') on every stimulus as an independent reading.

Two page-segmentation modes are run (3 = automatic, 11 = sparse text) and both outputs
are kept. Output: runs/ocr.jsonl {id, psm3, psm11, secs}; resumable.
"""
import concurrent.futures as cf, json, os, subprocess, time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TESS = os.path.join(ROOT, "data", "assets", "tess")
ENV = dict(os.environ, TESSDATA_PREFIX=os.path.join(TESS, "tessdata"), OMP_THREAD_LIMIT="1")


def ocr(path, psm):
    p = subprocess.run([os.path.join(TESS, "squashfs-root", "AppRun"), path, "-", "-l", "ukr", "--psm", str(psm)],
                       capture_output=True, text=True, env=ENV, timeout=300)
    return p.stdout


def main():
    stim = [json.loads(l)["id"] for l in open(os.path.join(ROOT, "data", "stimuli.jsonl"), encoding="utf-8")]
    out = os.path.join(ROOT, "runs", "ocr.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    todo = [s for s in stim if s not in done]
    print(len(done), "done,", len(todo), "to do")

    def work(sid):
        t0 = time.time()
        img = os.path.join(ROOT, "data", "images", sid + ".jpg")
        return {"id": sid, "psm3": ocr(img, 3), "psm11": ocr(img, 11), "secs": round(time.time() - t0, 1)}

    with cf.ThreadPoolExecutor(2) as ex, open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(ex.map(work, todo)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(i + 1, flush=True)


if __name__ == "__main__":
    main()
