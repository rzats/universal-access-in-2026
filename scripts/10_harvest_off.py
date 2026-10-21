"""Harvest real Ukrainian packaging photographs from Open Food Facts (OFF).

For every Ukrainian product with a Ukrainian ingredient transcription we take the raw
photographs selected as its Ukrainian front and ingredient panels, together with the
Google Cloud Vision OCR that OFF stores next to every raw photograph (word and symbol
boxes). The OCR is the ground truth; the contributor's ingredient transcription is an
independent human reading used as a quality gate (ingredient panels whose OCR does not
agree with the transcription are discarded).

Output: data/off/<code>_<imgid>.jpg (long side <= 1600 px), data/off/<code>_<imgid>.ocr.json,
data/off/candidates.jsonl. Resumable; polite rate limiting.
"""
import io, json, os, re, sys, time
import requests
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from textnorm import toks

UA = {"User-Agent": "uais-research/0.1 (zatserkovnyi.rostyslav@gmail.com)"}
OUT = os.path.join(ROOT, "data", "off")
UKR = re.compile(r"[іїєґ]")


import threading
_lock, _last = threading.Lock(), [0.0]


def get(url, **kw):
    if "world.openfoodfacts.org" in url:  # product API: stay under 100 requests per minute
        with _lock:
            wait = _last[0] + 0.7 - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
    for a in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(60)
        except requests.RequestException:
            pass
        time.sleep(5 * (a + 1))
    return None


def path_of(code):
    return "/".join([code[0:3], code[3:6], code[6:9], code[9:]]) if len(code) > 8 else code


def process(code, done):
    """Fetch one product; return candidate records for its Ukrainian front/ingredient photos."""
    recs = []
    pr = get(f"https://world.openfoodfacts.org/api/v2/product/{code}.json",
             params={"fields": "code,brands,product_name_uk,ingredients_text_uk,images"})
    if pr is None or "product" not in pr.json():
        return recs
    p = pr.json()["product"]
    if len(p.get("ingredients_text_uk") or "") < 40:
        return recs
    imgs = p.get("images", {})
    for panel in ["front_uk", "ingredients_uk"]:
        if panel not in imgs:
            continue
        imgid = str(imgs[panel].get("imgid"))
        key = f"{code}_{imgid}"
        if key in done:
            continue
        base = f"https://images.openfoodfacts.org/images/products/{path_of(code)}/{imgid}"
        rec = {"key": key, "code": code, "imgid": imgid, "panel": panel.split("_")[0],
               "brands": p.get("brands"), "product_name_uk": p.get("product_name_uk"),
               "ingredients_text_uk": p.get("ingredients_text_uk"), "ok": False}
        oj = get(base + ".json")
        try:
            fta = oj.json()["responses"][0]["fullTextAnnotation"]
        except Exception:
            rec["why"] = "no ocr"
            recs.append(rec)
            continue
        t = toks(fta["text"])
        letters = [x for x in t if not x.isdigit()]
        cyr = sum(bool(re.search(r"[а-яіїєґ]", x)) for x in letters) / max(1, len(letters))
        ukr = sum(bool(UKR.search(x)) for x in letters)
        tr = set(toks(p.get("ingredients_text_uk") or ""))
        agree = len(tr & set(t)) / max(1, len(tr))
        rec.update({"n_ocr_tokens": len(t), "cyr_share": round(cyr, 3), "n_ukr_letter_tokens": ukr,
                    "transcription_agreement": round(agree, 3)})
        ok = 20 <= len(t) <= 320 and cyr >= 0.8 and ukr >= 3 and (panel == "front_uk" or agree >= 0.6)
        if ok:
            im = get(base + ".jpg")
            if im is None:
                rec["why"] = "no image"
                ok = False
            else:
                img = Image.open(io.BytesIO(im.content)).convert("RGB")
                rec["orig_size"] = img.size
                s = min(1.0, 1600 / max(img.size))
                if s < 1:
                    img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.LANCZOS)
                rec["scale"] = s
                img.save(os.path.join(OUT, key + ".jpg"), quality=92)
                json.dump(fta, open(os.path.join(OUT, key + ".ocr.json"), "w", encoding="utf-8"), ensure_ascii=False)
                rec["url"] = base + ".jpg"
        rec["ok"] = ok
        recs.append(rec)
    return recs


def main():
    import concurrent.futures as cf
    os.makedirs(OUT, exist_ok=True)
    cand_path = os.path.join(OUT, "candidates.jsonl")
    prev = [json.loads(l) for l in open(cand_path, encoding="utf-8")] if os.path.exists(cand_path) else []
    done = {r["key"] for r in prev}
    seen = {r["code"] for r in prev}
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    n_ok = sum(r["ok"] for r in prev)
    out = open(cand_path, "a", encoding="utf-8")
    page = 1
    with cf.ThreadPoolExecutor(4) as ex:
        while n_ok < target and page <= 40:
            r = get("https://search.openfoodfacts.org/search",
                    params={"q": 'countries_tags:"en:ukraine"', "page_size": 100, "page": page, "fields": "code"})
            page += 1
            if r is None:
                continue
            codes = [h["code"] for h in r.json().get("hits", []) if h.get("code") and h["code"] not in seen]
            seen.update(codes)
            for recs in ex.map(lambda c: process(c, done), codes):
                for rec in recs:
                    n_ok += rec["ok"]
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            print("page", page - 1, "ok", n_ok, flush=True)
    print("done; ok candidates:", n_ok)


if __name__ == "__main__":
    main()
